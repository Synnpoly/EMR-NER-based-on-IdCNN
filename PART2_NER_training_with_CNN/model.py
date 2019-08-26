# encoding = utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from utils import result_to_json
from data_utils import create_input, iobes_iob, iob_iobes


# 双向LSTM或Id-CNN模型，找到x,y. y是双标签，x是文字word2vec映射成的词向量。
# 如何拟合x.y:拟合之前第一步提取x的特征，用BiLstm或idCNN对x做特征提取，+分类器（crf条件随机场）
# BiLSTM or idCNN + crf
# idCNN与cnn的区别是，idCNN的卷积核是扁的：找一句话之间的关系可以用扁的，
# 好处：可以有效地抗噪音：完形填空时，扁的卷积核它只会扫当前这句话，不会把上下文卷进来，抗的是上下文的躁
# CNN和RNN本质上没有太大差别，都是把局部的相关性体现出来，CNN体现在空间上，RNN体现在时间时序上

# crf：条件随机场。跟RNN类似，提供了一个分类结果，当然它也可以做特征提取。它的分类需要算一个联合概率
# 第一步，找到x,y
# 第二步，对x做特征提取、特征工程（之前所有的resnet等都是为特征工程服务的），对y做one_hot向量（或二分类）
# 第三步，去拟合，分类


class Model(object):
    # 用配置参数初始化模型
    def __init__(self, config):

        self.config = config

        self.lr = config["lr"]
        self.char_dim = config["char_dim"]  # 100
        self.lstm_dim = config["lstm_dim"]  # LSTM 神经元个数
        self.seg_dim = config["seg_dim"]  # 增加维度

        self.num_tags = config["num_tags"]  # 样本中tag个数
        self.num_chars = config["num_chars"]  # 样本中总字数
        self.num_segs = 4  # 0，1，2，3。0是不需要的字，1是第一个，2是中间的，3是最后一个

        # trainable=False 为不可训练参数
        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        # 神经网络类型，这里选用ID-CNN
        self.model_type = config['model_type']
        # ID-CNN神经网络层数
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        self.filter_width = 3
        self.num_filter = self.lstm_dim
        self.embedding_dim = self.char_dim + self.seg_dim
        self.repeat_times = 4
        self.cnn_output_width = 0

        # embeddings for chinese character and segmentation representation
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)
        # BiLSTM的参数
        if self.model_type == 'bilstm':
            # apply dropout before feed to lstm layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # bi-directional lstm layer
            model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)

            # logits for tags
            self.logits = self.project_layer_bilstm(model_outputs)
        # ID-CNN的参数
        elif self.model_type == 'idcnn':
            # apply dropout before feed to idcnn layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # ldcnn layer
            model_outputs = self.IDCNN_layer(model_inputs)

            # logits for tags
            self.logits = self.project_layer_idcnn(model_outputs)

        else:
            raise KeyError

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        # 优化方法self.opt：
        # SGD
        # Adam
        # AdaGrad
        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # 用梯度裁剪防止梯度爆炸，参数在config['clip]，将梯度控制在±clip范围内(这里是[-5,5]。)
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # 保存模型
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: 每个字用one-hot(独热)encode
        :param seg_inputs: 切词特征表示
        :param config: 是否使用切词特征表示
        :return: [1, num_steps, embedding size],
        """
        # 高:3 血:22 糖:23 和:24 高:3 血:22 压:25 char_inputs=[3,22,23,24,3,22,25]
        # 高血糖 和 高血压 seg_inputs 高血糖=[1,2,3] 和=[0] 高血压=[1,2,3]  seg_inputs=[1,2,3,0,1,2,3]
        embedding = []
        self.char_inputs_test = char_inputs
        self.seg_inputs_test = seg_inputs
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                name="char_embedding",
                shape=[self.num_chars, self.char_dim],
                initializer=self.initializer)
            # 输入char_inputs='常' 对应的字典的索引/编号/value为：8
            # self.char_lookup=[2677*100]的向量，char_inputs字对应在字典的索引/编号/key=[1]
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            # self.embedding1.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        # shape=[4*20]
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        self.embed_test = embed
        self.embedding_test = embedding
        return embed

    def biLSTM_layer(self, model_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                model_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    # ID-CNN layer
    def IDCNN_layer(self, model_inputs,
                    name=None):
        """
        :param model_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, cnn_output_width]
        """
        # tf.expand_dims会向tensor中插入一个维度，插入位置就是参数代表的位置（维度从0开始）。
        model_inputs = tf.expand_dims(model_inputs, 1)
        self.model_inputs_test = model_inputs
        reuse = False
        if self.dropout == 1.0:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            # shape=[1,3,120,100] 分别为：卷积核高度，宽度，输入数据channel数，卷积核个数
            # shape=[1, self.filter_width, self.embedding_dim,
            #            self.num_filter]
            # print(shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)

            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
            """
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer", use_cudnn_on_gpu=False)
            self.layerInput_test = layerInput
            finalOutFromLayers = []

            totalWidthForLastDim = 0
            # 4次卷积，尽量确保所有数据都能参与卷积运算。每轮3次卷积核膨胀，膨胀率分别为1,1,2.
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    # 1,1,2
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True
                                           if (reuse or j > 0) else False):
                        # w权重 其实就是卷积核参数：卷积核的高度，卷积核的宽度，图像通道数，卷积核个数
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        if j == 1 and i == 1:
                            self.w_test_1 = w
                        if j == 2 and i == 1:
                            self.w_test_2 = w
                        # b偏置
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        # tf.nn.atrous_conv2d(value,filters,rate,padding,name=None）
                        # 除去name参数用以指定该操作的name，与方法有关的一共四个参数：
                        # value：指需要做卷积的输入图像，要求是一个4维Tensor，具有[batch, height, width, channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
                        # filters：相当于CNN中的卷积核，要求是一个4维Tensor，具有[filter_height, filter_width, channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，同理这里第三维channels，就是参数value的第四维
                        # rate：
                        # 要求是一个int型的正数，正常的卷积操作应该会有stride（即卷积核的滑动步长），但是空洞卷积是没有stride参数的，
                        # 这一点尤其要注意。取而代之，它使用了新的rate参数，那么rate参数有什么用呢？它定义为我们在输入
                        # 图像上卷积时的采样间隔，可以理解为卷积核当中穿插了（rate-1）数量的“0”，
                        # 把原来的卷积核插出了很多“洞洞”，这样做卷积时就相当于对原图像的采样间隔变大了。
                        # 具体怎么插得，可以看后面更加详细的描述。此时我们很容易得出rate=1时，就没有0插入，
                        # 此时这个函数就变成了普通卷积。
                        # padding：填充值
                        # string类型的量，只能是”SAME”,”VALID”其中之一，这个值决定了不同边缘填充方式。
                        # 这里还有一个超参，卷积核滑动步长stride，默认为1.
                        # 结果返回一个Tensor，填充方式为“VALID”时，返回[batch,height-2*(filter_width-1),width-2*(filter_height-1),out_channels]的Tensor，填充方式为“SAME”时，返回[batch, height, width, out_channels]的Tensor #todo
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        self.conv_test = conv
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            # 在第4(axis=3)个维度也就是卷积核个数上拼接tensor，
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)
            # 从tensor中删除所有大小是1的维度
            # 给定张量输入，此操作返回相同类型的张量，并删除所有尺寸为1的尺寸。 如果不想删除所有尺寸1尺寸，可以通过指定squeeze_dims来删除特定尺寸1尺寸。
            finalOut = tf.squeeze(finalOut, [1])
            # 这里tf.reshape()中shape包含的-1为缺省值，可以自动计算
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut

    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim * 2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    # Project layer for ID-CNN by crownpku
    # Delete the hidden layer, and change bias initializer
    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param idcnn_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", initializer=tf.constant(0.001, shape=[self.num_tags]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    # todo
    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss" if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])],
                axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            # crf_log_likelihood在一个条件随机场里面计算标签序列的log-likelihood
            # inputs: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor,
            # 一般使用BILSTM处理之后输出转换为他要求的形状作为CRF层的输入.
            # tag_indices: 一个形状为[batch_size, max_seq_len] 的矩阵,其实就是真实标签.
            # sequence_lengths: 一个形状为 [batch_size] 的向量,表示每个序列的长度.
            # transition_params: 形状为[num_tags, num_tags] 的转移矩阵
            # log_likelihood: 标量,log-likelihood
            # transition_params: 形状为[num_tags, num_tags] 的转移矩阵
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths + 1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        生成给inputs赋值的字典
        :param is_train: 是否是训练模式的Flag
        :param batch: list train/evaluate data
        :return: feed_dict
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: 运行batch所需的tf.Session
        :param is_train: 表示是否是训练batch的flag
        :param batch: 数据集
        :return: 训练结果，损失
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _, char_lookup_out, seg_lookup_out, char_inputs_test, seg_inputs_test, embed_test, embedding_test, \
            model_inputs_test, layerInput_test, conv_test, w_test_1, w_test_2, char_inputs_test = sess.run(
                [self.global_step, self.loss, self.train_op, self.char_lookup, self.seg_lookup, self.char_inputs_test,
                 self.seg_inputs_test, \
                 self.embed_test, self.embedding_test, self.model_inputs_test, self.layerInput_test, self.conv_test,
                 self.w_test_1, self.w_test_2, self.char_inputs],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    # todo
    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]
        :param lengths: [batch_size]
        :param matrix: 状态转移矩阵
        :return:
        """
        # 这里是用Viterbi算法，从最后一个节点反向逐步求得最优路径节点，得到最优路径
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: tf.Session
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                # gold = iob_iobes([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                # pred = iob_iobes([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval(session=sess)
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)
