# encoding=utf8
import pickle
import itertools
from collections import OrderedDict
import os
import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager

root_path = os.getcwd() + os.sep

# tf.app.flags 添加命令行参数
# 训练模式开关
flags = tf.app.flags
flags.DEFINE_boolean("clean", True, "clean train folder")  # 设置为True：每次训练会清除上次训练日志
flags.DEFINE_boolean("train", False, "Whether train the model")
# configurations for model
flags.DEFINE_integer("seg_dim", 20, "Embedding size for segmentation, 0 if not used")  # embedding的增加维度
flags.DEFINE_integer("char_dim", 100, "Embedding size for characters")  # 字的维度，每个embedding有100个字向量
flags.DEFINE_integer("lstm_dim", 100, "Num of hidden units in LSTM, or num of filters in IDCNN")  # LSTM隐藏层单元数/卷积滤波器数
flags.DEFINE_string("tag_schema", "iobes", "tagging schema iobes or iob")  # tag类型

# configurations for training
flags.DEFINE_float("clip", 5, "Gradient clip")  # 梯度裁剪值
flags.DEFINE_float("dropout", 0.8, "Dropout rate")  # dropout的keep，先设置为.8了
flags.DEFINE_float("batch_size", 60, "batch size")  # batch size
flags.DEFINE_float("lr", 0.001, "Initial learning rate")  # 初始学习率
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")  # 优化训练方法，用的adam
flags.DEFINE_boolean("pre_emb", True, "Wither use pre-trained embedding")  # embedding的数据预处理
flags.DEFINE_boolean("zeros", True, "Wither replace digits with zero")  # 遇到生僻字用0替代
flags.DEFINE_boolean("lower", False, "Wither lower case")  # 是否需要将字母小写

flags.DEFINE_integer("max_epoch", 100, "maximum training epochs")  # 最大epoch，一个epoch是整个数据集都经过训练一次的周期
flags.DEFINE_integer("steps_check", 100, "steps per checkpoint")  # 每多少个batch输出一次损失函数
flags.DEFINE_string("ckpt_path", "ckpt", "Path to save model")  # 保存模型的路径
flags.DEFINE_string("summary_path", "summary", "Path to store summaries")  # 保存摘要和流程图的路径
flags.DEFINE_string("log_file", "train.log", "File for log")
flags.DEFINE_string("map_file", "./config/maps.pkl", "file for maps")  # 保存(映射)字典的文件
flags.DEFINE_string("vocab_file", "vocab.json", "File for vocab")  # 原始语料库文件
flags.DEFINE_string("config_file", "./config/config_file", "File for config")  # 配置文件
flags.DEFINE_string("tag_to_id_path", "./config/tag_to_id.txt", "File for tag_to_id.txt")
flags.DEFINE_string("id_to_tag_path", "./config/id_to_tag.txt", "File for id_to_tag.txt")
flags.DEFINE_string("script", "conlleval", "evaluation script")
flags.DEFINE_string("result_path", "result", "Path for results")
flags.DEFINE_string("emb_file", os.path.join(root_path + "data", "vec.txt"),
                    "Path for pre_trained embedding")  # 预处理的embedding数据集
flags.DEFINE_string("train_file", os.path.join(root_path + "data", "example.train"), "Path for train data")  # 训练集
flags.DEFINE_string("dev_file", os.path.join(root_path + "data", "example.dev"), "Path for dev data")  # 开发/验证集
flags.DEFINE_string("test_file", os.path.join(root_path + "data", "example.test"), "Path for test data")  # 测试集

flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")  # 选择NN模型：ID-CNN或BiLSTM
# flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")

FLAGS = flags.FLAGS  # 将上面的参数保存在FLAGS
# 断言确保上面的参数设置为正常值。
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


# 对训练结果进行评估dev/test
def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    # 加载数据集
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # 选择tag schema(IOB / IOBES)    I：中间，O：其他，B：开始 | E：结束，S：单个
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)
    update_tag_scheme(dev_sentences, FLAGS.tag_schema)
    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):  # 配置文件：char_to_id, id_to_char, tag_to_id, id_to_tag的数据
        # create dictionary for word
        if FLAGS.pre_emb:
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences, FLAGS.id_to_tag_path, FLAGS.tag_to_id_path)
        # with open('maps.txt','w',encoding='utf8') as f1:
        # f1.writelines(str(char_to_id)+" "+id_to_char+" "+str(tag_to_id)+" "+id_to_tag+'\n')
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)  #

    # prepare data, get a collection of list containing index
    # train_data[0][0]：一句话；
    # train_data[0][1]：单个字的编号；
    # train_data[0][2]：切词之后，切词特征：词的大小是一个字的话是0，词的大小是2以上的话：1,2....，2,3；
    # train_data[0][3]：每个字的标签
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), 0, len(test_data)))

    train_manager = BatchManager(train_data, FLAGS.batch_size)  # 按batch size将数据拆分
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        # tf.device("/cpu:0") 指定运行的GPU(默认为GPU:0)
        with tf.device("/cpu:0"):
            for i in range(100):
                # 按批次训练模型。这个是训练的开始，可以从这里倒着找整个网络怎么训练
                for batch in train_manager.iter_batch(shuffle=True):
                    step, batch_loss = model.run_step(sess, True, batch)
                    loss.append(batch_loss)
                    # 打印信息：
                    # iteration：迭代次数，也就是经过多少个epoch；
                    #
                    if step % FLAGS.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        logger.info("iteration:{} step:{}/{}, "
                                    "NER loss:{:>9.6f}".format(
                            iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                        loss = []

                # best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
                if i % 7 == 0:
                    save_model(sess, model, FLAGS.ckpt_path, logger)
            # evaluate(sess, model, "test", test_manager, id_to_tag, logger)


# 测试模式，输入测试句
def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        while True:
            # try:
            #     line = input("请输入测试句子:")
            #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            #     print(result)
            # except Exception as e:
            #     logger.info(e)

            line = input("请输入测试句子:")
            result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            print(result)


def main(is_train):
    if is_train:
        if FLAGS.clean:
            clean(FLAGS)
        train()  # 训练模式
    else:
        evaluate_line()  # 测试模式


if __name__ == "__main__":
    is_train = True
    tf.app.run(main(is_train))