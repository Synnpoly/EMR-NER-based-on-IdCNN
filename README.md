# EMR-NER-based-on-IdCNN
这是我学习的一个NLP入门项目，基于神经网络，主要使用膨胀卷积神经网络(ID-CNN)和条件随机场CRF实现的电子病历命名实体识别(NER)。<br>
用到的工具：
- [TensorFlow](https://tensorflow.google.cn/overview/?hl=zh_cn)
- [jieba中文分词工具包](https://github.com/fxsjy/jieba)


## 第一部分是对电子病历文本进行预处理，生成训练数据集
将source_data中的原始语料使用jieba分词，然后以字为单位进行标注，
这里的jieba加载(add_word)了事先准备的医学词典“DICT_NOW.csv”，里边含有医学名词以及标签。
然后进行分词,进而使用BIO标注法进行标注，即：B分词开头，I分词中，O不在tags列表中的其他词。

同时按照2:2:11的比例放在dev：test：train 三部分。
最终的输出是：
example.dev
example.test
example.train  
**标注效果展示：**
>双 B-ORG  
>侧 I-ORG  
>瞳 I-ORG  
>孔 I-ORG  
>头 B-SYM  
>痛 I-SYM  
>及 O  
>头 B-SYM  
>晕 I-SYM

## 第二部分是用ID-CNN模型训练电子病历NER。
[原理参考](https://blog.csdn.net/Kaiyuan_sjtu/article/details/89143573)  
**代码文件**
- main.py 主文件  
```python
def main(is_train):
    if is_train:
        if FLAGS.clean:
            clean(FLAGS)
        train()          # 训练模式
    else:
        evaluate_line()  # 测试模式
if __name__ == "__main__":
    is_train = True
    tf.app.run(main(is_train))
```
- model.py 神经网络模型设计；
- data_utils.py 从语料库生成训练可用数据的一些工具函数，如创建字典，maps，检查tags标注格式等的实现；
- utils.py 其他工具函数，如生成/清空paths，生成log，由ckpt创建模型等；
- loader.py 预处理和导入数据。

很多问题需要修改特别是test模块，包括训练中的可视化展示不完整，要努力学习呀。
