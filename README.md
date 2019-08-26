# EMR-NER-based-on-IdCNN
这是我学习的一个NLP入门项目，基于神经网络，主要使用膨胀卷积神经网络(ID-CNN)和条件随机场CRF实现的电子病历命名实体识别(NER)。<br>
用到的工具：
- [numpy](https://www.numpy.org/)
- [TensorFlow](https://tensorflow.google.cn/overview/?hl=zh_cn)
- [jieba中文分词工具包](https://github.com/fxsjy/jieba)


## 电子病历命名实体识别背景  
近年来,在电子病历文本上应用自然语言处理、信息抽取等技术服务于临床决策支持的研究倍受关注。这个过程分为两个不同的阶段:  
1.自然语言处理研究主要关注病历文本的预处理,包括句子边界识别、词性标注、句法分析等;  
2.信息抽取以自然语言处理研究为基础,主要关注病历文本中各类表达医疗知识的命名实体或医疗概念的识别和关系抽取。

## 实体标注细节 
通过分析电子病历，医生针对患者的诊疗活动可以概括为：通过患者自述（自诉症状） 和检查结果（检查项目）发现疾病的表现（症状），给出诊断结论（疾病），并基于诊断结论， 给出治疗措施（治疗方案）。这个过程可以看出，医疗活动主要涉及四类重要信息：症状、 疾病、检查和治疗，涉及的具体描述如下：   
1）疾病 DIS：泛指导致患者处于非健康状态的原因，比如：诊断、病史。  
2）疾病诊断分类 DT：疾病诊断相关分组，比如“高血压，极高危组”中的“极高危组”。   
3）症状 SYM：泛指疾病导致的不适和显示表达的检验检查结果，分为：自诉症状和体征（异 常检验检查结果）。   
4）检查 TES：泛指为了得到更多的由疾病导致的异常表现以支持诊断而采取的检查设备、 检查程序、检查项目等。   
5）治疗手段：泛指为了治愈疾病、缓解或改善症状而给予患者的 药物 DRU、手术 SUR和 措施 PRE等。  

## 代码实现
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
