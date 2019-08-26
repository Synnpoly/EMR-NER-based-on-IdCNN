# encoding=utf8
import os, jieba, csv
import jieba.posseg as pseg

# 用预备的医学词典生成train，dev，test数据集
c_root = os.getcwd() + os.sep + "source_data" + os.sep
dev = open("example.dev", 'w', encoding='utf8')
train = open("example.train", 'w', encoding='utf8')
test = open("example.test", 'w', encoding='utf8')
dicts = csv.reader(open("DICT_NOW.csv", 'r', encoding='utf8'))
# POStag和标点符号列表。转为set是为了加快后面遍历速度。
tags = set(['DIS', 'SYM', 'SGN', 'TES', 'DRU', 'SUR', 'PRE', 'PT', 'Dur', 'TP', 'REG', 'ORG', 'AT', 'PSB', 'DEG', 'FW', 'CL'])
symbols = set(['。', '？‘， ’?', '!', '！'])

# 将医学专有名词以及标签加入jieba词典中
for row in dicts:
    if len(row) == 2:
        # add_word保证添加的词语不会被cut掉
        jieba.add_word(row[0].strip(), tag=row[1].strip())
        # 调节单个词语的词频，使其能（或不能）被分出来。
        jieba.suggest_freq(row[0].strip())
split_num = 0
for file in os.listdir(c_root):
    # source_data中的original.txt文档是未经过分词的原始电子病历文档
    if "txtoriginal.txt" in file:
        fp = open(c_root + file, 'r', encoding='utf8')
        for line in fp:
            split_num += 1
            words = pseg.cut(line)
            for key, value in words:
                # print(key)
                # print(value)
                if value.strip() and key.strip():
                    import time

                    start_time = time.time()
                    # 此处以split_num%15生成index，以2:2:11的比例分配数据到dev,test,train数据集中
                    # index:1 dev数据集 2 test集 3 训练集
                    index = str(1) if split_num % 15 < 2 else str(
                        2) if split_num % 15 > 1 and split_num % 15 < 4 else str(3)
                    end_time = time.time()
                    # print("method one used time is {}".format(end_time-start_time))
                    # 用 IOB法 标记分词字符（B分词开头，I分词中，O tags不在tags列表中的分词）
                    if value not in tags:
                        value = 'O'
                        for achar in key.strip():
                            if achar and achar.strip() in symbols:
                                string = achar + " " + value.strip() + "\n" + "\n"
                                dev.write(string) if index == '1' else test.write(
                                    string) if index == '2' else train.write(string)
                            elif achar.strip() and achar.strip() not in symbols:
                                string = achar + " " + value.strip() + "\n"
                                dev.write(string) if index == '1' else test.write(
                                    string) if index == '2' else train.write(string)

                    elif value.strip() in tags:
                        begin = 0
                        for char in key.strip():
                            if begin == 0:
                                begin += 1
                                string1 = char + ' ' + 'B-' + value.strip() + '\n'
                                if index == '1':
                                    dev.write(string1)
                                elif index == '2':
                                    test.write(string1)
                                elif index == '3':
                                    train.write(string1)
                                else:
                                    pass
                            else:
                                string1 = char + ' ' + 'I-' + value.strip() + '\n'
                                if index == '1':
                                    dev.write(string1)
                                elif index == '2':
                                    test.write(string1)
                                elif index == '3':
                                    train.write(string1)
                                else:
                                    pass
                    else:
                        continue

dev.close()
train.close()
test.close()
