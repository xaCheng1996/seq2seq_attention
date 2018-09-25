#coding=utf-8
import numpy as np
import random
import re

padToken, goToken, eosToken, unknownToken = 2319, 337, 1464, 0

file = "./description.txt"
des = open(file, "r", encoding="utf-8").readlines()
des_dict = dict()
for i in des:
    j = i.split(":")
    # print(j[0] + "11" + j[1])
    des_dict[j[0]] = j[1]

class Batch:
    #batch类，里面包含了encoder输入，decoder输入以及他们的长度
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", "", string)
    # string = re.sub(r"？", "", string)
    # string = re.sub(r"～", "", string)
    # string = re.sub(r"！", "", string)
    # string = re.sub(r"‘", "", string)
    # string = re.sub(r"，", "", string)
    # string = re.sub(r"。", "", string)
    # string = re.sub(r"/", "", string)
    string = re.sub(r"！", "!", string)
    string = re.sub(r"，", ",", string)
    string = re.sub(r"\?", "", string)
    # string = re.sub(r"\s{2,}", "", string)
    string = re.sub(r"\[", "", string)
    string = re.sub(r"]", "", string)
    string = re.sub(r"【", "", string)
    string = re.sub(r"】", "", string)
    string = re.sub(r"\+", "", string)
    string = re.sub(r"、", "", string)
    string = re.sub(r"~", "", string)
    return string.strip().lower()


def loadDataset(input_data, target_data):
    print('Loading dataset from {}'.format(input_data))

    input_cor = list(open(input_data, "r", encoding="utf-8").readlines())
    target_cor = list(open(target_data, "r", encoding="utf-8").readlines())
    input_cor = [s.strip() for s in input_cor]
    target_cor = [s.strip() for s in target_cor]

    # Split by words
    x_text = [clean_str(sent) for sent in input_cor]
    y = [clean_str(sent) for sent in target_cor]

    vocab_size, word2id = word_to_id()
    id2word = id_to_word()
    word_list = []
    for i in x_text:
        word_tem = []
        k = i.split(" ")
        for j in k:
            if word2id.get(j) is not None and j != "食材":
                word_tem.append(word2id[j])
            else:
                if j == "商品品类" or j == "食材" or j == "菜系" or "业务性标签" or j == "用餐人数" or j == "制作方法":
                    continue
                else:
                    word_tem.append(unknownToken)
        word_list.append(word_tem)

    word_list_target = []
    for i in y:
        word_tem = []
        k = i.split(" ")
        for j in k:
            if word2id.get(j) is not None:
                word_tem.append(word2id[j])
            else:
                # print(j)
                # print(str(des_dict.get(j)))
                if des_dict.get(j) is not None or len(j) >= 3:
                    continue
                else:
                   # print(j)
                    word_tem.append(unknownToken)
        word_tem.append(eosToken)
        word_list_target.append(word_tem)
    return word_list, word_list_target, word2id, id2word, vocab_size


def createBatch(inputs, targets):
    '''
    根据给出的samples（就是一个batch的数据），进行padding并构造成placeholder所需要的数据形式
    :param samples: 一个batch的样本数据，列表，每个元素都是[question， answer]的形式，id
    :return: 处理完之后可以直接传入feed_dict的数据格式
    '''
    batch = Batch()
    batch.encoder_inputs_length = [len(x) for x in inputs]
    batch.decoder_targets_length = [len(y) for y in targets]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    for x in inputs:
        # 将source进行反序并PAD值本batch的最大长度
        source = list(reversed(x))
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(pad + source)

        # 将target进行PAD，并添加END符号
    for y in targets:
        target = y
        pad = [padToken] * (max_target_length - len(target))
        batch.decoder_targets.append(target + pad)
        # batch.target_inputs.append([goToken] + target + pad[:-1])
    return batch


def getBatches(input_data, target_data, batch_size):
    '''
    根据读取出来的所有数据和batch_size将原始数据分成不同的小batch。对每个batch索引的样本调用createBatch函数进行处理
    :param data: loadDataset函数读取之后的trainingSamples，就是QA对的列表
    :param batch_size: batch大小
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 列表，每个元素都是一个batch的样本数据，可直接传入feed_dict进行训练
    '''
    #每个epoch之前都要进行样本的shuffle

    np.random.seed(10)
    np.random.permutation(input_data)
    np.random.permutation(target_data)
    # print(input_data)
    # input_data = input_data[shuffle_indices]
    # target_data = target_data[shuffle_indices]

    batches = []
    data_len = len(input_data)

    def genNextSamples(data):
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    data_z = list(zip(input_data, target_data))
    for i in genNextSamples(data_z):
        x, y = zip(*i)
        batch = createBatch(x, y)
        batches.append(batch)
    return batches


def word_to_id():
    file_word_embedding = "./word_embedding_simple.txt"
    file_word_embedding_read = open(file_word_embedding, "r", encoding="utf-8").readlines()
    word_list = []
    word2id = dict()
    index = 0
    for ii in file_word_embedding_read:
        word = str(ii).split(":")[0]
        word = word.replace("	", "")
        # print(word)
        word2id[word] = index
        word_list.append(word)
        index += 1
    vocab_size = len(word_list)
    return vocab_size, word2id


def id_to_word():
    file_word_embedding = "./word_embedding_simple.txt"
    file_word_embedding_read = open(file_word_embedding, "r", encoding="utf-8").readlines()
    id2word = dict()
    # word_list = []
    index = 0
    for ii in file_word_embedding_read:
        word = str(ii).split(":")[0]
        word = word.replace("	", "")
        # print(word)
        id2word[index] = word
        index += 1
        # word_list.append(word)
    return id2word


def load_embedding():
    file_word_embedding = "./word_embedding_simple.txt"
    file_word_embedding_matrix = "/Users/xuang/statistics/embedding_matrix.txt"
    file_word_embedding_read = open(file_word_embedding, "r", encoding="utf-8").readlines()
    file_word_embedding_write = open(file_word_embedding_matrix, "a", encoding="utf-8")
    word_list = []
    index = 0
    for ii in file_word_embedding_read:
        word = str(ii).split("[")
        embedding = word[1].split("]")[0]
        embedding = str(embedding).split(",")
        embedding = [float(x) for x in embedding]
        file_word_embedding_write.write(str(embedding))
        file_word_embedding_write.write("\n")
        word_list.append(embedding)
        index += 1
    # for i in word_list:
    #     print(i)
    #     print(len(i))
    #     print("###########")
    # file_word_embedding_write.write(str(word_list))



# _, a = word_to_id(data)
# print(a)
# print(id_to_word(a))
# load_embedding()


def embedding_re():
    file_word_embedding_matrix = "./embedding_matrix.txt"
    fff = open(file_word_embedding_matrix, "r", encoding="utf-8").readlines()
    matrix = []
    for i in fff:
        word_list = []
        i = i[1:-2]
        j = str(i).split(",")
        word_list.append([float(x) for x in j])
        matrix.append(word_list)
    matrix = np.array(matrix).reshape([22466, 100])
    print("matrxin size:" + str(matrix.shape))
    # print(matrix[1])
    return matrix


# inputs, target, _, id2word, _ = loadDataset("./use_seg_x_test.txt", "./use_seg_y_test.txt")
# index = 0
# for i in target:
#     print(i)
#     if 0 in i:
#         index += 1
#
# print(index / len(target))
# print(id2word[1])


def id_transfer_word(data):
    file_word_embedding = "./word_embedding_simple.txt"
    file_word_embedding_read = open(file_word_embedding, "r", encoding="utf-8").readlines()
    word_list = []
    for ii in file_word_embedding_read:
        word = str(ii).split(":")[0]
        word = word.replace("	", "")
        # print(word)
        word_list.append(word)
    word_list = np.array(word_list)
    id_t_word = []
    for i in data:
        tem = []
        for j in i:
            if word_list[j] is None:
                tem.append("<UNK>")
            else:
                tem.append(word_list[j])
        id_t_word.append(tem)
    return id_t_word


# v, word2id = word_to_id()
# print(word2id.get("?"))
# print(word2id.get(","))
# print(word2id.get("，"))
# print(word2id.get(""))
