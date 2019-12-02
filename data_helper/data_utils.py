from bean.file_path import *
from keras.preprocessing import sequence
# word_index, sens, labels ==> sens_pad, label_pad


# 加载句子
def load_sens_labels_ml(file_path):

    sens = []
    labels = []
    with open(file_path, encoding='utf8') as f:
        sen = []
        label = []
        max_len = -1
        for line in f:
            value = line.strip().split(" ")
            if len(value) == 1:
                sens.append(sen)
                labels.append(label)
                if len(sen) > max_len:
                    max_len = len(sen)
                sen = []
                label = []
            else:
                sen.append(value[0])
                label.append(value[1])

    return sens, labels, max_len


def get_widx(sens):

    word_index = {}
    word_index['#OOV#'] = 0
    id = 1
    for sen in sens:
        for word in sen:
            if word.lower() not in word_index:
                word_index[word.lower()] = id
                id = id + 1

    return word_index


# 量化标签[O, B-A, B-I] ==> [0, 1, 2]
def quantitative_data(sens, labels, word_index):

    def quantitative_labels(labels):
        label_dic = {'O': 0, 'B-A': 1, 'I-A': 2}
        q_labels = []
        for l in labels:
            tmp = []
            for e in l:
                if e in label_dic:
                    tmp.append(label_dic[e])
                else:
                    tmp.append(0)
            q_labels.append(tmp)
        return q_labels

    def quantitative_sens(sens, word_index):
        q_sens = []
        for s in sens:
            tmp = []
            for word in s:
                if word.lower() in word_index:
                    tmp.append(word_index[word.lower()])
                else:
                    tmp.append(0)
            q_sens.append(tmp)
        return q_sens

    q_labels = quantitative_labels(labels)
    q_sens = quantitative_sens(sens, word_index)
    return q_sens, q_labels


def get_padding(data, max_len):
    padding_data = sequence.pad_sequences(data, maxlen=max_len, padding='post', truncating="post")
    return padding_data


def load_all(train_path, test_path):
    test_sens, test_labels, test_mxl = load_sens_labels_ml(test_path)
    train_sens, train_labels, train_mxl = load_sens_labels_ml(train_path)
    mx_len = max(test_mxl, train_mxl)

    word_index = get_widx(train_sens + test_sens)
    test_q_sens, test_q_labels = quantitative_data(test_sens, test_labels, word_index)
    train_q_sens, train_q_labels = quantitative_data(train_sens, train_labels, word_index)

    train_s1 = get_seq_len(train_sens)
    test_sl = get_seq_len(test_sens)

    test_sens_pad = get_padding(test_q_sens, mx_len)
    test_labels_pad = get_padding(test_q_labels, mx_len)
    train_sens_pad = get_padding(train_q_sens, mx_len)
    train_labels_pad = get_padding(train_q_labels, mx_len)

    return train_sens_pad, train_labels_pad, test_sens_pad, test_labels_pad, train_s1, test_sl, word_index


def get_seq_len(sens):
    return [len(sen) for sen in sens]


if __name__ == '__main__':
    test_sens, test_labels, test_mxl = load_sens_labels_ml(rest14_path[rest14_test_path])
    train_sens, train_labels, train_mxl = load_sens_labels_ml(rest14_path[rest14_train_path])
    word_index = get_widx(train_sens + test_sens)
    q_sens, q_labels = quantitative_data(test_sens, test_labels, word_index)
    train_s1 = get_seq_len(train_sens)
    test_sl = get_seq_len(test_sens)

    test_sens_pad = get_padding(q_sens, test_mxl)
    test_labels_pad = get_padding(q_labels, test_mxl)
