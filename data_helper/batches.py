import math
import tensorflow as tf
from data_helper.data_utils import *
from bean.file_path import *


def dataset_iterator(sens, labels, seq_len):
    batch_size = 64
    train_nums = len(sens)
    iterations = math.ceil(train_nums / batch_size)  # 总共可以划分出来的batch数量

    # 使用from_tensor_slices将数据放入队列，使用batch和repeat划分数据批次，且让数据序列无限延续
    dataset = tf.data.Dataset.from_tensor_slices((sens, labels, seq_len))
    dataset = dataset.batch(batch_size).repeat()

    # 使用生成器make_one_shot_iterator和get_next取数据
    iterator = dataset.make_one_shot_iterator()
    next_iterator = iterator.get_next()
    return iterations, next_iterator


if __name__ == '__main__':
    test_sens, test_labels, test_mxl = load_sens_labels_ml(rest14_path[rest14_test_path])
    train_sens, train_labels, train_mxl = load_sens_labels_ml(rest14_path[rest14_train_path])

    word_index = get_widx(train_sens + test_sens)

    train_s1 = get_seq_len(train_sens)
    test_sl = get_seq_len(test_sens)

    q_sens, q_labels = quantitative_data(test_sens, test_labels, word_index)

    test_sens_pad = get_padding(q_sens, test_mxl)
    test_labels_pad = get_padding(q_labels, test_mxl)

    iterations, next_iterator = dataset_iterator(test_sens_pad, test_labels_pad, test_sl)
    with tf.Session() as sess:
        for epoch in range(1):
            for iteration in range(iterations):
                sen_batche, labels_batch, seq_len_batch = sess.run(next_iterator)
                break
            break
