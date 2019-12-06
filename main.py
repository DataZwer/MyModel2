from bean.file_path import *
from bean.config import config_lstm
from model.lstm_crf import LSTM_CRF
import tensorflow as tf


def run_model(t, Model, train_path, test_path, embedding_path):
    for i in range(t):
        tf.reset_default_graph()  # 重置当前计算图
        with tf.Session() as sess:
            model = Model(config_lstm, sess, train_path, test_path, embedding_path)
            print("*******The %dth model*******" % (i + 1))
            predict_label_list = model.train()
        tf.get_default_graph().finalize()  # 获取当前计算图并结束它


if __name__ == '__main__':

    '''
    lap14_path[lap14_train_path]
    lap14_path[lap14_test_path]
    lap14_path[lap14_emb_path]

    rest14_path[rest14_train_path]
    rest14_path[rest14_test_path]
    rest14_path[rest14_emb_path]

    rest16_path[rest16_train_path]
    rest16_path[rest16_test_path]
    rest16_path[rest16_emb_path]

    rest15_path[rest15_train_path]
    rest15_path[rest15_test_path]
    rest15_path[rest15_emb_path]    
    '''
    run_model(1, LSTM_CRF, lap14_path[lap14_train_path], lap14_path[lap14_test_path], lap14_path[lap14_emb_path])
