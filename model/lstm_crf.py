import tensorflow as tf
from data_helper.data_utils import load_all
from data_helper.load_embeddings import load_glove_embedding

class LSTM_CRF(object):
    def __init__(self, config_lstm, sess, train_path, test_path, embedding_path):

        # 模型常量定义
        self.batch_size = config_lstm['batch_size']
        self.lstm_units = config_lstm['lstm_units']
        self.num_classes = config_lstm['num_classes']
        self.n_epochs = config_lstm['n_epochs']
        self.embedding_size = config_lstm['embedding_size']
        self.l2_reg_lambda = config_lstm['l2_reg_lambda']
        self.l2_loss = tf.constant(0.0)
        self.sess = sess
        self.train_path = train_path
        self.test_path = test_path
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.embedding_path = embedding_path

        # input
        self.train_sens = None
        self.train_labels = None
        self.test_sens = None
        self.test_labels = None
        self.max_len = None
        self.x_inputs = None
        self.y_inputs = None
        self.word_index = None

        # embedding
        self.glove_embedding = None

    def input_layer(self):
        self.train_sens, self.train_labels, self.test_sens, self.test_labels, self.word_index = load_all(self.train_path, self.test_path)
        self.x_inputs = tf.placeholder(tf.int32, [None, self.max_len], name='x_inputs')
        self.y_inputs = tf.placeholder(tf.float32, [None, self.num_classes], name='y_inputs')

    def embedding_layer(self):
        # glove
        embedding_maxtrix = load_glove_embedding(
            word_index=self.word_index,
            trimmed_filename=self.embedding_path,
            load=True,
        )
        glove_w2v = tf.Variable(embedding_maxtrix, dtype=tf.float32, name='glove_w2v')
        tmp = tf.nn.embedding_lookup(glove_w2v, self.x_inputs)
        self.glove_embedding = tf.nn.dropout(tmp, self.dropout_keep_prob)
