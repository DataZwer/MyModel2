import tensorflow as tf
from data_helper.batches import dataset_iterator
from data_helper.data_utils import load_all
from data_helper.load_embeddings import load_glove_embedding
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode


class LSTM_CRF(object):
    def __init__(self, config_lstm, sess, train_path, test_path, embedding_path):

        # 模型常量定义
        self.batch_size = config_lstm['batch_size']
        self.lstm_units = config_lstm['lstm_units']
        self.num_classes = config_lstm['num_classes']
        self.n_epochs = config_lstm['n_epochs']
        self.embedding_size = config_lstm['embedding_size']
        self.l2_reg_lambda = config_lstm['l2_reg_lambda']
        self.clip_grad = config_lstm['clip']  # ??
        self.optimizer = 'Adam'
        self.l2_loss = tf.constant(0.0)
        self.sess = sess
        self.train_path = train_path
        self.test_path = test_path
        self.embedding_path = embedding_path

        # input
        self.train_sens = None
        self.train_labels = None
        self.test_sens = None
        self.test_labels = None
        self.train_sl = None
        self.test_sl = None
        self.max_len = None
        self.x_inputs = None
        self.y_inputs = None
        self.word_index = None
        self.sequence_lengths = None
        self.lr_pl = None
        self.dropout_keep_prob = None

        # 没必要的删一点
        self.glove_embedding = None
        self.logits = None
        self.CRF = None
        self.loss = None
        self.transition_params = None
        self.train_op = None

    def input_layer(self):

        self.train_sens, self.train_labels, self.test_sens, self.test_labels, self.train_sl, self.test_sl, self.word_index = \
            load_all(self.train_path, self.test_path)
        with tf.variable_scope("input"):
            self.x_inputs = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
            self.y_inputs = tf.placeholder(tf.int32, shape=[None, None], name="labels")
            self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

            self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")  # l2正则化
            self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    def embedding_layer(self):

        # glove embedding
        embedding_maxtrix = load_glove_embedding(
            word_index=self.word_index,
            trimmed_filename=self.embedding_path,
            load=True,
        )

        with tf.variable_scope("embedding"):
            glove_w2v = tf.Variable(embedding_maxtrix, dtype=tf.float32, name='glove_w2v')
            tmp = tf.nn.embedding_lookup(glove_w2v, self.x_inputs)
            self.glove_embedding = tf.nn.dropout(tmp, self.dropout_keep_prob)

    def bi_lstm_layer(self):

        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.lstm_units)
            cell_bw = LSTMCell(self.lstm_units)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.glove_embedding,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_keep_prob)  # [b_s, m_x, 300]

        with tf.variable_scope("proj"):

            W = tf.get_variable(name="W",
                                shape=[2 * self.lstm_units, self.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_classes],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.lstm_units])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_classes])  # 每个标签的得分[b_s, ml, num_class]

    def loss_op(self):
        log_likelihood, self.transition_params = crf_log_likelihood(
            inputs=self.logits,
            tag_indices=self.y_inputs,
            sequence_lengths=self.sequence_lengths
        )  # 对数似然函数
        self.loss = -tf.reduce_mean(log_likelihood)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def build_graph(self):
        self.input_layer()
        self.embedding_layer()
        self.bi_lstm_layer()
        self.loss_op()
        self.trainstep_op()

    def train(self):
        self.build_graph()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # 划分数据batch
        iterations, next_iterator = dataset_iterator(
            self.train_sens,  # 句子单词id
            self.train_labels,  # OBI标签
            self.train_sl  # 实际句子长度
        )

        for epoch in range(self.n_epochs):
            train_loss = 0.0
            count = 0

            print("-----------Now we begin the %dth epoch-----------" % (epoch + 1))
            for iter in range(iterations):
                sen_batch, labels_batch, seq_len_batch = self.sess.run(next_iterator)

                if len(sen_batch) < self.batch_size:
                    continue

                f_dict = {
                    self.x_inputs: sen_batch,
                    self.y_inputs: labels_batch,
                    self.sequence_lengths: seq_len_batch,
                    self.lr_pl: 0.001,
                    self.dropout_keep_prob: 0.5
                }

                _, loss, step_num_ = self.sess.run([self.train_op, self.loss,  self.global_step], feed_dict=f_dict)
                count = count + 1
                train_loss = train_loss + loss

            print("The loss value in this epoch is %.3f " % (train_loss / count))
            predict_label_list = self.test()
            if (epoch + 1) == 16:
                return predict_label_list

    def test(self):

        count = 0
        iterations_test, next_iterator_test = dataset_iterator(
            self.test_sens,  # 句子单词id
            self.test_labels,  # OBI标签
            self.test_sl  # 实际句子长度
        )

        predict_label_list = []
        for iter in range(iterations_test):
            sen_batch_test, labels_batch_test, seq_len_batch_test = self.sess.run(next_iterator_test)

            if len(sen_batch_test) < self.batch_size:
                continue

            f_dict = {
                self.x_inputs: sen_batch_test,
                self.y_inputs: labels_batch_test,
                self.sequence_lengths: seq_len_batch_test,
                self.lr_pl: None,
                self.dropout_keep_prob: 1.0
            }

            count = count + 1

            predict_labels = self.predict_one_batch(f_dict, seq_len_batch_test)
            predict_label_list.extend(predict_labels)

        return predict_label_list

    def predict_one_batch(self, feed_dict, seq_len_batch_test):

        logits, transition_params = self.sess.run([self.logits, self.transition_params], feed_dict=feed_dict)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_batch_test):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)

        return label_list
