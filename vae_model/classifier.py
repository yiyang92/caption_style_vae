import tensorflow as tf

from utils.rnn_model import make_rnn_cell

class Classifier(object):
    def __init__(self, images_fv, vocab_size, embed_size, lstm_hidden):
        """Classifier"""
        self.images_fv = images_fv
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_hidden = lstm_hidden

    def q_y_x(self, captions, lengths, n_classes):
        """
        Returns:
            x_logits: classifier unnormalized log probabilities
        """
        with tf.variable_scope("net"):
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                        "dec_embeddings", [self.vocab_size,
                                           self.embed_size],
                        dtype=tf.float32)
                vect_inputs = tf.nn.embedding_lookup(embedding, captions)
            keep_prob = tf.placeholder_with_default(1.0, (),
                                                    name='classifier_drop')
            cell_0 = make_rnn_cell([self.lstm_hidden],
                                   base_cell=tf.contrib.rnn.LSTMCell,
                                   dropout_keep_prob=keep_prob)
            zero_state0 = cell_0.zero_state(
                batch_size=tf.shape(self.images_fv)[0], dtype=tf.float32)
            initial_state = zero_state0
            # _, initial_state = cell_0(self.images_fv, zero_state0)
            # captions LSTM
            outputs, final_state = tf.nn.dynamic_rnn(cell_0,
                                                     inputs=vect_inputs,
                                                     sequence_length=lengths,
                                                     initial_state=initial_state,
                                                     swap_memory=True,
                                                     dtype=tf.float32)
            y_logits = tf.layers.dense(final_state[0][1], n_classes)
        return y_logits
