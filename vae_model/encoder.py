import tensorflow as tf
from tensorflow import layers
import zhusuan as zs
from utils.rnn_model import make_rnn_cell


class Encoder():
    def __init__(self, images_fv, lstm_hidden, prior,
                 latent_size, vocab_size, embed_size, z_samples, params):
        """
        Args:
            images_fv: image features mapping to word embeddings
        """
        self.images_fv = images_fv
        self.lstm_hidden = lstm_hidden
        self.prior = prior
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.z_samples = z_samples
        self.params = params

    def q_z_xy(self, captions, labels, lengths, images=None):
        """Calculate approximate posterior q(z|x, y, f(I))
        Returns:
            model: zhusuan model object, can be used for getting probabilities
        """
        if images is not None:
            self.images_fv = images
        with zs.BayesianNet() as model:
            # encoder and decoder have different embeddings but the same image features
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                            "enc_embeddings", [self.vocab_size,
                                               self.embed_size],
                            dtype=tf.float32)
                vect_inputs = tf.nn.embedding_lookup(embedding, captions)
            with tf.name_scope(name="net") as scope1:
                cell_0 = make_rnn_cell(
                    [self.lstm_hidden],
                    base_cell=tf.contrib.rnn.LSTMCell)
                zero_state0 = cell_0.zero_state(
                    batch_size=tf.shape(self.images_fv)[0],
                    dtype=tf.float32)
                # run this cell to get initial state
                added_shape = self.embed_size + self.params.n_classes
                im_f = tf.layers.dense(self.images_fv, added_shape)
                _, initial_state0 = cell_0(im_f, zero_state0)
                # c = h = tf.layers.dense(self.images_fv,
                #                         self.params.decoder_hidden,
                #                         name='dec_init_map')
                # initial_state0 = (tf.nn.rnn_cell.LSTMStateTuple(c, h), )
                # x, y
                y = tf.tile(tf.expand_dims(labels, 1),
                            [1, tf.shape(vect_inputs)[1], 1])
                vect_inputs = tf.concat([vect_inputs, tf.to_float(y)], 2)
                outputs, final_state = tf.nn.dynamic_rnn(cell_0,
                                                         inputs=vect_inputs,
                                                         sequence_length=lengths,
                                                         initial_state=initial_state0,
                                                         swap_memory=True,
                                                         dtype=tf.float32,
                                                         scope=scope1)
            # [batch_size, 2 * lstm_hidden_size]
            # final_state = ((c, h), )
            final_state = final_state[0][1]
            lz_mean = layers.dense(inputs=final_state,
                                   units=self.latent_size,
                                   activation=None)
            lz_logstd = layers.dense(inputs=final_state,
                                     units=self.latent_size,
                                     activation=None)
            lz_std = tf.exp(lz_logstd)
            # define latent variable`s Stochastic Tensor
            # add mu_k, sigma_k, CVAe ag-cvae
            tm_list = []  # means
            tl_list = []  # log standard deviations
            z = zs.Normal('z', mean=lz_mean, std=lz_std, group_ndims=1,
                          n_samples=self.z_samples)
        return model, tm_list, tl_list
