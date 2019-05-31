import os
import tensorflow as tf
import numpy as np
import zhusuan as zs

# Data
from utils.data import Data
from ops.inference import inference
# Model parts
from vae_model.classifier import Classifier
from vae_model.encoder import Encoder
from vae_model.decoder import Decoder
# Image features extraction
from utils.image_embeddings import vgg16, ResNet

# TODO: finish it !

class ScvaeModel():
    def __init__(self, params, data, mode="train_eval"):
        assert mode in ("train_eval", "out_gen"), "train_eval or out_gen"
        self._params = params
        self._data = data
        self._img_size = [224, 224]  # Resnet input size
        if mode == "train_eval":
            self._batch_size = params.batch_size
            self.__build_model()
    
    def __set_placeholders(self):
        # labelled placeholders
        self._cap_enc_l = tf.placeholder(
            tf.int32, [None, None], name='cap_enc_l')
        self._cap_dec_l = tf.placeholder(
            tf.int32, [None, None], name='cap_dec_l')
        self._y_labels = tf.placeholder(
            tf.int32, [None], name='y_labels')
        self._cap_len_l = tf.placeholder(
            tf.int32, [None], name='cap_len_l')
        # unlabelled placeholders
        self._cap_enc_u = tf.placeholder(
            tf.int32, [None, None], name='cap_enc_u')
        self._cap_dec_u = tf.placeholder(
            tf.int32, [None, None], name='cap_dec_u')
        self._cap_len_u = tf.placeholder(
            tf.int32, [None], name='cap_len_u')
        # Image paths
        self._impaths = tf.placeholder(tf.string, [None], "impaths")
    
    def __img_embeddings(self):
        import multiprocessing
        def _parse_function(filename):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image_resized = tf.image.resize_images(
                image_decoded, self._img_size)
            return image_resized
        dataset = tf.data.Dataset.from_tensor_slices(self._impaths)
        dataset = dataset.map(
            _parse_function, num_parallel_calls=multiprocessing.cpu_count())
        dataset = dataset.batch(self._batch_size)
        iterator = dataset.make_initializable_iterator()
        self._iterator = iterator
        images = iterator.get_next()
        resnet = ResNet(50)  # Have checkpoints fo ResNet 50
        self._img_embed_size = resnet.final_size
        is_training = tf.constant(False)
        im_features = resnet(images, is_training)
        self._im_features = im_features

    def __log_joint(self, observed):
        n_x = tf.shape(observed['x'])[1]
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            model, _, _ = decoder.px_z_y(observed, n_x=n_x)
        log_px_zy, log_pz, log_py = model.local_log_prob(['x', 'z', 'y'])
        log_px_zy = tf.Print(log_px_zy, [tf.shape(log_px_zy), tf.shape(log_pz),
                                        tf.shape(log_py)], first_n=1)
        return log_px_zy + log_pz + log_py

    def __build_model(self):
        y_one_hot = tf.one_hot(
            self._y_labels, self._params.n_classes, dtype=tf.int32)
        # ss-vae model
        vocab_size = self._data.dictionary.vocab_size
        classifier = Classifier(
            self._im_features, 
            vocab_size,
            self._params.embed_size, 
            self._params.class_hidden)
        encoder = Encoder(
            self._im_features, 
            self._params.encoder_hidden,
            'Normal',
            self._params.latent_size,
            vocab_size,
            self._params.embed_size,
            self._params.gen_z_samples,
            self._params)
        decoder = Decoder(
            self._im_features,
            self._params,
            self._data.dictionary,
            n_classes=self._params.n_classes)
        # labelled lower bound calculation
        with tf.variable_scope('variational'):
            variational, _, _ = encoder.q_z_xy(
                self._cap_enc_l,
                y_one_hot,
                self._cap_len_l)    
        qz_samples, log_qz = variational.query(
            'z', outputs=True, local_log_prob=True)
        # Resizing to [n_samples, b_size, caption_len]
        x_labelled_obs = tf.tile(tf.expand_dims(self._cap_enc_l, 0),
                         [self._params.gen_z_samples, 1, 1])
        y_labeled_obs = tf.tile(tf.expand_dims(y_one_hot, 0),
                        [self._params.gen_z_samples, 1, 1])
        l_b = zs.variational.elbo(self.__log_joint,
                          observed={'x': x_labelled_obs, 'y': y_labeled_obs},
                          latent={'z': [qz_samples, log_qz]}, axis=0)
        labeled_lower_bound = tf.reduce_mean(l_b)
        # Unlabelled lower bound
        # tile y to sum later
        n_classes = self._params.n_classes
        y_diag = tf.diag(tf.ones(n_classes, dtype=tf.int32))
        y_u = tf.reshape(tf.tile(tf.expand_dims(y_diag, 0), 
        [tf.shape(self._cap_enc_u)[0], 1, 1]), [-1, n_classes])
        x_u = tf.reshape(tf.tile(tf.expand_dims(self._cap_enc_u, 1), [
            1, n_classes, 1]), [-1, tf.shape(self._cap_enc_u)[1]])
        len_u = tf.reshape(tf.tile(tf.expand_dims(
            self._cap_len_u, 1), [1, n_classes]),
            [n_classes * tf.shape(self._cap_len_u)[0]])
        # decoder inps
        dec_u = tf.reshape(tf.tile(tf.expand_dims(self._cap_dec_u, 1), [
            1, n_classes, 1]), [-1, tf.shape(self._cap_dec_u)[1]])
        # images
        images_u = tf.reshape(
            tf.tile(tf.expand_dims(self._im_features, 1), 
            [1, n_classes, 1]), [-1, self._params.embed_size])
        with tf.variable_scope("variational", reuse=tf.AUTO_REUSE):
            variational, _, _ = encoder.q_z_xy(x_u, y_u, len_u, images_u)
        qz_samples, log_qz = variational.query(
            'z', outputs=True, local_log_prob=True)
        y_unlabeled_obs = tf.tile(tf.expand_dims(y_u, 0),
                          [self._params.gen_z_samples, 1, 1])
        x_unlabeled_obs = tf.tile(tf.expand_dims(x_u, 0),
                                [self._params.gen_z_samples, 1, 1])
        decoder.captions = dec_u
        decoder.lengths = len_u
        decoder.images_fv = images_u
        lb_z = zs.variational.elbo(self.__log_joint,
                                observed={'x': x_unlabeled_obs,
                                        'y': y_unlabeled_obs},
                                latent={'z': [qz_samples, log_qz]}, axis=0)
        # sum over y
        lb_z = tf.reshape(lb_z, [-1, n_classes])
        with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):
            qy_logits_u = classifier.q_y_x(
                self._cap_enc_u, self._cap_len_u, n_classes)
        qy_u = tf.reshape(tf.nn.softmax(qy_logits_u), [-1, tf.shape(y_one_hot)[1]])
        qy_u += 1e-8
        qy_u /= tf.reduce_sum(qy_u, 1, keep_dims=True)
        log_qy_u = tf.log(qy_u)  # [batch_size, n_classes]
        unlabeled_lower_bound = tf.reduce_mean(
            tf.reduce_sum(qy_u * (lb_z - log_qy_u), 1))
        # Build classifier
        with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):
            qy_logits_l = classifier.q_y_x(
                self._cap_enc_l, self._cap_len_l, n_classes)
        qy_l = tf.nn.softmax(qy_logits_l)
        pred_y = tf.argmax(qy_l, 1)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred_y,
                                            tf.argmax(y_one_hot, 1)), tf.float32))
        onehot_cat = zs.distributions.OnehotCategorical(qy_logits_l)
        log_qy_x = onehot_cat.log_prob(y_one_hot)
        beta = 0.1 * self._params.batch_size # 0.1 * batch_size
        classifier_cost = -beta * tf.reduce_mean(log_qy_x)
        self.cost = -(labeled_lower_bound + unlabeled_lower_bound - classifier_cost) / 2.
