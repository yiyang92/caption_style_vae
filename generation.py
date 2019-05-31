# caption generation API
import tensorflow as tf
import pickle
import cv2

from vae_model.decoder import Decoder
from utils.image_embeddings import vgg16, ResNet


class CaptionGen():
    def __init__(self, checkpoint, params_path, dict_path, cnn_w_path=None):
        self._checkpoint = checkpoint
        self._dict = self._load_dict(dict_path)
        self._params = self._load_params(params_path)
        try:  # for backward compatibility with old params
            self._img_embed_ = self._params.img_embed
        except:
            self._img_embed_ = "vgg"
        if self._img_embed_ == "resnet":
            self._cnn_w_path = self._params.resnet_cp_path
            self._img_embed_dim = 2048
        else:
            try:
                self._cnn_w_path = self._params.vgg_weights_path
            except:
                self._cnn_w_path = cnn_w_path
            self._img_embed_dim = 4096
        # if need some spetial path
        if cnn_w_path is not None:
            self._cnn_w_path = cnn_w_path
        self._built = False
        self._fgraph_built = False
    
    @property
    def model_built(self):
        return self._built

    def _load_dict(self, vocab_path):
        with open(vocab_path, 'rb') as rf:
            return pickle.load(rf)

    def load_image(self, image_path, shape=(224, 224)):
        img = cv2.imread(image_path)
        img = cv2.resize(img, shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # handle grayscale input images
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
        return img

    def _build_feature_graph(self):
        im_embed = tf.Graph()
        with im_embed.as_default():
            self.input_img = tf.placeholder(tf.float32, [224, 224, 3])
            input_image = tf.expand_dims(self.input_img, 0)
            if self._img_embed_ == "resnet":
                image_embeddings = ResNet(50)
                is_training = tf.constant(False)
                features = image_embeddings(input_image, is_training)
                self._resnet_saver = tf.train.Saver()
            else:
                image_embeddings = vgg16(input_image)
                features = image_embeddings.fc2
        return im_embed, features, image_embeddings

    def extract_features(self, image):
        if not self._fgraph_built:
            embed_graph, features, im_emb = self._build_feature_graph()
            f_extr_sess = self._set_session(embed_graph)
            if self._img_embed_ == "resnet":
                self._resnet_saver.restore(
                    f_extr_sess, self._cnn_w_path)
            else:
                im_emb.load_weights(self._cnn_w_path, f_extr_sess)
            self._fgraph_built = True
            self._features = features
            self.f_extr_sess = f_extr_sess
        feed = {self.input_img: image}
        return self.f_extr_sess.run(self._features, feed_dict=feed)

    def _lbls_idx(self, style_label):
        label = None
        try:
            # [hum_c, rom_c, act_c]
            gen_labels = {'humorous': 0, 'romantic': 1, 'actual': 2}
            label = gen_labels[style_label]
            labels_idx = {idx:lbl for lbl, idx in gen_labels.items()}
        except KeyError:
            pass
        try:
            gen_labels = {"positive": 0, "negative": 1, "actual": 2}
            label = gen_labels[style_label]
            labels_idx = {idx:lbl for lbl, idx in gen_labels.items()}
        except KeyError:
            pass
        if label == None:
            raise KeyError("Labels not right")
        return label, labels_idx

    def _load_params(self, params_path):
        """Load serialized Parameters class, for convenience"""
        with open(params_path, 'rb') as rf:
            params = pickle.load(rf)
            return params

    def _get_saver(self):
        return tf.train.Saver(tf.trainable_variables())

    def _set_session(self, graph=tf.get_default_graph()):
        return tf.InteractiveSession(graph=graph)

    def _restore_gen_weights(self, sess):
        saver = self._get_saver()
        saver.restore(save_path=self._checkpoint, sess=sess)

    def build_model(self):
        params = self._params
        # placeholders
        cap_dec_l = tf.placeholder(tf.int32, [None, None], name='cap_dec_l')
        cap_len_l = tf.placeholder(tf.int32, [None], name='cap_len_l')
        self.image_batch = tf.placeholder(
            tf.float32, [None, self._img_embed_dim], name='image_batch')
        images_fv = tf.layers.dense(self.image_batch, params.embed_size,
                                    name='imf_emb')
        self.decoder = Decoder(images_fv, cap_dec_l, cap_len_l, params,
                               self._dict, n_classes=params.n_classes)
        # initialize network
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            _, _, _ = self.decoder.px_z_y({}, cap_dec_l, cap_len_l, False)
        print("Model built")
        self._built = True

    def predict(self, im_path, style_label):
        if not self._built:
            self.build_model()
        label, labels_idx = self._lbls_idx(style_label)
        labels = [label]
        img = self.load_image(im_path)
        features = self.extract_features(img)
        # images.shape==[1, 4096]
        sess = self._set_session()
        self._restore_gen_weights(sess)
        sent, _ = self.decoder.online_inference(
            sess, ['000_000_000'], features, 
            self.image_batch, labels, labels_names=labels_idx)
        return sent

    
if __name__ == "__main__":
    # Solely for testing purposes
    # need to provide dictionary, parameters, checkpoint
    # cap_gen = CaptionGen('./checkpoints/07.ckpt',
    #                      './pickles/params_07.pickle',
    #                      './pickles/capt_dict_07.pickle',
    #                      './utils/vgg16_weights.npz')
    cap_gen = CaptionGen('./checkpoints/setn_cap_10k.ckpt',
                        './pickles/params_setn_cap_10k.pickle',
                        './pickles/capt_dict_senticap_vocab.pickle')
    sent = cap_gen.predict(
        './images/COCO_val2014_000000000196.jpg', 'positive')
    print(sent[0]['caption'])
