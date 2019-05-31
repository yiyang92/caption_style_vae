import os
from glob import glob
import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm

from random import shuffle
from utils.captions import Dictionary
from utils.image_utils import load_image
from utils.image_embeddings import vgg16, ResNet


class Data():
    def __init__(self, images_dir, params, pickles_dir='./pickles',
                 keep_words=1, n_classes=3, all_30k_set=False,
                 partial=0.0, vocab_fn='00'):
        """Data loader class.
        Args:
            images_dir: images directory
            pickles_dir: data directory, should contain captions_tr.pkl,
        captions_val.pkl and captions_test.pkl, whick can be obtained by
        launching preprocessing script
            keep_words: minimum number of times word appers to be included in
        vocabulary
            n_classes: how many classes to include in training
            all_30k_set: whether to use additional f30k data, used in exps.
            partial: adding part of f30k data
        """
        self.images_dir = images_dir
        self.pickles_dir = pickles_dir
        self.params = params
        # labelled (+ unlabelled)
        if all_30k_set:
            self.train_captions = self._load_captions('captions_tr.pkl')
        else:
            self.train_captions = self._load_captions('captions_ltr.pkl')
            if partial > 0.0:
                self._add_partially(partial)
        self.val_captions = self._load_captions('captions_val.pkl')
        self.test_captions = self._load_captions('captions_test.pkl')
        print("Train data: ", len(self.train_captions.keys()))
        self.dictionary = Dictionary(self.train_captions, keep_words)
        self._save_dict(vocab_fn)
        # Image embeddings
        self.img_embed = params.img_embed
        if self.img_embed == "resnet":
            self.weights_path = params.resnet_cp_path
        elif self.img_embed == "vgg":
            self.weights_path = params.vgg_weights_path
        else:
            raise ValueError("Must choose between VGG16 or ResNet")
        self.im_features = self._extract_features_from_dir()
        # number of classes
        self.n_classes = n_classes

    @property
    def train_set_len(self):
        return len(self.train_captions.keys())

    def _save_dict(self, vocab_fn):
        with open('./pickles/capt_dict_{}.pickle'.format(
                vocab_fn), 'wb') as wf:
            pickle.dump(file=wf, obj=self.dictionary)

    def _load_captions(self, f_name):
        with open(os.path.join(self.pickles_dir, f_name), 'rb') as rf:
            return pickle.load(rf)

    def _add_partially(self, percentage):
        # TODO: vocabulary?
        """Partially adds f30k data for experiments."""
        # presave for future usages
        save_dict = 'train_fpart_{}'.format(percentage)
        if os.path.exists('./pickles/' + save_dict):
            self.train_captions = self._load_captions(save_dict)
        else:
            # load 'captions_tr.pkl'
            f30k = self._load_captions('captions_tr.pkl')
            total_f30 = len(f30k.keys()) - len(self.train_captions.keys())
            # check whether image name in f30kstyle
            im_stylized = set(self.train_captions.keys())
            fl30k_ctr = 0
            limit = total_f30 * percentage
            for imn in f30k:
                if fl30k_ctr >= limit:
                    break
                if imn in im_stylized:
                    continue
                else:
                    self.train_captions[imn] = f30k[imn]
                    fl30k_ctr += 1
            with open('./pickles/' + save_dict, 'wb') as wf:
                pickle.dump(self.train_captions, wf)
            print("f30k part: {}".format(fl30k_ctr))
            print("stylized part: {}".format(len(im_stylized)))

    def get_batch(self, batch_size, set='train', im_features=True,
                  get_names=False, label=None):
        """Get batch."""
        # if select inly one caption
        imn_batch = [None] * batch_size
        if set == 'train':
            self._iterable = self.train_captions.copy()
        elif set == 'val':
            self._iterable = self.val_captions.copy()
        else:
            self._iterable = self.test_captions.copy()
        im_names = list(self._iterable.keys())
        shuffle(im_names)
        for i, item in enumerate(im_names):
            inx = i % batch_size
            imn_batch[inx] = item
            if inx == batch_size - 1:
                # images or features
                images = self._get_images(imn_batch, im_features)
                labelled, unlabelled, labels,\
                    lengths_l, lengths_u = self._form_captions_batch(
                        imn_batch, self._iterable, label)
                # 'romantic', 'humorous', 'actual' (optionally) types
                ret = (labelled, unlabelled, labels,
                       images, lengths_l, lengths_u)
                if get_names:
                    ret += (imn_batch,)
                yield ret
                imn_batch = [None] * batch_size
        if imn_batch[0]:
            imn_batch = [item for item in imn_batch if item]
            images = self._get_images(imn_batch, im_features)
            labelled, unlabelled, labels,\
                lengths_l, lengths_u = self._form_captions_batch(
                    imn_batch, self._iterable, label)
            ret = (labelled, unlabelled, labels,
                   images, lengths_l, lengths_u)
            if get_names:
                ret += (imn_batch,)
            yield ret

    def _form_captions_batch(self, imn_batch, captions, label=None):
        # randomly choose 2 captions, labelled, one unlabelled
        labels = np.zeros(len(imn_batch))
        labelled, unlabelled = [], []
        lengths_l = np.zeros((len(imn_batch)))
        lengths_u = np.zeros((len(imn_batch)))
        for i, imn in enumerate(imn_batch):
            cap_dict = captions[imn]
            try:  # in case if no stylized captions
                hum_c, rom_c = cap_dict['humorous'], cap_dict['romantic']
                hum_c = self.dictionary.index_caption(hum_c[0])
                rom_c = self.dictionary.index_caption(rom_c[0])
                cap_list = [hum_c, rom_c]
            except:
                cap_list = []
            act_c_list = cap_dict['actual']
            # randomly choose one of the 5 actual captions
            rand_cap = np.random.randint(low=0, high=len(act_c_list))
            act_c = self.dictionary.index_caption(act_c_list[rand_cap])
            # important, label-label_index correspondance
            cap_list.append(act_c)
            # what will be labelled, what unlabelled
            l_index = np.random.randint(low=0, high=len(cap_list), size=2)
            only_act = False
            if len(cap_list) == 1:
                only_act = True
                rand_cap = np.random.randint(low=0, high=len(act_c_list))
                act_c = self.dictionary.index_caption(act_c_list[rand_cap])
                cap_list.append(act_c)  # unlabeled input = also actual
            if label is None:
                labelled.append(cap_list[l_index[0]])
                unlabelled.append(cap_list[l_index[1]])
                if only_act:
                    labels[i] = 2  # dont want to change labels order (for now)
                else:
                    labels[i] = l_index[0]
            else:
                labelled.append(cap_list[label])
                unlabelled.append(cap_list[l_index[1]])
                labels[i] = label
            lengths_l[i] = len(labelled[i]) - 1
            lengths_u[i] = len(unlabelled[i]) - 1
        pad_l = len(max(labelled, key=len))
        pad_u = len(max(unlabelled, key=len))
        captions_l_inp = np.array([cap[:-1] + [0] * (
            pad_l - len(cap)) for cap in labelled])
        captions_l_lbl = np.array([cap[1:] + [0] * (
            pad_l - len(cap)) for cap in labelled])
        labelled = (captions_l_inp, captions_l_lbl)
        captions_u_inp = np.array([cap[:-1] + [0] * (
            pad_u - len(cap)) for cap in unlabelled])
        captions_u_lbl = np.array([cap[1:] + [0] * (
            pad_u - len(cap)) for cap in unlabelled])
        unlabelled = (captions_u_inp, captions_u_lbl)
        return labelled, unlabelled, labels, lengths_l, lengths_u

    def _get_images(self, imn_batch, im_features=True):
        images = []
        if im_features:
            for name in imn_batch:
                images.append(self.im_features[name])
        else:
            for name in imn_batch:
                img = load_image(os.path.join(self.images_dir, name))
                images.append(img)
        return np.stack(np.squeeze(images))

    def _extract_features_from_dir(self, save_pickle=True, im_shape=(224,
                                                                     224)):
        """
        Args:
            data_dir: image data directory
            save_pickle: bool, will serialize feature_dict and save it into
        ./pickle directory
            im_shape: desired images shape
        Returns:
            feature_dict: dictionary of the form {image_name: feature_vector}
        """
        feature_dict = {}
        data_dir = self.images_dir
        if self.img_embed == "resnet":
            embed_file = os.path.join("./pickles/", "img_embed_res.pickle")
        else:
            embed_file = os.path.join("./pickles/", "img_embed_vgg.pickle")
        try:
            with open(embed_file, 'rb') as rf:
                print("Loading prepared feature vector from {}".format(
                    embed_file
                ))
                feature_dict = pickle.load(rf)
        except:
            print("Extracting features")
            if not os.path.exists("./pickles"):
                os.makedirs("./pickles")
            im_embed = tf.Graph()
            with im_embed.as_default():
                input_img = tf.placeholder(tf.float32, [None,
                                                        im_shape[0],
                                                        im_shape[1], 3])
                if self.img_embed == "resnet":
                    image_embeddings = ResNet(50)
                    is_training = tf.constant(False)
                    features = image_embeddings(input_img, is_training)
                    saver = tf.train.Saver()
                else:
                    image_embeddings = vgg16(input_img)
                    features = image_embeddings.fc2
                gpu_options = tf.GPUOptions(
                    visible_device_list=self.params.gpu, 
                    allow_growth=True)
                config = tf.ConfigProto(gpu_options=gpu_options)
            with tf.Session(graph=im_embed, config=config) as sess:
                if len(list(glob(data_dir + '*.jpg'))) == 0:
                    raise FileNotFoundError()
                if self.img_embed == "resnet":
                    print("loading resnet weights")
                    saver.restore(sess, self.weights_path)
                else:
                    print("loading vgg16 imagenet weights")
                    image_embeddings.load_weights(self.weights_path, sess)
                for img_path in tqdm(glob(data_dir + '*.jpg')):
                    img = load_image(img_path)
                    img = np.expand_dims(img, axis=0)
                    f_vector = sess.run(features, {input_img: img})
                    # ex. COCO_val2014_0000000XXXXX.jpg
                    feature_dict[img_path.split('/')[-1]] = f_vector
            if save_pickle:
                with open(embed_file, 'wb') as wf:
                    pickle.dump(feature_dict, wf)
        return feature_dict
