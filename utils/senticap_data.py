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


class SenticapData():
    def __init__(
        self, images_dir, params, keep_words, vocab_fn, senticap_dir):
        """Data loader class."""
        self.images_dir = images_dir
        self.params = params
        #
        self.train_captions = self._load_captions(
            os.path.join(senticap_dir, "train_sc.pickle"))
        self.val_captions = self._load_captions(
            os.path.join(senticap_dir, "val_sc.pickle"))
        self.test_captions = self._load_captions(
            os.path.join(senticap_dir, "test_sc.pickle"))
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
        self.n_classes = 3  # Positive and ndegative sentiment + factual

    @property
    def train_set_len(self):
        return len(self.train_captions.keys())
    
    def _save_dict(self, vocab_fn):
        with open('./pickles/capt_dict_{}.pickle'.format(vocab_fn), 'wb') as wf:
            pickle.dump(file=wf, obj=self.dictionary)

    def _load_captions(self, f_name):
        with open(f_name, 'rb') as rf:
            return pickle.load(rf)

    def get_batch(self, batch_size, set='train', im_features=True,
                  get_names=False, label=None, add_act=True):
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
                        imn_batch, self._iterable, label, add_act=add_act)
                # 'positive', 'negative', 'actual' (optionally) types
                ret = (labelled, unlabelled, labels,
                       images, lengths_l, lengths_u)
                if get_names:
                    ret += (imn_batch,)
                yield ret
                imn_batch = [None] * batch_size
        if imn_batch[0]:
            imn_batch = [item for item in imn_batch if item]
            images = self._get_images(imn_batch, im_features)
            if len(images.shape) == 1:
                images = np.expand_dims(images, 0)
            labelled, unlabelled, labels,\
                lengths_l, lengths_u = self._form_captions_batch(
                    imn_batch, self._iterable, label, add_act=add_act)
            ret = (labelled, unlabelled, labels,
                   images, lengths_l, lengths_u)
            if get_names:
                ret += (imn_batch,)
            yield ret

    def _form_captions_batch(
        self, imn_batch, captions, label=None, add_act=True):
        # randomly choose 2 captions, labelled, one unlabelled
        labels = np.zeros(len(imn_batch))
        labelled, unlabelled = [], []
        lengths_l = np.zeros((len(imn_batch)))
        lengths_u = np.zeros((len(imn_batch)))
        gen_labels = {"positive": 0, "negative": 1, "actual": 2}
        for i, imn in enumerate(imn_batch):
            cap_dict = captions[imn]
            pos = cap_dict.get("positive", 0)
            neg = cap_dict.get("negative", 0)
            caps = [(pos, "positive"), (neg, "negative")]
            # Adding random choice
            cap_list = []
            for cap in caps:
                if cap[0] == 0:
                    continue
                shuffle(cap[0])
                cap_ = self.dictionary.index_caption(cap[0][0])
                cap_list.append((cap_, cap[1]))
            if add_act:  # Add act if available
                act_c_list = cap_dict['actual']
                if len(act_c_list) != 0:
                    # randomly choose one of the 5 actual captions
                    rand_cap = np.random.randint(low=0, high=len(act_c_list))
                    act_c = self.dictionary.index_caption(act_c_list[rand_cap])
                    # important, label-label_index correspondance
                    cap_list.append((act_c, "actual"))
            if len(cap_list) == 0:
                raise ValueError("No captions in caplist")
            # what will be labelled, what unlabelled
            l_index = np.random.randint(low=0, high=len(cap_list), size=2)
            only_act = False
            if len(cap_list) == 1 and cap_list[0][1] == "actual": # only if actual
                only_act = True
                rand_cap = np.random.randint(low=0, high=len(act_c_list))
                act_c = self.dictionary.index_caption(act_c_list[rand_cap])
                cap_list.append((act_c, "actual"))  # unlabeled input = also actual
            if label is None:
                l_capt = cap_list[l_index[0]]
                labelled.append(l_capt[0])
                unlabelled.append(cap_list[l_index[1]][0])
                if only_act:
                    labels[i] = gen_labels["actual"]
                else:
                    labels[i] = gen_labels[l_capt[1]]
            else:
                labelled.append(np.zeros((1)))
                unlabelled.append(np.zeros(1))
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
        # Impaths in form folder/imname.jpg
        # ex. val2014/COCO_val2014_0000000XXXXX.jpg
        im_paths = list(
            self.train_captions.keys()) + list(
                self.val_captions.keys()) + list(self.test_captions.keys())
        if self.img_embed == "resnet":
            embed_file = os.path.join("./pickles/", "sc_img_embed_res.pickle")
        else:
            embed_file = os.path.join("./pickles/", "sc_img_embed_vgg.pickle")
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
                # gpu_options = tf.GPUOptions(
                #     visible_device_list=self.params.gpu, 
                #     allow_growth=True)
                config = tf.ConfigProto()
            with tf.Session(graph=im_embed, config=config) as sess:
                if self.img_embed == "resnet":
                    print("loading resnet weights")
                    saver.restore(sess, self.weights_path)
                else:
                    print("loading vgg16 imagenet weights")
                    image_embeddings.load_weights(self.weights_path, sess)
                for img_path in tqdm(im_paths):
                    # Get the full path
                    img_path_ = os.path.join(self.images_dir, img_path)
                    img = load_image(img_path_) 
                    img = np.expand_dims(img, axis=0)
                    f_vector = sess.run(features, {input_img: img})
                    feature_dict[img_path] = f_vector
            if save_pickle:
                with open(embed_file, 'wb') as wf:
                    pickle.dump(feature_dict, wf)
        return feature_dict
