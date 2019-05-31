import re

import tensorflow as tf
from tensorflow import layers
import zhusuan as zs
from utils.rnn_model import make_rnn_cell, rnn_placeholders
import numpy as np

from utils.top_n import TopN, Beam


class Decoder():
    """Decoder class."""

    def __init__(self, images_fv, captions, lengths,
                 params, data_dict, n_classes=2):
        """
        Args:
            images_fv: image features mapping to word embeddings
            captions: captions input placeholder
            lengths: caption length without zero-padding, placeholder
            params: Parameters() class instance
            data_dict: Dictionary() class instance, used for caption generators
        dynamic_rnn lengths
        """
        self.images_fv = images_fv
        self.captions = captions
        self.lengths = lengths
        self.params = params
        self.data_dict = data_dict
        self.cap_clusters = None
        self.n_classes = n_classes

    def px_z_y(self, observed, captions=None, lengths=None, gen_mode=False,
               n_x=None):
        """
        Args:
            observed: for q, parametrized by encoder, used during training
        Returns:
            model: zhusuan model object, can be used for getting probabilities
        """
        if captions is not None and lengths is not None:
            self.captions = captions
            self.lengths = lengths
        if n_x is None:
            n_x = tf.shape(self.images_fv)[0]
        with zs.BayesianNet(observed) as model:
            z_mean = tf.zeros([n_x, self.params.latent_size])
            z = zs.Normal('z', mean=z_mean, std=self.params.std,
                          group_ndims=1,
                          n_samples=self.params.gen_z_samples)
            tf.summary.histogram("distributions/z", z)
            y_logits = tf.zeros([n_x, self.n_classes])
            y = zs.OnehotCategorical('y', y_logits,
                                     n_samples=self.params.gen_z_samples)
            with tf.variable_scope("net"):
                embedding = tf.get_variable(
                        "dec_embeddings", [self.data_dict.vocab_size,
                                           self.params.embed_size],
                        dtype=tf.float32)
                # word dropout
                before = tf.reshape(self.captions, [-1])
                word_drop_keep = self.params.word_dropout_keep
                if gen_mode:
                    word_drop_keep = 1.0
                captions = tf.nn.dropout(tf.to_float(self.captions),
                                         word_drop_keep)
                after = tf.reshape(tf.to_int32(captions), [-1])
                mask_after = tf.to_int32(tf.not_equal(before, after))
                to_unk = mask_after * self.data_dict.word2idx['<UNK>']
                captions = tf.reshape(tf.add(after, to_unk),
                                      [tf.shape(self.images_fv)[0], -1])
                vect_inputs = tf.nn.embedding_lookup(embedding, captions)
                dec_lstm_drop = self.params.dec_lstm_drop
                if gen_mode:
                    dec_lstm_drop = 1.0
                cell_0 = make_rnn_cell(
                    [self.params.decoder_hidden],
                    base_cell=tf.contrib.rnn.LSTMCell,
                    dropout_keep_prob=dec_lstm_drop)
                # zero_state0 = cell_0.zero_state(
                #     batch_size=tf.shape(self.images_fv)[0],
                #     dtype=tf.float32)
                # run this cell to get initial state
                added_shape = self.params.gen_z_samples * self.params.n_classes +\
                 self.params.embed_size
                # added_shape = self.params.embed_size
                # f_mapping = tf.layers.dense(self.images_fv, added_shape,
                #                             name='f_emb2')
                c = h = tf.layers.dense(self.images_fv,
                                        self.params.decoder_hidden,
                                        name='dec_init_map')
                initial_state0 = (tf.nn.rnn_cell.LSTMStateTuple(c, h), )
                # vector z, mapped into embed_dim
                z = tf.concat([z, tf.to_float(y)], 2)
                z = tf.reshape(z, [n_x, (self.params.latent_size
                                         + self.n_classes)\
                                   * self.params.gen_z_samples])
                z_dec = layers.dense(z, added_shape, name='z_rnn')
                _, z_state = cell_0(z_dec, initial_state0)
                initial_state = rnn_placeholders(z_state)
                # concat with inputs
                y_re = tf.to_float(
                    tf.reshape(y, [tf.shape(self.images_fv)[0],
                                   self.params.gen_z_samples * self.params.n_classes]))
                y = tf.tile(tf.expand_dims(y_re, 1),
                            [1, tf.shape(vect_inputs)[1], 1])
                vect_inputs = tf.concat([vect_inputs, y], 2)
                # vect_inputs = tf.Print(vect_inputs, [tf.shape(vect_inputs)],
                #                        first_n=1)
                # captions LSTM
                outputs, final_state = tf.nn.dynamic_rnn(cell_0,
                                                         inputs=vect_inputs,
                                                         sequence_length=self.lengths,
                                                         initial_state=initial_state,
                                                         swap_memory=True,
                                                         dtype=tf.float32)
            # output shape [batch_size, seq_length, self.params.decoder_hidden]
            if gen_mode:
                # only interested in the last output
                outputs = outputs[:, -1, :]
            outputs_r = tf.reshape(outputs, [-1, cell_0.output_size])
            x_logits = tf.layers.dense(outputs_r,
                                       units=self.data_dict.vocab_size,
                                       name='rnn_logits')
            x_logits_r = tf.reshape(x_logits, [tf.shape(outputs)[0],
                                               tf.shape(outputs)[1], -1])
            x = zs.Categorical('x', x_logits_r, group_ndims=1)
            # for generating
            sample = None
            if gen_mode:
                if self.params.sample_gen == 'sample':
                    sample = tf.multinomial(
                        x_logits / self.params.temperature, 1)[0][0]
                elif self.params.sample_gen == 'beam_search':
                    sample = tf.nn.softmax(x_logits)
                else:
                    sample = tf.nn.softmax(x_logits)
        return model, x_logits, (initial_state, final_state, sample)

    def online_inference(self, sess, picture_ids, in_pictures, image_ph,
                         labels, stop_word='<EOS>', ground_truth=None,
                         labels_names=None):
        """Generate caption, given batch of pictures and their ids (names).
        Args:
            sess: tf.Session() object
            picture_ids: list of picture ids in shape [batch_size]
            in_pictures: input pictures
            stop_word: when stop caption generation
            image_f_inputs: image placeholder
        Returns:
            cap_list: list of format [{'image_id', caption: ''}]
            cap_raw: list of generated caption indices
        """
        # get stop word index from dictionary
        stop_word_idx = self.data_dict.word2idx['<EOS>']
        cap_list = [None] * in_pictures.shape[0]
        # set placeholders
        captions = tf.placeholder(tf.int32, [1, None])
        lengths = tf.placeholder(tf.int32, [None], name='seq_length')
        labels_ph = tf.placeholder(tf.int32, [None], name='labels')
        y_one_hot = tf.one_hot(labels_ph, self.n_classes, dtype=tf.int32)
        labels_tiled = tf.tile(tf.expand_dims(y_one_hot, 0),
                               [self.params.gen_z_samples, 1, 1])
        # initialize caption generator
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            _, _, states = self.px_z_y({'y': labels_tiled},
                                       captions, lengths, True)
        init_state, out_state, sample = states
        cap_raw = []
        for i in range(len(in_pictures)):
            state = None
            if ground_truth is not None:
                b_index = self.data_dict.word2idx['<BOS>']
                e_index = self.data_dict.word2idx['<EOS>']
                g_truth = ' '.join([self.data_dict.idx2word[word]
                                    for word in ground_truth[i]
                                    if word not in [b_index, e_index, 0]])
            if self.params.data == "stylenet":
                image_id = int(picture_ids[i].split('.')[0])
            else:
                image_id = int(re.findall(r"[0-9]+", picture_ids[i])[2])
            cap_list[i] = {'image_id': image_id,
                           'caption': ' ',
                           'label': labels_names[int(labels[i])]}
            if ground_truth is not None:
                cap_list[i].update({'ground_truth': g_truth})
            sentence = ['<BOS>']
            cur_it = 0
            gen_word_idx = 0
            cap_raw.append([])
            while (cur_it < self.params.gen_max_len):
                input_seq = [self.data_dict.word2idx[word] for word in sentence]
                feed = {captions: np.array(input_seq)[-1].reshape([1, 1]),
                        lengths: [len(input_seq)],
                        image_ph: np.expand_dims(in_pictures[i], 0),
                        labels_ph: [labels[i]]}
                # for the first decoder step, the state is None
                if state is not None:
                    feed.update({init_state: state})
                next_word_probs, state = sess.run([sample, out_state],
                                                  feed)
                next_word_probs = next_word_probs.ravel()
                t = self.params.temperature
                next_word_probs = next_word_probs**(
                    1/t) / np.sum(next_word_probs**(1/t))
                gen_word_idx = np.argmax(next_word_probs)
                # elif self.params.sample_gen == 'sample':
                #     gen_word_idx = next_word_probs
                gen_word = self.data_dict.idx2word[gen_word_idx]
                sentence += [gen_word]
                cap_raw[i].append(gen_word_idx)
                cur_it += 1
                if gen_word_idx == stop_word_idx:
                    break
            cap_list[i]['caption'] = ' '.join([word for word in sentence
                                               if word not in ['<BOS>',
                                                               '<EOS>']])
            # print(cap_list[i]['caption'] + ' ' + cap_list[i]['label'])
            # print("Ground truth caption: ", cap_list[i]['ground_truth'])
        return cap_list, cap_raw

    def beam_search(self, sess, picture_ids, in_pictures, image_ph,
                    labels, ground_truth=None, beam_size=2,
                    ret_beams=False, len_norm_f=0.7, labels_names=None):
        """Generate captions using beam search algorithm
        Args:
            sess: tf.Session
            picture_ids: list of picture ids in shape [batch_size]
            in_pictures: input pictures
            beam_size: keep how many beam candidates
            ret_beams: whether to return several beam canditates
            image_f_inputs: image placeholder
            len_norm_f: beam search length normalization parameter
        Returns:
            cap_list: list of format [{'image_id', caption: ''}]
                or (if ret_beams)
            cap_list: list of format [[{'image_id', caption: '' * beam_size}]]
        """
        # get stop word index from dictionary
        start_word_idx = self.data_dict.word2idx['<BOS>']
        stop_word_idx = self.data_dict.word2idx['<EOS>']
        cap_list = [None] * in_pictures.shape[0]
        # set placeholders
        captions = tf.placeholder(tf.int32, [1, None])
        lengths = tf.placeholder(tf.int32, [None], name='seq_length')
        labels_ph = tf.placeholder(tf.int32, [None], name='labels')
        y_one_hot = tf.one_hot(labels_ph, self.n_classes, dtype=tf.int32)
        labels_tiled = tf.tile(tf.expand_dims(y_one_hot, 0),
                               [self.params.gen_z_samples, 1, 1])
        # initialize caption generator
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            _, _, states = self.px_z_y({'y': labels_tiled},
                                       captions, lengths, True)
        init_state, out_state, sample = states
        for im in range(len(in_pictures)):
            state = None
            if ground_truth is not None:
                b_index = self.data_dict.word2idx['<BOS>']
                e_index = self.data_dict.word2idx['<EOS>']
                g_truth = ' '.join([self.data_dict.idx2word[word]
                                    for word in ground_truth[im]
                                    if word not in [b_index, e_index, 0]])
            if self.params.data == "stylenet":
                image_id = int(picture_ids[im].split('.')[0])
            else:
                image_id = int(re.findall(r"[0-9]+", picture_ids[im])[2])
            cap_list[im] = {'image_id': image_id,
                           'caption': ' ',
                           'label': labels_names[int(labels[im])]}
            if ground_truth is not None:
                cap_list[im].update({'ground_truth': g_truth})
            # initial feed
            seed = start_word_idx
            feed = {captions: np.array(seed).reshape([1, 1]),
                    lengths: [1],
                    image_ph: np.expand_dims(in_pictures[im], 0),
                    labels_ph: [labels[im]]}
            # probs are normalized probs
            probs, state = sess.run([sample, out_state], feed)
            # initial Beam, pushed to the heap (TopN class)
            # inspired by tf/models/im2txt
            initial_beam = Beam(sentence=[seed],
                                state=state,
                                logprob=0.0,
                                score=0.0)
            partial_captions = TopN(beam_size)
            partial_captions.push(initial_beam)
            complete_captions = TopN(beam_size)

            # continue to generate, until max_len
            for _ in range(self.params.gen_max_len - 1):
                partial_captions_list = partial_captions.extract()
                partial_captions.reset()
                # get last word in the sentence
                input_feed = [(c.sentence[-1],
                               len(c.sentence)) for c in partial_captions_list]
                state_feed = [c.state for c in partial_captions_list]
                # get states and probs for every beam
                probs_list, states_list = [], []
                for inp_length, state in zip(input_feed, state_feed):
                    inp, length = inp_length
                    feed = {captions: np.array(inp).reshape([1, 1]),
                            lengths: [length],
                            image_ph: np.expand_dims(in_pictures[im], 0),
                            init_state: state,
                            labels_ph: [labels[im]]}
                    probs, new_state = sess.run([sample, out_state], feed)
                    probs_list.append(probs)
                    states_list.append(new_state)
                # for every beam get candidates and append to list
                for i, partial_caption in enumerate(partial_captions_list):
                    cur_state = states_list[i]
                    cur_probs = probs_list[i]
                    # sort list probs, enumerate to remember indices
                    w_probs = list(enumerate(cur_probs.ravel()))
                    w_probs.sort(key=lambda x: -x[1])
                    # keep n probs
                    w_probs = w_probs[:beam_size]
                    for w, p in w_probs:
                        if p < 1e-12:
                            continue  # Avoid log(0).
                        sentence = partial_caption.sentence + [w]
                        logprob = partial_caption.logprob + np.log(p)
                        score = logprob
                        # complete caption, got <EOS>
                        if w == stop_word_idx:
                            if len_norm_f > 0:
                                score /= len(sentence)**len_norm_f
                            beam = Beam(sentence, cur_state, logprob, score)
                            complete_captions.push(beam)
                        else:
                            beam = Beam(sentence, cur_state, logprob, score)
                            partial_captions.push(beam)
                if partial_captions.size() == 0:
                    # When all captions are complete
                    break
            # If we have no complete captions then fall back to the partial captions.
            # But never output a mixture of complete and partial captions because a
            # partial caption could have a higher score than all the complete captions.
            if not complete_captions.size():
                complete_captions = partial_captions
            # find the best beam
            beams = complete_captions.extract(sort=True)
            if not ret_beams:
                best_beam = beams[0]
                capt = [self.data_dict.idx2word[word] for
                                                   word in best_beam.sentence
                                                   if word not in [seed,
                                                                   stop_word_idx]]
                cap_list[im]['caption'] = ' '.join(capt)
            print(cap_list[im]['caption'] + ' ' + cap_list[im]['label'])
            # print("Ground truth caption: ", cap_list[im]['ground_truth'])
            # return list of beam candidates
            if ret_beams:
                c_list = []
                for c in beams:
                    capt = [self.data_dict.idx2word[word] for
                                                       word in c.sentence
                                                       if word not in [seed,
                                                                       stop_word_idx]]
                    c_list.append(' '.join(capt))
                cap_list[im]['caption'] = c_list
        return cap_list
