# inference op
import json
import os
import numpy as np


def inference(params, decoder, data, image_ph, saver, sess, all_labels=False):
    print("Restoring from checkpoint")
    saver.restore(sess, "./checkpoints/{}.ckpt".format(
        params.checkpoint))
    # validation set
    captions_gen = []
    print("Generating captions for {} file".format(params.gen_set))
    label = None
    try:
        # [hum_c, rom_c, act_c]
        gen_labels = {'humorous': 0, 'romantic': 1, 'actual': 2}
        label = gen_labels[params.gen_label]
        labels_idx = {idx:lbl for lbl, idx in gen_labels.items()}
    except KeyError:
        pass
    try:
        gen_labels = {"positive": 0, "negative": 1, "actual": 2}
        label = gen_labels[params.gen_label]
        labels_idx = {idx:lbl for lbl, idx in gen_labels.items()}
    except KeyError:
        pass
    if label is None:
        raise KeyError("Label is not right")
    if params.sample_gen == 'beam_search':
        print(
            "Beam search generation, might take time, beam size: {}".format(
                params.beam_size))
    for labelled, _, gt_labels, images,\
            lengths_l, lengths_u, names in data.get_batch(100,
                                                          set=params.gen_set,
                                                          get_names=True,
                                                          label=label):
        image_names = names
        labels = np.ones(len(names)) * label
        if params.sample_gen == 'beam_search':
            sent = decoder.beam_search(sess, image_names, images,
                                       image_ph, labels,
                                       ground_truth=labelled[1],
                                       beam_size=params.beam_size,
                                       labels_names=labels_idx)
        else:
            sent, _ = decoder.online_inference(sess, image_names, images,
                                               image_ph, labels,
                                               ground_truth=labelled[1],
                                               labels_names=labels_idx)
        captions_gen += sent
    print("Generated {} captions".format(len(captions_gen)))
    res_dir = "./results"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    gen_file = os.path.join(res_dir, "{}_{}_{}.json".format(
        params.gen_set, params.data, params.gen_name))
    if os.path.exists(gen_file):
        print("")
        os.remove(gen_file)
    with open(gen_file, 'w') as wj:
        print("saving val json file into ", gen_file)
        json.dump(captions_gen, wj)
