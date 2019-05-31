# test batch generation
import os
import tensorflow as tf
import numpy as np
import zhusuan as zs

from utils.data import Data
from utils.senticap_data import SenticapData
from utils.parameters import Parameters
from ops.inference import inference
from vae_model.classifier import Classifier
from vae_model.encoder import Encoder
from vae_model.decoder import Decoder

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='1'
# parameters
params = Parameters()
params.parse_args()
n_classes = params.n_classes
if params.save_params:
    params.save()
# data
global_step = tf.Variable(0, trainable=False, name="gl_step")
images_dir = params.images_dir
if params.data == "stylenet":
    data = Data(images_dir, params, keep_words=params.keep_words,
                all_30k_set=params.use_f30k, partial=params.add_f,
                vocab_fn=params.checkpoint)
elif params.data == "senticap":
    IMAGES_DIR = "/home/luoyy16/datasets-large/mscoco/coco/images/"
    SENTICAP_DIR = "./senticap"
    data = SenticapData(
        IMAGES_DIR, 
        params, 
        keep_words=params.keep_words,
        vocab_fn="senticap_vocab",
        senticap_dir=SENTICAP_DIR)
# labelled placeholders
cap_enc_l = tf.placeholder(tf.int32, [None, None], name='cap_enc_l')
cap_dec_l = tf.placeholder(tf.int32, [None, None], name='cap_dec_l')
y_labels = tf.placeholder(tf.int32, [None], name='y_labels')
y_one_hot = tf.one_hot(y_labels, n_classes, dtype=tf.int32)
cap_len_l = tf.placeholder(tf.int32, [None], name='cap_len_l')
# unlabelled placeholders
cap_enc_u = tf.placeholder(tf.int32, [None, None], name='cap_enc_u')
cap_dec_u = tf.placeholder(tf.int32, [None, None], name='cap_dec_u')
cap_len_u = tf.placeholder(tf.int32, [None], name='cap_len_u')
# image features inputs
if params.img_embed == "vgg":
    cnn_feature_size = 4096
else:
    cnn_feature_size = 2048
image_batch = tf.placeholder(tf.float32, [None, cnn_feature_size], name='image_batch')
images_fv = tf.layers.dense(image_batch, params.embed_size, name='imf_emb')
# ss-vae model
classifier = Classifier(images_fv, data.dictionary.vocab_size,
                        params.embed_size, params.class_hidden)
encoder = Encoder(images_fv, params.encoder_hidden, 'Normal',
                  params.latent_size, data.dictionary.vocab_size,
                  params.embed_size, params.gen_z_samples, params)
decoder = Decoder(images_fv, cap_dec_u, cap_len_u, params,
                  data.dictionary, n_classes=params.n_classes)


def log_joint(observed):
    n_x = tf.shape(observed['x'])[1]
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        model, _, _ = decoder.px_z_y(observed, n_x=n_x)
    log_px_zy, log_pz, log_py = model.local_log_prob(['x', 'z', 'y'])
    log_px_zy = tf.Print(log_px_zy, [tf.shape(log_px_zy), tf.shape(log_pz),
                                     tf.shape(log_py)], first_n=1)
    return log_px_zy + log_pz + log_py


# labelled
with tf.variable_scope('variational'):
    variational, _, _ = encoder.q_z_xy(cap_enc_l, y_one_hot, cap_len_l)
qz_samples, log_qz = variational.query('z', outputs=True, local_log_prob=True)
decoder.captions = cap_dec_l
decoder.lengths = cap_len_l
decoder.images_fv = images_fv
x_labelled_obs = tf.tile(tf.expand_dims(cap_enc_l, 0),
                         [params.gen_z_samples, 1, 1])
y_labeled_obs = tf.tile(tf.expand_dims(y_one_hot, 0),
                        [params.gen_z_samples, 1, 1])
l_b = zs.variational.elbo(log_joint,
                          observed={'x': x_labelled_obs, 'y': y_labeled_obs},
                          latent={'z': [qz_samples, log_qz]}, axis=0)
labeled_lower_bound = tf.reduce_mean(l_b)
tf.summary.scalar("loss/labelled_lb", labeled_lower_bound)

# unlabelled
# tile y to sum later
y_diag = tf.diag(tf.ones(n_classes, dtype=tf.int32))
y_u = tf.reshape(tf.tile(tf.expand_dims(y_diag, 0), [tf.shape(
    cap_enc_u)[0], 1, 1]), [-1, n_classes])
x_u = tf.reshape(tf.tile(tf.expand_dims(cap_enc_u, 1), [
    1, n_classes, 1]), [-1, tf.shape(cap_enc_u)[1]])
len_u = tf.reshape(tf.tile(tf.expand_dims(
    cap_len_u, 1), [1, n_classes]), [n_classes * tf.shape(cap_len_u)[0]])
# decoder inps
dec_u = tf.reshape(tf.tile(tf.expand_dims(cap_dec_u, 1), [
    1, n_classes, 1]), [-1, tf.shape(cap_dec_u)[1]])
# images
images_u = tf.reshape(tf.tile(tf.expand_dims(images_fv, 1), [
    1, n_classes, 1]), [-1, params.embed_size])
with tf.variable_scope("variational", reuse=tf.AUTO_REUSE):
    variational, _, _ = encoder.q_z_xy(x_u, y_u, len_u, images_u)
qz_samples, log_qz = variational.query('z', outputs=True, local_log_prob=True)
decoder.captions = dec_u
decoder.lengths = len_u
decoder.images_fv = images_u
y_unlabeled_obs = tf.tile(tf.expand_dims(y_u, 0),
                          [params.gen_z_samples, 1, 1])
x_unlabeled_obs = tf.tile(tf.expand_dims(x_u, 0),
                          [params.gen_z_samples, 1, 1])
lb_z = zs.variational.elbo(log_joint,
                           observed={'x': x_unlabeled_obs,
                                     'y': y_unlabeled_obs},
                           latent={'z': [qz_samples, log_qz]}, axis=0)
# sum over y
lb_z = tf.reshape(lb_z, [-1, n_classes])
with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):
    qy_logits_u = classifier.q_y_x(cap_enc_u, cap_len_u, n_classes)
qy_u = tf.reshape(tf.nn.softmax(qy_logits_u), [-1, tf.shape(y_one_hot)[1]])
qy_u += 1e-8
qy_u /= tf.reduce_sum(qy_u, 1, keep_dims=True)
log_qy_u = tf.log(qy_u)  # [batch_size, n_classes]
unlabeled_lower_bound = tf.reduce_mean(
    tf.reduce_sum(qy_u * (lb_z - log_qy_u), 1))
tf.summary.scalar("loss/unlabelled_lb", unlabeled_lower_bound)

# Build classifier
with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE):
    qy_logits_l = classifier.q_y_x(cap_enc_l, cap_len_l, n_classes)
qy_l = tf.nn.softmax(qy_logits_l)
pred_y = tf.argmax(qy_l, 1)
acc = tf.reduce_mean(tf.cast(tf.equal(pred_y,
                                      tf.argmax(y_one_hot, 1)), tf.float32))
onehot_cat = zs.distributions.OnehotCategorical(qy_logits_l)
log_qy_x = onehot_cat.log_prob(y_one_hot)
beta = 0.1 * params.batch_size # 0.1 * batch_size
classifier_cost = -beta * tf.reduce_mean(log_qy_x)
tf.summary.scalar("loss/classifier", classifier_cost)
# Joint cost
cost = -(labeled_lower_bound + unlabeled_lower_bound - classifier_cost) / 2.
tf.summary.scalar("loss/combined_loss", cost)
# Optimization
gradients = tf.gradients(cost, tf.trainable_variables())
clipped_grad, _ = tf.clip_by_global_norm(gradients, params.lstm_clip_by_norm)
grads_vars = zip(clipped_grad, tf.trainable_variables())
optimizer = tf.train.AdamOptimizer(params.learning_rate).apply_gradients(
    grads_vars, 
    global_step=global_step)

# aditional tensors
t_name = 'classifier/net/classifier_drop:0'
classifier_drop = tf.get_default_graph().get_tensor_by_name(t_name)
# training
saver = tf.train.Saver(tf.trainable_variables(),
                       max_to_keep=params.max_checkpoints_to_keep)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.set_random_seed(1234)
np.random.seed(1234)
# logging
summarize = tf.summary.merge_all()
with tf.Session(config=config) as sess:
    if params.mode == "training":
        sess.run(tf.global_variables_initializer())
        if params.logging:
            summary_writer = tf.summary.FileWriter(params.LOG_DIR,
                                                   sess.graph)
            summary_writer.add_graph(sess.graph)
        if params.restore:
            print("Restoring from checkpoint")
            saver.restore(sess, "./checkpoints/{}.ckpt".format(
                params.checkpoint))
        for e in range(params.num_epochs):
            lbs, ubs, accs = [], [], []
            for labelled, unlabelled, labels, images,\
                    lengths_l, lengths_u in data.get_batch(params.batch_size):
                feed = {cap_enc_l: labelled[1],
                        cap_dec_l: labelled[0],
                        cap_len_l: lengths_l,
                        y_labels: labels,
                        cap_enc_u: unlabelled[1],
                        cap_dec_u: unlabelled[0],
                        cap_len_u: lengths_u,
                        image_batch: images,
                        classifier_drop: params.class_lstm_drop}
                lb, _, ulb, acc_, g_s = sess.run([labeled_lower_bound,
                                             optimizer, unlabeled_lower_bound,
                                             acc, global_step],
                                            feed_dict=feed)

                if params.logging:
                    summaries = sess.run(summarize, feed_dict=feed)
                    summary_writer.add_summary(summaries, g_s)
                lbs.append(lb)
                ubs.append(ulb)
                accs.append(acc_)
            # validation
            if e % 2 == 0:
                print("Epoch: ", e)
                print("Train llb: {}, "
                      "train ulb: {}, train class acc: {}".format(np.mean(lbs),
                                                                  np.mean(ulb),
                                                                  np.mean(accs)))
                lbs, ubs, accs = [], [], []
                for labelled, unlabelled, labels, images,\
                        lengths_l, lengths_u in data.get_batch(
                            params.batch_size, set='val'):
                        feed = {cap_enc_l: labelled[1],
                                cap_dec_l: labelled[0],
                                cap_len_l: lengths_l,
                                y_labels: labels,
                                cap_enc_u: unlabelled[1],
                                cap_dec_u: unlabelled[0],
                                cap_len_u: lengths_u,
                                image_batch: images}
                        lb, ulb, acc_ = sess.run([labeled_lower_bound,
                                                  unlabeled_lower_bound, acc],
                                                 feed_dict=feed)
                        lbs.append(lb)
                        ubs.append(ulb)
                        accs.append(acc_)
                print("Val llb: {}, "
                      "val ulb: {}, val class acc: {}".format(np.mean(lbs),
                                                              np.mean(ubs),
                                                              np.mean(accs)))
            if e % 4 == 0:
                # save model
                if not os.path.exists("./checkpoints"):
                    os.makedirs("./checkpoints")
                save_path = saver.save(sess, "./checkpoints/{}.ckpt".format(
                    params.checkpoint))
                print("Model saved in file: %s" % save_path)

    if params.mode == "inference":
        decoder.images_fv = images_fv
        inference(params, decoder, data, image_batch, saver, sess)
