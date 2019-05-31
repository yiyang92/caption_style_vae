class Parameters():
    def parse_args(self):
        import argparse
        import os
        # some parameters with ### sign are not used (maybe will be used)
        parser = argparse.ArgumentParser(description="Specify training parameters, "
                                         "all parameters also can be "
                                         "directly specify in the "
                                         "Parameters class")
        parser.add_argument('--lr', default=0.0005,
                            help='learning rate', dest='lr')
        parser.add_argument('--embed_dim', default=256,
                            help='embedding size', dest='embed')
        parser.add_argument('--enc_hid', default=512,
                            help='encoder state size', dest='enc_hid')
        parser.add_argument('--dec_hid', default=512,
                            help='decoder state size', dest='dec_hid')
        parser.add_argument('--latent', default=150,
                            help='latent space size', dest='latent')
        parser.add_argument('--restore', help='whether restore',
                            action="store_true")
        parser.add_argument('--gpu', help="specify GPU number")
        parser.add_argument('--epochs', default=40,
                            help="number of training epochs")
        parser.add_argument('--bs', default=32,
                            help="Batch size")
        parser.add_argument('--no_encoder',
                            help="use this if want to run baseline lstm",
                            action="store_true") ###
        parser.add_argument('--temperature', default=0.6,
                            help="set temperature parameter for generation")
        parser.add_argument('--gen_name', default="00",
                            help="prefix of generated json nam")
        parser.add_argument('--gen_z_samples', default=1,
                            help="#z samples")
        parser.add_argument('--ann_param', default=0,
                            help="annealing speed, more slower") ###
        parser.add_argument('--dec_lstm_drop', default=0.9,
                            help="decoder lstm dropout")
        parser.add_argument('--sample_gen', default='greedy',
                            help="'greedy', 'sample', 'beam_search'",
                            choices=["greedy", "sample", "beam_search"])
        parser.add_argument('--checkpoint', default="last_run",
                            help="specify checkpoint name, default=last_run")
        parser.add_argument('--optimizer', default="Adam",
                            choices=['SGD', 'Adam', 'Momentum'],
                            help="SGD or Adam")
        parser.add_argument('--c_v', default=False,
                            help="Whether to use cluster vectors",
                            action="store_true") ###
        parser.add_argument('--std', default=0.1,
                            help="z~N(0, std), during the test time")
        parser.add_argument('--save_params',
                            help="save params class into pickle",
                            action="store_true")
        parser.add_argument('--fine_tune',
                            help="fine_tune",
                            action="store_true")
        parser.add_argument('--mode', default="training",
                            choices=['training', 'inference'],
                            help="specify training or inference")
        parser.add_argument('--gen_label', default="actual",
                            choices=['actual', 'humorous', 'romantic', 
                            "positive", "negative"],
                            help="generation label")
        parser.add_argument('--beam_size', default=3,
                            help="beam size (default:5)")
        # For senticap data only use test set for correctness
        parser.add_argument('--gen_set', default="test",
                            choices=['val', 'test'],
                            help="generation set for evaluation")
        parser.add_argument('--word_drop_keep',
                            default=1.0,
                            help="Decoder word dropout rate")
        parser.add_argument('--keep_words',
                            default=1,
                            help='minimum word occurence', type=int)
        # For "stylenet" data only
        parser.add_argument('--dont_use_f30k', default=True,
                            action='store_false',
                            help="whether to use f30k additional data")
        # percentage of f30k, for experiments, set f30k to false first
        parser.add_argument('--add_f', default=0.0,
                            help="percentage of f30k")
        # Resnet is recommended
        parser.add_argument('--img_embed', default="resnet",
                            help="Resnet or vgg",
                            choices=["resnet", "vgg"])
        # Whether to use tensorboard for logging
        parser.add_argument('--logging', default=False,
                            help="Log or not",
                            action="store_true")
        parser.add_argument('--logdir', default='./model_logs/',
                            help="Logdir")
        parser.add_argument(
            '--data', default="stylenet",
            choices=["stylenet","senticap"])
          

        args = parser.parse_args()
        self.learning_rate = float(args.lr)
        self.embed_size = int(args.embed)
        self.encoder_hidden = int(args.enc_hid)
        self.decoder_hidden = int(args.dec_hid)
        self.latent_size = int(args.latent)
        self.restore = args.restore
        self.num_epochs = int(args.epochs)
        self.no_encoder = args.no_encoder
        self.temperature = float(args.temperature)
        self.gen_name = args.gen_name
        self.gen_z_samples = int(args.gen_z_samples)
        self.ann_param = float(args.ann_param)
        self.dec_lstm_drop = float(args.dec_lstm_drop)
        self.sample_gen = args.sample_gen
        self.checkpoint = args.checkpoint
        self.optimizer = args.optimizer
        self.use_c_v = args.c_v
        self.batch_size = int(args.bs)
        self.std = float(args.std)
        self.save_params = args.save_params
        self.fine_tune = args.fine_tune
        self.mode = args.mode
        self.gen_label = args.gen_label
        self.beam_size = int(args.beam_size)
        self.gen_set = args.gen_set
        self.word_dropout_keep = float(args.word_drop_keep)
        self.keep_words = int(args.keep_words)
        self.use_f30k = args.dont_use_f30k
        self.add_f = float(args.add_f)
        self.img_embed = args.img_embed
        self.logging = args.logging
        self.LOG_DIR = args.logdir
        self.data = args.data
        self.gpu = args.gpu
        # CUDA settings
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        # Some technical arguments
        self.n_classes = 3
        self.vgg_weights_path = './utils/vgg16_weights.npz'
        self.resnet_cp_path = './utils/resnet_v1_imagenet/model.ckpt-257706'
        self.images_dir = "/home/luoyy/datasets/flickr30k-images/"
        # Some additional model parameters
        self.class_hidden = 512
        self.lstm_clip_by_norm = 5.0  # This number is from Google NIC code
        self.max_checkpoints_to_keep = 5
        self.class_lstm_drop = 1.0
        # Generation parameters
        self.gen_max_len = 30

    def save(self):
        import pickle
        save_file = "./pickles/params_{}.pickle".format(self.checkpoint)
        print("saving parameters of current run into {}".format(save_file))
        with open(save_file, 'wb') as wf:
            pickle.dump(obj=self, file=wf)
