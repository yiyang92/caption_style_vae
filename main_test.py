from utils.parameters import Parameters
from utils.senticap_data import SenticapData


IMAGES_DIR = "/home/luoyy16/datasets-large/mscoco/coco/images/"
SENTICAP_DIR = "./senticap"
params = Parameters()
params.parse_args()
n_classes = params.n_classes


data = SenticapData(
    IMAGES_DIR, 
    params, 
    keep_words=params.keep_words,
    vocab_fn="senticap_vocab",
    senticap_dir=SENTICAP_DIR)

# Need data.dictionary.vocab_size, data.get_batch
for labelled, unlabelled, labels, images,\
                    lengths_l, lengths_u in data.get_batch(32, set="train"):
    print(labelled[0].shape)
    print(images.shape)
    