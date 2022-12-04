import torch

# # multi
EMBEDDING_PATH = "gensim_train/word2vec.model"

TRAIN_FILE="data/new_split/train.json"
DEV_FILE="data/new_split/valid.json"
TEST_FILE="data/new_split/test.json"

PRETRAIN = False
FREEZE = True
PRE_MODEL = "electra"
MODEL = "lstm"


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMB_DIM = 256
HID_DIM = 256 if PRETRAIN else 256
SEQ_LEN = {"fact": 300, "claim":300, "view": 300}
MAX_DEC_LEN = 300
MIN_DEC_LEN = 35
BATCH_SIZE = 64 if FREEZE else 32
EPOCHS = 25 if PRETRAIN else 10
 
