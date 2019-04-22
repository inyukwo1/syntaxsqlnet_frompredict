import torch
import argparse
from utils import *
from word_embedding import WordEmbedding
from models.from_predictor import FromPredictor
from hyperparameters import H_PARAM
import time
from models.bert_container import BertContainer

SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']


def prepare_from_model(args):
    N_word = H_PARAM["N_word"]
    B_word = H_PARAM["B_word"]
    FROM_N_h = H_PARAM["FROM_N_H"]
    N_depth = H_PARAM["N_DEPTH"]

    if torch.cuda.is_available():
        GPU = True
    else:
        GPU = False

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word))
    bert_model = BertContainer()
    model = FromPredictor(N_word=N_word, N_h=FROM_N_h, N_depth=N_depth, gpu=GPU, use_hs=True, bert=bert_model.bert,
                          onefrom=args.onefrom)

    if GPU:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.load_state_dict(torch.load(args.load_path + "/from_models.dump", map_location=device))
    bert_model.main_bert.load_state_dict(
        torch.load("{}/bert_from_models.dump".format(args.load_path), map_location=device))
    bert_model.bert_param.load_state_dict(
        torch.load("{}/bert_from_params.dump".format(args.load_path), map_location=device))
    bert_model.eval()
    embed_layer = WordEmbedding(word_emb, N_word, gpu=GPU, SQL_TOK=SQL_TOK, use_bert=True, trainable=False)
    return model, embed_layer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onefrom', action='store_true')
    parser.add_argument('--data_root', type=str, default='',
            help='root path for generated_data')
    parser.add_argument('--load_path', type=str, default='',
            help='root path for load dir')
    args = parser.parse_args()

    model, embed_layer = prepare_from_model(args)

    component = "from"
    if args.onefrom:
        component = "onefrom"
    dev_data = load_train_dev_dataset(component, "dev", "full", args.data_root)
    acc = from_acc(model, embed_layer, dev_data, 1)
