import json
import torch
import datetime
import argparse
import numpy as np
from utils import *
from models.net_utils import encode_question
from word_embedding import WordEmbedding
from models.agg_predictor import AggPredictor
from models.col_predictor import ColPredictor
from models.desasc_limit_predictor import DesAscLimitPredictor
from models.having_predictor import HavingPredictor
from models.keyword_predictor import KeyWordPredictor
from models.multisql_predictor import MultiSqlPredictor
from models.op_predictor import OpPredictor
from models.root_teminal_predictor import RootTeminalPredictor
from models.andor_predictor import AndOrPredictor
from models.from_predictor import FromPredictor
from pytorch_pretrained_bert import BertModel
from hyperparameters import H_PARAM
from models.bert_container import BertContainer


def random_seed_set(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tqdm', action='store_true',
            help='If set, use tqdm.')
    parser.add_argument('--onefrom', action='store_true')
    parser.add_argument('--use_lstm', action="store_true")
    parser.add_argument('--save_dir', type=str, default='',
            help='set model save directory.')
    parser.add_argument('--data_root', type=str, default='',
            help='root path for generated_data')
    parser.add_argument('--epoch',type=int,default=500,
                        help='number of epoch for training')

    args = parser.parse_args()

    random_seed_set(H_PARAM["random_seed"])

    N_word = H_PARAM["N_word"]
    B_word = H_PARAM["B_word"]
    N_h = H_PARAM["N_H"]
    FROM_N_h = H_PARAM["FROM_N_H"]
    N_depth = H_PARAM["N_DEPTH"]

    if torch.cuda.is_available():
        GPU = True
    else:
        GPU = False
    learning_rate = H_PARAM["learning_rate"]
    bert_learning_rate = H_PARAM["bert_learning_rate"]

    train_component = "from"
    if args.onefrom:
        train_component = "onefrom"
    train_data = load_train_dev_dataset(train_component, "train", "full", args.data_root)
    dev_data = load_train_dev_dataset(train_component, "dev", "full", args.data_root)
    prepared_tables = prepare_tables(train_data, "std")

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word))
    print("finished load word embedding")
    bert_model = BertContainer()
    model = FromPredictor(N_word=N_word, N_h=FROM_N_h, N_depth=N_depth, gpu=GPU, use_hs=True, bert=bert_model.bert, onefrom=args.onefrom, use_lstm=args.use_lstm)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    print("finished build model")

    print_flag = False
    embed_layer = WordEmbedding(word_emb, N_word, gpu=GPU, SQL_TOK=SQL_TOK, use_bert=True)
    print("start training")
    best_acc = 0.0
    for i in range(args.epoch):
        print(('Epoch %d @ %s'%(i+1, datetime.datetime.now())), flush=True)
        bert_model.train()
        print((' Loss = %s'% from_train(GPU,
               model, optimizer, H_PARAM["batch_size"], args.onefrom, embed_layer, train_data, use_tqdm=args.tqdm, bert_model=bert_model, use_lstm=args.use_lstm)))
        bert_model.eval()
        if i % 10 == 9:
            acc = from_acc(model, embed_layer, dev_data,  1, use_lstm=args.use_lstm)
            if acc > best_acc:
                best_acc = acc
                print("Save model...")
                torch.save(model.state_dict(), args.save_dir+"/from_models.dump")
                torch.save(bert_model.main_bert.state_dict(), args.save_dir+"/bert_from_models.dump")
                torch.save(bert_model.bert_param.state_dict(), args.save_dir+"/bert_from_params.dump")

