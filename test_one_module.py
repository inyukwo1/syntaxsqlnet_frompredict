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
from models.find_predictor import FindPredictor
from pytorch_pretrained_bert import BertModel
from hyperparameters import H_PARAM
import time
from models.bert_container import BertContainer

TRAIN_COMPONENTS = ('multi_sql','keyword','col','op','agg','root_tem','des_asc','having','andor', 'from')
SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert', action='store_true',
            help='If set, use bert to encode question.')
    parser.add_argument('--data_root', type=str, default='',
            help='root path for generated_data')
    parser.add_argument('--load_path', type=str, default='',
            help='root path for load dir')
    parser.add_argument('--train_component',type=str,default='',
                        help='set train components,available:[multi_sql,keyword,col,op,agg,root_tem,des_asc,having,andor]')
    parser.add_argument('--history_type', type=str, default='full', choices=['full','part','no'], help='full, part, or no history')
    parser.add_argument('--table_type', type=str, default='std', choices=['std','no'], help='standard, hierarchical, or no table info')
    args = parser.parse_args()
    use_hs = True
    if args.history_type == "no":
        args.history_type = "full"
        use_hs = False

    N_word = H_PARAM["N_word"]
    B_word = H_PARAM["B_word"]
    N_h = H_PARAM["N_H"]
    FROM_N_h = H_PARAM["FROM_N_H"]
    N_depth = H_PARAM["N_DEPTH"]
    USE_SMALL=False
    BATCH_SIZE=1

    if torch.cuda.is_available():
        GPU = True
    else:
        GPU = False
    if args.bert:
        BERT = True
    else:
        BERT = False
    if args.train_component not in TRAIN_COMPONENTS:
        print("Invalid train component")
        exit(1)
    dev_data = load_train_dev_dataset(args.train_component, "dev", args.history_type, args.data_root)

    start_time = time.time()
    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), load_used=False, use_small=USE_SMALL)
    print("finished load word embedding: {}".format(time.time() - start_time))
    model = None
    bert_model = BertContainer()
    if args.train_component == "multi_sql":
        model = MultiSqlPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs, bert=bert)
    elif args.train_component == "keyword":
        model = KeyWordPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs, bert=bert)
    elif args.train_component == "col":
        model = ColPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs, bert=bert)
    elif args.train_component == "op":
        model = OpPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs, bert=bert)
    elif args.train_component == "agg":
        model = AggPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs, bert=bert)
    elif args.train_component == "root_tem":
        model = RootTeminalPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs, bert=bert)
    elif args.train_component == "des_asc":
        model = DesAscLimitPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs, bert=bert)
    elif args.train_component == "having":
        model = HavingPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs, bert=bert)
    elif args.train_component == "andor":
        model = AndOrPredictor(N_word=N_word, N_h=N_h, N_depth=N_depth, gpu=GPU, use_hs=use_hs, bert=bert)
    elif args.train_component == "from":
        model = FindPredictor(N_word=N_word, N_h=FROM_N_h, N_depth=N_depth, gpu=GPU, use_hs=use_hs, bert=bert_model.bert)
    print("finished build model")

    print_flag = False
    if GPU:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.load_state_dict(torch.load(args.load_path + "/{}_models.dump".format(args.train_component), map_location=device))

    if BERT:
        bert_model.main_bert.load_state_dict(torch.load("{}/bert_from_models.dump".format(args.load_path), map_location=device))
        bert_model.bert_param.load_state_dict(torch.load("{}/bert_from_params.dump".format(args.load_path), map_location=device))
        bert_model.eval()
    embed_layer = WordEmbedding(word_emb, N_word, gpu=GPU, SQL_TOK=SQL_TOK, use_bert=BERT, trainable=False)
    if args.train_component == "from":
        acc = from_acc(model, embed_layer, dev_data, 1)
    else:
        acc = epoch_acc(model, BATCH_SIZE, args.train_component, embed_layer, dev_data, table_type=args.table_type)
    print("finished: {}".format(time.time() - start_time))
