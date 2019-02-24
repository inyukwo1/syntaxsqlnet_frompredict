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
import time

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

    N_word=300
    B_word=42
    N_h = 300
    N_depth=2
    USE_SMALL=False
    BATCH_SIZE=48

    if torch.cuda.is_available():
        GPU = True
    else:
        GPU = False
    if args.bert:
        BERT = True
    else:
        BERT = False
    # TRAIN_ENTRY=(False, True, False)  # (AGG, SEL, COND)
    # TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
    learning_rate = 1e-4
    bert_learning_rate = 1e-5
    if args.train_component not in TRAIN_COMPONENTS:
        print("Invalid train component")
        exit(1)
    dev_data = load_train_dev_dataset(args.train_component, "dev", args.history_type, args.data_root)
    # sql_data, table_data, val_sql_data, val_table_data, \
    #         test_sql_data, test_table_data, \
    #         TRAIN_DB, DEV_DB, TEST_DB = load_dataset(args.dataset, use_small=USE_SMALL)

    start_time = time.time()
    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), load_used=False, use_small=USE_SMALL)
    print("finished load word embedding: {}".format(time.time() - start_time))
    #word_emb = load_concat_wemb('glove/glove.42B.300d.txt', "/data/projects/paraphrase/generation/para-nmt-50m/data/paragram_sl999_czeng.txt")
    model = None
    if BERT:
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        if GPU:
            bert_model.cuda()
        def berter(q, q_len):
            return encode_question(bert_model, q, q_len)
        bert = berter
    else:
        bert_model = None
        bert = None
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
        model = FindPredictor(N_word=N_word, N_h=N_h, N_depth=N_depth, gpu=GPU, use_hs=use_hs, bert=bert)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    if BERT:
        optimizer_bert = torch.optim.Adam(bert_model.parameters(), lr=bert_learning_rate)
    else:
        optimizer_bert = None
    print("finished build model")

    print_flag = False
    model.load_state_dict(torch.load(args.load_path))
    embed_layer = WordEmbedding(word_emb, N_word, gpu=GPU, SQL_TOK=SQL_TOK, use_bert=BERT, trainable=False)
    acc = epoch_acc(model, BATCH_SIZE, args.train_component, embed_layer, dev_data, table_type=args.table_type)
    print("finished: {}".format(time.time() - start_time))
