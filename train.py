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
from models.bert_container import BertContainer


def random_seed_set(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

TRAIN_COMPONENTS = ('multi_sql','keyword','col','op','agg','root_tem','des_asc','having','andor', 'from')
SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tqdm', action='store_true',
            help='If set, use tqdm.')
    parser.add_argument('--bert', action='store_true',
            help='If set, use bert to encode question.')
    parser.add_argument('--toy', action='store_true',
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--save_dir', type=str, default='',
            help='set model save directory.')
    parser.add_argument('--data_root', type=str, default='',
            help='root path for generated_data')
    parser.add_argument('--train_emb', action='store_true',
            help='Train word embedding.')
    parser.add_argument('--train_component',type=str,default='',
                        help='set train components,available:[multi_sql,keyword,col,op,agg,root_tem,des_asc,having,andor]')
    parser.add_argument('--epoch',type=int,default=500,
                        help='number of epoch for training')
    parser.add_argument('--history_type', type=str, default='full', choices=['full','part','no'], help='full, part, or no history')
    parser.add_argument('--table_type', type=str, default='std', choices=['std','no'], help='standard, hierarchical, or no table info')
    args = parser.parse_args()
    use_hs = True
    if args.history_type == "no":
        args.history_type = "full"
        use_hs = False

    random_seed_set(H_PARAM["random_seed"])

    N_word = H_PARAM["N_word"]
    B_word = H_PARAM["B_word"]
    N_h = H_PARAM["N_H"]
    FROM_N_h = H_PARAM["FROM_N_H"]
    N_depth = H_PARAM["N_DEPTH"]
    if args.toy:
        USE_SMALL=True
        BATCH_SIZE=20
    else:
        USE_SMALL=False

    if torch.cuda.is_available():
        GPU = True
    else:
        GPU = False
    if args.bert:
        BERT = True
    else:
        BERT = False
    learning_rate = H_PARAM["learning_rate"]
    bert_learning_rate = H_PARAM["bert_learning_rate"]
    if args.train_component not in TRAIN_COMPONENTS:
        print("Invalid train component")
        exit(1)
    train_data = load_train_dev_dataset(args.train_component, "train", args.history_type, args.data_root)
    dev_data = load_train_dev_dataset(args.train_component, "dev", args.history_type, args.data_root)
    prepared_tables = prepare_tables(train_data, args.table_type)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), load_used=args.train_emb, use_small=USE_SMALL)
    print("finished load word embedding")
    model = None
    if BERT:
        bert_model = BertContainer()
    else:
        bert_model = None
    if args.train_component == "multi_sql":
        model = MultiSqlPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs)
    elif args.train_component == "keyword":
        model = KeyWordPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs)
    elif args.train_component == "col":
        model = ColPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs)
    elif args.train_component == "op":
        model = OpPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs)
    elif args.train_component == "agg":
        model = AggPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs)
    elif args.train_component == "root_tem":
        model = RootTeminalPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs)
    elif args.train_component == "des_asc":
        model = DesAscLimitPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs)
    elif args.train_component == "having":
        model = HavingPredictor(N_word=N_word,N_h=N_h,N_depth=N_depth, gpu=GPU, use_hs=use_hs)
    elif args.train_component == "andor":
        model = AndOrPredictor(N_word=N_word, N_h=N_h, N_depth=N_depth, gpu=GPU, use_hs=use_hs)
    elif args.train_component == "from":
        model = FindPredictor(N_word=N_word, N_h=FROM_N_h, N_depth=N_depth, gpu=GPU, use_hs=use_hs, bert=bert_model.bert)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    print("finished build model")

    print_flag = False
    embed_layer = WordEmbedding(word_emb, N_word, gpu=GPU, SQL_TOK=SQL_TOK, use_bert=BERT, trainable=args.train_emb)
    print("start training")
    best_acc = 0.0
    for i in range(args.epoch):
        print(('Epoch %d @ %s'%(i+1, datetime.datetime.now())), flush=True)
        bert_model.train()
        print((' Loss = %s'% epoch_train(GPU,
               model, optimizer, H_PARAM["batch_size"], args.train_component, embed_layer, train_data, prepared_tables, table_type=args.table_type, use_tqdm=args.tqdm, bert_model=bert_model)))
        bert_model.eval()
        if args.train_component == "from":
            acc = from_acc(model, embed_layer, dev_data, H_PARAM["batch_size"])
        else:
            acc = epoch_acc(model, 1, args.train_component,embed_layer, dev_data, table_type=args.table_type)
        if acc > best_acc:
            best_acc = acc
            print("Save model...")
            torch.save(model.state_dict(), args.save_dir+"/{}_models.dump".format(args.train_component))
            if BERT:
                print("Save bert")
                torch.save(bert_model.state_dict(), args.save_dir+"/bert_{}_models.dump".format(args.train_component))

