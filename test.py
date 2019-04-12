import json
import torch
import datetime
import argparse
import numpy as np
from utils import *
from models.net_utils import encode_question
from models.bert_container import BertContainer
from supermodel import SuperModel
from pytorch_pretrained_bert import BertModel

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_emb', action='store_true',
            help='Train word embedding.')
    parser.add_argument('--toy', action='store_true',
                        help='If set, use small data; used for fast debugging.')
    parser.add_argument('--models', type=str, help='path to saved model')
    parser.add_argument('--test_data_path',type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--history_type', type=str, default='full', choices=['full','part','no'], help='full, part, or no history')
    parser.add_argument('--table_type', type=str, default='std', choices=['std','hier','no'], help='standard, hierarchical, or no table info')
    parser.add_argument('--schema_compound', type=int, default=0)
    parser.add_argument('--with_from', action='store_true')
    args = parser.parse_args()
    use_hs = True
    if args.history_type == "no":
        args.history_type = "full"
        use_hs = False
    if args.schema_compound != 0:
        H_PARAM["dev_db_compound_num"] = args.schema_compound

    N_word=300
    B_word=42
    N_depth=2
    if args.toy:
        USE_SMALL=True
        GPU=True
        BATCH_SIZE=2 #20
    else:
        USE_SMALL=False
        GPU=True
        BATCH_SIZE=2 #64
    if not torch.cuda.is_available():
        GPU=False
    data = json.load(open(args.test_data_path))

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
            load_used=args.train_emb, use_small=USE_SMALL)

    bert_model = BertContainer()

    model = SuperModel(word_emb, N_word=N_word, gpu=GPU, trainable_emb = args.train_emb, table_type=args.table_type, use_hs=use_hs, bert=bert_model.bert, with_from=args.with_from)

    # agg_m, sel_m, cond_m = best_model_name(args)
    # torch.save(model.state_dict(), "saved_models/{}_models.dump".format(args.train_component))

    print("Loading from modules...")
    if GPU:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.multi_sql.load_state_dict(torch.load("{}/multi_sql_models.dump".format(args.models), map_location=device))
    model.key_word.load_state_dict(torch.load("{}/keyword_models.dump".format(args.models), map_location=device))
    model.col.load_state_dict(torch.load("{}/col_models.dump".format(args.models), map_location=device))
    model.op.load_state_dict(torch.load("{}/op_models.dump".format(args.models), map_location=device))
    model.agg.load_state_dict(torch.load("{}/agg_models.dump".format(args.models), map_location=device))
    model.root_teminal.load_state_dict(torch.load("{}/root_tem_models.dump".format(args.models), map_location=device))
    model.des_asc.load_state_dict(torch.load("{}/des_asc_models.dump".format(args.models), map_location=device))
    model.having.load_state_dict(torch.load("{}/having_models.dump".format(args.models), map_location=device))
    if args.with_from:
        model.from_table.load_state_dict(torch.load("{}/from_models.dump".format(args.models), map_location=device))
        bert_model.main_bert.load_state_dict(torch.load("{}/bert_from_models.dump".format(args.models), map_location=device))
        bert_model.bert_param.load_state_dict(torch.load("{}/bert_from_params.dump".format(args.models), map_location=device))
        bert_model.eval()

    test_acc(model, BATCH_SIZE, data, args.output_path)
    #test_exec_acc()
