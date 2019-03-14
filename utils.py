import re
import io
import json
import numpy as np
import os
import signal
from preprocess_train_dev_data import get_table_dict
from models.net_utils import encode_question
import tqdm
import random
import pandas as pd
import copy
import os.path
import pickle
import torch


def load_train_dev_dataset(component,train_dev,history, root):
    return json.load(open("{}/{}_{}_{}_dataset.json".format(root, history,train_dev,component)))


def to_batch_seq(data, idxes, st, ed):
    q_seq = []
    history = []
    label = []
    for i in range(st, ed):
        q_seq.append(data[idxes[i]]['question_tokens'])
        history.append(data[idxes[i]]["history"])
        label.append(data[idxes[i]]["label"])
    return q_seq,history,label


# CHANGED
def to_batch_tables(data, idxes, st,ed, table_type):
    # col_lens = []
    col_seq = []
    tname_seqs = []
    par_tnum_seqs = []
    foreign_keys = []
    for i in range(st, ed):
        ts = data[idxes[i]]["ts"]
        tname_toks = [x.split(" ") for x in ts[0]]
        col_type = ts[2]
        cols = [x.split(" ") for xid, x in ts[1]]
        tab_seq = [xid for xid, x in ts[1]]
        cols_add = []
        for tid, col, ct in zip(tab_seq, cols, col_type):
            col_one = [ct]
            if tid == -1:
                tabn = ["all"]
            else:
                if table_type == "no":
                    tabn = []
                elif table_type == "struct":
                    tabn = []
                else:
                    tabn = tname_toks[tid]
            for t in tabn:
                if t not in col:
                    col_one.append(t)
            col_one.extend(col)
            cols_add.append(col_one)
        col_seq.append(cols_add)
        tname_seqs.append(tname_toks)
        par_tnum_seqs.append(tab_seq)
        foreign_keys.append(ts[3])

    return col_seq, tname_seqs, par_tnum_seqs, foreign_keys


def prepare_tables(data, table_type):
    prepared = {}
    for datum in data:
        ts = datum["ts"]
        tname_toks = [x.split(" ") for x in ts[0]]
        col_type = ts[2]
        cols = [x.split(" ") for xid, x in ts[1]]
        tab_seq = [xid for xid, x in ts[1]]
        cols_add = []
        for tid, col, ct in zip(tab_seq, cols, col_type):
            col_one = [ct]
            if tid == -1:
                tabn = ["all"]
            else:
                if table_type == "no":
                    tabn = []
                elif table_type == "struct":
                    tabn = []
                else:
                    tabn = tname_toks[tid]
            for t in tabn:
                if t not in col:
                    col_one.append(t)
            col_one.extend(col)
            cols_add.append(col_one)
        foreign_keys = ts[3]
        for idx, _ in enumerate(foreign_keys):
            foreign_keys[idx][0] = foreign_keys[idx][0] - 1
            foreign_keys[idx][1] = foreign_keys[idx][1] - 1
        for par_num in tab_seq[1:]:
            assert par_num < len(tname_toks)
        for f, p in foreign_keys:
            assert f < len(cols_add[1:])
            assert p < len(cols_add[1:])
        prepared[ts[4]] = (cols_add[1:], tname_toks, tab_seq[1:], foreign_keys)
    return prepared


def augment_batch_tables(col_seq, tname_seqs, par_tnum_seqs, foreign_keys, prepared_tables):
    ret_col_seq = []
    ret_tname_seqs = []
    ret_par_tnum_seqs = []
    ret_foreign_keys = []
    for one_col_seq, one_tname_seqs, one_par_tnum_seqs, one_foreign_keys in zip(col_seq, tname_seqs, par_tnum_seqs, foreign_keys):
        if len(one_tname_seqs) == 1 or random.randint(0, 100) < 50:
            one_col_seq, one_tname_seqs, one_par_tnum_seqs, one_foreign_keys = \
                augment_table(one_col_seq, one_tname_seqs, one_par_tnum_seqs, one_foreign_keys, prepared_tables)
        ret_col_seq.append(one_col_seq)
        ret_tname_seqs.append(one_tname_seqs)
        ret_par_tnum_seqs.append(one_par_tnum_seqs)
        ret_foreign_keys.append(one_foreign_keys)
    return ret_col_seq, ret_tname_seqs, ret_par_tnum_seqs, ret_foreign_keys


def augment_table(one_col_seq, one_tname_seqs, one_par_tnum_seqs, one_foreign_keys, prepared_tables):
    add_num = random.randint(3, 7)
    choosed_tables = []
    one_col_seq, one_tname_seqs, one_par_tnum_seqs, one_foreign_keys = copy.deepcopy((one_col_seq, one_tname_seqs, one_par_tnum_seqs, one_foreign_keys))
    for _ in range(add_num):
        choosed_table = random.choice(list(prepared_tables.keys()))
        choosed_tables.append(choosed_table)
        if prepared_tables[choosed_table][0] == one_col_seq:
            continue
        new_col_seq, new_tname_seqs, new_par_tnum_seqs, new_foreign_keys = prepared_tables[choosed_table]
        ret_par_tnum_seqs = []
        ret_foreign_keys = []
        for f, p in new_foreign_keys:
            ret_foreign_keys.append([f + len(one_col_seq), p + len(one_col_seq)])
        for par_tnum in new_par_tnum_seqs:
            ret_par_tnum_seqs.append(par_tnum + len(one_tname_seqs))
        one_col_seq += new_col_seq
        one_tname_seqs += new_tname_seqs
        one_par_tnum_seqs += ret_par_tnum_seqs
        one_foreign_keys += ret_foreign_keys
    for par_num in one_par_tnum_seqs:
        assert par_num < len(one_tname_seqs)
    for f, p in one_foreign_keys:
        assert f < len(one_col_seq)
        assert p < len(one_col_seq)
    return one_col_seq, one_tname_seqs, one_par_tnum_seqs, one_foreign_keys


def to_batch_from_candidates(par_tab_nums, data, idxes, st, ed):
    # col_lens = []
    from_candidates = []
    for idx, i in enumerate(range(st, ed)):
        table_candidate = data[idxes[i]]["from"]
        col_candidates = [0]
        for col, par in enumerate(par_tab_nums[idx]):
            if str(par) in table_candidate:
                col_candidates.append(col)
        from_candidates.append(col_candidates)

    return from_candidates

## used for training in train.py
def epoch_train(gpu, model, optimizer, batch_size, component,embed_layer,data, prepared_tables, table_type, use_tqdm, optimizer_bert, optimizer_encoder, bert):
    model.train()
    newdata = []
    for entry in data:
        if len(entry["ts"][0]) > 1:
            newdata.append(entry)
    data = newdata
    perm=np.random.permutation(len(data))
    cum_loss = 0.0
    st = 0
    total_err = 0

    for _ in tqdm.tqdm(range(len(data) // batch_size), disable=not use_tqdm):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, history, label = to_batch_seq(data, perm, st, ed)
        q_emb_var, q_len = embed_layer.gen_x_q_batch(q_seq)
        hs_emb_var, hs_len = embed_layer.gen_x_history_batch(history)
        score = 0.0
        loss = 0.0
        if component == "multi_sql":
            mkw_emb_var = embed_layer.gen_word_list_embedding(["none","except","intersect","union"],(ed-st))
            mkw_len = np.full(q_len.shape, 4,dtype=np.int64)
            # print("mkw_emb:{}".format(mkw_emb_var.size()))
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, mkw_emb_var=mkw_emb_var, mkw_len=mkw_len)
        elif component == "keyword":
            #where group by order by
            # [[0,1,2]]
            kw_emb_var = embed_layer.gen_word_list_embedding(["where", "group by", "order by"],(ed-st))
            mkw_len = np.full(q_len.shape, 3, dtype=np.int64)
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, kw_emb_var=kw_emb_var, kw_len=mkw_len)
        elif component == "col":
            #col word embedding
            # [[0,1,3]]
            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(data, perm, st, ed, table_type)
            from_candidates = to_batch_from_candidates(par_tab_nums, data, perm, st, ed)
            col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len = embed_layer.gen_col_batch(col_seq, tab_seq)
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, from_candidates)

        elif component == "op":
            #B*index
            gt_col = np.zeros(q_len.shape,dtype=np.int64)
            index = 0
            for i in range(st,ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1

            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len = embed_layer.gen_col_batch(col_seq, tab_seq)
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == "agg":
            # [[0,1,3]]
            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len = embed_layer.gen_col_batch(col_seq, tab_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == "root_tem":
            #B*0/1
            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len = embed_layer.gen_col_batch(col_seq, tab_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(data[perm[i]]["history"])
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == "des_asc":
            # B*0/1
            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len = embed_layer.gen_col_batch(col_seq, tab_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == 'having':
            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len = embed_layer.gen_col_batch(col_seq, tab_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == "andor":
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len)

        elif component == "from":
            tabs = []
            cols = []
            for i in range(st, ed):
                tabs.append(data[perm[i]]['ts'][0])
                cols.append(data[perm[i]]["ts"][1])
            q_emb, q_len,  table_cols, table_col_num_lens, table_col_name_lens, table_col_type_ids, special_tok_id, table_locs = embed_layer.gen_bert_batch_with_table(q_seq, tabs, cols)
            score = model.forward(q_emb, q_len, hs_emb_var, hs_len,  table_cols, table_col_num_lens, table_col_name_lens, table_col_type_ids, special_tok_id, table_locs)
        loss = model.loss(score, label)

        err = model.check_acc(score, label)
        total_err += err

        # print("loss {}".format(loss.data.cpu().numpy()))
        if gpu:
            cum_loss += loss.data.cpu().numpy()*(ed - st)
        else:
            cum_loss += loss.data.numpy()*(ed - st)
        optimizer.zero_grad()
        if optimizer_bert:
            optimizer_bert.zero_grad()
        if optimizer_encoder:
            optimizer_encoder.zero_grad()
        loss.backward()
        optimizer.step()
        if optimizer_bert:
            optimizer_bert.step()
        if optimizer_encoder:
            optimizer_encoder.step()

        st = ed
    print(("Train {} acc total acc: {}".format(component, 1 - total_err * 1.0 / len(data))), flush=True)

    return cum_loss / len(data)

## used for development evaluation in train.py
def epoch_acc(model, batch_size, component, embed_layer,data, table_type, error_print=False, train_flag = False):
    model.eval()
    perm = list(range(len(data)))
    st = 0
    total_number_error = 0.0
    total_p_error = 0.0
    total_error = 0.0
    print(("dev data size {}".format(len(data))))
    while st < len(data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, history, label = to_batch_seq(data, perm, st, ed)
        q_emb_var, q_len = embed_layer.gen_x_q_batch(q_seq)
        hs_emb_var, hs_len = embed_layer.gen_x_history_batch(history)
        score = 0.0

        if component == "multi_sql":
            #none, except, intersect,union
            #truth B*index(0,1,2,3)
            # print("hs_len:{}".format(hs_len))
            # print("q_emb_shape:{} hs_emb_shape:{}".format(q_emb_var.size(), hs_emb_var.size()))
            mkw_emb_var = embed_layer.gen_word_list_embedding(["none","except","intersect","union"],(ed-st))
            mkw_len = np.full(q_len.shape, 4,dtype=np.int64)
            # print("mkw_emb:{}".format(mkw_emb_var.size()))
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, mkw_emb_var=mkw_emb_var, mkw_len=mkw_len)
        elif component == "keyword":
            #where group by order by
            # [[0,1,2]]
            kw_emb_var = embed_layer.gen_word_list_embedding(["where", "group by", "order by"],(ed-st))
            mkw_len = np.full(q_len.shape, 3, dtype=np.int64)
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, kw_emb_var=kw_emb_var, kw_len=mkw_len)
        elif component == "col":
            #col word embedding
            # [[0,1,3]]
            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(data, perm, st, ed, table_type)
            from_candidates = to_batch_from_candidates(par_tab_nums, data, perm, st, ed)
            col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len = embed_layer.gen_col_batch(col_seq, tab_seq)
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, from_candidates)
        elif component == "op":
            #B*index
            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len = embed_layer.gen_col_batch(col_seq, tab_seq)
            gt_col = np.zeros(q_len.shape,dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st,ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == "agg":
            # [[0,1,3]]
            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len = embed_layer.gen_col_batch(col_seq, tab_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1

            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == "root_tem":
            #B*0/1
            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len = embed_layer.gen_col_batch(col_seq, tab_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(data[perm[i]]["history"])
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == "des_asc":
            # B*0/1
            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len = embed_layer.gen_col_batch(col_seq, tab_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == 'having':
            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len = embed_layer.gen_col_batch(col_seq, tab_seq)
            gt_col = np.zeros(q_len.shape, dtype=np.int64)
            # print(ed)
            index = 0
            for i in range(st, ed):
                # print(i)
                gt_col[index] = data[perm[i]]["gt_col"]
                index += 1
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col=gt_col)

        elif component == "andor":
            score = model.forward(q_emb_var, q_len, hs_emb_var, hs_len)
        elif component == "from":
            tabs = []
            cols = []
            for i in range(st, ed):
                tabs.append(data[perm[i]]['ts'][0])
                cols.append(data[perm[i]]["ts"][1])
            q_emb, q_len, table_cols, table_col_num_lens, table_col_name_lens, table_col_type_ids, special_tok_id, table_locs = embed_layer.gen_bert_batch_with_table(
                q_seq, tabs, cols)
            score = model.forward(q_emb, q_len, hs_emb_var, hs_len, table_cols, table_col_num_lens, table_col_name_lens,
                                  table_col_type_ids, special_tok_id, table_locs)
        # print("label {}".format(label))
        if component in ("agg","col","keyword","op"):
            num_err, p_err, err = model.check_acc(score, label)
            total_number_error += num_err
            total_p_error += p_err
            total_error += err
        else:
            err = model.check_acc(score, label)
            total_error += err
        st = ed

    if component in ("agg","col","keyword","op"):
        print(("Dev {} acc number predict acc:{} partial acc: {} total acc: {}".format(component,1 - total_number_error*1.0/len(data),1 - total_p_error*1.0/len(data),  1 - total_error*1.0/len(data))))
        return 1 - total_error*1.0/len(data)
    else:
        print(("Dev {} acc total acc: {}".format(component,1 - total_error*1.0/len(data))))
        return 1 - total_error*1.0/len(data)


def timeout_handler(num, stack):
    print("Received SIGALRM")
    raise Exception("Timeout")

## used in test.py
def test_acc(model, batch_size, data,output_path):
    table_dict = get_table_dict("./data/tables.json")
    f = open(output_path,"w")
    for item in data[:]:
        db_id = item["db_id"]
        if db_id not in table_dict: print(("Error %s not in table_dict" % db_id))
        # signal.signal(signal.SIGALRM, timeout_handler)
        # signal.alarm(2) # set timer to prevent infinite recursion in SQL generation
        sql = model.forward([item["question_toks"]]*batch_size,[],table_dict[db_id])
        if sql is not None:
            print(sql)
            sql = model.gen_sql(sql,table_dict[db_id])
        else:
            sql = "select a from b"
        print(sql)
        print("")
        f.write("{}\n".format(sql))
    f.close()


def load_word_emb(file_name, load_used=False, use_small=False):
    if not load_used:
        cached_file_path = file_name + ".cache"
        if os.path.isfile(cached_file_path):
            with open(cached_file_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(('Loading word embedding from %s'%file_name))
            ret = {}
            with open(file_name) as inf:
                for idx, line in enumerate(inf):
                    if (use_small and idx >= 5000):
                        break
                    info = line.strip().split(' ')
                    if info[0].lower() not in ret:
                        ret[info[0]] = np.array([float(x) for x in info[1:]])
            with open(cached_file_path, 'wb') as f:
                pickle.dump(ret, f)
            return ret
    else:
        print ('Load used word embedding')
        with open('../alt/glove/word2idx.json') as inf:
            w2i = json.load(inf)
        with open('../alt/glove/usedwordemb.npy') as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val
