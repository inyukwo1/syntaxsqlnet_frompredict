import re
import io
import json
import numpy as np
import os
import signal
from preprocess_train_dev_data import get_table_dict
import tqdm
import random
from random import shuffle
import pandas as pd
import torch
import os.path
import pickle


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


def generate_from_candidates(par_tab_nums, foreign_keys, real_froms):
    B = len(par_tab_nums)
    def extract_from(par_tab_num, foreign_key):
        max_par_tab_num = par_tab_num[-1]
        synthetic_from = dict()
        start_table = random.randint(0, max_par_tab_num)
        synthetic_from[start_table] = set()
        while random.randint(0, 100) < 50:
            shuffle(foreign_key)
            for pair in foreign_key:
                parent_1 = par_tab_num[pair[0]]
                parent_2 = par_tab_num[pair[1]]
                if parent_1 in synthetic_from and parent_2 not in synthetic_from:
                    synthetic_from[parent_1].add(pair[0])
                    synthetic_from[parent_2] = set()
                    synthetic_from[parent_2].add(pair[1])
                    break

                elif parent_2 in synthetic_from and parent_1 not in synthetic_from:
                    synthetic_from[parent_2].add(pair[1])
                    synthetic_from[parent_1] = set()
                    synthetic_from[parent_1].add(pair[0])
                    break

        for tab in synthetic_from:
            synthetic_from[tab] = list(synthetic_from[tab])
            synthetic_from[tab].sort()
        return synthetic_from

    from_candidates = []
    labels = []
    for par_tab_num, foreign_key, real_from in zip(par_tab_nums, foreign_keys, real_froms):
        candidates = []
        while len(candidates) < 10:  # HARD-CODED
            synthetic_from = extract_from(par_tab_num, foreign_key)
            if synthetic_from != real_from:
                candidates.append(synthetic_from)
        candidates.append(real_from)
        label = [0.] * 10 + [1.]
        together = list(zip(candidates, label))
        shuffle(together)
        candidates, label = zip(*together)
        from_candidates.append(candidates)
        labels.append(label)
    labels = torch.from_numpy(np.array(labels, dtype=np.float32))
    if torch.cuda.is_available():
        labels = labels.cuda()
    return from_candidates, labels


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
def epoch_train(gpu, model, optimizer, batch_size, component,embed_layer,data, table_type, use_tqdm, optimizer_bert, use_from):
    model.train()
    # newdata = []
    # # for entry in data:
    # #     if len(entry["ts"][0]) > 1:
    # #         newdata.append(entry)
    # data = newdata
    perm=np.random.permutation(len(data))
    cum_loss = 0.0
    st = 0

    for _ in tqdm.tqdm(range(len(data) // batch_size), disable=not use_tqdm):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, history,label = to_batch_seq(data, perm, st, ed)
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
            if not use_from:
                from_candidates = None
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
            real_froms = []
            for one_from in label:
                real_from = dict()
                for tab in one_from:
                    real_from[int(tab)] = one_from[tab]
                    real_from[int(tab)].sort()
                real_froms.append(real_from)
            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len = embed_layer.gen_col_batch(
                col_seq, tab_seq)
            candidate_schemas, label = generate_from_candidates(par_tab_nums, foreign_keys, real_froms)
            score = model.forward(par_tab_nums, candidate_schemas, q_emb_var, q_len, hs_emb_var, hs_len,
                                  col_emb_var, col_len, col_name_len, table_emb_var, table_len, table_name_len)
        loss = model.loss(score, label)
        # print("loss {}".format(loss.data.cpu().numpy()))
        if gpu:
            cum_loss += loss.data.cpu().numpy()*(ed - st)
        else:
            cum_loss += loss.data.numpy()*(ed - st)
        optimizer.zero_grad()
        if optimizer_bert:
            optimizer_bert.zero_grad()
        loss.backward()
        optimizer.step()
        if optimizer_bert:
            optimizer_bert.step()

        st = ed

    return cum_loss / len(data)


## used for development evaluation in train.py
def epoch_acc(model, batch_size, component, embed_layer,data, table_type, error_print=False, train_flag = False, use_from = False):
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
            if not use_from:
                from_candidates = None
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
            real_froms = []
            for one_from in label:
                real_from = dict()
                for tab in one_from:
                    real_from[int(tab)] = one_from[tab]
                    real_from[int(tab)].sort()
                real_froms.append(real_from)
            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(data, perm, st, ed, table_type)
            col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len = embed_layer.gen_col_batch(
                col_seq, tab_seq)
            candidate_schemas, label = generate_from_candidates(par_tab_nums, foreign_keys, real_froms)
            score = model.forward(par_tab_nums, candidate_schemas, q_emb_var, q_len, hs_emb_var, hs_len,
                                  col_emb_var, col_len, col_name_len, table_emb_var, table_len, table_name_len)
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
