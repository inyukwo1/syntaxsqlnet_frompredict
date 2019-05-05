import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from graph_utils import *
import random


AGG_OPS = ('none', 'maximum', 'minimum', 'count', 'sum', 'average')


class WordEmbedding(nn.Module):
    def __init__(self, word_emb, N_word, gpu, SQL_TOK, use_bert, trainable=False):
        super(WordEmbedding, self).__init__()
        self.trainable = trainable
        self.N_word = N_word
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK
        self.use_bert = use_bert
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        self.sep_embedding = np.random.rand(self.N_word)

        if trainable:
            print("Using trainable embedding")
            self.w2i, word_emb_val = word_emb
            # tranable when using pretrained model, init embedding weights using prev embedding
            self.embedding = nn.Embedding(len(self.w2i), N_word)
            self.embedding.weight = nn.Parameter(torch.from_numpy(word_emb_val.astype(np.float32)))
        else:
            # else use word2vec or glove
            self.word_emb = word_emb
            print("Using fixed embedding")

    def word_find(self, word):
        word = ''.join([i for i in word if i.isalpha()])
        word = word.lower()
        return self.word_emb.get(word, np.zeros(self.N_word, dtype=np.float32))

    def gen_x_q_batch(self, q):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_q in enumerate(q):
            q_val = []
            for ws in one_q:
                q_val.append(self.word_find(ws))

            val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [np.zeros(self.N_word, dtype=np.float32)])  #<BEG> and <END>
            val_len[i] = 1 + len(q_val) + 1
        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)

        return val_inp_var, val_len

    def gen_joingraph_encoding_nobert(self, q, tables, table_cols, foreign_keys, primary_keys, labels):
        B = len(q)
        val_embs = []
        table_embs = []
        table_embs_len = np.zeros(B, dtype=np.int64)
        val_len = np.zeros(B, dtype=np.int64)
        anses = []
        for idx, one_q in enumerate(q):
            parent_tables = []
            table_name = tables[idx]
            for t, c in table_cols[idx]:
                parent_tables.append(t)

            if random.randint(0, 100) < 7:
                true_graph = 1.
                generated_graph = str_graph_to_num_graph(labels[idx])
            else:
                true_graph = 0.
                generated_graph = generate_random_graph_generate(len(tables[idx]), parent_tables, foreign_keys[idx])
                if graph_checker(generated_graph, labels[idx], foreign_keys[idx], primary_keys[idx]):
                    true_graph = 1.
            anses.append(true_graph)
            q_val = []
            for ws in one_q:
                q_val.append(self.word_find(ws))


            col_name_dict = {}
            for table_num in generated_graph:
                col_name_dict[table_num] = []
            renamed_table_cols = deepcopy(table_cols[idx])
            for col_idx, [par_tab, col_name] in enumerate(renamed_table_cols):
                if par_tab in generated_graph:
                    if col_idx in generated_graph[par_tab]:
                        if col_idx in primary_keys:
                            for f, p in foreign_keys:
                                if parent_tables[f] in generated_graph and f in generated_graph[parent_tables[f]] and p == col_idx:
                                    _, col_name = renamed_table_cols[f]
                            for f, p in foreign_keys:
                                if parent_tables[f] in generated_graph and f in generated_graph[parent_tables[f]] and p == col_idx:
                                    renamed_table_cols[f][1] = col_name
                        col_name_dict[par_tab].append(col_name)
                    else:
                        col_name_dict[par_tab].append(col_name)

            # one_batch_table_embs = []
            # one_batch_table_len = []
            # for table_num in generated_graph:
            #     one_table_embs = [np.zeros(self.N_word, dtype=np.float32)] + [self.word_find(x) for x in table_name[table_num]]
            #     col_names = col_name_dict[table_num]
            #     for col_name in col_names:
            #         one_table_embs += [self.sep_embedding] + [self.word_find(x) for x in col_name]
            #     one_table_embs += [np.zeros(self.N_word, dtype=np.float32)]
            #     one_batch_table_embs.append(one_table_embs)
            #     one_batch_table_len.append(len(one_table_embs))
            # table_embs.append(one_batch_table_embs)
            # table_embs_len.append(one_batch_table_len)

            # one_table_embs = []
            # for table_num in generated_graph:
            #     one_table_embs += [self.sep_embedding] + [self.word_find(x) for x in table_name[table_num]]
            # for table_num in generated_graph:
            #     col_names = col_name_dict[table_num]
            #     for col_name in col_names:
            #         one_table_embs += [self.sep_embedding] + [self.word_find(x) for x in col_name]
            # one_table_embs += [np.zeros(self.N_word, dtype=np.float32)]
            #
            # table_embs.append(one_table_embs)
            # table_embs_len[idx] = len(one_table_embs)

            cols_seq = []
            for table_num in generated_graph:
                col_names = col_name_dict[table_num]
                for col_name in col_names:
                    one_col_seq = table_name[table_num].split(" ")
                    one_col_seq += col_name.split(" ")
                    cols_seq.append(one_col_seq)
            table_embs_len[idx] = len(cols_seq)
            table_embs += cols_seq

            val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [
                np.zeros(self.N_word, dtype=np.float32)])  # <BEG> and <END>
            val_len[idx] = 1 + len(q_val) + 1

        table_embs_var, table_name_len = self.str_list_to_batch(table_embs)

        max_val_len = max(val_len)
        val_emb_array = np.zeros((B, max_val_len, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        anses = torch.tensor(anses)
        if self.gpu:
            val_inp = val_inp.cuda()
            anses = anses.cuda()
        val_inp_var = Variable(val_inp)

        return val_inp_var, val_len, table_embs_var, table_name_len, table_embs_len, anses

    def gen_joingraph_eval_nobert(self, one_q, one_tables, one_cols, foreign_keys, primary_keys):
        parent_nums = []
        for par_tab, _ in one_cols:
            parent_nums.append(par_tab)
        table_graph_lists = []
        for tab in range(len(one_tables)):
            table_graph_lists += list(generate_four_hop_path_from_seed(tab, parent_nums, foreign_keys))

        simple_graph_lists = []
        for graph in table_graph_lists:
            new_graph = deepcopy(graph)
            for k in new_graph:
                for idx, l in enumerate(new_graph[k]):
                    new_graph[k][idx] = l[0]
            simple_graph_lists.append(new_graph)
        B = len(table_graph_lists)
        val_embs = []
        table_embs = []
        table_embs_len = np.zeros(B, dtype=np.int64)
        val_len = np.zeros(B, dtype=np.int64)
        for b in range(B):
            q_val = []
            for ws in one_q:
                q_val.append(self.word_find(ws))
            generated_graph = table_graph_lists[b]
            col_name_dict = {}
            for table_num in generated_graph:
                col_name_dict[table_num] = []
            renamed_table_cols = deepcopy(one_cols)
            for col_idx, [par_tab, col_name] in enumerate(renamed_table_cols):
                if par_tab in generated_graph:
                    if col_idx in generated_graph[par_tab]:
                        if col_idx in primary_keys:
                            for f, p in foreign_keys:
                                if parent_nums[f] in generated_graph and f in generated_graph[parent_nums[f]] and p == col_idx:
                                    _, col_name = renamed_table_cols[f]
                            for f, p in foreign_keys:
                                if parent_nums[f] in generated_graph and f in generated_graph[parent_nums[f]] and p == col_idx:
                                    renamed_table_cols[f][1] = col_name
                        col_name_dict[par_tab].append(col_name)
                    else:
                        col_name_dict[par_tab].append(col_name)

            # one_batch_table_embs = []
            # one_batch_table_len = []
            # for table_num in generated_graph:
            #     one_table_embs = [np.zeros(self.N_word, dtype=np.float32)] + [self.word_find(x) for x in
            #                                                                   one_tables[table_num]]
            #     col_names = col_name_dict[table_num]
            #     for col_name in col_names:
            #         one_table_embs += [self.sep_embedding] + [self.word_find(x) for x in col_name]
            #     one_table_embs += [np.zeros(self.N_word, dtype=np.float32)]
            #     one_batch_table_embs.append(one_table_embs)
            #     one_batch_table_len.append(len(one_table_embs))
            # table_embs.append(one_batch_table_embs)
            # table_embs_len.append(one_batch_table_len)

            # one_table_embs = []
            # for table_num in generated_graph:
            #     one_table_embs += [self.sep_embedding] + [self.word_find(x) for x in one_tables[table_num]]
            #
            # for table_num in generated_graph:
            #     col_names = col_name_dict[table_num]
            #     for col_name in col_names:
            #         one_table_embs += [self.sep_embedding] + [self.word_find(x) for x in col_name]
            # one_table_embs += [np.zeros(self.N_word, dtype=np.float32)]
            # table_embs.append(one_table_embs)
            # table_embs_len[b] = len(one_table_embs)

            cols_seq = []
            for table_num in generated_graph:
                col_names = col_name_dict[table_num]
                for col_name in col_names:
                    one_col_seq = one_tables[table_num].split(" ")
                    one_col_seq += col_name.split(" ")
                    cols_seq.append(one_col_seq)
            table_embs_len[b] = len(cols_seq)
            table_embs += cols_seq

            val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [np.zeros(self.N_word, dtype=np.float32)])  # <BEG> and <END>
            val_len[b] = 1 + len(q_val) + 1

        table_embs_var, table_name_len = self.str_list_to_batch(table_embs)
        max_len = max(val_len)
        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var, val_len, table_embs_var, table_name_len, table_embs_len, simple_graph_lists, table_graph_lists


    def gen_x_q_bert_batch(self, q):
        tokenized_q = []
        q_len = np.zeros(len(q), dtype=np.int64)
        for idx, one_q in enumerate(q):
            tokenized_one_q = self.bert_tokenizer.tokenize(" ".join(one_q))
            indexed_one_q = self.bert_tokenizer.convert_tokens_to_ids(tokenized_one_q)
            tokenized_q.append(indexed_one_q)
            q_len[idx] = len(indexed_one_q)
        max_len = max(q_len)
        for tokenized_one_q in tokenized_q:
            tokenized_one_q += [0] * (max_len - len(tokenized_one_q))
        tokenized_q = torch.LongTensor(tokenized_q)
        if self.gpu:
            tokenized_q = tokenized_q.cuda()
        return tokenized_q, q_len

    def encode_one_q_with_bert(self, one_q, table_name, table_cols, parent_tables, foreign_keys, primary_keys, table_graph):
        input_q = "[CLS] " + " ".join(one_q)
        one_q_q_len = len(self.bert_tokenizer.tokenize(input_q))
        for table_num in table_graph:
            input_q += " [SEP] " + table_name[table_num]
        # table_names = [tables[idx][table_num] for table_num in generated_graph]
        # input_q += " ".join(table_names)
        col_name_dict = {}
        for table_num in table_graph:
            col_name_dict[table_num] = []
        renamed_table_cols = deepcopy(table_cols)
        for col_idx, [par_tab, col_name] in enumerate(renamed_table_cols):
            if par_tab in table_graph:
                if col_idx in table_graph[par_tab]:
                    if col_idx in primary_keys:
                        for f, p in foreign_keys:
                            if parent_tables[f] in table_graph and f in table_graph[parent_tables[f]] and p == col_idx:
                                _, col_name = renamed_table_cols[f]
                        for f, p in foreign_keys:
                            if parent_tables[f] in table_graph and f in table_graph[parent_tables[f]] and p == col_idx:
                                renamed_table_cols[f][1] = col_name
                    col_name_dict[par_tab].append(col_name)
                else:
                    col_name_dict[par_tab].append(col_name)
        col_name_list = [l for k, l in col_name_dict.items()]
        col_name_len_list = [len(l) for l in col_name_list]
        sep_embeddings = list(range(len(table_graph)))
        for k_idx, k in enumerate(table_graph):
            for cidx in range(max(col_name_len_list)):
                l = col_name_dict[k]
                if cidx < len(l):
                    input_q += " [SEP] " + l[cidx]
                    sep_embeddings.append(k_idx)

        tokenozed_one_q = self.bert_tokenizer.tokenize(input_q)
        indexed_one_q = self.bert_tokenizer.convert_tokens_to_ids(tokenozed_one_q)

        sep_embeddings_per_loc = []
        cur_sep_cnt = -1
        for token_idx, token in enumerate(tokenozed_one_q):
            if token == '[SEP]':
                cur_sep_cnt += 1
                sep_embeddings_per_loc.append(sep_embeddings[cur_sep_cnt])
            else:
                sep_embeddings_per_loc.append(-1)
        return one_q_q_len, indexed_one_q, sep_embeddings_per_loc

    def gen_bert_batch_with_table(self, q, tables, table_cols, foreign_keys, primary_keys, labels):
        tokenized_q = []

        q_len = []
        q_q_len = []
        anses = []
        sep_embeddings = []
        for idx, one_q in enumerate(q):
            parent_tables = []
            for t, c in table_cols[idx]:
                parent_tables.append(t)

            if random.randint(0, 100) < 7:
                true_graph = 1.
                generated_graph = str_graph_to_num_graph(labels[idx])
            else:
                true_graph = 0.
                generated_graph = generate_random_graph_generate(len(tables[idx]), parent_tables, foreign_keys[idx])
                if graph_checker(generated_graph, labels[idx], foreign_keys[idx], primary_keys[idx]):
                    true_graph = 1.
            anses.append(true_graph)

            one_q_q_len, indexed_one_q, one_sep_embeddings \
                = self.encode_one_q_with_bert(one_q, tables[idx], table_cols[idx], parent_tables, foreign_keys[idx], primary_keys[idx], generated_graph)
            q_q_len.append(one_q_q_len)
            tokenized_q.append(indexed_one_q)
            q_len.append(len(indexed_one_q))
            sep_embeddings.append(one_sep_embeddings)

        max_len = max(q_len)
        for tokenized_one_q in tokenized_q:
            tokenized_one_q += [0] * (max_len - len(tokenized_one_q))
        tokenized_q = torch.LongTensor(tokenized_q)
        anses = torch.tensor(anses)
        if self.gpu:
            tokenized_q = tokenized_q.cuda()
            anses = anses.cuda()
        return tokenized_q, q_len, q_q_len, anses, sep_embeddings

    def gen_bert_for_eval(self, one_q, one_tables, one_cols, foreign_keys, primary_keys):
        tokenized_q = []
        parent_nums = []
        sep_embeddings = []

        for par_tab, _ in one_cols:
            parent_nums.append(par_tab)
        table_graph_lists = []
        for tab in range(len(one_tables)):
            table_graph_lists += list(generate_four_hop_path_from_seed(tab, parent_nums, foreign_keys))

        simple_graph_lists = []
        for graph in table_graph_lists:
            new_graph = deepcopy(graph)
            for k in new_graph:
                for idx, l in enumerate(new_graph[k]):
                    new_graph[k][idx] = l[0]
            simple_graph_lists.append(new_graph)
        B = len(table_graph_lists)
        q_len = []
        q_q_len = []
        for b in range(B):

            one_q_q_len, indexed_one_q, one_sep_embeddings \
                = self.encode_one_q_with_bert(one_q, one_tables, one_cols, parent_nums, foreign_keys, primary_keys,
                                              simple_graph_lists[b])
            q_q_len.append(one_q_q_len)
            tokenized_q.append(indexed_one_q)
            q_len.append(len(indexed_one_q))
            sep_embeddings.append(one_sep_embeddings)

        max_len = max(q_len)
        for tokenized_one_q in tokenized_q:
            tokenized_one_q += [0] * (max_len - len(tokenized_one_q))
        tokenized_q = torch.LongTensor(tokenized_q)
        if self.gpu:
            tokenized_q = tokenized_q.cuda()
        return tokenized_q, q_len, q_q_len, simple_graph_lists, table_graph_lists, sep_embeddings

    def gen_x_history_batch(self, history):
        B = len(history)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_history in enumerate(history):
            history_val = []
            for item in one_history:
                #col
                if isinstance(item, list) or isinstance(item, tuple):
                    emb_list = []
                    ws = item[0].split() + item[1].split()
                    ws_len = len(ws)
                    for w in ws:
                        emb_list.append(self.word_find(w))
                    if ws_len == 0:
                        raise Exception("word list should not be empty!")
                    elif ws_len == 1:
                        history_val.append(emb_list[0])
                    else:
                        history_val.append(sum(emb_list) / float(ws_len))
                #ROOT
                elif isinstance(item,str):
                    if item == "ROOT":
                        item = "root"
                    elif item == "asc":
                        item = "ascending"
                    elif item == "desc":
                        item = "descending"
                    if item in (
                    "none", "select", "from", "where", "having", "limit", "intersect", "except", "union", 'not',
                    'between', '=', '>', '<', 'in', 'like', 'is', 'exists', 'root', 'ascending', 'descending'):
                        history_val.append(self.word_find(item))
                    elif item == "orderBy":
                        history_val.append((self.word_find("order") +
                                            self.word_find("by")) / 2)
                    elif item == "groupBy":
                        history_val.append((self.word_find("group") +
                                            self.word_find("by")) / 2)
                    elif item in ('>=', '<=', '!='):
                        history_val.append((self.word_find(item[0]) +
                                            self.word_find(item[1])) / 2)
                elif isinstance(item,int):
                    history_val.append(self.word_find(AGG_OPS[item]))
                else:
                    print(("Warning: unsupported data type in history! {}".format(item)))

            val_embs.append(history_val)
            val_len[i] = len(history_val)
        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)

        return val_inp_var, val_len


    def gen_word_list_embedding(self,words,B):
        val_emb_array = np.zeros((B,len(words), self.N_word), dtype=np.float32)
        for i,word in enumerate(words):
            if len(word.split()) == 1:
                emb = self.word_find(word)
            else:
                word = word.split()
                emb = (self.word_find(word[0]) + self.word_find(word[1]))/2
            for b in range(B):
                val_emb_array[b,i,:] = emb
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var


    def gen_col_batch(self, cols, tables):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)
        tab_len = np.zeros(len(tables), dtype=np.int64)

        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[b] = len(one_cols)
        name_inp_var, name_len = self.str_list_to_batch(names)

        table_names = []
        for b, one_tables in enumerate(tables):
            table_names = table_names + one_tables
            tab_len[b] = len(one_tables)
        table_name_inp_var, table_name_len = self.str_list_to_batch(table_names)
        return name_inp_var, name_len, col_len, table_name_inp_var, table_name_len, tab_len

    def str_list_to_batch(self, str_list):
        """get a list var of wemb of words in each column name in current bactch"""
        B = len(str_list)

        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_str in enumerate(str_list):
            if self.trainable:
                val = [self.w2i.get(x, 0) for x in one_str]
            else:
                val = [self.word_find(x) for x in one_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros(
                    (B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        return val_inp_var, val_len
