import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from pytorch_pretrained_bert import BertTokenizer


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
        if self.use_bert:
            return self.gen_x_q_bert_batch(q)
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

    def gen_bert_batch_with_table(self, q, tables, table_cols):
        tokenized_q = []
        tokenized_t_c = []
        q_len = np.zeros(len(q), dtype=np.int64)
        t_c_num_len = []
        t_c_name_len = []
        table_locs = []
        t_c_type_ids = []
        for idx, one_q in enumerate(q):
            input_q = "[CLS] " + " ".join(one_q)
            input_t_c_list = []
            one_t_c_name_len = []
            one_t_c_type_ids = []
            for table_num, table_name in enumerate(tables[idx]):
                input_t_c = "[CLS] " + table_name
                for par_tab, col_name in table_cols[idx]:
                    if par_tab == table_num:
                        input_t_c += " [SEP] " + col_name
                tokenized_one_t_c = self.bert_tokenizer.tokenize(input_t_c)
                table_token_len = -1
                for tok_idx, token in enumerate(tokenized_one_t_c):
                    if token == "[SEP]":
                        table_token_len = tok_idx - 1
                        break
                assert table_token_len != -1
                one_t_c_type_ids.append([0] * table_token_len)
                indexed_one_t_c = self.bert_tokenizer.convert_tokens_to_ids(tokenized_one_t_c)
                input_t_c_list.append(indexed_one_t_c)
                one_t_c_name_len.append(len(indexed_one_t_c))
            t_c_type_ids.append(one_t_c_type_ids)
            t_c_name_len.append(one_t_c_name_len)
            t_c_num_len.append(len(input_t_c_list))
            tokenized_t_c.append(input_t_c_list)
            tokenozed_one_q = self.bert_tokenizer.tokenize(input_q)
            indexed_one_q = self.bert_tokenizer.convert_tokens_to_ids(tokenozed_one_q)
            tokenized_q.append(indexed_one_q)
            q_len[idx] = len(indexed_one_q)
            table_loc = []
            for i in range(len(input_t_c_list)):
                table_loc.append(len(tokenozed_one_q) + 7 * i)
            table_locs.append(table_loc)
        max_len = max(q_len)
        for tokenized_one_q in tokenized_q:
            tokenized_one_q += [0] * (max_len - len(tokenized_one_q))
        max_name_len = max(map(max, t_c_name_len))
        max_num_len = max(t_c_num_len)
        for tokenized_t_c_list in tokenized_t_c:
            for one_tokenized_t_c in tokenized_t_c_list:
                one_tokenized_t_c += [0] * (max_name_len - len(one_tokenized_t_c))
            tokenized_t_c_list += [[0] * max_name_len] * (max_num_len - len(tokenized_t_c_list))
        for one_t_c_type_ids in t_c_type_ids:
            for type_ids in one_t_c_type_ids:
                type_ids += [1] * (max_name_len - len(type_ids))
            one_t_c_type_ids += [[0] * max_name_len] * (max_num_len - len(one_t_c_type_ids))
        t_c_type_ids = torch.LongTensor(t_c_type_ids)
        tokenized_q = torch.LongTensor(tokenized_q)
        tokenized_t_c = torch.LongTensor(tokenized_t_c)
        special_tok_id = self.bert_tokenizer.convert_tokens_to_ids(["[SEP]"])
        special_tok_id = torch.LongTensor(special_tok_id).view(1, 1, -1)
        if self.gpu:
            tokenized_q = tokenized_q.cuda()
            t_c_type_ids = t_c_type_ids.cuda()
            tokenized_t_c = tokenized_t_c.cuda()
            special_tok_id = special_tok_id.cuda()
        return tokenized_q, q_len, tokenized_t_c, t_c_num_len, t_c_name_len, t_c_type_ids, special_tok_id, table_locs

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
                        item == "descending"
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
