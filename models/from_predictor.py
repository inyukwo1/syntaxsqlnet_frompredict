import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, col_tab_name_encode, encode_question, SIZE_CHECK, seq_conditional_weighted_num
from pytorch_pretrained_bert import BertModel
from models.schema_encoder import SchemaEncoder, SchemaAggregator


class FromPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu, use_hs, bert=None):
        super(FromPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu
        self.use_hs = use_hs
        self.threshold = 0.5

        self.use_bert = True if bert else False
        if bert:
            self.q_bert = bert
            self.encoded_num = 768
        else:
            self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
            self.encoded_num = N_h

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.schema_encoder = SchemaEncoder(N_h, 200)
        self.schema_aggregator = SchemaAggregator(N_h)

        self.q_table_num_att = nn.Linear(self.encoded_num, N_h)
        self.q_col_num_att = nn.Linear(self.encoded_num, N_h)
        self.hs_table_num_att = nn.Linear(N_h, N_h)
        self.hs_col_num_att = nn.Linear(N_h, N_h)

        self.schema_num_out = nn.Linear(N_h, N_h)
        self.q_table_num_out = nn.Linear(N_h, N_h)
        self.q_col_num_out = nn.Linear(N_h, N_h)
        self.hs_table_num_out = nn.Linear(N_h, N_h)
        self.hs_col_num_out = nn.Linear(N_h, N_h)
        self.table_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 6))

        self.q_table_att = nn.Linear(self.encoded_num, N_h)
        self.q_col_att = nn.Linear(self.encoded_num, N_h)
        self.hs_table_att = nn.Linear(N_h, N_h)
        self.hs_col_att = nn.Linear(N_h, N_h)

        self.schema_out = nn.Linear(N_h, N_h)
        self.q_table_out = nn.Linear(N_h, N_h)
        self.q_col_out = nn.Linear(N_h, N_h)
        self.hs_table_out = nn.Linear(N_h, N_h)
        self.hs_col_out = nn.Linear(N_h, N_h)

        self.table_att = nn.Sequential(nn.Tanh(), nn.Linear(N_h, N_h), nn.Sigmoid())

        if gpu:
            self.cuda()

    def forward(self, parent_tables, foreign_keys, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len,
                table_emb_var, table_len, table_name_len):

        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        max_col_len = max(col_len)
        max_table_len = max(table_len)
        B = len(q_len)
        if self.use_bert:
            q_enc = self.q_bert(q_emb_var, q_len)
        else:
            q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        assert list(q_enc.size()) == [B, max_q_len, self.encoded_num]
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        table_tensors, col_tensors, batch_graph = self.schema_encoder(parent_tables, foreign_keys,
                                                                      col_emb_var, col_name_len, col_len,
                                                                      table_emb_var, table_name_len, table_len)
        aggregated_schema = self.schema_aggregator(batch_graph)
        SIZE_CHECK(table_tensors, [B, max_table_len, self.N_h])
        SIZE_CHECK(col_tensors, [B, max_col_len, self.N_h])

        q_table_weighted_num_num = seq_conditional_weighted_num(self.q_table_num_att, q_enc, q_len, table_tensors,
                                                            table_len).sum(1)
        hs_table_weighted_num_num = seq_conditional_weighted_num(self.hs_table_num_att, hs_enc, hs_len, table_tensors,
                                                             table_len).sum(1)
        q_col_weighted_num_num = seq_conditional_weighted_num(self.q_col_num_att, q_enc, q_len, col_tensors, col_len).sum(1)
        hs_col_weighted_num_num = seq_conditional_weighted_num(self.hs_col_num_att, hs_enc, hs_len, col_tensors, col_len).sum(1)

        x = self.schema_out(F.relu(aggregated_schema))
        x = x + self.q_table_out(q_table_weighted_num_num)
        x = x + int(self.use_hs) * self.hs_table_out(hs_table_weighted_num_num)
        x = x + self.q_col_out(q_col_weighted_num_num)
        x = x + int(self.use_hs) * self.hs_col_out(hs_col_weighted_num_num)
        table_num_score = self.table_num_out(x)


        q_table_weighted_num = seq_conditional_weighted_num(self.q_table_att, q_enc, q_len, table_tensors, table_len).sum(1)
        hs_table_weighted_num = seq_conditional_weighted_num(self.hs_table_att, hs_enc, hs_len, table_tensors, table_len).sum(1)
        q_col_weighted_num = seq_conditional_weighted_num(self.q_col_att, q_enc, q_len, col_tensors, col_len).sum(1)
        hs_col_weighted_num = seq_conditional_weighted_num(self.hs_col_att, hs_enc, hs_len, col_tensors, col_len).sum(1)

        x = self.schema_out(F.relu(aggregated_schema))
        x = x + self.q_table_out(q_table_weighted_num)
        x = x + int(self.use_hs) * self.hs_table_out(hs_table_weighted_num)
        x = x + self.q_col_out(q_col_weighted_num)
        x = x + int(self.use_hs) * self.hs_col_out(hs_col_weighted_num)

        SIZE_CHECK(x, [B, self.N_h])
        table_score = (self.table_att(table_tensors) * x.unsqueeze(1)).sum(2)
        SIZE_CHECK(table_score, [B, max_table_len])
        for idx, num in enumerate(table_len.tolist()):
            if num < max_table_len:
                table_score[idx, num:] = -100

        return table_num_score, table_score

    def loss(self, score, truth):
        table_num_score, table_score = score
        num_truth = [len(one_truth) - 1 for one_truth in truth]
        data = torch.from_numpy(np.array(num_truth))
        if self.gpu:
            data = data.cuda()
        truth_num_var = Variable(data)
        loss = F.cross_entropy(table_num_score, truth_num_var)

        B = len(truth)
        SIZE_CHECK(table_score, [B, None])
        max_table_len = list(table_score.size())[1]
        label = []
        for one_truth in truth:
            label.append([1. if str(i) in one_truth else 0. for i in range(max_table_len)])
        label = torch.from_numpy(np.array(label)).type(torch.FloatTensor)
        if self.gpu:
            label = label.cuda()
        label = Variable(label)
        loss += F.binary_cross_entropy_with_logits(table_score, label)
        return loss

    def check_acc(self, score, truth):
        num_err, err, tot_err = 0, 0, 0
        B = len(truth)
        pred = []
        if self.gpu:
            table_num_score, table_score = [sc.data.cpu().numpy() for sc in score]
        else:
            table_num_score, table_score = [sc.data.numpy() for sc in score]

        for b in range(B):
            cur_pred = {}
            table_num = np.argmax(table_num_score[b]) + 1
            cur_pred["table_num"] = table_num
            cur_pred["table"] = np.argsort(-table_score[b])[:table_num]
            pred.append(cur_pred)
        for b, (p, t) in enumerate(zip(pred, truth)):
            table_num, tab = p['table_num'], p['table']
            flag = True
            if table_num != len(t): # double check truth format and for test cases
                num_err += 1
                flag = False
            fk_list = []
            regular = []
            for l in t:
                if isinstance(l, list):
                    fk_list.append(l)
                else:
                    regular.append(l)

            if flag: #double check
                for c in tab:
                    for fk in fk_list:
                        if c in fk:
                            fk_list.remove(fk)
                    for r in regular:
                        if c == r:
                            regular.remove(r)

                if len(fk_list) != 0 or len(regular) != 0:
                    err += 1
                    flag = False

            if not flag:
                tot_err += 1

        return np.array((num_err, err, tot_err))

    def score_to_tables(self, score):
        if self.gpu:
            table_num_score, table_score = [sc.data.cpu().numpy() for sc in score]
        else:
            table_num_score, table_score = [sc.data.numpy() for sc in score]
        B = len(score)
        batch_tables = []
        for b in range(B):
            cur_pred = {}
            table_num = np.argmax(table_num_score[b]) + 1
            cur_pred["table_num"] = table_num
            cur_pred["table"] = np.argsort(-table_score[b])[:table_num]
            batch_tables.append(cur_pred["table"])
        return batch_tables

