import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, col_tab_name_encode, encode_question, SIZE_CHECK, seq_conditional_weighted_num
from pytorch_pretrained_bert import BertModel
from models.schema_encoder import SchemaEncoder, SchemaAggregator


class FindPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu, use_hs, bert=None):
        super(FindPredictor, self).__init__()
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

        self.schema_encoder = SchemaEncoder(N_word, N_h, N_h)
        self.schema_aggregator = SchemaAggregator(N_h)

        self.q_table_att = nn.Linear(self.encoded_num, N_h)
        self.q_col_att = nn.Linear(self.encoded_num, N_h)
        self.hs_table_att = nn.Linear(N_h, N_h)
        self.hs_col_att = nn.Linear(N_h, N_h)

        self.schema_out = nn.Linear(N_h, N_h)
        self.q_hs_table_out = nn.Sequential(nn.Linear(self.encoded_num + 2 * N_h, N_h), nn.ReLU(), nn.Linear(N_h, N_h), nn.ReLU(), nn.Linear(N_h, N_h))
        self.q_hs_col_out = nn.Sequential(nn.Linear(self.encoded_num + 2 * N_h, N_h), nn.ReLU(), nn.Linear(N_h, N_h), nn.ReLU(), nn.Linear(N_h, N_h))
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

        q_table_weighted_num = seq_conditional_weighted_num(self.q_table_att, q_enc, q_len, table_tensors, table_len)
        SIZE_CHECK(q_table_weighted_num, [B, max_table_len, self.encoded_num])
        hs_table_weighted_num = seq_conditional_weighted_num(self.hs_table_att, hs_enc, hs_len, table_tensors, table_len)
        SIZE_CHECK(hs_table_weighted_num, [B, max_table_len, self.N_h])
        q_col_weighted_num = seq_conditional_weighted_num(self.q_col_att, q_enc, q_len, col_tensors, col_len)
        hs_col_weighted_num = seq_conditional_weighted_num(self.hs_col_att, hs_enc, hs_len, col_tensors, col_len)

        x = torch.cat((q_table_weighted_num, hs_table_weighted_num, self.schema_out(F.relu(aggregated_schema)).unsqueeze(1).expand(-1, max_table_len, -1)), dim=2)
        table_score = self.q_hs_table_out(x).sum(2)
        SIZE_CHECK(table_score, [B, max_table_len])
        for idx, num in enumerate(table_len.tolist()):
            if num < max_table_len:
                table_score[idx, num:] = -100

        x = torch.cat((q_col_weighted_num, hs_col_weighted_num, self.schema_out(F.relu(aggregated_schema)).unsqueeze(1).expand(-1, max_col_len, -1)), dim=2)
        col_score = self.q_hs_col_out(x).sum(2)
        for idx, num in enumerate(col_len.tolist()):
            if num < max_col_len:
                col_score[idx, num:] = -100

        return table_score, col_score

    def loss(self, score, truth):
        table_score, col_score = score
        num_truth = [len(one_truth) - 1 for one_truth in truth]
        data = torch.from_numpy(np.array(num_truth))
        if self.gpu:
            data = data.cuda()

        loss = 0

        B = len(truth)
        SIZE_CHECK(table_score, [B, None])
        max_table_len = list(table_score.size())[1]
        max_col_len = list(col_score.size())[1]
        label = []
        fcol_label = []
        for one_truth in truth:
            label.append([1. if str(i) in one_truth else 0. for i in range(max_table_len)])
            one_f_cols = []
            for flist in one_truth.values():
                one_f_cols += flist
            fcol_label.append([1. if i in one_f_cols else 0. for i in range(max_col_len)])
        label = torch.from_numpy(np.array(label)).type(torch.FloatTensor)
        fcol_label = torch.from_numpy(np.array(fcol_label)).type(torch.FloatTensor)
        if self.gpu:
            label = label.cuda()
            fcol_label = fcol_label.cuda()
        label = Variable(label)
        fcol_label = Variable(fcol_label)
        loss += F.binary_cross_entropy_with_logits(table_score, label)
        loss += F.binary_cross_entropy_with_logits(col_score, fcol_label)
        return loss

    def check_acc(self, score, truth):
        err = 0
        B = len(truth)
        pred = []
        if self.gpu:
            table_score, col_score = [F.tanh(sc).data.cpu().numpy() for sc in score]
        else:
             table_score, col_score = [F.tanh(sc).data.numpy() for sc in score]

        for b in range(B):
            cur_pred = {}
            cur_pred["table"] = []
            for i, val in enumerate(table_score[b]):
                if val > 0.:
                    cur_pred["table"].append(i)
            pred.append(cur_pred)
        for b, (p, t) in enumerate(zip(pred, truth)):
            tab = p['table']
            fk_list = []
            regular = []
            for l in t:
                l = int(l)
                if isinstance(l, list):
                    fk_list.append(l)
                else:
                    regular.append(l)
            if set(tab) != set(regular):
                err += 1

        return np.array(err)

    def score_to_tables(self, score):
        score = F.sigmoid(score)
        if self.gpu:
            score = [sc.data.cpu().numpy() for sc in score]
        else:
            score = [sc.data.numpy() for sc in score]
        B = len(score)
        batch_tables = []
        for b in range(B):
            tables = []
            for entry in range(len(score[b])):
                if score[b][entry] > self.threshold:
                    tables.append(entry)
            batch_tables.append(tables)
        return batch_tables

