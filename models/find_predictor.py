import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, col_tab_name_encode, encode_question, SIZE_CHECK, seq_conditional_weighted_num
from pytorch_pretrained_bert import BertModel
from models.schema_encoder import SchemaEncoder


class FindPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu, use_hs, bert=None):
        super(FindPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu
        self.use_hs = use_hs

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

        self.schema_encoder = SchemaEncoder(N_h)

        self.q_table_att = nn.Linear(self.encoded_num, N_h)
        self.q_col_att = nn.Linear(self.encoded_num, N_h)
        self.hs_table_att = nn.Linear(N_h, N_h)
        self.hs_col_att = nn.Linear(N_h, N_h)

        self.q_table_out = nn.Linear(N_h, N_h)
        self.q_col_out = nn.Linear(N_h, N_h)
        self.hs_table_out = nn.Linear(N_h, N_h)
        self.hs_col_out = nn.Linear(N_h, N_h)

        self.table_att = nn.Sequential(nn.Tanh(), nn.Linear(N_h, N_h))

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
        SIZE_CHECK(table_tensors, [B, max_table_len, self.N_h])
        SIZE_CHECK(col_tensors, [B, max_col_len, self.N_h])

        q_table_weighted_num = seq_conditional_weighted_num(self.q_table_att, q_enc, q_len, table_tensors, table_len).sum(1)
        hs_table_weighted_num = seq_conditional_weighted_num(self.hs_table_att, hs_enc, hs_len, table_tensors, table_len).sum(1)
        q_col_weighted_num = seq_conditional_weighted_num(self.q_col_att, q_enc, q_len, col_tensors, col_len).sum(1)
        hs_col_weighted_num = seq_conditional_weighted_num(self.hs_col_att, hs_enc, hs_len, col_tensors, col_len).sum(1)

        x = self.q_table_out(q_table_weighted_num)
        x = x + int(self.use_hs) * self.hs_table_out(hs_table_weighted_num)
        x = x + self.q_col_out(q_col_weighted_num)
        x = x + int(self.use_hs) * self.hs_col_out(hs_col_weighted_num)

        SIZE_CHECK(x, [B, self.N_h])
        table_score = (self.table_att(table_tensors) * x.unsqueeze(1)).sum(2)
        SIZE_CHECK(table_score, [B, max_table_len])
        for idx, num in enumerate(table_len.tolist()):
            if num < max_table_len:
                table_score[idx, num:] = -100

        return table_score

    def loss(self, score, truth):
        B = len(truth)
        SIZE_CHECK(score, [B, None])
        max_table_len = list(score.size())[1]
        label = []
        for one_truth in truth:
            label.append([1. if str(i) in one_truth else 0. for i in range(max_table_len)])

        label = torch.from_numpy(np.array(label)).type(torch.FloatTensor)
        if self.gpu:
            label = label.cuda()
        label = Variable(label)
        loss = F.binary_cross_entropy_with_logits(score, label)
        return loss

    def check_acc(self, score, truth):
        err = 0
        score = F.sigmoid(score)
        B = len(truth)
        if self.gpu:
            score = score.cpu()
        score = score.data.numpy()
        threshold = 0.5
        for b in range(B):
            if score[b] > threshold and str(b) not in truth[b]:
                err += 1
            if score[b] <= threshold and str(b) in truth[b]:
                err += 1

        return np.array(err)
