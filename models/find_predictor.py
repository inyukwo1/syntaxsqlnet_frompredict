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
            self.encoded_num = 1024
        else:
            self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
            self.encoded_num = N_h

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.outer1 = nn.Sequential(nn.Linear(N_h + self.encoded_num, N_h), nn.ReLU())
        self.outer2 = nn.Sequential(nn.Linear(N_h, 1))
        if gpu:
            self.cuda()

    def forward(self, q_emb, q_len, hs_emb_var, hs_len):

        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        B = len(q_len)
        max_table_len = 0

        if self.use_bert:
            q_enc = self.q_bert(q_emb, q_len)
        else:
            q_enc, _ = run_lstm(self.q_lstm, q_emb, q_len)
        assert list(q_enc.size()) == [B, max_q_len, self.encoded_num]
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        hs_enc = hs_enc[:, 0, :]
        q_enc = q_enc[:, 0, :]
        q_enc = torch.cat((q_enc, hs_enc), dim=1)
        x = self.outer1(q_enc)
        x = self.outer2(x).squeeze()
        return x

    def loss(self, score, labels):
        loss = F.binary_cross_entropy_with_logits(score, labels)
        return loss

    def check_acc(self, score, truth):
        err = 0
        B = len(truth)
        pred = []
        if self.gpu:
            score = F.tanh(score).data.cpu().numpy()
            truth = truth.cpu().numpy()
        else:
            score = F.tanh(score).data.numpy()
            truth = truth.numpy()

        for b in range(B):
            if score[b] > 0. and truth[b] < 0.5:
                err += 1
            elif score[b] <= 0. and truth[b] > 0.5:
                err += 1
        return np.array(err)

    def check_eval_acc(self, score, label):
        if self.gpu:
            score = F.tanh(score).data.cpu().numpy()
        else:
            score = F.tanh(score).data.numpy()
        correct = True
        for b in range(len(score)):
            if score[b] > 0. and str(b) not in label:
                correct = False
            elif score[b] < 0. and str(b) in label:
                correct = False

        print(score)
        print("=======")
        print(label)
        print("@@@@@@@@@@@@@@{}".format(correct))
        if correct:
            return 1.
        else:
            return 0.

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

