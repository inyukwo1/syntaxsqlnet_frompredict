import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, seq_conditional_weighted_num, SIZE_CHECK
from pytorch_pretrained_bert import BertModel


class MultiSqlPredictor(nn.Module):
    '''Predict if the next token is (multi SQL key words):
        NONE, EXCEPT, INTERSECT, or UNION.'''
    def __init__(self, N_word, N_h, N_depth, gpu, use_hs, bert=None):
        super(MultiSqlPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu
        self.use_hs = use_hs

        self.use_bert = True if bert else False
        if bert:
            self.q_bert = bert
            encoded_num = 768
        else:
            self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
            encoded_num = N_h

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.mkw_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_att = nn.Linear(encoded_num, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.multi_out_q = nn.Linear(encoded_num, N_h)
        self.multi_out_hs = nn.Linear(N_h, N_h)
        self.multi_out_c = nn.Linear(N_h, N_h)
        self.multi_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

        if gpu:
            self.cuda()

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, mkw_emb_var, mkw_len):
        # print("q_emb_shape:{} hs_emb_shape:{}".format(q_emb_var.size(), hs_emb_var.size()))
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        B = len(q_len)

        # q_enc: (B, max_q_len, hid_dim)
        # hs_enc: (B, max_hs_len, hid_dim)
        # mkw: (B, 4, hid_dim)
        if self.use_bert:
            q_enc = self.q_bert(q_emb_var, q_len)
        else:
            q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        mkw_enc, _ = run_lstm(self.mkw_lstm, mkw_emb_var, mkw_len)

        # Compute attention values between multi SQL key words and question tokens.
        q_weighted = seq_conditional_weighted_num(self.q_att, q_enc, q_len, mkw_enc)
        SIZE_CHECK(q_weighted, [B, 4, self.N_h])

        # Same as the above, compute SQL history embedding weighted by key words attentions
        hs_weighted = seq_conditional_weighted_num(self.hs_att, hs_enc, hs_len, mkw_enc)

        # Compute prediction scores=
        mulit_score = self.multi_out(self.multi_out_q(q_weighted) + int(self.use_hs)* self.multi_out_hs(hs_weighted) + self.multi_out_c(mkw_enc)).view(B, 4)

        return mulit_score


    def loss(self, score, truth):
        data = torch.from_numpy(np.array(truth))
        if self.gpu:
            data = data.cuda()
        truth_var = Variable(data)
        loss = self.CE(score, truth_var)

        return loss


    def check_acc(self, score, truth):
        err = 0
        B = len(score)
        pred = []
        for b in range(B):
            if self.gpu:
                argmax_score = np.argmax(score[b].data.cpu().numpy())
            else:
                argmax_score = np.argmax(score[b].data.numpy())
            pred.append(argmax_score)
        for b, (p, t) in enumerate(zip(pred, truth)):
            if p != t:
                err += 1

        return err
