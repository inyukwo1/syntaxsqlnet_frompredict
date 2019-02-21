import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, col_tab_name_encode, plain_conditional_weighted_num, SIZE_CHECK
from pytorch_pretrained_bert import BertModel

class DesAscLimitPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu, use_hs, bert=None):
        super(DesAscLimitPredictor, self).__init__()
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

        self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_att = nn.Linear(encoded_num, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.dat_out_q = nn.Linear(encoded_num, N_h)
        self.dat_out_hs = nn.Linear(N_h, N_h)
        self.dat_out_c = nn.Linear(N_h, N_h)
        self.dat_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 4)) #for 4 desc/asc limit/none combinations

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if gpu:
            self.cuda()

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col):
        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        max_col_len = max(col_len)
        B = len(q_len)

        if self.use_bert:
            q_enc = self.q_bert(q_emb_var, q_len)
        else:
            q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        col_enc, _ = col_tab_name_encode(col_emb_var, col_name_len, col_len, self.col_lstm)

        # get target/predicted column's embedding
        # col_emb: (B, hid_dim)
        col_emb = []
        for b in range(B):
            col_emb.append(col_enc[b, gt_col[b]])
        col_emb = torch.stack(col_emb) # [B, dim]
        q_weighted = plain_conditional_weighted_num(self.q_att, q_enc, q_len, col_emb)

        # Same as the above, compute SQL history embedding weighted by column attentions
        hs_weighted = plain_conditional_weighted_num(self.hs_att, hs_enc, hs_len, col_emb)
        # dat_score: (B, 4)
        dat_score = self.dat_out(self.dat_out_q(q_weighted) + int(self.use_hs)* self.dat_out_hs(hs_weighted) + self.dat_out_c(col_emb))

        return dat_score


    def loss(self, score, truth):
        loss = 0
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
