import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, col_tab_name_encode, plain_conditional_weighted_num, SIZE_CHECK
from pytorch_pretrained_bert import BertModel


class OpPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu, use_hs, bert=None):
        super(OpPredictor, self).__init__()
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

        self.q_num_att = nn.Linear(encoded_num, N_h)
        self.hs_num_att = nn.Linear(N_h, N_h)
        self.op_num_out_q = nn.Linear(encoded_num, N_h)
        self.op_num_out_hs = nn.Linear(N_h, N_h)
        self.op_num_out_c = nn.Linear(N_h, N_h)
        self.op_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 2)) #for 1-2 op num, could be changed

        self.q_att = nn.Linear(encoded_num, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.op_out_q = nn.Linear(encoded_num, N_h)
        self.op_out_hs = nn.Linear(N_h, N_h)
        self.op_out_c = nn.Linear(N_h, N_h)
        self.op_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 11)) #for 11 operators

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
        col_emb = torch.stack(col_emb)

        # Predict op number
        q_weighted_num = plain_conditional_weighted_num(self.q_num_att, q_enc, q_len, col_emb)

        # Same as the above, compute SQL history embedding weighted by column attentions
        hs_weighted_num = plain_conditional_weighted_num(self.hs_num_att, hs_enc, hs_len, col_emb)
        # op_num_score: (B, 2)
        op_num_score = self.op_num_out(self.op_num_out_q(q_weighted_num) + int(self.use_hs)* self.op_num_out_hs(hs_weighted_num) + self.op_num_out_c(col_emb))
        SIZE_CHECK(op_num_score, [B, 2])

        # Compute attention values between selected column and question tokens.
        q_weighted = plain_conditional_weighted_num(self.q_att, q_enc, q_len, col_emb)

        # Same as the above, compute SQL history embedding weighted by column attentions
        hs_weighted = plain_conditional_weighted_num(self.hs_att, hs_enc, hs_len, col_emb)

        # Compute prediction scores
        # op_score: (B, 10)
        op_score = self.op_out(self.op_out_q(q_weighted) + int(self.use_hs)* self.op_out_hs(hs_weighted) + self.op_out_c(col_emb))
        SIZE_CHECK(op_score, [B, 11])

        score = (op_num_score, op_score)

        return score

    def loss(self, score, truth):
        loss = 0
        B = len(truth)
        op_num_score, op_score = score
        truth = [t if len(t) <= 2 else t[:2] for t in truth]
        # loss for the op number
        truth_num = [len(t)-1 for t in truth] #num_score 0 maps to 1 in truth
        data = torch.from_numpy(np.array(truth_num))
        if self.gpu:
            data = data.cuda()
        truth_num_var = Variable(data)
        loss += self.CE(op_num_score, truth_num_var)
        # loss for op
        T = len(op_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][truth[b]] = 1
        data = torch.from_numpy(np.array(truth_prob))
        if self.gpu:
            data = data.cuda()
        truth_var = Variable(data)
        #loss += self.mlsml(op_score, truth_var)
        #loss += self.bce_logit(op_score, truth_var)
        pred_prob = self.sigm(op_score)
        bce_loss = -torch.mean( 3*(truth_var * \
                torch.log(pred_prob+1e-10)) + \
                (1-truth_var) * torch.log(1-pred_prob+1e-10) )
        loss += bce_loss

        return loss


    def check_acc(self, score, truth):
        num_err, err, tot_err = 0, 0, 0
        B = len(truth)
        pred = []
        if self.gpu:
            op_num_score, op_score = [x.data.cpu().numpy() for x in score]
        else:
            op_num_score, op_score = [x.data.numpy() for x in score]

        for b in range(B):
            cur_pred = {}
            op_num = np.argmax(op_num_score[b]) + 1 #num_score 0 maps to 1 in truth, must have at least one op
            cur_pred['op_num'] = op_num
            cur_pred['op'] = np.argsort(-op_score[b])[:op_num]
            pred.append(cur_pred)

        for b, (p, t) in enumerate(zip(pred, truth)):
            op_num, op = p['op_num'], p['op']
            flag = True
            if op_num != len(t):
                num_err += 1
                flag = False
            if flag and set(op) != set(t):
                err += 1
                flag = False
            if not flag:
                tot_err += 1

        return np.array((num_err, err, tot_err))
