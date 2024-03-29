import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, col_tab_name_encode, seq_conditional_weighted_num, SIZE_CHECK


class ColPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu, use_hs, bert=None):
        super(ColPredictor, self).__init__()
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
        self.col_num_out_q = nn.Linear(encoded_num, N_h)
        self.col_num_out_hs = nn.Linear(N_h, N_h)
        self.col_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 6)) # num of cols: 1-3

        self.q_att = nn.Linear(encoded_num, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.col_out_q = nn.Linear(encoded_num, N_h)
        self.col_out_c = nn.Linear(N_h, N_h)
        self.col_out_hs = nn.Linear(N_h, N_h)
        self.col_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if gpu:
            self.cuda()

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, col_candidates=None):

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

        # Predict column number: 1-3
        q_weighted_num = seq_conditional_weighted_num(self.q_num_att, q_enc, q_len, col_enc, col_len).sum(1)
        SIZE_CHECK(q_weighted_num, [B, self.N_h])

        # Same as the above, compute SQL history embedding weighted by column attentions
        hs_weighted_num = seq_conditional_weighted_num(self.hs_num_att, hs_enc, hs_len, col_enc, col_len).sum(1)
        SIZE_CHECK(hs_weighted_num, [B, self.N_h])
        # self.col_num_out: (B, 3)
        col_num_score = self.col_num_out(self.col_num_out_q(q_weighted_num) + int(self.use_hs) * self.col_num_out_hs(hs_weighted_num))

        # Predict columns.
        q_weighted = seq_conditional_weighted_num(self.q_att, q_enc, q_len, col_enc)

        # Same as the above, compute SQL history embedding weighted by column attentions
        hs_weighted = seq_conditional_weighted_num(self.hs_att, hs_enc, hs_len, col_enc)
        # Compute prediction scores
        # self.col_out.squeeze(): (B, max_col_len)
        col_score = self.col_out(self.col_out_q(q_weighted) + int(self.use_hs)* self.col_out_hs(hs_weighted) + self.col_out_c(col_enc)).view(B,-1)

        for idx, num in enumerate(col_len):
            if num < max_col_len:
                col_score[idx, num:] = -100
            for col_num in range(num):
                if col_candidates is not None:
                    if col_num not in col_candidates[idx]:
                        col_score[idx, col_num] = -100

        score = (col_num_score, col_score)

        return score

    def loss(self, score, truth):
        #here suppose truth looks like [[[1, 4], 3], [], ...]
        loss = 0
        B = len(truth)
        col_num_score, col_score = score
        #loss for the column number
        truth_num = [len(t) - 1 for t in truth] # double check truth format and for test cases
        data = torch.from_numpy(np.array(truth_num))
        if self.gpu:
            data = data.cuda()
        truth_num_var = Variable(data)

        loss += self.CE(col_num_score, truth_num_var)
        #loss for the key words
        T = len(col_score[0])
        # print("T {}".format(T))
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            gold_l = []
            for t in truth[b]:
                if isinstance(t, list):
                    gold_l.extend(t)
                else:
                    gold_l.append(t)
            truth_prob[b][gold_l] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            data = data.cuda()
        truth_var = Variable(data)
        #loss += self.mlsml(col_score, truth_var)
        #loss += self.bce_logit(col_score, truth_var) # double check no sigmoid
        pred_prob = self.sigm(col_score)
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
            col_num_score, col_score = [x.data.cpu().numpy() for x in score]
        else:
            col_num_score, col_score = [x.data.numpy() for x in score]

        for b in range(B):
            cur_pred = {}
            col_num = np.argmax(col_num_score[b]) + 1 #double check
            cur_pred['col_num'] = col_num
            cur_pred['col'] = np.argsort(-col_score[b])[:col_num]
            pred.append(cur_pred)

        for b, (p, t) in enumerate(zip(pred, truth)):
            col_num, col = p['col_num'], p['col']
            flag = True
            if col_num != len(t): # double check truth format and for test cases
                num_err += 1
                flag = False
            #to eval col predicts, if the gold sql has JOIN and foreign key col, then both fks are acceptable
            fk_list = []
            regular = []
            for l in t:
                if isinstance(l, list):
                    fk_list.append(l)
                else:
                    regular.append(l)

            if flag: #double check
                for c in col:
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
