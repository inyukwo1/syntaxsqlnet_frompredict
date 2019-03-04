import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules import BatchNorm1d, Dropout
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
            self.q_encode = nn.Sequential(nn.Linear(self.encoded_num, N_h), nn.ReLU())
        else:
            self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
            self.encoded_num = N_h
            self.q_encode = nn.Sequential(nn.Linear(self.encoded_num * 2, N_h), nn.ReLU())

        self.c_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.t_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.hs_encode = nn.Sequential(nn.Linear(N_h * 2, N_h), nn.ReLU())

        self.q_encode_dropout = Dropout()
        self.q_hs_dropout = Dropout()

        self.t_self_layer1 = nn.Sequential(nn.Linear(N_h * 3, N_h), nn.ReLU(), Dropout())
        self.t_self_layer2 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.t_self_layer3 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.t_self_layer4 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.t_self_layer5 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())

        self.tc_layer1 = nn.Sequential(nn.Linear(N_h * 4, N_h), nn.ReLU(), Dropout())
        self.tc_layer2 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.tc_layer3 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.tc_layer4 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.tc_layer5 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())

        self.tsc_layer1 = nn.Sequential(nn.Linear(N_h * 4, N_h), nn.ReLU(), Dropout())
        self.tsc_layer2 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.tsc_layer3 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.tsc_layer4 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.tsc_layer5 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())

        self.not_t_self_layer1 = nn.Sequential(nn.Linear(N_h * 3, N_h), nn.ReLU(), Dropout())
        self.not_t_self_layer2 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.not_t_self_layer3 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.not_t_self_layer4 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.not_t_self_layer5 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())

        self.not_tsc_layer1 = nn.Sequential(nn.Linear(N_h * 4, N_h), nn.ReLU(), Dropout())
        self.not_tsc_layer2 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.not_tsc_layer3 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.not_tsc_layer4 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.not_tsc_layer5 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())

        self.added_layer1 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.batchnorm1 = BatchNorm1d(11)
        self.added_layer2 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.batchnorm2 = BatchNorm1d(11)
        self.added_layer3 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.batchnorm3 = BatchNorm1d(11)
        self.added_layer4 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU(), Dropout())
        self.added_layer5 = nn.Linear(N_h, 1)
        if gpu:
            self.cuda()

    def forward(self, par_tab_nums, candidate_schemas, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len,
                table_emb_var, table_len, table_name_len):

        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        max_col_len = max(col_len)
        max_table_len = max(table_len)
        B = len(q_len)
        if self.use_bert:
            q_enc = self.q_bert(q_emb_var, q_len)
            new_q_enc = q_enc[:, 0 ,:].view(B, 768)
        else:
            q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
            new_q_enc = []
            for b in range(len(q_enc)):
                new_q_enc.append(torch.cat((q_enc[b, 0], q_enc[b, q_len[b] - 1])))
            new_q_enc = torch.stack(new_q_enc)
        c_enc, _ = col_tab_name_encode(col_emb_var, col_name_len, col_len, self.c_lstm)
        t_enc, _ = col_tab_name_encode(table_emb_var, table_name_len, table_len, self.t_lstm)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)

        new_h_enc = []
        for b in range(len(q_enc)):
            new_h_enc.append(torch.cat((hs_enc[b, 0], hs_enc[b, hs_len[b] - 1])))
        new_h_enc = torch.stack(new_h_enc)
        new_h_enc = self.q_hs_dropout(new_h_enc)
        h_enc = self.hs_encode(new_h_enc)
        new_q_enc = self.q_encode_dropout(new_q_enc)
        q_enc = self.q_encode(new_q_enc)

        if self.gpu:
            q_enc = q_enc.cpu()
            h_enc = h_enc.cpu()
            c_enc = c_enc.cpu()
            t_enc = t_enc.cpu()
        table_tensors_batch = []
        table_col_tensors_batch = []
        table_subcol_tensors_batch = []
        not_table_tensors_batch = []
        not_table_subcol_tensors_batch = []

        max_tab_seq_len = 0
        max_col_tab_seq_len = 0
        max_subcol_tab_seq_len = 0
        max_not_tab_seq_len = 0
        max_not_subcol_tab_seq_len = 0
        for b, candidate_schema in enumerate(candidate_schemas):
            table_tensors_candidates = []
            col_tensors_candidates = []
            table_subcol_tensors_candidates = []
            not_table_tensors_candidates = []
            not_table_subcol_tensors_candidates = []
            for schema in candidate_schema:
                table_tensors = []
                table_col_tensors = []
                table_subcol_tensors = []
                not_table_tensors = []
                not_table_subcol_tensors = []
                max_table_num = par_tab_nums[b][-1]
                for tab in range(max_table_num + 1):
                    if tab in schema:
                        table_tensors.append(torch.cat((t_enc[b, tab], q_enc[b], h_enc[b])))
                        for col in schema[tab]:
                            table_col_tensors.append(torch.cat((t_enc[b, tab], c_enc[b, col], q_enc[b], h_enc[b])))
                        for colnum, partab in enumerate(par_tab_nums[b]):
                            if partab == tab:
                                table_subcol_tensors.append(
                                    torch.cat((t_enc[b, tab], c_enc[b, colnum], q_enc[b], h_enc[b])))
                    else:
                        not_table_tensors.append(torch.cat((t_enc[b, tab], q_enc[b], h_enc[b])))
                        for colnum, partab in enumerate(par_tab_nums[b]):
                            if partab == tab:
                                not_table_subcol_tensors.append(
                                    torch.cat((t_enc[b, tab], c_enc[b, colnum], q_enc[b], h_enc[b])))
                table_tensors = torch.stack(table_tensors)
                table_subcol_tensors = torch.stack(table_subcol_tensors)
                if table_col_tensors:
                    table_col_tensors = torch.stack(table_col_tensors)
                else:
                    table_col_tensors = torch.zeros((1, self.N_h * 4))
                if not_table_tensors:
                    not_table_tensors = torch.stack(not_table_tensors)
                else:
                    not_table_tensors = torch.zeros((1, self.N_h * 3))
                if not_table_subcol_tensors:
                    not_table_subcol_tensors = torch.stack(not_table_subcol_tensors)
                else:
                    not_table_subcol_tensors = torch.zeros((1, self.N_h * 4))
                table_tensors_candidates.append(table_tensors)
                col_tensors_candidates.append(table_col_tensors)
                table_subcol_tensors_candidates.append(table_subcol_tensors)
                not_table_tensors_candidates.append(not_table_tensors)
                not_table_subcol_tensors_candidates.append(not_table_subcol_tensors)
                if len(table_tensors) > max_tab_seq_len:
                    max_tab_seq_len = len(table_tensors)
                if len(table_col_tensors) > max_col_tab_seq_len:
                    max_col_tab_seq_len = len(table_col_tensors)
                if len(table_subcol_tensors) > max_subcol_tab_seq_len:
                    max_subcol_tab_seq_len = len(table_subcol_tensors)
                if len(not_table_tensors) > max_not_tab_seq_len:
                    max_not_tab_seq_len = len(not_table_tensors)
                if len(not_table_subcol_tensors) > max_not_subcol_tab_seq_len:
                    max_not_subcol_tab_seq_len = len(not_table_subcol_tensors)
            table_tensors_batch.append(table_tensors_candidates)
            table_col_tensors_batch.append(col_tensors_candidates)
            table_subcol_tensors_batch.append(table_subcol_tensors_candidates)
            not_table_tensors_batch.append(not_table_tensors_candidates)
            not_table_subcol_tensors_batch.append(not_table_subcol_tensors_candidates)
        for b in range(B):
            for cand_idx in range(11):
                cand_tensor = table_tensors_batch[b][cand_idx]
                if len(cand_tensor) < max_tab_seq_len:
                    table_tensors_batch[b][cand_idx] = torch.cat((cand_tensor, torch.zeros((max_tab_seq_len - len(cand_tensor), self.N_h * 3))))
                col_cand_tensor = table_col_tensors_batch[b][cand_idx]
                if len(col_cand_tensor) < max_col_tab_seq_len:
                    table_col_tensors_batch[b][cand_idx] = torch.cat((col_cand_tensor, torch.zeros((max_col_tab_seq_len - len(col_cand_tensor), self.N_h * 4))))
                subcol_cand_tensor = table_subcol_tensors_batch[b][cand_idx]
                if len(subcol_cand_tensor) < max_subcol_tab_seq_len:
                    table_subcol_tensors_batch[b][cand_idx] = torch.cat((subcol_cand_tensor, torch.zeros((max_subcol_tab_seq_len - len(subcol_cand_tensor), self.N_h * 4))))
                not_cand_tensor = not_table_tensors_batch[b][cand_idx]
                if len(not_cand_tensor) < max_not_tab_seq_len:
                    not_table_tensors_batch[b][cand_idx] = torch.cat((not_cand_tensor, torch.zeros((max_not_tab_seq_len - len(not_cand_tensor), self.N_h * 3))))
                not_subcol_cand_tensor = not_table_subcol_tensors_batch[b][cand_idx]
                if len(not_subcol_cand_tensor) < max_not_subcol_tab_seq_len:
                    not_table_subcol_tensors_batch[b][cand_idx] = torch.cat((not_subcol_cand_tensor, torch.zeros((max_not_subcol_tab_seq_len - len(not_subcol_cand_tensor), self.N_h * 4))))
            table_tensors_batch[b] = torch.stack(table_tensors_batch[b])
            table_col_tensors_batch[b] = torch.stack(table_col_tensors_batch[b])
            table_subcol_tensors_batch[b] = torch.stack(table_subcol_tensors_batch[b])
            not_table_tensors_batch[b] = torch.stack(not_table_tensors_batch[b])
            not_table_subcol_tensors_batch[b] = torch.stack(not_table_subcol_tensors_batch[b])
        table_tensors_batch = torch.stack(table_tensors_batch)
        table_col_tensors_batch = torch.stack(table_col_tensors_batch)
        table_subcol_tensors_batch = torch.stack(table_subcol_tensors_batch)
        not_table_tensors_batch = torch.stack(not_table_tensors_batch)
        not_table_subcol_tensors_batch = torch.stack(not_table_subcol_tensors_batch)
        if self.gpu:
            table_tensors_batch = table_tensors_batch.cuda()
            table_col_tensors_batch = table_col_tensors_batch.cuda()
            table_subcol_tensors_batch = table_subcol_tensors_batch.cuda()
            not_table_tensors_batch = not_table_tensors_batch.cuda()
            not_table_subcol_tensors_batch = not_table_subcol_tensors_batch.cuda()

        SIZE_CHECK(table_tensors_batch, [B, 11, None, self.N_h * 3])
        SIZE_CHECK(table_col_tensors_batch, [B, 11, None, self.N_h * 4])
        SIZE_CHECK(table_subcol_tensors_batch, [B, 11, None, self.N_h * 4])
        x1 = self.t_self_layer1(table_tensors_batch)
        x2 = self.t_self_layer2(x1)
        x3 = self.t_self_layer3(x2) + x2
        x4 = self.t_self_layer4(x3) + x1
        x5 = self.t_self_layer5(x4)
        table_tensors_batch = torch.sum(x5, dim=2)

        x1 = self.tc_layer1(table_col_tensors_batch)
        x2 = self.tc_layer2(x1)
        x3 = self.tc_layer3(x2) + x2
        x4 = self.tc_layer4(x3) + x1
        x5 = self.tc_layer5(x4) + x3
        table_col_tensors_batch = torch.sum(x5, dim=2)

        x1 = self.tsc_layer1(table_subcol_tensors_batch)
        x2 = self.tsc_layer2(x1)
        x3 = self.tsc_layer3(x2) + x2
        x4 = self.tsc_layer4(x3) + x1
        x5 = self.tsc_layer5(x4) + x3
        table_subcol_tensors_batch = torch.sum(x5, dim=2)

        x1 = self.not_t_self_layer1(not_table_tensors_batch)
        x2 = self.not_t_self_layer2(x1)
        x3 = self.not_t_self_layer3(x2) + x2
        x4 = self.not_t_self_layer4(x3) + x1
        x5 = self.not_t_self_layer5(x4)
        not_table_tensors_batch = torch.sum(x5, dim=2)
        x1 = self.not_tsc_layer1(not_table_subcol_tensors_batch)
        x2 = self.not_tsc_layer2(x1)
        x3 = self.not_tsc_layer3(x2) + x2
        x4 = self.not_tsc_layer4(x3) + x1
        x5 = self.not_tsc_layer5(x4) + x3
        not_table_subcol_tensors_batch = torch.sum(x5, dim=2)

        table_tensors_batch = torch.add(table_tensors_batch, table_col_tensors_batch)
        table_tensors_batch = torch.add(table_tensors_batch, table_subcol_tensors_batch)
        table_tensors_batch = torch.add(table_tensors_batch, not_table_tensors_batch)
        table_tensors_batch = torch.add(table_tensors_batch, not_table_subcol_tensors_batch)

        SIZE_CHECK(table_tensors_batch, [B, 11, self.N_h])
        x1 = self.added_layer1(table_tensors_batch)
        x1 = self.batchnorm1(x1)
        x2 = self.added_layer2(x1)
        x2 = self.batchnorm2(x2) + x1
        x3 = self.added_layer3(x2)
        x3 = self.batchnorm3(x3) + x2
        table_tensors_batch = self.added_layer4(x3)
        table_tensors_batch = self.added_layer5(table_tensors_batch).squeeze()

        return table_tensors_batch

    def loss(self, score, truth):
        return F.binary_cross_entropy_with_logits(score, truth)

    def check_acc(self, score, truth):
        err = 0
        if self.gpu:
            score = score.data.cpu().numpy()
            truth = truth.cpu().numpy()
        else:
            score = score.data.numpy()
            truth = truth.numpy()
        score = np.argmax(score, axis=1)
        truth = np.argmax(truth, axis=1)
        for b in range(len(score)):
            if score[b] != truth[b]:
                err+=1
        return err

    def score_to_tables(self, score):
        #TODO
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

