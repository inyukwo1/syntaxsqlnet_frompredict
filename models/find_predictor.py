import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, col_tab_name_encode, encode_question, SIZE_CHECK, seq_conditional_weighted_num
from pytorch_pretrained_bert import BertModel
from models.schema_bert import SchemaBert


def graph_maker(tab_list, foreign_keys, parent_tables):
    graph = {}
    for tab in tab_list:
        graph[tab] = []
    for tab in tab_list:
        if graph[tab]:
            continue
        for f, p in foreign_keys:
            if parent_tables[f] == tab and parent_tables[p] in tab_list:
                graph[tab] = [f]
                graph[parent_tables[p]].append(p)
            elif parent_tables[p] == tab and parent_tables[f] in tab_list:
                graph[tab] = [p]
                graph[parent_tables[f]].append(f)
    while len(graph) > 1:
        delete = False
        for t in graph:
            if not graph[t]:
                delete = True
                break
        if not delete:
            break
        graph.pop(t)
    return graph


def graph_checker(graph1, graph2):
    if len(graph1) != len(graph2):
        return False
    for t in graph1:
        if str(t) not in graph2:
            return False
        t_list = graph1[t]
        t_list.sort()
        graph2_t_list = graph2[str(t)]
        graph2_t_list.sort()
        if t_list != graph2_t_list:
            return False
    return True


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

        self.outer = nn.Linear(N_h + self.encoded_num, 1)
        if gpu:
            self.cuda()

    def forward(self, q_emb, q_len, hs_emb_var, hs_len,  table_cols, table_col_num_lens, table_col_name_lens, table_col_type_ids, special_tok_id, table_locs, parent_tabs, foreign_keys):

        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        B = len(q_len)
        max_table_len = 0
        for loc in table_locs:
            if max_table_len < len(loc):
                max_table_len = len(loc)

        if self.use_bert:
            q_enc = self.q_bert(q_emb, q_len,  table_cols, table_col_num_lens, table_col_name_lens, table_col_type_ids, special_tok_id, parent_tabs, foreign_keys)
        else:
            q_enc, _ = run_lstm(self.q_lstm, q_emb, q_len)
        _, max_q_len, _ = list(q_enc.size())
        assert list(q_enc.size()) == [B, max_q_len, self.encoded_num]
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        hs_enc = hs_enc[:,0,:].unsqueeze(1).expand(B, max_q_len, self.N_h)
        q_enc = torch.cat((q_enc, hs_enc), dim=2)
        x = self.outer(q_enc).squeeze(2)
        SIZE_CHECK(x, [B, max_q_len])
        if self.gpu:
            x = x.cpu()
        newx = []
        for b, table_loc in enumerate(table_locs):
            slice = []
            for i in range(max_q_len):
                if i in table_loc:
                    slice.append(x[b, i] + x[b, i+1] + x[b, i+2] + x[b, i+3] + x[b, i+4] + x[b, i+5] + x[b, i+6])
            slice = torch.stack(slice)
            if len(slice) < max_table_len:
                padding = torch.from_numpy(np.full((max_table_len - len(slice), ), -100., dtype=np.float32))
                slice = torch.cat((slice, padding))
            newx.append(slice)
        newx = torch.stack(newx)

        if self.gpu:
            newx = newx.cuda()
        return newx

    def loss(self, score, truth):
        num_truth = [len(one_truth) - 1 for one_truth in truth]
        data = torch.from_numpy(np.array(num_truth))
        if self.gpu:
            data = data.cuda()

        loss = 0

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
        loss += F.binary_cross_entropy_with_logits(score, label)
        return loss

    def check_acc(self, score, truth, foreign_keys, parent_tables, log=False):
        err = 0
        graph_err = 0
        hueristic_err = 0
        B = len(truth)
        pred = []
        if self.gpu:
            score = F.tanh(score).data.cpu().numpy()
        else:
            score = F.tanh(score).data.numpy()

        for b in range(B):
            cur_pred = {}
            cur_pred["table"] = []
            cur_pred["itable"] = []
            for i, val in enumerate(score[b]):
                if val > 0.:
                    cur_pred["table"].append(i)
                    cur_pred["itable"].append(i)
            if not cur_pred["itable"]:
                cur_pred["itable"].append(np.argmax(score[b]))
            pred.append(cur_pred)
        for b, (p, t) in enumerate(zip(pred, truth)):
            tab = p['table']
            itab = p["itable"]
            regular = []
            for l in t:
                l = int(l)
                regular.append(l)
            if set(tab) != set(regular):
                err += 1
            igraph = graph_maker(itab, foreign_keys[b], parent_tables[b])
            graph = graph_maker(tab, foreign_keys[b], parent_tables[b])
            if log:
                print(tab)
                print("============")
                print(graph)
                print("=======")
                print(igraph)
                print("========")
                print(t)
                print("@@@@@@@@@@{}".format(graph_checker(igraph, t)), flush=True)
            if not graph_checker(graph, t):
                graph_err += 1
            if not graph_checker(igraph, t):
                hueristic_err += 1

        return err, graph_err, hueristic_err

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

