import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, SIZE_CHECK, seq_conditional_weighted_num, col_tab_name_encode
from graph_utils import *


def sql_graph_maker(tab_list, foreign_keys, parent_tables):
    graph = {}
    for tab in tab_list:
        graph[tab] = []
    for tab in tab_list:
        if graph[tab]:
            continue
        for f, p in foreign_keys:
            if parent_tables[f] == tab and parent_tables[p] in tab_list:
                graph[tab].append((f, p, parent_tables[p]))
                graph[parent_tables[p]].append((p, f, tab))
            elif parent_tables[p] == tab and parent_tables[f] in tab_list:
                graph[tab].append((p, f, parent_tables[f]))
                graph[parent_tables[f]].append((f, p, tab))

    def unreachable(graph, t):
        reached = set()
        reached.add(t)
        added = True
        while added:
            added = False
            for tab in graph:
                for edge in graph[tab]:
                    if edge[2] in reached and tab not in reached:
                        reached.add(tab)
                        added = True
                    if tab in reached and edge[2] not in reached:
                        reached.add(edge[2])
                        added = True
        for tab in graph:
            if tab not in reached:
                return True
        return False

    while len(graph) > 1: # TODO pop lower score first
        delete = False
        for t in graph:
            if unreachable(graph, t):
                delete = True
                break
        if not delete:
            break
        graph.pop(t)
        for another_t in graph:
            for edge in graph[another_t]:
                if edge[2] == t:
                    graph[another_t].remove(edge)
                    break
    return graph


def graph_maker(tab_list, foreign_keys, parent_tables):
    # graph = {}
    # for tab in tab_list:
    #     graph[tab] = []
    # for tab in tab_list:
    #     if graph[tab]:
    #         continue
    #     for f, p in foreign_keys:
    #         if parent_tables[f] == tab and parent_tables[p] in tab_list:
    #             graph[tab] = [f]
    #             graph[parent_tables[p]].append(p)
    #         elif parent_tables[p] == tab and parent_tables[f] in tab_list:
    #             graph[tab] = [p]
    #             graph[parent_tables[f]].append(f)
    # while len(graph) > 1:
    #     delete = False
    #     for t in graph:
    #         if not graph[t]:
    #             delete = True
    #             break
    #     if not delete:
    #         break
    #     graph.pop(t)
    sql_graph = sql_graph_maker(tab_list, foreign_keys, parent_tables)
    for t in sql_graph:
        for idx, _ in enumerate(sql_graph[t]):
            sql_graph[t][idx] = sql_graph[t][idx][0]
        sql_graph[t] = list(set(sql_graph[t]))
    graph = sql_graph
    return graph


class FromPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu, use_hs, bert, onefrom, use_lstm=False):
        super(FromPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu
        self.use_hs = use_hs
        self.threshold = 0.5

        self.q_bert = bert
        self.onefrom = onefrom
        self.use_lstm = use_lstm
        self.encoded_num = 1024

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        if self.onefrom:
            self.onefrom_vec = nn.Parameter(torch.zeros(N_h))
        if self.use_lstm:
            self.main_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                                     num_layers=N_depth, batch_first=True,
                                     dropout=0.3, bidirectional=True)
            self.q_att = nn.Linear(N_h, N_h)
            self.hs_att= nn.Linear(N_h, N_h)
            self.q_out = nn.Linear(N_h, N_h)
            self.hs_out = nn.Linear(N_h, N_h)
            self.tab_out = nn.Linear(N_h, N_h)
            self.table_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                                     num_layers=N_depth, batch_first=True,
                                     dropout=0.3, bidirectional=True)

            self.table_onemore_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                                     num_layers=N_depth, batch_first=True,
                                     dropout=0.3, bidirectional=True)
            self.encoded_num = N_h
            self.q_score_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

            self.q_num_att = nn.Linear(N_h, N_h)
            self.q_num_out = nn.Linear(N_h, N_h)
            self.tab_num_out = nn.Linear(N_h, N_h)
            self.q_num_score_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 6))

        self.outer1 = nn.Sequential(nn.Linear(N_h + self.encoded_num, N_h), nn.ReLU())
        self.outer2 = nn.Sequential(nn.Linear(N_h, 1))
        if gpu:
            self.cuda()

    def forward(self, q_emb, q_len, q_q_len, hs_emb_var, hs_len, sep_embeddings, table_emb=None, table_len=None, table_name_len=None):
        B = len(q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        if self.use_lstm:
            table_len = np.array(table_len)
            q_enc, _ = run_lstm(self.main_lstm, q_emb, q_len)
            t_enc, _ = col_tab_name_encode(table_emb, table_name_len, table_len, self.table_lstm)
            q_weighted_ox_num = seq_conditional_weighted_num(self.q_num_att, q_enc, q_len, t_enc, table_len).sum(1)
            tab_num_score = self.q_num_score_out(self.q_num_out(q_weighted_ox_num))

            q_weighted_ox = seq_conditional_weighted_num(self.q_att, q_enc, q_len, t_enc, table_len)
            tab_score = self.q_score_out(self.q_out(q_weighted_ox) + self.tab_out(t_enc)).view(B, -1)
            for idx, num in enumerate(table_len):
                if num < max(table_len):
                    tab_score[idx, num:] = -100
            return torch.sigmoid(tab_num_score), tab_score
        else:
            q_enc = self.q_bert(q_emb, q_len, q_q_len, sep_embeddings)
        if self.onefrom:
            hs_enc = self.onefrom_vec.view(1, -1).expand(B,  -1)
            hs_enc = hs_enc[:, 0, :]

        q_enc = q_enc[:, 0, :]
        q_enc = torch.cat((q_enc, hs_enc), dim=1)
        x = self.outer1(q_enc)
        x = self.outer2(x).squeeze(1)
        return x

    def loss(self, score, labels):
        num_score, score = score
        num_label = [len(t) - 1 for t in labels]
        num_label = torch.from_numpy(np.array(num_label))
        if self.gpu:
            num_label = num_label.cuda()
        num_label = Variable(num_label)
        torch_label = torch.zeros_like(score)
        B, T = list(score.size())
        for b in range(B):
            for t in labels:
                torch_label[b, t] = 1.
        loss = F.cross_entropy(num_score, num_label) + F.binary_cross_entropy_with_logits(score, torch_label)
        return loss

    def check_acc(self, score, truth):
        num_err = 0
        exact_err = 0
        fifth_err = 0
        seventh_err = 0
        B = len(truth)
        pred = []
        num_score, score = score
        if self.gpu:
            num_score = num_score.data.cpu().numpy()
            score = score.data.cpu().numpy()
        else:
            num_score = num_score.data.numpy()
            score = score.data.numpy()
        for b in range(B):
            tab_num = np.argmax(num_score[b]) + 1
            if tab_num != len(truth[b]):
                num_err += 1
            exact_wrong = False
            fifth_wrong = False
            seventh_wrong = False
            top_tables = np.argsort(-score[b])
            for t in truth[b]:
                if t not in top_tables[:tab_num]:
                    exact_wrong = True
                if t not in top_tables[:5]:
                    fifth_wrong = True
                if t not in top_tables[:7]:
                    seventh_wrong = True
            if exact_wrong:
                exact_err += 1
            if fifth_wrong:
                fifth_err += 1
            if seventh_wrong:
                seventh_err += 1
        return num_err, exact_err, fifth_err, seventh_err

    def check_acc_eval(self, score, truth, queries, tables):
        num_err = 0
        exact_err = 0
        fifth_err = 0
        seventh_err = 0
        B = len(truth)
        pred = []
        num_score, score = score
        if self.gpu:
            num_score = num_score.data.cpu().numpy()
            score = score.data.cpu().numpy()
        else:
            num_score = num_score.data.numpy()
            score = score.data.numpy()
        for b in range(B):
            print("==============")
            print(score[b])
            print("truth:")
            print(truth[b])
            print("queries")
            print(queries[b])
            print("tabbles")
            print(tables[b])
            tab_num = np.argmax(num_score[b]) + 1
            if tab_num != len(truth[b]):
                num_err += 1
            exact_wrong = False
            fifth_wrong = False
            seventh_wrong = False
            top_tables = np.argsort(-score[b])
            for t in truth[b]:
                if t not in top_tables[:tab_num]:
                    exact_wrong = True
                if t not in top_tables[:5]:
                    fifth_wrong = True
                if t not in top_tables[:7]:
                    seventh_wrong = True
            if exact_wrong:
                exact_err += 1
            if fifth_wrong:
                fifth_err += 1
            if seventh_wrong:
                seventh_err += 1
        return num_err, exact_err, fifth_err, seventh_err

    def score_to_tables(self, score, foreign_keys, parent_tables):
        if self.gpu:
            score = torch.tanh(score).data.cpu().numpy()
        else:
            score = torch.tanh(score).data.numpy()
        tabs = []

        for b in range(len(score)):
            if score[b] > 0.:
                tabs.append(b)
        if not tabs:
            tabs.append(np.argmax(score))
        sorted_score_arg = np.argsort(score)
        new_tabs = []
        for tab in sorted_score_arg:
            if tab in tabs:
                new_tabs.append(tab)
        tabs = new_tabs
        predict_graph = sql_graph_maker(tabs, foreign_keys, parent_tables)
        return list(predict_graph.keys()), predict_graph


