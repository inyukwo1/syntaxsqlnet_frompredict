import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, SIZE_CHECK
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
    def __init__(self, N_word, N_h, N_depth, gpu, use_hs, bert, onefrom):
        super(FromPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu
        self.use_hs = use_hs
        self.threshold = 0.5

        self.q_bert = bert
        self.onefrom = onefrom
        self.encoded_num = 1024

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.outer1 = nn.Sequential(nn.Linear(N_h + self.encoded_num, N_h), nn.ReLU())
        self.outer2 = nn.Sequential(nn.Linear(N_h, 1))
        if self.onefrom:
            self.onefrom_vec = nn.Parameter(torch.zeros(N_h))
        if gpu:
            self.cuda()

    def forward(self, q_emb, q_len, q_q_len, hs_emb_var, hs_len, sep_embeddings):
        B = len(q_len)

        q_enc = self.q_bert(q_emb, q_len, q_q_len, sep_embeddings)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        hs_enc = hs_enc[:, 0, :]
        if self.onefrom:
            hs_enc = self.onefrom_vec.view(1, -1).expand(B,  -1)

        q_enc = q_enc[:, 0, :]
        q_enc = torch.cat((q_enc, hs_enc), dim=1)
        x = self.outer1(q_enc)
        x = self.outer2(x).squeeze(1)
        return x

    def loss(self, score, labels):
        loss = F.binary_cross_entropy_with_logits(score, labels)
        return loss

    def check_acc(self, score, truth):
        err = 0
        B = len(truth)
        pred = []
        if self.gpu:
            score = torch.tanh(score).data.cpu().numpy()
            truth = truth.cpu().numpy()
        else:
            score = F.tanh(score).data.numpy()
            truth = truth.numpy()
        for b in range(B):
            if score[b] > 0. and truth[b] < 0.5:
                err += 1
            elif score[b]<= 0. and truth[b] > 0.5:
                err += 1
        return np.array(err)

    def check_eval_acc(self, score, table_graph_list, graph, foreign_keys, primary_keys, parent_tables, table_names, column_names, question):
        table_num_ed = len(table_names)
        for predicted_graph, sc in zip(table_graph_list, score):
            print("$$$$$$$$$$$$$$$$$")
            print(predicted_graph)
            print("~~~~~")
            print(sc)
        correct = False

        selected_graph = table_graph_list[np.argmax(score)]
        graph_correct = graph_checker(selected_graph, graph, foreign_keys, primary_keys)
        print("#### " + " ".join(question))
        for idx, table_name in enumerate(table_names):
            print("Table {}: {}".format(idx, table_name))
            for col_idx, [par_tab, col_name] in enumerate(column_names):
                if par_tab == idx:
                    print("   {}: {}".format(col_idx, col_name))

        print("=======")
        print(selected_graph)
        print("========")
        print(graph)
        print("@@@@@@@@@@@@@@{}, {}".format(correct, graph_correct))
        return graph_correct

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
