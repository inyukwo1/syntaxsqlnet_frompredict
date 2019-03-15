import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, col_tab_name_encode, encode_question, SIZE_CHECK, seq_conditional_weighted_num
from pytorch_pretrained_bert import BertModel


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

    def check_eval_acc(self, score, graph, foreign_keys, parent_tables):
        if self.gpu:
            score = F.tanh(score).data.cpu().numpy()
        else:
            score = F.tanh(score).data.numpy()
        correct = True
        graph_correct = True
        tabs = []

        for b in range(len(score)):
            if score[b] > 0.:
                tabs.append(b)
            if score[b] > 0. and str(b) not in graph:
                correct = False
            elif score[b] < 0. and str(b) in graph:
                correct = False

        if not tabs:
            tabs.append(np.argmax(score))
        predict_graph = graph_maker(tabs, foreign_keys, parent_tables)
        if not graph_checker(predict_graph, graph):
            graph_correct = False
        print(score)
        print("=======")
        print(predict_graph)
        print("========")
        print(graph)
        print("@@@@@@@@@@@@@@{}, {}".format(correct, graph_correct))
        return correct, graph_correct

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

