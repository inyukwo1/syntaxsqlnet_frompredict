import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, SIZE_CHECK
from graph_utils import *


def graph_maker_joincondition(tab_list, foreign_keys, parent_tables, label):
    graph = {}
    for tab in tab_list:
        graph[tab] = []
    for tab in tab_list:
        if graph[tab]:
            continue
        for f, p in foreign_keys:
            if parent_tables[f] == tab and parent_tables[p] in tab_list:
                label_tab = str(tab)
                label_tab1 = str(parent_tables[p])
                if label_tab in label and f in label[label_tab] and label_tab1 in label and p in label[label_tab1]:
                    graph[tab].append(f)
                    graph[parent_tables[p]].append(p)
                    break
            elif parent_tables[p] == tab and parent_tables[f] in tab_list:
                label_tab = str(tab)
                label_tab1 = str(parent_tables[f])
                if label_tab in label and p in label[label_tab] and label_tab1 in label and f in label[label_tab1]:
                    graph[tab].append(p)
                    graph[parent_tables[f]].append(f)
                    break
    return graph


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
                break
            elif parent_tables[p] == tab and parent_tables[f] in tab_list:
                graph[tab].append((p, f, parent_tables[f]))
                graph[parent_tables[f]].append((f, p, tab))
                break

    # def unreachable(graph, t):
    #     reached = set()
    #     reached.add(t)
    #     added = True
    #     while added:
    #         added = False
    #         for tab in graph:
    #             for edge in graph[tab]:
    #                 if edge[2] in reached and tab not in reached:
    #                     reached.add(tab)
    #                     added = True
    #                 if tab in reached and edge[2] not in reached:
    #                     reached.add(edge[2])
    #                     added = True
    #     for tab in graph:
    #         if tab not in reached:
    #             return True
    #     return False
    #
    # while len(graph) > 1: # TODO pop lower score first
    #     delete = False
    #     for t in graph:
    #         if unreachable(graph, t):
    #             delete = True
    #             break
    #     if not delete:
    #         break
    #     graph.pop(t)
    #     for another_t in graph:
    #         for edge in graph[another_t]:
    #             if edge[2] == t:
    #                 graph[another_t].remove(edge)
    #                 break
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


class WikiSQLNum(nn.Module):
    def __init__(self):
        super(WikiSQLNum, self).__init__()
        self.tab_lstm = nn.LSTM(input_size=100, hidden_size=100//2,
                num_layers=2, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.tab_self_attention = nn.Linear(100, 1)
        self.tab_hidden = nn.Linear(100, 2 * 100)
        self.tab_cell = nn.Linear(100, 2 * 100)
        self.q_lstm = nn.LSTM(input_size=100, hidden_size=100//2,
                num_layers=2, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.q_self_attention = nn.Linear(100, 1)
        self.num_out = nn.Sequential(nn.Linear(100, 100),
                                     nn.Tanh(),
                                     nn.Linear(100, 4))  # max number (4 + 1)

    def forward(self, q_embs, q_len, tab_embs, tab_len):
        B = len(q_len)
        tab_enc, _ = run_lstm(self.tab_lstm, tab_embs, tab_len)

        tab_att = self.tab_self_attention(tab_enc).squeeze(2)
        for b, one_tab_len in enumerate(tab_len):
            if one_tab_len < max(tab_len):
                tab_att[b, one_tab_len:] = -1000
        tab_att = F.softmax(tab_att, dim=1)
        attentioned_tab_enc = torch.mul(tab_enc, tab_att.unsqueeze(2)).sum(1)
        SIZE_CHECK(attentioned_tab_enc, [B, 100])
        hidden0 = self.tab_hidden(attentioned_tab_enc)
        hidden0 = hidden0.view(B, 2 * 2, 100 // 2)
        hidden0 = hidden0.transpose(0, 1).contiguous()
        cell0 = self.tab_cell(attentioned_tab_enc)
        cell0 = cell0.view(B, 2 * 2, 100 // 2)
        cell0 = cell0.transpose(0, 1).contiguous()

        q_enc, _ = run_lstm(self.q_lstm, q_embs, q_len, (hidden0, cell0))
        q_att = self.q_self_attention(q_enc).squeeze(2)
        for b, one_q_len in enumerate(q_len):
            if one_q_len < max(q_len):
                q_att[b, one_q_len:] = -1000
        q_att = F.softmax(q_att, dim=1)
        attentioned_q_enc = torch.mul(q_enc, q_att.unsqueeze(2)).sum(1)
        num = self.num_out(attentioned_q_enc)
        return num


class WikiSQLCol(nn.Module):
    def __init__(self):
        super(WikiSQLCol, self).__init__()
        self.q_lstm = nn.LSTM(input_size=100, hidden_size=100//2,
                num_layers=2, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.tab_lstm = nn.LSTM(input_size=100, hidden_size=100//2,
                num_layers=2, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.coatt = nn.Linear(100, 100)
        self.W_c = nn.Linear(100, 100)
        self.W_hs = nn.Linear(100, 100)
        self.W_out = nn.Sequential(
            nn.Tanh(), nn.Linear(2 * 100, 1)
        )

    def forward(self, q_embs, q_len, tab_embs, tab_len):
        B = len(q_len)
        q_enc, _ = run_lstm(self.q_lstm, q_embs, q_len)
        tab_enc, _ = run_lstm(self.tab_lstm, tab_embs, tab_len)
        att = torch.bmm(tab_enc, self.coatt(q_enc).transpose(1, 2))
        for b, one_q_len in enumerate(q_len):
            if one_q_len < max(q_len):
                att[b, :, one_q_len:] = -1000
        att = F.softmax(att, dim=2).unsqueeze(3)
        q_enc = q_enc.unsqueeze(1)
        c_n = torch.mul(q_enc, att).sum(2)

        y = torch.cat([self.W_c(c_n), self.W_hs(tab_enc)], dim=2)
        score = self.W_out(y).squeeze(2)
        for b, one_tab_len in enumerate(tab_len):
            if one_tab_len < max(tab_len):
                score[b, one_tab_len:] = -1000

        return score


class FromPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu, use_hs, bert, onefrom, wikisql_style=False):
        super(FromPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu
        self.use_hs = use_hs
        self.threshold = 0.5

        self.q_bert = bert
        self.onefrom = onefrom
        self.wikisql_style = wikisql_style
        self.encoded_num = 1024

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.outer1 = nn.Sequential(nn.Linear(N_h + self.encoded_num, N_h), nn.ReLU())
        self.outer2 = nn.Sequential(nn.Linear(N_h, 1))
        if self.wikisql_style:
            self.from_num = WikiSQLNum()
            self.from_cols = WikiSQLCol()
            self.w_num = nn.Linear(self.encoded_num, 4)
        if self.onefrom:
            self.onefrom_vec = nn.Parameter(torch.zeros(N_h))
        if gpu:
            self.cuda()

    def forward(self, q_emb, q_len, q_q_len, hs_emb_var, hs_len, sep_embeddings, tab_locations=None):
        B = len(q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        hs_enc = hs_enc[:, 0, :]

        q_enc = self.q_bert(q_emb, q_len, q_q_len, sep_embeddings)
        if tab_locations:
            new_q_emb = []
            new_q_len = np.zeros(B, dtype=np.int)
            for b, one_tab_locations in enumerate(tab_locations):
                new_q_len[b] = one_tab_locations[0]-1
            max_q_len = np.max(new_q_len)
            for b, one_tab_locations in enumerate(tab_locations):
                one_q_enc = q_enc[b, :new_q_len[b], :100]
                if new_q_len[b] < max_q_len:
                    padding = torch.zeros((max_q_len - new_q_len[b]), 100).float()
                    if torch.cuda.is_available():
                        padding = padding.cuda()
                    one_q_enc = torch.cat((one_q_enc, padding), dim=0)
                new_q_emb.append(one_q_enc)
            new_q_emb = torch.stack(new_q_emb)

            tab_emb = []
            tab_len = np.zeros(B, dtype=np.int)
            for b, one_tab_locations in enumerate(tab_locations):
                tab_len[b] = len(one_tab_locations)
            max_tab_len = np.max(tab_len)
            for b, one_tab_locations in enumerate(tab_locations):
                one_tab_emb = []
                for tab_loc in one_tab_locations:
                    one_tab_emb.append(q_enc[b, tab_loc, :100])
                while len(one_tab_emb) < max_tab_len:
                    padding = torch.zeros_like(q_enc[0, 0, :100])
                    one_tab_emb.append(padding)
                one_tab_emb = torch.stack(one_tab_emb)
                tab_emb.append(one_tab_emb)
            tab_emb = torch.stack(tab_emb)
            return self.from_num(new_q_emb, new_q_len, tab_emb, tab_len), self.from_cols(new_q_emb, new_q_len, tab_emb, tab_len)

        if self.onefrom:
            hs_enc = self.onefrom_vec.view(1, -1).expand(B, -1)

        q_enc = q_enc[:, 0, :]
        q_enc = torch.cat((q_enc, hs_enc), dim=1)
        x = self.outer1(q_enc)
        x = self.outer2(x).squeeze(1)
        return x

    def loss(self, score, labels):
        if self.wikisql_style:
            num_score, tab_score = score
            nums = []
            for label in labels:
                if len(label) <= 4:
                    nums.append(len(label) - 1)
                else:
                    nums.append(3)
            nums = torch.tensor(nums)

            if torch.cuda.is_available():
                nums = nums.cuda()
            num_score = F.sigmoid(num_score)
            num_loss = F.cross_entropy(num_score, nums)
            tabs = torch.zeros_like(tab_score)
            for b, one_labels in enumerate(labels):
                for t in one_labels:
                    tabs[b, int(t)] = 1
            tab_loss = F.binary_cross_entropy_with_logits(tab_score, tabs)
            return num_loss + tab_loss

        loss = F.binary_cross_entropy_with_logits(score, labels)
        return loss

    def check_acc(self, score, truth):
        if self.wikisql_style:
            num_err = 0
            err = 0
            num_score, score = score
            if self.gpu:
                num_score = num_score.cpu()
                num_score = num_score.data
                num_score = num_score.numpy()
                score = score.data.cpu().numpy()
            else:
                num_score = num_score.data.numpy()
                score = score.data.numpy()
            for b in range(len(truth)):
                if len(truth[b]) != np.argmax(num_score[b]) + 1:
                    num_err += 1
                sel_tabs = np.argsort(-score[b])[:len(truth[b])]
                wrong = False
                for t in truth[b]:
                    if int(t) not in sel_tabs:
                        wrong = True
                if wrong:
                    err += 1
            return num_err, err

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

    def wikisql_acc(self, num_score, tab_score, label, tables):
        if self.gpu:
            num_score = torch.tanh(num_score).data.cpu().numpy()
            tab_score = torch.tanh(tab_score).data.cpu().numpy()
        else:
            num_score = torch.tanh(num_score).data.numpy()
            tab_score = torch.tanh(tab_score).data.numpy()
        tabs = []
        for b in range(len(num_score)):
            print("number score: {}".format(num_score[b]))
            print("tab score: {}".format(tab_score[b]))
            print("label: {}".format(label[b]))
            num_tab = np.argmax(num_score[b]) + 1
            tabs.append(np.argsort(-tab_score[b])[:num_tab])
        foreign_keys = []
        primary_keys = []
        parent_nums = []
        for table in tables:
            foreign_keys.append(table["foreign_keys"])
            primary_keys.append(table["primary_keys"])
            parent_nums.append([par_num for par_num, _ in table["column_names"]])
        err_num = 0.
        tab_err_num = 0.
        for b in range(len(num_score)):
            predict_graph = graph_maker(tabs[b], foreign_keys[b], parent_nums[b])
            correct_joincond_graph = graph_maker_joincondition(tabs[b], foreign_keys[b], parent_nums[b], label[b])
            correct = graph_checker(predict_graph, label[b], foreign_keys[b], primary_keys[b])
            if not correct:
                err_num += 1.
            tab_correct = graph_checker(correct_joincond_graph, label[b], foreign_keys[b], primary_keys[b])
            if not tab_correct:
                tab_err_num += 1.
        return err_num, tab_err_num

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

    def score_to_tables(self, num_score, tab_score, foreign_keys, parent_tables):
        if self.gpu:
            num_score = torch.tanh(num_score).data.cpu().numpy()
            tab_score = torch.tanh(tab_score).data.cpu().numpy()
        else:
            num_score = torch.tanh(num_score).data.numpy()
            tab_score = torch.tanh(tab_score).data.numpy()
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

