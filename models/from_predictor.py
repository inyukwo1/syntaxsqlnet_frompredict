import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, SIZE_CHECK
from graph_utils import *
from pytorch_pretrained_bert import *


def position_encoding_init(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # apply cos on 1st,3rd,5th...emb_dim
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=4,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


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


positional_encoding = position_encoding_init(1000, 300).cuda()


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
            tempbert = BertModel.from_pretrained("bert-base-uncased")
            tempbert.config.hidden_size = 300
            tempbert.config.max_position_embeddings = 1000
            tempbert.config.num_hidden_layers = 3
            self.bert = BertModel(tempbert.config)
            self.encoded_num = 300

        self.outer1 = nn.Sequential(nn.Linear(N_h + self.encoded_num, N_h), nn.ReLU())
        self.outer2 = nn.Sequential(nn.Linear(N_h, 1))
        if gpu:
            self.cuda()

    def forward(self, q_emb, q_len, q_q_len, hs_emb_var, hs_len, sep_embeddings):
        B = len(q_len)
        if self.use_lstm:
            _, max_q_len, _ = list(q_emb.size())
            q_emb = q_emb + positional_encoding.unsqueeze(0)[:, :max_q_len, :]
            attention_mask = np.zeros((B, 1, 1, max_q_len), dtype=np.float32)
            for b, q in enumerate(q_len):
                if q < max(q_len):
                    attention_mask[b, 0, 0, q:] -= 10000.0
            attention_mask = torch.from_numpy(attention_mask).cuda()
            q_enc = self.bert.encoder(q_emb, attention_mask, False)[-1]

        else:
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
        # for predicted_graph, sc in zip(table_graph_list, score):
        #     print("$$$$$$$$$$$$$$$$$")
        #     print(predicted_graph)
        #     print("~~~~~")
        #     print(sc)
        correct = False

        selected_graph = table_graph_list[np.argmax(score)]
        graph_correct = graph_checker(selected_graph, graph, foreign_keys, primary_keys)
        # print("#### " + " ".join(question))
        # for idx, table_name in enumerate(table_names):
        #     print("Table {}: {}".format(idx, table_name))
        #     for col_idx, [par_tab, col_name] in enumerate(column_names):
        #         if par_tab == idx:
        #             print("   {}: {}".format(col_idx, col_name))
        #
        # print("=======")
        # print(selected_graph)
        # print("========")
        # print(graph)
        # print("@@@@@@@@@@@@@@{}, {}".format(correct, graph_correct))
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

