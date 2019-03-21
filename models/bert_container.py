from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer
import torch
import torch.nn as nn
import numpy as np
from hyperparameters import H_PARAM


def embedding_tensor_listify(tensor):
    D1, D2, D3 = list(tensor.size())
    list_tensor = []
    for d1 in range(D1):
        one_list_tensor = []
        for d2 in range(D2):
            one_list_tensor.append(tensor[d1, d2])
        list_tensor.append(one_list_tensor)
    return list_tensor


def list_tensor_tensify(list_tensor):
    for idx, one_list_tensor in enumerate(list_tensor):
        list_tensor[idx] = torch.stack(one_list_tensor)
    list_tensor = torch.stack(list_tensor)
    return list_tensor


class BertContainer:
    def __init__(self):
        self.main_bert = BertModel.from_pretrained('bert-large-cased')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        table_marker_tensor = torch.Tensor(1024).random_()
        table_sep_marker_tensor = torch.Tensor(1024).random_()
        if torch.cuda.is_available():
            self.main_bert.cuda()
            table_marker_tensor = table_marker_tensor.cuda()
            table_sep_marker_tensor = table_sep_marker_tensor.cuda()
        self.table_marker = nn.Parameter(table_marker_tensor)
        self.table_sep_marker = nn.Parameter(table_sep_marker_tensor)

        self.other_optimizer = torch.optim.Adam(self.other_parameters(), lr=H_PARAM["learning_rate"])
        self.main_bert_optimizer = torch.optim.Adam(self.main_bert.parameters(), lr=H_PARAM["bert_learning_rate"])

    def bert(self, inp, inp_len, table_locs, table_sep_locs):
        [batch_num, max_seq_len] = list(inp.size())
        mask = np.zeros((batch_num, max_seq_len), dtype=np.float32)
        for idx, len in enumerate(inp_len):
            mask[idx, :len] = np.ones(len, dtype=np.float32)
        mask = torch.LongTensor(mask)
        if torch.cuda.is_available():
            mask = mask.cuda()

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.main_bert.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.main_bert.embeddings(inp)
        list_embedding_output = embedding_tensor_listify(embedding_output)
        for b, one_tensor_list in enumerate(list_embedding_output):
            for idx, tensor in enumerate(one_tensor_list):
                if idx in table_locs[b]:
                    one_tensor_list[idx] = tensor + self.table_marker
                if idx in table_sep_locs[b]:
                    one_tensor_list[idx] = tensor + self.table_sep_marker
        embedding_output = list_tensor_tensify(list_embedding_output)
        encoded_layers = self.main_bert.encoder(embedding_output, extended_attention_mask)
        return encoded_layers[-1]

    def other_parameters(self):
        yield self.table_marker
        yield self.table_sep_marker

    def train(self):
        self.main_bert.train()

    def eval(self):
        self.main_bert.eval()

    def zero_grad(self):
        self.main_bert_optimizer.zero_grad()
        self.other_optimizer.zero_grad()

    def step(self):
        self.main_bert_optimizer.step()
        self.other_optimizer.step()
