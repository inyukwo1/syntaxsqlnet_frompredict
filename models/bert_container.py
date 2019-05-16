from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer
import torch
import copy
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


class BertParameterWrapper(nn.Module):
    def __init__(self):
        super(BertParameterWrapper, self).__init__()
        self.pos0 = nn.Parameter(torch.rand(1024) / 30)
        self.pos1 = nn.Parameter(torch.rand(1024) / 30)
        self.pos2 = nn.Parameter(torch.rand(1024) / 30)
        self.pos3 = nn.Parameter(torch.rand(1024) / 30)
        self.pos4 = nn.Parameter(torch.rand(1024) / 30)
        self.pos5 = nn.Parameter(torch.rand(1024) / 30)


class BertContainer:
    def __init__(self):
        self.main_bert = BertModel.from_pretrained('bert-large-cased')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        # self.bert_param = BertParameterWrapper()
        if torch.cuda.is_available():
            self.main_bert.cuda()
        # self.other_optimizer = torch.optim.Adam(self.bert_param.parameters(), lr=H_PARAM["learning_rate"])
        self.main_bert_optimizer = torch.optim.Adam(self.main_bert.parameters(), lr=H_PARAM["bert_learning_rate"])

    def bert(self, inp, inp_len, q_inp_len, sep_embeddings):
        [batch_num, max_seq_len] = list(inp.size())
        mask = np.zeros((batch_num, max_seq_len), dtype=np.float32)
        for idx, leng in enumerate(inp_len):
            mask[idx, :leng] = np.ones(leng, dtype=np.float32)
        [batch_num, max_seq_len] = list(inp.size())
        emb_mask = np.ones((batch_num, max_seq_len), dtype=np.float32)
        for idx, leng in enumerate(q_inp_len):
            emb_mask[idx, :leng] = np.zeros(leng, dtype=np.float32)
        mask = torch.LongTensor(mask)
        emb_mask = torch.LongTensor(emb_mask)
        if torch.cuda.is_available():
            mask = mask.cuda()
            emb_mask = emb_mask.cuda()

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.main_bert.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.main_bert.embeddings(inp, emb_mask)
        x = embedding_output
        for layer_num, layer_module in enumerate(self.main_bert.encoder.layer):
            x = layer_module(x, extended_attention_mask)

        return x

    def train(self):
        self.main_bert.train()

    def eval(self):
        self.main_bert.eval()

    def zero_grad(self):
        self.main_bert_optimizer.zero_grad()
        # self.other_optimizer.zero_grad()

    def step(self):
        self.main_bert_optimizer.step()
        #self.other_optimizer.step()
