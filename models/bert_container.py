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


class BertContainer:
    def __init__(self):
        self.main_bert = BertModel.from_pretrained('bert-large-cased')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        info_adder = nn.Sequential(nn.Linear(1024, 1024), nn.Sigmoid())
        self.foreign_info_adder = nn.ModuleList([copy.deepcopy(info_adder) for _ in range(24)])
        if torch.cuda.is_available():
            self.main_bert.cuda()
            self.foreign_info_adder.cuda()

        self.other_optimizer = torch.optim.Adam(self.foreign_info_adder.parameters(), lr=H_PARAM["learning_rate"])
        self.main_bert_optimizer = torch.optim.Adam(self.main_bert.parameters(), lr=H_PARAM["bert_learning_rate"])

    def bert(self, inp, inp_len, selected_tables):
        [batch_num, max_seq_len] = list(inp.size())
        mask = np.zeros((batch_num, max_seq_len), dtype=np.float32)
        for idx, leng in enumerate(inp_len):
            mask[idx, :leng] = np.ones(leng, dtype=np.float32)
        mask = torch.LongTensor(mask)
        if torch.cuda.is_available():
            mask = mask.cuda()

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.main_bert.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.main_bert.embeddings(inp)

        x = embedding_output
        for layer_num, layer_module in enumerate(self.main_bert.encoder.layer):
            x = layer_module(x, extended_attention_mask)
            right_tensors = []
            cur = 0
            for one_selected in selected_tables:
                right_tensors.append(torch.zeros_like(x[0, 0]))
                for idx in range(len(one_selected) - 1):
                    right_tensors.append(x[cur + idx, 0])
                cur += len(one_selected)
            assert cur == batch_num
            right_tensors = torch.stack(right_tensors).unsqueeze(1)
            right_tensors = self.foreign_info_adder[layer_num](right_tensors)
            left_tensors = []
            cur = 0
            for one_selected in selected_tables:
                for idx in range(1, len(one_selected)):
                    left_tensors.append(x[cur + idx, 0])
                left_tensors.append(torch.zeros_like(x[0, 0]))
                cur += len(one_selected)
            assert cur == batch_num
            left_tensors = torch.stack(left_tensors).unsqueeze(1)
            left_tensors = self.foreign_info_adder[layer_num](left_tensors)
            padding = torch.zeros_like(right_tensors).expand(-1, max_seq_len - 2, -1)
            y = torch.cat((torch.zeros_like(right_tensors), (right_tensors + left_tensors), padding), dim=1)
            if 3 <= layer_num <= 6:
                x = x + y

        return x

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
