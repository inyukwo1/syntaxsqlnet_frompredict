import torch
from torch import nn
import numpy as np
from copy import deepcopy
from pytorch_pretrained_bert import BertModel
from models.net_utils import SIZE_CHECK


def make_padded_tensor(arr):  # 2-D array
    arr_len = [len(att) for att in arr]
    max_arr_len = max(arr_len)
    arr = [att + [0.] * (max_arr_len - len(att)) for att in arr]
    arr = np.array(arr)
    return torch.from_numpy(arr)


class SchemaBert(nn.Module):
    def __init__(self):
        super(SchemaBert, self).__init__()
        self.main_bert = BertModel.from_pretrained('bert-large-cased')
        self.table_cols_encoder = deepcopy(self.main_bert.encoder.layer[0])

    def forward(self, input_ids, input_id_lens, table_cols, table_col_num_lens, table_col_name_lens, table_col_type_ids, special_tok_id):
        B = len(input_ids)
        _, max_table_col_num_lens, max_table_col_name_lens = list(table_cols.size())
        attention_mask = [[1.] * (input_id_lens[b] + 2 * table_col_num_lens[b]) for b in range(B)]
        attention_mask = make_padded_tensor(attention_mask)
        token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        question_embedding = self.main_bert.embeddings(input_ids, torch.zeros_like(token_type_ids))
        special_tok_embedding = self.main_bert.embeddings(special_tok_id, torch.ones_like(special_tok_id))
        special_tok_embedding = special_tok_embedding.view(-1)
        table_cols = table_cols.view(B * max_table_col_num_lens, max_table_col_name_lens)
        table_cols_embedding = self.main_bert.embeddings(table_cols, table_col_type_ids)
        encoded_table_cols = self.table_cols_encoder(table_cols_embedding)
        SIZE_CHECK(encoded_table_cols, [B * max_table_col_num_lens, max_table_col_name_lens, -1])
        encoded_table_cols = encoded_table_cols[:, 0, :].view(B, max_table_col_num_lens, -1)
        padded_encoded_table_cols = []
        for b in range(B):
            one_padded_tensor = []
            for idx in range(max_table_col_num_lens):
                one_padded_tensor.append(torch.cat((special_tok_embedding, encoded_table_cols[b, idx])))
            one_padded_tensor = torch.stack(one_padded_tensor)
            padded_encoded_table_cols.append(one_padded_tensor)
        table_added_question_embedding = []
        for b in range(B):
            question_tensor = question_embedding[b, :input_id_lens[b]]
            question_tensor = torch.cat((question_tensor, padded_encoded_table_cols[b]), dim=0)
            if input_id_lens[b] < max_table_col_num_lens:
                padding = torch.zeros((max_table_col_num_lens - input_id_lens[b], 1024))
                question_tensor = torch.cat((question_tensor, padding), dim=0)
            table_added_question_embedding.append(question_tensor)
        table_added_question_embedding = torch.stack(table_added_question_embedding)
        encoded_layers = self.main_bert.encoder(table_added_question_embedding,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)
        return encoded_layers


