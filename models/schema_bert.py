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
        self.table_embedder = deepcopy(self.main_bert.embeddings)
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
        special_tok_embedding = self.table_embedder(special_tok_id, torch.ones_like(special_tok_id))
        special_tok_embedding = special_tok_embedding.view(-1)
        table_col_attention_mask = np.zeros((B, max_table_col_num_lens, max_table_col_name_lens), dtype=np.float32)
        for b in range(B):
            for i in range(table_col_num_lens[b]):
                for j in range(table_col_name_lens[b][i]):
                    table_col_attention_mask[b, i, j] = 1.
        table_col_attention_mask = torch.from_numpy(table_col_attention_mask).view(B * max_table_col_num_lens, max_table_col_name_lens)
        table_col_attention_mask = table_col_attention_mask.unsqueeze(1).unsqueeze(2)
        table_col_attention_mask = table_col_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        table_col_attention_mask = (1.0 - table_col_attention_mask) * -10000.0
        if torch.cuda.is_available():
            table_col_attention_mask = table_col_attention_mask.cuda()
            extended_attention_mask = extended_attention_mask.cuda()

        table_cols = table_cols.view(B * max_table_col_num_lens, max_table_col_name_lens)
        table_col_type_ids = table_col_type_ids.view(B * max_table_col_num_lens, max_table_col_name_lens)
        table_cols_embedding = self.table_embedder(table_cols, table_col_type_ids)
        encoded_table_cols = self.table_cols_encoder(table_cols_embedding, table_col_attention_mask)
        SIZE_CHECK(encoded_table_cols, [B * max_table_col_num_lens, max_table_col_name_lens, -1])
        encoded_table_cols = encoded_table_cols[:, 0, :].view(B, max_table_col_num_lens, -1)
        padded_encoded_table_cols = []
        for b in range(B):
            one_padded_tensor = []
            for idx in range(table_col_num_lens[b]):
                one_padded_tensor.append(torch.cat((special_tok_embedding.unsqueeze(0), encoded_table_cols[b, idx].unsqueeze(0)), dim=0))
            one_padded_tensor = torch.stack(one_padded_tensor).view(-1, 1024)
            padded_encoded_table_cols.append(one_padded_tensor)
        table_added_question_embedding = []
        for b in range(B):
            question_tensor = question_embedding[b, :input_id_lens[b]]
            question_tensor = torch.cat((question_tensor, padded_encoded_table_cols[b]), dim=0)
            table_added_question_embedding.append(question_tensor)
        max_q_t_len = 0
        for question_tensor in table_added_question_embedding:
            if max_q_t_len < len(question_tensor):
                max_q_t_len = len(question_tensor)
        for b, question_tensor in enumerate(table_added_question_embedding):
            if len(question_tensor) < max_q_t_len:
                padding = torch.zeros((max_q_t_len - len(question_tensor), 1024))
                if torch.cuda.is_available():
                    padding = padding.cuda()
                question_tensor = torch.cat((question_tensor, padding), dim=0)
                table_added_question_embedding[b] = question_tensor
        table_added_question_embedding = torch.stack(table_added_question_embedding)
        encoded_layers = self.main_bert.encoder(table_added_question_embedding,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)
        return encoded_layers[-1]


