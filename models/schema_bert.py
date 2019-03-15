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
        self.table_cols_embeddings = deepcopy(self.main_bert.embeddings)
        self.foreign_notifier = nn.Sequential(nn.Linear(1024, 1024), nn.Tanh())
        self.primary_notifier = nn.Sequential(nn.Linear(1024, 1024), nn.Tanh())
        self.table_cols_encoder = deepcopy(self.main_bert.encoder.layer[0])

    def make_mask(self, mask_tensor):
        mask_tensor = mask_tensor.unsqueeze(1).unsqueeze(2)
        mask_tensor = mask_tensor.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        mask_tensor = (1.0 - mask_tensor) * -10000.0
        if torch.cuda.is_available():
            mask_tensor = mask_tensor.cuda()
        return mask_tensor

    def forward(self, input_ids, input_id_lens, table_cols, table_col_num_lens, table_col_name_lens, table_col_type_ids, special_tok_id, parent_nums, foreign_keys):
        B = len(input_ids)
        _, max_table_col_num_lens, max_table_col_name_lens = list(table_cols.size())
        attention_mask = [[1.] * (input_id_lens[b] + 3 * table_col_num_lens[b]) for b in range(B)]
        attention_mask = make_padded_tensor(attention_mask)
        extended_attention_mask = self.make_mask(attention_mask)
        question_embedding = self.main_bert.embeddings(input_ids)

        # special_tok_embedding = self.main_bert.embeddings.word_embeddings(special_tok_id)
        # special_tok_embedding = special_tok_embedding.view(-1)
        table_col_attention_mask = np.zeros((B, max_table_col_num_lens, max_table_col_name_lens), dtype=np.float32)
        for b in range(B):
            for i in range(table_col_num_lens[b]):
                for j in range(table_col_name_lens[b][i]):
                    table_col_attention_mask[b, i, j] = 1.
        table_col_attention_mask = torch.from_numpy(table_col_attention_mask).view(B * max_table_col_num_lens, max_table_col_name_lens)
        table_col_attention_mask = self.make_mask(table_col_attention_mask)

        table_cols = table_cols.view(B * max_table_col_num_lens, max_table_col_name_lens)
        table_col_type_ids = table_col_type_ids.view(B * max_table_col_num_lens, max_table_col_name_lens)
        table_cols_embedding = self.main_bert.embeddings(table_cols, table_col_type_ids)
        encoded_table_cols = self.table_cols_encoder(table_cols_embedding, table_col_attention_mask)
        SIZE_CHECK(encoded_table_cols, [B * max_table_col_num_lens, max_table_col_name_lens, -1])
        encoded_table_cols = encoded_table_cols[:, :3, :].view(B, max_table_col_num_lens, 3, -1)
        temp_encoded_table_cols = []
        for b in range(B):
            temp_encoded_table_cols_one = []
            for i in range(max_table_col_num_lens):
                temp_encoded_table_cols_one.append(encoded_table_cols[b, i])
            temp_encoded_table_cols.append(temp_encoded_table_cols_one)
        for b in range(B):
            for f, p in foreign_keys[b]:
                f_parent = parent_nums[b][f]
                p_parent = parent_nums[b][p]
                temp_encoded_table_cols[b][f_parent] = temp_encoded_table_cols[b][f_parent] + self.primary_notifier(temp_encoded_table_cols[b][p_parent])
                temp_encoded_table_cols[b][p_parent] = temp_encoded_table_cols[b][p_parent] + self.foreign_notifier(temp_encoded_table_cols[b][f_parent])
        for b in range(B):
            temp_encoded_table_cols[b] = torch.stack(temp_encoded_table_cols[b])
        temp_encoded_table_cols = torch.stack(temp_encoded_table_cols)
        encoded_table_cols = temp_encoded_table_cols.view(B, max_table_col_num_lens, 3, -1)

        padded_encoded_table_cols = []
        for b in range(B):
            one_padded_tensor = []
            for idx in range(table_col_num_lens[b]):
                word_embedding = encoded_table_cols[b, idx]
                one_padded_tensor.append(word_embedding)
            one_padded_tensor = torch.cat(one_padded_tensor, dim=0)
            embedding_device = one_padded_tensor.device
            position_embedding = self.main_bert.embeddings.position_embeddings(torch.arange(input_id_lens[b], input_id_lens[b] + len(one_padded_tensor), dtype=torch.long, device=embedding_device))
            one_embedding = self.main_bert.embeddings.token_type_embeddings(torch.ones(len(one_padded_tensor), dtype=torch.long, device=embedding_device))
            embedding = one_padded_tensor + position_embedding + one_embedding
            embedding = self.main_bert.embeddings.LayerNorm(embedding)
            embedding = self.main_bert.embeddings.dropout(embedding)
            padded_encoded_table_cols.append(embedding)
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


