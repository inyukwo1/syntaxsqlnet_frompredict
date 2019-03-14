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
        # self.table_embedder = deepcopy(self.main_bert.embeddings)
        # self.table_cols_encoder = deepcopy(self.main_bert.encoder.layer[0])

    def forward(self, input_ids, input_id_lens, table_cols, table_col_num_lens, table_col_name_lens, table_col_type_ids, special_tok_id):
        B = len(input_ids)
        _, max_table_col_num_lens, max_table_col_name_lens = list(table_cols.size())
        attention_mask = [[1.] * (input_id_lens[b] + table_col_num_lens[b] + sum(table_col_name_lens[b])) for b in range(B)]
        attention_mask = make_padded_tensor(attention_mask)
        token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.main_bert.embeddings.word_embeddings(input_ids)
        position_embeddings = self.main_bert.embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.main_bert.embeddings.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        question_embedding = self.main_bert.embeddings.LayerNorm(embeddings)

        special_tok_embedding = self.main_bert.embeddings.word_embeddings(special_tok_id)
        special_tok_embedding = special_tok_embedding.view(-1)
        table_col_attention_mask = np.zeros((B, max_table_col_num_lens, max_table_col_name_lens), dtype=np.float32)
        for b in range(B):
            for i in range(table_col_num_lens[b]):
                for j in range(table_col_name_lens[b][i]):
                    table_col_attention_mask[b, i, j] = 1.
        # table_col_attention_mask = torch.from_numpy(table_col_attention_mask).view(B * max_table_col_num_lens, max_table_col_name_lens)
        # table_col_attention_mask = table_col_attention_mask.unsqueeze(1).unsqueeze(2)
        # table_col_attention_mask = table_col_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # table_col_attention_mask = (1.0 - table_col_attention_mask) * -10000.0
        if torch.cuda.is_available():
            # table_col_attention_mask = table_col_attention_mask.cuda()
            extended_attention_mask = extended_attention_mask.cuda()

        table_cols = table_cols.view(B * max_table_col_num_lens, max_table_col_name_lens)
        table_col_type_ids = table_col_type_ids.view(B * max_table_col_num_lens, max_table_col_name_lens)
        table_cols_embedding = self.main_bert.embeddings.word_embeddings(table_cols)
        table_cols_embedding = table_cols_embedding.view(B, max_table_col_num_lens, max_table_col_name_lens, -1)
        # encoded_table_cols = self.table_cols_encoder(table_cols_embedding, table_col_attention_mask)
        # SIZE_CHECK(encoded_table_cols, [B * max_table_col_num_lens, max_table_col_name_lens, -1])
        # encoded_table_cols = encoded_table_cols[:, 0, :].view(B, max_table_col_num_lens, -1)
        padded_encoded_table_cols = []
        for b in range(B):
            one_padded_tensor = []
            for idx in range(table_col_num_lens[b]):
                word_embedding = torch.cat((special_tok_embedding.unsqueeze(0), table_cols_embedding[b, idx, :table_col_name_lens[b][idx]]), dim=0)
                one_padded_tensor.append(word_embedding)
            one_padded_tensor = torch.cat(one_padded_tensor, dim=0)
            embedding_device = one_padded_tensor.device
            position_embedding = self.main_bert.embeddings.position_embeddings(torch.arange(input_id_lens[b], input_id_lens[b] + len(one_padded_tensor), dtype=torch.long, device=embedding_device))
            zero_embedding = self.main_bert.embeddings.token_type_embeddings(torch.zeros(len(one_padded_tensor), dtype=torch.long, device=embedding_device))
            embedding = one_padded_tensor + position_embedding + zero_embedding
            embedding = self.main_bert.embeddings.LayerNorm(embedding)
            #embedding = self.main_bert.embeddings.dropout(embedding)
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
        table_added_question_embedding =  self.main_bert.embeddings.dropout(table_added_question_embedding)
        encoded_layers = self.main_bert.encoder(table_added_question_embedding,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)
        return encoded_layers[-1]


