import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.net_utils import run_lstm, col_tab_name_encode, encode_question, SIZE_CHECK, seq_conditional_weighted_num
from pytorch_pretrained_bert import BertModel
from models.schema_encoder import SchemaEncoder, SchemaAggregator


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.relu(output)

        return output, attention_weights


class FromPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu, use_hs, bert=None):
        super(FromPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu
        self.use_hs = use_hs
        self.threshold = 0.5

        self.use_bert = True if bert else False
        if bert:
            self.q_bert = bert
            self.encoded_num = 768
        else:
            self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
            self.encoded_num = N_h

        self.hs_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.tab_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h // 2,
                                num_layers=N_depth, batch_first=True,
                                dropout=0.3, bidirectional=True)

        self.table_column_attention1 = Attention(N_h)
        self.table_column_attention2 = Attention(N_h)
        self.table_column_attention3 = Attention(N_h)
        self.column_attention1 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU())
        self.column_attention2 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU())
        self.column_attention3 = nn.Sequential(nn.Linear(N_h, N_h), nn.ReLU())

        self.q_num_att = nn.Linear(self.encoded_num, N_h)
        self.hs_num_att = nn.Linear(N_h, N_h)
        self.col_num_out_q = nn.Linear(self.encoded_num, N_h)
        self.col_num_out_hs = nn.Linear(N_h, N_h)
        self.col_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 6))  # num of cols: 1-3

        self.q_att = nn.Linear(self.encoded_num, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.col_out_q = nn.Linear(self.encoded_num, N_h)
        self.col_out_c = nn.Linear(N_h, N_h)
        self.col_out_hs = nn.Linear(N_h, N_h)
        self.col_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax()  # dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

        if gpu:
            self.cuda()

    def forward(self, parent_tables, foreign_keys, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len,
                table_emb_var, table_len, table_name_len):

        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        max_tab_len = max(table_len)
        B = len(q_len)
        if self.use_bert:
            q_enc = self.q_bert(q_emb_var, q_len)
        else:
            q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        col_enc, _ = col_tab_name_encode(col_emb_var, col_name_len, col_len, self.col_lstm)
        tab_enc, _ = col_tab_name_encode(table_emb_var, table_name_len, table_len, self.tab_lstm)

        st_ed_par_tables = []
        max_sted_len = 0
        for b, batch_max_table in enumerate(table_len):
            st = 1
            ed = 1
            batch_st_ed_tables = []
            for i in range(batch_max_table):
                while parent_tables[b][ed] == i:
                    ed += 1
                    if ed >= len(parent_tables[b]):
                        break
                batch_st_ed_tables.append((st, ed))
                if ed - st > max_sted_len:
                    max_sted_len = ed - st
                st = ed
            st_ed_par_tables.append(batch_st_ed_tables)
        if torch.cuda.is_available():
            col_enc = col_enc.cpu()
            tab_enc = tab_enc.cpu()
        new_tab_enc = []
        for b in range(len(table_len)):
            new_tab_enc.append(tab_enc[b, :table_len[b]])
        tab_enc = torch.cat(new_tab_enc, dim=0)
        new_col_enc = []
        for b in range(len(col_enc)):
            for table_num, (st, ed) in enumerate(st_ed_par_tables[b]):
                subcol = col_enc[b, st:ed]
                padding = torch.from_numpy(np.zeros((max_sted_len - (ed - st), self.N_h), dtype=np.float32))
                padded_col = torch.cat((subcol, padding),
                                       dim=0)
                new_col_enc.append(padded_col)
        if torch.cuda.is_available():
            tab_enc = tab_enc.cuda()
            col_enc = torch.stack(new_col_enc).cuda()
        else:
            col_enc = torch.stack(new_col_enc)
        tab_enc = tab_enc.view(-1, 1, self.N_h)
        col_enc = col_enc.view(-1, max_sted_len, self.N_h)
        tab_enc, _ = self.table_column_attention1(tab_enc, col_enc)
        col_enc = self.column_attention1(col_enc)
        tab_enc, _ = self.table_column_attention2(tab_enc, col_enc)
        col_enc = self.column_attention2(col_enc)
        tab_enc, _ = self.table_column_attention3(tab_enc, col_enc)
        if torch.cuda.is_available():
            tab_enc = tab_enc.cpu()
        tab_enc = tab_enc.view(-1, self.N_h)
        new_tab_enc = []
        st = 0
        for b in range(len(table_len)):
            tab_tensor = tab_enc[st: st+table_len[b]]
            padding = torch.from_numpy(np.zeros((max_tab_len - table_len[b], self.N_h), dtype=np.float32))
            new_tab_enc.append(torch.cat((tab_tensor, padding), dim=0))
        tab_enc = torch.stack(new_tab_enc)
        if torch.cuda.is_available():
            tab_enc = tab_enc.cuda()

        # Predict column number: 1-3
        q_weighted_num = seq_conditional_weighted_num(self.q_num_att, q_enc, q_len, tab_enc, table_len).sum(1)
        SIZE_CHECK(q_weighted_num, [B, self.N_h])

        # Same as the above, compute SQL history embedding weighted by column attentions
        hs_weighted_num = seq_conditional_weighted_num(self.hs_num_att, hs_enc, hs_len, tab_enc, table_len).sum(1)
        SIZE_CHECK(hs_weighted_num, [B, self.N_h])
        # self.col_num_out: (B, 3)
        col_num_score = self.col_num_out(
            self.col_num_out_q(q_weighted_num) + int(self.use_hs) * self.col_num_out_hs(hs_weighted_num))

        # Predict columns.
        q_weighted = seq_conditional_weighted_num(self.q_att, q_enc, q_len, tab_enc)

        # Same as the above, compute SQL history embedding weighted by column attentions
        hs_weighted = seq_conditional_weighted_num(self.hs_att, hs_enc, hs_len, tab_enc)
        # Compute prediction scores
        # self.col_out.squeeze(): (B, max_col_len)
        tab_score = self.col_out(
            self.col_out_q(q_weighted) + int(self.use_hs) * self.col_out_hs(hs_weighted) + self.col_out_c(
                tab_enc)).view(B, -1)

        for idx, num in enumerate(table_len):
            if num < max_tab_len:
                tab_score[idx, num:] = -100

        score = (col_num_score, tab_score)

        return score

    def loss(self, score, truth):
        table_num_score, table_score = score
        num_truth = [len(one_truth) - 1 for one_truth in truth]
        data = torch.from_numpy(np.array(num_truth))
        if self.gpu:
            data = data.cuda()
        truth_num_var = Variable(data)
        loss = F.cross_entropy(table_num_score, truth_num_var)

        B = len(truth)
        SIZE_CHECK(table_score, [B, None])
        max_table_len = list(table_score.size())[1]
        label = []
        for one_truth in truth:
            label.append([1. if str(i) in one_truth else 0. for i in range(max_table_len)])
        label = torch.from_numpy(np.array(label)).type(torch.FloatTensor)
        if self.gpu:
            label = label.cuda()
        label = Variable(label)
        loss += F.binary_cross_entropy_with_logits(table_score, label)
        return loss

    def check_acc(self, score, truth):
        num_err, err, tot_err = 0, 0, 0
        B = len(truth)
        pred = []
        if self.gpu:
            table_num_score, table_score = [sc.data.cpu().numpy() for sc in score]
        else:
            table_num_score, table_score = [sc.data.numpy() for sc in score]

        for b in range(B):
            cur_pred = {}
            table_num = np.argmax(table_num_score[b]) + 1
            cur_pred["table_num"] = table_num
            cur_pred["table"] = np.argsort(-table_score[b])[:table_num]
            pred.append(cur_pred)
        for b, (p, t) in enumerate(zip(pred, truth)):
            table_num, tab = p['table_num'], p['table']
            flag = True
            if table_num != len(t): # double check truth format and for test cases
                num_err += 1
                flag = False
            fk_list = []
            regular = []
            for l in t:
                if isinstance(l, list):
                    fk_list.append(l)
                else:
                    regular.append(l)

            if flag: #double check
                for c in tab:
                    for fk in fk_list:
                        if c in fk:
                            fk_list.remove(fk)
                    for r in regular:
                        if c == r:
                            regular.remove(r)

                if len(fk_list) != 0 or len(regular) != 0:
                    err += 1
                    flag = False

            if not flag:
                tot_err += 1

        return np.array((num_err, err, tot_err))

    def score_to_tables(self, score):
        if self.gpu:
            table_num_score, table_score = [sc.data.cpu().numpy() for sc in score]
        else:
            table_num_score, table_score = [sc.data.numpy() for sc in score]
        B = len(score)
        batch_tables = []
        for b in range(B):
            cur_pred = {}
            table_num = np.argmax(table_num_score[b]) + 1
            cur_pred["table_num"] = table_num
            cur_pred["table"] = np.argsort(-table_score[b])[:table_num]
            batch_tables.append(cur_pred["table"])
        return batch_tables

