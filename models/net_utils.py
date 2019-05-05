from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import sys


def SIZE_CHECK(tensor, size):
    for idx, dim in enumerate(size):
        if dim is None:
            size[idx] = list(tensor.size())[idx]
    if list(tensor.size()) != size:
        print("tensor size: {}, expected {}".format(list(tensor.size()), size), file=sys.stderr)
    assert list(tensor.size()) == size


def seq_conditional_weighted_num(attention_layer, predicate_tensor, predicate_len, conditional_tensor,
                                 conditional_len=None):
    max_predicate_len = max(predicate_len)
    if conditional_len is not None:
        max_conditional_len = max(conditional_len)
    else:
        max_conditional_len = None
    B = len(predicate_len)
    SIZE_CHECK(predicate_tensor, [B, max_predicate_len, None])
    co_attention = torch.bmm(conditional_tensor, attention_layer(predicate_tensor).transpose(1, 2))
    for idx, num in enumerate(predicate_len):
        if num < max_predicate_len:
            co_attention[idx, :, num:] = -100
    if conditional_len is not None:
        for idx, num in enumerate(conditional_len):
            if num < max_conditional_len:
                co_attention[idx, num:, :] = -100
    softmaxed_attention = F.softmax(co_attention.view(-1, max_predicate_len), dim=1)\
        .view(B, -1, max_predicate_len)
    weighted = (predicate_tensor.unsqueeze(1) * softmaxed_attention.unsqueeze(3)).sum(2)
    SIZE_CHECK(weighted, [B, None, None])
    return weighted


def plain_conditional_weighted_num(att, predicate_tensor, predicate_len, conditional_tensor):
    max_predicate_len = max(predicate_len)
    B = len(predicate_len)

    SIZE_CHECK(predicate_tensor, [B, max_predicate_len, None])
    SIZE_CHECK(conditional_tensor, [B, None])

    co_attention = torch.bmm(conditional_tensor.unsqueeze(1), att(predicate_tensor).transpose(1, 2))\
        .view(B, max_predicate_len)
    for idx, num in enumerate(predicate_len):
        if num < max_predicate_len:
            co_attention[idx, num:] = -100
    co_attention = F.softmax(co_attention, dim=1)
    weighted = (predicate_tensor * co_attention.unsqueeze(2))
    weighted = weighted.sum(1)
    return weighted


def encode_question(bert, inp, inp_len):
    [batch_num, max_seq_len] = list(inp.size())
    mask = np.zeros((batch_num, max_seq_len), dtype=np.float32)
    for idx, len in enumerate(inp_len):
        mask[idx, :len] = np.ones(len, dtype=np.float32)
    mask = torch.LongTensor(mask)
    if torch.cuda.is_available():
        mask = mask.cuda()
    encoded, _ = bert(input_ids=inp, attention_mask=mask)
    return encoded[-1]


def run_lstm(lstm, inp, inp_len, hidden=None):
    # Run the LSTM using packed sequence.
    # This requires to first sort the input according to its length.
    sort_perm = np.array(sorted(list(range(len(inp_len))),
        key=lambda k:inp_len[k], reverse=True))
    sort_inp_len = inp_len[sort_perm]
    sort_perm_inv = np.argsort(sort_perm)
    if inp.is_cuda:
        sort_perm = torch.LongTensor(sort_perm).cuda()
        sort_perm_inv = torch.LongTensor(sort_perm_inv).cuda()

    lstm_inp = nn.utils.rnn.pack_padded_sequence(inp[sort_perm],
            sort_inp_len, batch_first=True)
    if hidden is None:
        lstm_hidden = None
    else:
        lstm_hidden = (hidden[0][:, sort_perm], hidden[1][:, sort_perm])

    sort_ret_s, sort_ret_h = lstm(lstm_inp, lstm_hidden)
    ret_s = nn.utils.rnn.pad_packed_sequence(
            sort_ret_s, batch_first=True)[0][sort_perm_inv]
    ret_h = (sort_ret_h[0][:, sort_perm_inv], sort_ret_h[1][:, sort_perm_inv])
    return ret_s, ret_h


def col_tab_name_encode(name_inp_var, name_len, col_len, enc_lstm):
    #Encode the columns.
    #The embedding of a column name is the last state of its LSTM output.
    name_hidden, _ = run_lstm(enc_lstm, name_inp_var, name_len)
    name_out = name_hidden[tuple(range(len(name_len))), name_len-1]
    ret = torch.FloatTensor(
            len(col_len), max(col_len), name_out.size()[1]).zero_()
    if name_out.is_cuda:
        ret = ret.cuda()

    st = 0
    for idx, cur_len in enumerate(col_len):
        ret[idx, :cur_len] = name_out.data[st:st+cur_len]
        st += cur_len
    ret_var = Variable(ret)

    return ret_var, col_len
    
