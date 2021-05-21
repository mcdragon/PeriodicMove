import torch
import torch.nn as nn
import math
from torch.nn import Module, Parameter
import torch.nn.functional as F

class PosEncoder(nn.Module):
    def __init__(self, length, hidden_size):
        super().__init__()
        freqs = torch.Tensor(
            [10000 ** (-i / hidden_size) if i % 2 == 0 else -10000 ** ((1 - i) / hidden_size) for i in range(hidden_size)]).unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(hidden_size)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(hidden_size, 1).to(torch.float)
        self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freqs), phases)), requires_grad=False)

    def forward(self, x, transpose=True):
        if not transpose:
            x = x + self.pos_encoding
            return x
        return x + self.pos_encoding.transpose(0,1)


class PosEmbedding(nn.Module):
    def __init__(self, length, hidden_size):
        super().__init__()
        self.pos_embedding = nn.Embedding(length, hidden_size)

    def forward(self, x, transpose=None):
        if transpose is not None:
            raise ValueError("For now, transpose is not supported by PosEmbedding.")
        return x + self.pos_embedding.weight


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        edge_in_hidden = self.linear_edge_in(hidden)
        edge_out_hidden = self.linear_edge_out(hidden)
        input_in = torch.matmul(A[:, :, :A.shape[1]], edge_in_hidden) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], edge_out_hidden) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hidden_size
        self.n_node = n_node
        #self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, hidden, A):
        hidden = self.gnn(A, hidden)
        return hidden


class PeriodMigrationCrossMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.linear_his_one = nn.Linear(hidden_size, hidden_size*n_heads, bias=True)
        self.linear_his_two = nn.Linear(hidden_size, hidden_size*n_heads, bias=True)
        self.compress_layer = nn.Linear(hidden_size*n_heads, hidden_size)

    def forward(self, seq_hidden_current, seq_hidden_history):
        # seq_hidden_history: batch_size, history_size, seq_length, hidden_size
        # seq_hidden_current: batch_size, seq_length, hidden_size

        batch_size = seq_hidden_history.shape[0]

        # batch_size, seq_length, hidden_size*n_heads
        his_q1 = self.linear_his_one(seq_hidden_current)
        # batch_size, history_size, seq_length, hidden_size*n_heads
        his_q2 = self.linear_his_two(seq_hidden_history)

        # batch_size*n_heads, seq_len, hidden_size
        his_q1 = torch.cat(his_q1.split(self.hidden_size, dim=-1), dim=0)
        # batch_size*n_heads, history_size, seq_len, hidden_size
        his_q2 = torch.cat(his_q2.split(self.hidden_size, dim=-1), dim=0)
        # batch_size*n_heads, hidden_size, history_size*seq_len
        his_q2_reshaped = his_q2.view(his_q2.shape[0], -1, his_q2.shape[-1]).transpose(1,2)
        # batch_size*n_heads, seq_len, history_size, seq_len
        att_weights = torch.softmax(torch.bmm(his_q1, his_q2_reshaped).view(his_q1.shape[0], his_q1.shape[1], -1, his_q1.shape[1]), -1)
        # batch_size*n_heads, seq_len, history_size, hidden_size
        seq_hidden_history_attn = (att_weights.unsqueeze(-1) * his_q2.unsqueeze(1)).sum(-2).transpose(1,2)
        # batch_size, seq_len, history_size, hidden_size*n_heads
        seq_hidden_history_attn = torch.cat(seq_hidden_history_attn.split(batch_size,0), -1)

        # residual connection
        seq_hidden_history = self.compress_layer(seq_hidden_history_attn)+seq_hidden_history
        #seq_hidden_history = self.compress_layer(seq_hidden_history_attn)

        return seq_hidden_history


class GNN_GNN_model(nn.Module):
    """
    Historical Trajectory Modeling: GNN
    Current trajectory Modeling: GNN
    """
    def __init__(self, parameters, n_node, history_size):
        super(GNN_GNN_model, self).__init__()
        self.hidden_size = parameters.hidden_size
        # 有两个embedding层  一个在当前 一个在SessionGraph中
        self.embedding = nn.Embedding(n_node, parameters.hidden_size)
        print("Warning: the length of PosEncoder in GNN_GNN_model is fixed at 48.")
        self.pos_encoding_layer = PosEncoder(48, parameters.hidden_size)
        #self.pos_encoding_layer = PosEmbedding(48, parameters.hidden_size)
        self.GNN = SessionGraph(parameters, n_node)
        #self.linear_his_one = nn.Linear(parameters.hidden_size, parameters.hidden_size, bias=True)
        #self.linear_his_two = nn.Linear(parameters.hidden_size, parameters.hidden_size, bias=True)
        self.cross_attn_layer = PeriodMigrationCrossMultiHeadAttention(parameters.hidden_size, parameters.cross_n_heads)
        self.linear_one = nn.Linear(parameters.hidden_size, parameters.hidden_size, bias=True)
        self.linear_two = nn.Linear(parameters.hidden_size, parameters.hidden_size, bias=True)
        self.linear_three = nn.Linear(parameters.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(parameters.hidden_size * 2, parameters.hidden_size, bias=True)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.history_size = history_size
        self.init_weight()

    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_embedding(self, indexes=None):
        if indexes is None:
            return self.embedding.weight
        else:
            return self.embedding(indexes)

    def forward(self, history_trajs_masks, historical_alias_inputs, historical_input_A, historical_inputs, current_alias_inputs, current_input_A, current_inputs, cur_mask_pos):
        # history_trajs_masks: batch_size, history_size

        # Historical trajectory GNN
        historical_sessions = self.embedding(historical_inputs)
        historical_sessions = self.GNN(historical_sessions, historical_input_A)
        #historical_sessions = self.embedding(historical_inputs)
        get_history = lambda i: historical_sessions[i][historical_alias_inputs[i]]
        # batch_size * history_length, seq_length, hidden_size
        seq_hidden_history = torch.stack([get_history(i) for i in torch.arange(len(historical_alias_inputs)).long()])
        # batch_size * history_length, seq_length, hidden_size
        seq_hidden_history = self.pos_encoding_layer(seq_hidden_history)
        # batch_size * history_size, seq_length, hidden_size -> batch_size, history_size, seq_length, hidden_size
        seq_hidden_history = seq_hidden_history.view(-1, self.history_size, seq_hidden_history.shape[-2], seq_hidden_history.shape[-1])

        # Current trajectory GNN
        current_sessions = self.embedding(current_inputs)
        current_sessions = self.GNN(current_sessions, current_input_A)
        #current_sessions = self.embedding(current_inputs)
        get_current = lambda i: current_sessions[i][current_alias_inputs[i]]
        # batch_size, seq_length, hidden_size
        seq_hidden_current = torch.stack([get_current(i) for i in torch.arange(len(current_alias_inputs)).long()])
        # batch_size, seq_length, hidden_size
        seq_hidden_current = self.pos_encoding_layer(seq_hidden_current)

        # Period migration Cross Attention
        # batch_size, history_size, seq_length, hidden_size
        seq_hidden_history = self.cross_attn_layer(seq_hidden_current, seq_hidden_history)

        # Attention Mechainsm
        # batch_size, history_size, seq_length, hidden_size
        q1 = self.linear_one(seq_hidden_history)
        # batch_size, seq_length, latent_size -> batch_size, 1, seq_length, hidden_size
        q2 = self.linear_two(seq_hidden_current).unsqueeze(1)
        # batch_size, history_size, seq_length, 1
        alpha = self.linear_three(torch.sigmoid(q1 + q2)) * history_trajs_masks.view(q1.shape[0], q1.shape[1], 1, 1)
        # batch_size, seq_length, hidden_size
        historical_information = torch.sum(alpha * seq_hidden_history.float(), 1)
        # batch_size, seq_length, hidden_size
        #historical_information = seq_hidden_history.mean(1)

        # concat historical and current information
        # batch_size, seq_length, hidden_size
        hybrid_embedding_ = self.linear_transform(torch.cat([historical_information, seq_hidden_current], -1))
        # cur_mask_pos: batch_size, mask_num -> index: batch_size * mask_num
        index = (cur_mask_pos+torch.arange(cur_mask_pos.shape[0]).to(cur_mask_pos.device).view(-1,1)*hybrid_embedding_.shape[1]).view(-1)
        # hybrid_embedding_: batch_size, seq_length, hidden_size -> batch_size * seq_length, hidden_size
        # batch_size * mask_num, hidden_size
        hybrid_embedding = hybrid_embedding_.view(-1, self.hidden_size).index_select(0, index)

        # dropout
        hybrid_embedding = self.dropout(hybrid_embedding)

        # calculate score
        # n, hidden_size
        # The first three are pad, * and <m>.
        candidate_poi = self.embedding.weight[3:]
        # scores: batch_size * mask_num, n
        scores = torch.matmul(hybrid_embedding, candidate_poi.transpose(1, 0))

        score = F.log_softmax(scores, dim=-1)
        return score

