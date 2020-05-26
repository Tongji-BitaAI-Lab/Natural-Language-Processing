"""Sequence to Sequence models."""
import torch
import torch.nn as nn
from torch.nn import Module, LSTM, Linear, ModuleList
import torch.nn.functional as F

class FeedForward(Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super(FeedForward, self).__init__()
        self.dropout = dropout
        self.linear1 = Linear(input_size, hidden_size)
        self.linear2 = Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj

def att_mask(tensor, mask):
    masked_tensor = tensor + (1. - mask.float()) * -1e10
    return masked_tensor


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None):
        """Propogate input through the network.

        input: batch x dim
        context: batch x sourceL x dim
        """
        if input.size(1) == 1:
            input = input.view(-1, input.size(-1))
        target = self.linear_in(input)
        target = target.unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL

        if mask is not None:
            attn = att_mask(attn, mask)

        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        # print("DONE PRINTING INPUT SIZE")
        return h_tilde, attn


class LSTMAttentionDot(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True):
        """Initialize params."""
        super(LSTMAttentionDot, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.attention_layer = SoftDotAttention(hidden_size)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""

        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            hx_modified = self.hidden_weights(hx)
            gates = self.input_weights(input) + hx_modified

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate_sigm = torch.sigmoid(ingate)
            forgetgate_sigm = torch.sigmoid(forgetgate)
            cellgate_tanh = torch.tanh(cellgate)
            outgate_sigm = torch.sigmoid(outgate)

            cy = (forgetgate_sigm * cx) + (ingate_sigm * cellgate_tanh)
            hy = outgate_sigm * torch.tanh(cy)  # n_b x hidden_dim
            h_tilde, alpha = self.attention_layer(hy, ctx, ctx_mask)

            return h_tilde, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            output.append(hidden[0])

        cat_output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = cat_output.transpose(0, 1)
        else:
            output = cat_output
        return output, hidden
