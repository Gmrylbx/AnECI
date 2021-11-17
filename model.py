import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sparse_dropout


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation=F.leaky_relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()

        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero
        self.weight = glorot_init(input_dim, output_dim)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs

        if self.training and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless:  # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight

        out = torch.sparse.mm(support, xw)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), support


class AnECI(nn.Module):
    def __init__(self, input_dim, output_dim, hidden, dropout, num_features_nonzero):
        super(AnECI, self).__init__()
        assert hidden[-1] == output_dim
        assert len(dropout) == len(hidden)

        self.num_layer = len(hidden)
        self.hidden = [input_dim] + hidden
        self.sft = nn.Softmax(dim=1)
        # print('input dim:', input_dim)
        # print('output dim:', output_dim)
        # print('num_features_nonzero:', num_features_nonzero)
        self.layers = nn.Sequential()
        for i in range(self.num_layer):
            self.layers.add_module('Conv' + str(i),
                                   GraphConvolution(self.hidden[i], self.hidden[i + 1], num_features_nonzero,
                                                    activation=F.leaky_relu, dropout=dropout[i],
                                                    is_sparse_inputs=bool(i == 0)))

    def forward(self, inputs):
        x, support = inputs
        out = self.layers((x, support))
        prob = self.sft(out[0])
        return prob, out[0]

    def embedding(self, inputs):
        x, support = inputs
        out = self.layers((x, support))
        return out[0]
