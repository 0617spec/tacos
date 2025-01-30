import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj

import pdb

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation,device):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.disc2 = Discriminator2(n_h)
        self.device =device

    def loss(self, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, msk=None, samp_bias1=None, samp_bias2=None):
        
        h_0 = self.gcn(seq1, adj)
        # print(h_0.shape)
        # if aug_type == 'edge':

        #     h_1 = self.gcn(seq1, aug_adj1, sparse)
        #     h_3 = self.gcn(seq1, aug_adj2, sparse)

        # elif aug_type == 'mask':

        #     h_1 = self.gcn(seq3, adj, sparse)
        #     h_3 = self.gcn(seq4, adj, sparse)

        # elif aug_type == 'node' or aug_type == 'subgraph':

        h_1 = self.gcn(seq3, aug_adj1)
        h_3 = self.gcn(seq4, aug_adj2)
        # print(h_1.shape)
        # print(h_3.shape)
        # else:
            # assert False
            
        c_1 = self.read(h_1, msk)
        c_1= self.sigm(c_1)

        c_3 = self.read(h_3, msk)
        c_3= self.sigm(c_3)

        h_2 = self.gcn(seq2, adj)

        ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)

        ret = ret1 + ret2
        
        return ret

    # Detach the return variables
    def forward(self, seq, adj, msk=None):
        h_1 = self.gcn(seq, adj)
        # c = self.read(h_1, msk)

        return h_1

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act):
        super(GCN, self).__init__()
        self.gcn = GCNConv(in_ft, out_ft)
        self.act = nn.PReLU() if act == 'prelu' else act

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        return self.act(x)
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class Discriminator2(nn.Module):
    def __init__(self, n_h):
        super(Discriminator2, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c_x = torch.unsqueeze(c, 1)
        # c_x = c_x.expand_as(h_pl)
        c_x = c
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits