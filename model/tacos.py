import torch.nn as nn
import torch
import numpy as np
from typing import Any, Optional, Callable
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from .dgi import DGI

class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: Callable,
                 base_model: Any = GCNConv,
                 k: int = 2,):

        super(Decoder, self).__init__()
        self.base_model = base_model
        assert k >= 2
        self.k = k 
        

        self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.activation = activation
        

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (nodes, features)
    def forward(self, seq, adj, sparse=True):

        seq_fts = self.fc(seq)
        if sparse:
            out = torch.spmm(adj, seq_fts)
        else:
            out = torch.bmm(adj, seq_fts)

        if self.bias is not None:
            out += self.bias
        return self.act(out)

class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: Callable,
                 base_model: Any = GCNConv,
                 k: int = 2,):

        super(Encoder, self).__init__()
        self.base_model = base_model
        assert k >= 2
        self.k = k 
        

        self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.activation = activation
        

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x
        
class CSGCL(torch.nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 num_hidden: int,
                 num_proj_hidden: int,
                 tau: float = 0.5):

        super(CSGCL, self).__init__()
        self.encoder = encoder
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.num_hidden = num_hidden

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self,
                   z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def _sim(self,
             z1: torch.Tensor,
             z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def _infonce(self,
                  z1: torch.Tensor,
                  z2: torch.Tensor) -> torch.Tensor:

        temp = lambda x: torch.exp(x / self.tau)
        refl_sim = temp(self._sim(z1, z1))
        between_sim = temp(self._sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def _batched_infonce(self,
                          z1: torch.Tensor,
                          z2: torch.Tensor,
                          batch_size: int) -> torch.Tensor:
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            # print(i)
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self._sim(z1[mask], z1))
            between_sim = f(self._sim(z1[mask], z2))
            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)
        
    def _team_up(self,
                 z1: torch.Tensor,
                 z2: torch.Tensor,
                 cs: torch.Tensor,
                 current_ep: int,
                 t0: int,
                 gamma_max: int) -> torch.Tensor:
        gamma = min(max(0, (current_ep - t0) / 100), gamma_max)
        temp = lambda x: torch.exp(x / self.tau)
        refl_sim = temp(self._sim(z1, z1) + gamma * cs + gamma * cs.unsqueeze(dim=1))
        between_sim = temp(self._sim(z1, z2) + gamma * cs + gamma * cs.unsqueeze(dim=1))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def _batched_team_up(self,
                         z1: torch.Tensor,
                         z2: torch.Tensor,
                         cs: torch.Tensor,
                         current_ep: int,
                         t0: int,
                         gamma_max: int,
                         batch_size: int) -> torch.Tensor:
        gamma = min(max(0, (current_ep - t0) / 100), gamma_max)
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        temp = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = temp(self._sim(z1[mask], z1) + gamma * cs + gamma * cs.unsqueeze(dim=1)[mask])
            between_sim = temp(self._sim(z1[mask], z2) + gamma * cs + gamma * cs.unsqueeze(dim=1)[mask])

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def infonce(self,
                z1: torch.Tensor,
                z2: torch.Tensor,
                mean: bool = True,
                batch_size: Optional[int] = None) -> torch.Tensor:
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self._infonce(h1, h2)
            l2 = self._infonce(h2, h1)
        else:
            l1 = self._batched_infonce(h1, h2, batch_size)
            l2 = self._batched_infonce(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def team_up_loss(self,
                     z1: torch.Tensor,
                     z2: torch.Tensor,
                     cs: torch.Tensor,
                     current_ep: int,
                     t0: int = 0,
                     gamma_max: int = 1,
                     mean: bool = True,
                     batch_size: Optional[int] = None) -> torch.Tensor:

        h1 = self.projection(z1)
        h2 = self.projection(z2)
        # cs = torch.from_numpy(cs).to(h1.device)
        if batch_size is None:
            l1 = self._team_up(h1, h2, cs, current_ep, t0, gamma_max)
            l2 = self._team_up(h2, h1, cs, current_ep, t0, gamma_max)
        else:
            l1 = self._batched_team_up(h1, h2, cs, current_ep, t0, gamma_max, batch_size)
            l2 = self._batched_team_up(h2, h1, cs, current_ep, t0, gamma_max, batch_size)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret

class TacosModel(nn.Module):
    
    def __init__(self,input_dimension,latent_dimension=30,device='cpu',base='csgcl'):
        super(TacosModel,self).__init__()
        # csgcl hyperparam
        
        self.input_dimension = input_dimension
        num_hidden = latent_dimension
        num_proj_hidden = 256
        tau = 0.6
        layer_num = 2
        
        self.base = base
        
        
        
        
        self.device = device

        encoder = Encoder(self.input_dimension,
                          num_hidden,
                          torch.nn.PReLU(),
                          base_model=GCNConv,
                          k=layer_num).to(device)
        if self.base == 'csgcl':
            self.base_model = CSGCL(encoder,
                                num_hidden,
                                num_proj_hidden,
                                tau).to(device)
        elif self.base == 'gcn':
            self.base_model = encoder
        elif self.base == 'graphcl':
            self.base_model = DGI(self.input_dimension,num_hidden,'prelu',device).to(device)

        else:
            print('not implement')
            assert False
        
        # self.decoder = GCN(num_hidden, self.input_dimension, act='prelu').to(device)
        self.decoder = Decoder(num_hidden, self.input_dimension, torch.nn.PReLU(),base_model=GCNConv,k=layer_num).to(device)
        
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        if self.base == 'csgcl':
            return self.base_model(x, edge_index)
        elif self.base=='gcn':
            return self.base_model(x, edge_index)
        elif self.base=='graphcl':
            return self.base_model(x,edge_index)
            
        else:
            print('not implement')
            assert False
            return 0
    
    def ced(self,edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        p: float,
        threshold: float = 1.) -> torch.Tensor:

        edge_weight = edge_weight / edge_weight.mean() * (1. - p)
        edge_weight = edge_weight.where(edge_weight > (1. - threshold), torch.ones_like(edge_weight) * (1. - threshold))
        edge_weight = edge_weight.where(edge_weight < 1, torch.ones_like(edge_weight) * 1)
        
        sel_mask = torch.bernoulli(edge_weight).to(torch.bool)
        return edge_index[:, sel_mask]
    
    def cav(self,feature: torch.Tensor,
            node_cs:  torch.Tensor,
            p: float,
            max_threshold: float = 0.7) -> torch.Tensor:
        x = feature.abs()
        device = feature.device
        w = x.t() @ node_cs
        w[torch.nonzero(w == 0)] = w.max()  # for redundant attributes of Cora
        w = w.log()
        w = (w.max() - w) / (w.max() - w.min())
        w = w / w.mean() * p
        w = w.where(w < max_threshold, max_threshold * torch.ones(1).to(device))
        w = w.where(w > 0, torch.zeros(1).to(device))
        drop_mask = torch.bernoulli(w).to(torch.bool)
        feature = feature.clone()
        feature[:, drop_mask] = 0.
        return feature
    
    # def graphcl_loss(self,sug)
    
    def csgcl_loss(self,Y,edge_list,edge_weight,node_cs,epoch,
                   ced_drop_rate_1 = 0.2,
                   ced_drop_rate_2 = 0.7,
                   cav_drop_rate_1 = 0.1,
                   cav_drop_rate_2 = 0.2,
                   ced_thr = 1.,
                   cav_thr = 1.,
                   t0 = 500,
                   gamma = 1.0,
                   batch = None,
                   ):

        edge_index_1 = self.ced(edge_list, edge_weight, p=ced_drop_rate_1, threshold=ced_thr)
        edge_index_2 = self.ced(edge_list, edge_weight, p=ced_drop_rate_2, threshold=ced_thr)

        x1 = self.cav(Y, node_cs, cav_drop_rate_1, max_threshold=cav_thr)
        x2 = self.cav(Y, node_cs, cav_drop_rate_2, max_threshold=cav_thr)
        z1 = self.base_model(x1, edge_index_1)
        z2 = self.base_model(x2, edge_index_2)


        # gamma的值随着epoch而变化
        base_loss = self.base_model.team_up_loss(z1, z2,
                            cs=node_cs,
                            current_ep=epoch,
                            t0=t0,
                            gamma_max=gamma,
                            batch_size=batch)

            
        return base_loss
    
    def spatial_loss(self,z,coord,regularization_acceleration=True,edge_subset_sz=1000000):
        penalty=0
        if regularization_acceleration:
            # for i in range(len(z_list)):
            # slicez1 = z_list[i]
            # coord1 = coord_list[i]
            slicez1 = z
            coord1 =coord
            #slice1
            cell_random_subset_11, cell_random_subset_21 = torch.randint(0, slicez1.shape[0], (edge_subset_sz,)).to(self.device), torch.randint(0, slicez1.shape[0], (edge_subset_sz,)).to(self.device)
            z11, z21 = torch.index_select(slicez1, 0, cell_random_subset_11), torch.index_select(slicez1, 0, cell_random_subset_21)
            c11, c21 = torch.index_select(coord1, 0, cell_random_subset_11), torch.index_select(coord1, 0,
                                                                                                cell_random_subset_11)
            pdist1 = torch.nn.PairwiseDistance(p=2)

            z_dists1 = pdist1(z11, z21)
            z_dists1 = z_dists1 / torch.max(z_dists1)

            sp_dists1 = pdist1(c11, c21)
            sp_dists1 = sp_dists1 / torch.max(sp_dists1)
            n_items1 = z_dists1.size(dim=0)
            penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists1, sp_dists1)), n_items1).to(self.device)
            penalty += penalty_1
        else:
            # for i in range(len(z_list)):
            # slicez1 = z_list[i]
            # coord1 = coord_list[i]
            slicez1 = z
            coord1 =coord
            z_dists1 = torch.cdist(slicez1, slicez1, p=2)  ####### here should use z individually?????
            z_dists1 = torch.div(z_dists1, torch.max(z_dists1)).to(self.device)
            sp_dists1 = torch.cdist(coord1, coord1, p=2)
            sp_dists1 = torch.div(sp_dists1, torch.max(sp_dists1)).to(self.device)
            n_items1 = slicez1.size(dim=0) * slicez1.size(dim=0)
            penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists1, sp_dists1)), n_items1).to(self.device)
            penalty += penalty_1
        return penalty
    
    def _cross_loss(self,z_list,a,p,n,alpha=1.0):
        z = torch.cat(z_list, dim=0)
        anchor_arr = z[a,]
        positive_arr = z[p,]
        negative_arr = z[n,]
        triplet_loss = torch.nn.TripletMarginLoss(margin=alpha, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        return tri_output
        
    # def cross_loss(self,X_tuple,mnn_graph_list,alpha=1.0,negative= True):
    #     slicez1 = X_tuple[0]
    #     slicez2 = torch.cat(X_tuple[1:], dim=0)
        
    #     # none_zero_list =torch.nonzero(mnn_graph).to(self.device)
        
        
        
        
        
    #     a_idx = mnn_graph_list[:,0]
    #     p_idx = mnn_graph_list[:,1]
        
        
    #     a_slice,p_slice = torch.index_select(slicez1,0,a_idx),torch.index_select(slicez2,0,p_idx)
        
    #     if negative:
    #         n_idx = torch.randint(0, slicez1.shape[0], (a_idx.shape[0],)).to(self.device)
    #         n_slice = torch.index_select(slicez1,0,n_idx)
    #     # pdist_3 = torch.nn.PairwiseDistance(p=2)
    #     pdist3 = torch.nn.PairwiseDistance(p=2)
    #     p_dist = pdist3(a_slice, p_slice)
    #     if negative:
    #         n_dist = pdist3(a_slice, n_slice)
    #         cross_slice_total = torch.sum(torch.max(p_dist-n_dist+alpha,torch.zeros(p_dist.shape).to(self.device)))
    #     else:
    #         cross_slice_total = torch.sum(torch.max(p_dist,torch.zeros(p_dist.shape).to(self.device)))
        
    #     cross_loss = torch.div(cross_slice_total,mnn_graph_list.shape[0])

    #     return cross_loss
    
    def reconstr_loss(self,
                      z: torch.Tensor,
                      edge_index,
                      X: torch.Tensor,
                      ):


        z_rec = self.decoder(z, edge_index)
        recon_loss = torch.mean((z_rec - X) ** 2)
        

        return recon_loss, z_rec
    