import torch.nn as nn
import torch
import numpy as np
from typing import Any, Optional, Callable
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from scipy import sparse
import random
import os
import networkx as nx
from sklearn.neighbors import kneighbors_graph

import gudhi
from .utils import graph_alpha

class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: Callable,
                 base_model: Any = GCNConv,
                 k: int = 2,
                 skip: bool = False):

        super(Encoder, self).__init__()
        self.base_model = base_model
        assert k >= 2
        self.k = k 
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


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
                     cs: np.ndarray,
                     current_ep: int,
                     t0: int = 0,
                     gamma_max: int = 1,
                     mean: bool = True,
                     batch_size: Optional[int] = None) -> torch.Tensor:

        h1 = self.projection(z1)
        h2 = self.projection(z2)
        cs = torch.from_numpy(cs).to(h1.device)
        if batch_size is None:
            l1 = self._team_up(h1, h2, cs, current_ep, t0, gamma_max)
            l2 = self._team_up(h2, h1, cs, current_ep, t0, gamma_max)
        else:
            l1 = self._batched_team_up(h1, h2, cs, current_ep, t0, gamma_max, batch_size)
            l2 = self._batched_team_up(h2, h1, cs, current_ep, t0, gamma_max, batch_size)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
    

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

# class GraphEncoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super(GraphEncoder, self).__init__()
#         self.conv = GCNConv(in_channels, hidden_channels, cached=False)
#         self.prelu = nn.PReLU(hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
#         self.prelu2 = nn.PReLU(hidden_channels)

#     def forward(self, x, edge_index):
#         x = self.conv(x, edge_index)
#         x = self.prelu(x)
#         x = self.conv2(x, edge_index)
#         x = self.prelu2(x)
#         return x

class Tacos_model(nn.Module):
    def __init__(self,Y,coord,n_list,D,regularization_acceleration=True,edge_subset_sz=1000000,spatial_regularization=0.9,latent_dimension=20,lamb=0.1,device='cpu',use_CSGCL=True,edge_list=None):
        super(Tacos_model,self).__init__()
        self.device = device
        self.Y = Y.to(device)

   
        self.n_list = n_list # cell number
        
        N1 = n_list[0]
        n_rest = 0
        for i in n_list[1:]:
            n_rest+=i
        N2 = n_rest
        
        self.N1 = N1
        self.N2 = N2
        
        # self.N2 = N2 # cell number
        self.D = D # gene number
        # self.mnn_graph = mnn_graph
        # self.none_zero_list =torch.nonzero(mnn_graph).to(device)
        self.lamb = lamb

        self.latent_dimention = latent_dimension
        self.device = device
        self.use_CSGCL = use_CSGCL
     
        self.spatial_regularization = spatial_regularization
        # self.balance_factor = balance_factor
        self.coord1 = coord[:N1,]
        self.coord2 = coord[N1:,]
        

        # if not self.spatial_regularization==0:#spaceflow constraint
        #     self.regularization_acceleration = regularization_acceleration


        self.regularization_acceleration = regularization_acceleration
        self.edge_subset_sz = edge_subset_sz

        # spatial_graph1 = graph_alpha(self.coord1)
        # spatial_graph2 = graph_alpha(self.coord2)


        self.coord= coord.to(device)
        self.coord1 = self.coord1.to(device)
        self.coord2 = self.coord2.to(device)
        
        if edge_list is None:
        
            n_total = sum(n_list)
            arr_total = np.zeros((n_total,n_total))

            flag = 0
            for n in n_list:
                cor = self.coord[flag:flag+n,].cpu()
                spatial_graph = graph_alpha(cor).toarray()
                arr_total[flag:flag+n,flag:flag+n] = spatial_graph[:n, :n]
                flag+=n



            # righttop = np.zeros((N1,N2))
            # leftbottem = np.zeros((N2,N1))
            # first_row = np.hstack((spatial_graph1.toarray(),righttop))  #横向合并
            # second_row = np.hstack((leftbottem,spatial_graph2.toarray()))  #横向合并
            # whole_graph = np.vstack((first_row,second_row))  #纵向合并

            self.spatial_graph = sparse.csr_matrix(arr_total)
            self.edge_list = sparse_mx_to_torch_edge_list(self.spatial_graph).to(device)
        else:
            self.edge_list = edge_list


        # csgcl hyperparam
        
        
        num_hidden = latent_dimension
        num_proj_hidden = 256
        tau = 0.6
        layer_num = 2
        self.ced_drop_rate_1 = 0.2
        self.ced_drop_rate_2 = 0.7
        self.cav_drop_rate_1 = 0.1
        self.cav_drop_rate_2 = 0.2
        self.ced_thr = 1.
        self.cav_thr = 1.
        self.t0 = 1000
        self.gamma = 10

        encoder = Encoder(self.D,
                          num_hidden,
                          torch.nn.PReLU(),
                          base_model=GCNConv,
                          k=layer_num).to(device)

        self.base_model = CSGCL(encoder,
                                num_hidden,
                                num_proj_hidden,
                                tau).to(device)
        
    def init_CSGCL_communities(self,node_cs,edge_weight,edge_list):
        self.node_cs = node_cs
        self.edge_weight = edge_weight
        self.edge_list = edge_list.to(self.device)
        
    def ced(self,edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        p: float,
        threshold: float = 1.) -> torch.Tensor:
        # print(1)
        edge_weight = edge_weight / edge_weight.mean() * (1. - p)
        edge_weight = edge_weight.where(edge_weight > (1. - threshold), torch.ones_like(edge_weight) * (1. - threshold))
        edge_weight = edge_weight.where(edge_weight < 1, torch.ones_like(edge_weight) * 1)
        
        sel_mask = torch.bernoulli(edge_weight).to(torch.bool)
        # print(sel_mask.shape)
        # print(edge_index.shape)
        return edge_index[:, sel_mask]

    def cav(self,feature: torch.Tensor,
            node_cs: np.ndarray,
            p: float,
            max_threshold: float = 0.7) -> torch.Tensor:
        x = feature.abs()
        device = feature.device
        w = x.t() @ torch.tensor(node_cs).to(device)
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


    def forward(self,epoch,sub_graph):
        # z, neg_z, summary = self.DGI_model(self.Y, self.edge_list)
        z = self.base_model(self.Y, self.edge_list)
        # print('csgcl')

        slice_list = []
        coord_list = []
        
        flag = 0
        for i in range(len(self.n_list)):
            slice_list.append(z[flag:flag+self.n_list[i],])
            coord_list.append(self.coord[flag:flag+self.n_list[i],])
            flag+=self.n_list[i]
        
        # for i in range(len(self.n_list)):
        #     if i ==0:
        #         slice_list.append(z[:self.n_list[i],])
        #     else:
        #         slice_list.append(z[self.n_list[i-1]:self.n_list[i],])
            
        # slicez1 = z[:self.N1,]
        # slicez2 = z[self.N1:,]


        if not self.spatial_regularization==0:#spaceflow constraint
            penalty=0
            if self.regularization_acceleration:
                for i in range(len(slice_list)):
                    slicez1 = slice_list[i]
                    coord1 = coord_list[i]
                    #slice1
                    cell_random_subset_11, cell_random_subset_21 = torch.randint(0, slicez1.shape[0], (self.edge_subset_sz,)).to(self.device), torch.randint(0, slicez1.shape[0], (self.edge_subset_sz,)).to(self.device)
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

                # slice2
                # cell_random_subset_12, cell_random_subset_22 = torch.randint(0, slicez2.shape[0], (self.edge_subset_sz,)).to(self.device), torch.randint(0, slicez2.shape[0], (self.edge_subset_sz,)).to(self.device)
                # z12, z22 = torch.index_select(slicez2, 0, cell_random_subset_12), torch.index_select(slicez2, 0, cell_random_subset_22)
                # c12, c22 = torch.index_select(self.coord2, 0, cell_random_subset_12), torch.index_select(self.coord2, 0,
                #                                                                                     cell_random_subset_12)
                # pdist2 = torch.nn.PairwiseDistance(p=2)

                # z_dists2 = pdist2(z12, z22)
                # z_dists2 = z_dists2 / torch.max(z_dists2)

                # sp_dists2 = pdist2(c12, c22)
                # sp_dists2 = sp_dists2 / torch.max(sp_dists2)
                # n_items2 = z_dists2.size(dim=0)
            else:
                for i in range(len(slice_list)):
                    slicez1 = slice_list[i]
                    coord1 = coord_list[i]
                    z_dists1 = torch.cdist(slicez1, slicez1, p=2)  ####### here should use z individually?????
                    z_dists1 = torch.div(z_dists1, torch.max(z_dists1)).to(self.device)
                    sp_dists1 = torch.cdist(coord1, coord1, p=2)
                    sp_dists1 = torch.div(sp_dists1, torch.max(sp_dists1)).to(self.device)
                    n_items1 = slicez1.size(dim=0) * slicez1.size(dim=0)
                    penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists1, sp_dists1)), n_items1).to(self.device)
                    penalty += penalty_1

                # z_dists2 = torch.cdist(slicez2, slicez2, p=2)  ####### here should use z individually?????
                # z_dists2 = torch.div(z_dists2, torch.max(z_dists2)).to(self.device)
                # sp_dists2 = torch.cdist(self.coord2, self.coord2, p=2)
                # sp_dists2 = torch.div(sp_dists2, torch.max(sp_dists2)).to(self.device)
                # n_items2 = slicez2.size(dim=0) * slicez2.size(dim=0)
            # penalty = torch.sum(torch.mul(1.0 - x_dist, self.sp_dists))
            # penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists1, sp_dists1)), n_items1).to(self.device)
            # penalty_2 = torch.div(torch.sum(torch.mul(1.0 - z_dists2, sp_dists2)), n_items2).to(self.device)
            # penalty = penalty_1+penalty_2

        # trip
        if not sub_graph==None:
            self.none_zero_list =torch.nonzero(sub_graph).to(self.device)
        if self.lamb!=0:
            slicez1 = z[:self.N1,]
            slicez2 = z[self.N1:,]
            
            a_idx = self.none_zero_list[:,0]
            p_idx = self.none_zero_list[:,1]
            n_idx = torch.randint(0, slicez1.shape[0], (a_idx.shape[0],)).to(self.device)
            
            a_slice,p_slice,n_slice = torch.index_select(slicez1,0,a_idx),torch.index_select(slicez2,0,p_idx),torch.index_select(slicez1,0,n_idx)
            # pdist_3 = torch.nn.PairwiseDistance(p=2)
            
            pdist3 = torch.nn.PairwiseDistance(p=2)
            p_dist,n_dist = pdist3(a_slice, p_slice),pdist3(a_slice, n_slice)
            cross_slice_total = torch.sum(torch.max(p_dist-n_dist+2.0,torch.zeros(p_dist.shape).to(self.device)))

            # # normal
            # a_idx = self.none_zero_list[:,0]
            # p_idx = self.none_zero_list[:,1]       
            # a_slice, p_slice = torch.index_select(slicez1,0,a_idx), torch.index_select(slicez2,0,p_idx)
            # pdist3 = torch.nn.PairwiseDistance(p=2)
            # p_dist= pdist3(a_slice, p_slice)
            # cross_slice_total = torch.sum(p_dist)


            
            
            
            cross_slice = torch.div(cross_slice_total,self.none_zero_list.shape[0])


        ### csgcl base loss
        if self.use_CSGCL:

            edge_index_1 = self.ced(self.edge_list, self.edge_weight, p=self.ced_drop_rate_1, threshold=self.ced_thr)
            edge_index_2 = self.ced(self.edge_list, self.edge_weight, p=self.ced_drop_rate_2, threshold=self.ced_thr)

            x1 = self.cav(self.Y, self.node_cs, self.cav_drop_rate_1, max_threshold=self.cav_thr)
            x2 = self.cav(self.Y, self.node_cs, self.cav_drop_rate_2, max_threshold=self.cav_thr)
            z1 = self.base_model(x1, edge_index_1)
            z2 = self.base_model(x2, edge_index_2)
        

            # gamma的值随着epoch而变化
            base_loss = self.base_model.team_up_loss(z1, z2,
                                cs=self.node_cs,
                                current_ep=epoch,
                                t0=self.t0,
                                gamma_max=self.gamma,
                                batch_size=None)
        else:
            base_loss = 0

        # dgi_loss = self.DGI_model.loss(z, neg_z, summary)
        str0 = f'base_loss: {base_loss:2f}, '
        
        
        loss1 =  base_loss
        if not self.spatial_regularization==0:#spaceflow constraint
            loss2 = loss1 + self.spatial_regularization*penalty
            str1 = f'penalty: {penalty:2f}, '
        else:
            loss2 = loss1
            str1=''
        
        if not self.lamb==0:
            loss = loss2+self.lamb*cross_slice
            str2 = f'cross_loss:{cross_slice:2f},cross_distance:{cross_slice_total:2f}'
        else:
            loss = loss2
            str2=''
        
        str_total = f'total: {loss:2f} '
        
        str_all = str_total+str0+str1+str2

        print(str_all)


        return loss,z