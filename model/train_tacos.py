import os
import torch
import random
import numpy as np
import anndata
import scipy.sparse as sp
from typing import Optional
from .tacos import TacosModel
from . import utils
# from .tacos import TacosModel
from scipy.sparse import csr_matrix

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import pickle
import time

from thop import profile


import anndata
import scipy.sparse as sp
from typing import Optional
from .tacos import TacosModel
from . import utils
# from .tacos import TacosModel
from scipy.sparse import csr_matrix

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import pickle
import time

from thop import profile



class Tacos(object):
    # def __init__(self,adata:anndata.AnnData,latent_dim=30,random_seed=42,gpu=0,com_detection='louvain',init_embedding = False,init_args:Optional[dict]=None,check_detect = False,path=None):
    def __init__(self,adata:anndata.AnnData,latent_dim=30,random_seed=42,gpu=0,com_detection='louvain',check_detect = False,path=None):
    # def __init__(self,adata:anndata.AnnData,latent_dim=30,random_seed=42,gpu=0,com_detection='louvain',init_embedding = False,init_args:Optional[dict]=None,check_detect = False,path=None):
    def __init__(self,adata:anndata.AnnData,latent_dim=30,random_seed=42,gpu=0,com_detection='louvain',check_detect = False,path=None):
        if not random_seed==None:
            self.random_seed = random_seed
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.device =  f"cuda:{gpu}" if (torch.cuda.is_available() and (not gpu is None)) else 'cpu'
        print(f'use device {self.device} to train...')
        
        save_dir = os.path.dirname(path)
        if not os.path.exists(path):
            os.makedirs(path)
        
        num_list = [i for i in adata.obs['batch'].value_counts()]
        self.latent_dim = latent_dim
        if isinstance(adata.X,csr_matrix):
            expr = adata.X.toarray()
        else:
            expr = adata.X
        X = torch.from_numpy(expr).to(torch.float32)
        
        self.X = X
        self.num_list = num_list
        self.adata = adata
        
        # use initial embedding to construct graph
        if  (check_detect and path is not None) and os.path.isfile(path+'edge_index.pt'):
            # graph_list = utils.construct_graph(adata.obsm['spatial'],num_list)
            edge_index = torch.load(path+'edge_index.pt')
        else:
            print('building graphs based on coordinate...')
            graph_list = utils.construct_graph(adata.obsm['spatial'],num_list)
            # graph = utils.merge_csr_graphs(graph_list)
            # edge_index = utils.adj_to_edge_index(graph)
            edge_index_list = [utils.adj_to_edge_index(i) for i in graph_list]
            edge_index = edge_index_list
            # edge_index = utils.adj_to_edge_index(graph)
            
            if check_detect and path is not None:
                torch.save(edge_index, path+'edge_index.pt')
                print('edge_index saved')

        
        if (check_detect and path is not None) and os.path.isfile(path+'edge_weight.pt'):
            edge_weight = torch.load(path+'edge_weight.pt')
            with open(path+'node_cs.pkl', 'rb') as f:
                node_cs = pickle.load(f)
            # node_cs = np.load(path+'node_cs.npy')
            print('load nodecs, edge_weight saved')
        else:
            edge_weight= []
            node_cs = []
            assert len(edge_index) == len(num_list)
            flag = 0
            for i in range(len(edge_index)):
                ew,nc = utils.community_augmentation(X[flag:flag+num_list[i],],edge_index[i],com_detection)
                edge_weight.append(ew)
                node_cs.append(nc)
                flag+=num_list[i]
            
            # edge_weight, node_cs = utils.community_augmentation(X, edge_index, com_detection)
            if check_detect and path is not None:
                torch.save(edge_weight, path+'edge_weight.pt')
                with open(path+'node_cs.pkl', 'wb') as f:
                    pickle.dump(node_cs, f)
                # np.save(path+'node_cs.npy', node_cs)
                print('node_cs, edge_weight saved')
        # if init_embedding:
        #     assert not init_args is None,'lack initial args'
        #     self.init_embedding,_ = self.runner(edge_index,edge_weight,node_cs,init_args,path)
            
        #     print('Building graphs based on initial embedding...')
        #     graph_list_ = utils.construct_graph(self.init_embedding,num_list)
        #     graph_ = utils.merge_csr_graphs(graph_list_)
        #     edge_index_ = utils.adj_to_edge_index(graph_).to(self.device)
        #     graph_weighted_ = graph_ + sp.eye(int(X.shape[0]))
        #     graph_weighted_ = utils.normalize_adj(graph_weighted_)  # coo
        #     sp_weighted_graph_ = utils.sparse_mx_to_torch_sparse_tensor(graph_weighted_).to(self.device)
        #     # self.graph  = sp_weighted_graph_
        #     edge_weight_, node_cs_ = utils.community_augmentation(self.init_embedding, edge_index_, com_detection)
        #     self.edge_index = edge_index_
        #     self.edge_weight = edge_weight_
        #     self.node_cs = node_cs_
            
        # else:
            
        self.init_embedding  = None
        self.edge_index = edge_index # device
        
        self.device =  f"cuda:{gpu}" if (torch.cuda.is_available() and (not gpu is None)) else 'cpu'
        print(f'use device {self.device} to train...')
        
        save_dir = os.path.dirname(path)
        if not os.path.exists(path):
            os.makedirs(path)
        
        num_list = [i for i in adata.obs['batch'].value_counts()]
        self.latent_dim = latent_dim
        if isinstance(adata.X,csr_matrix):
            expr = adata.X.toarray()
        else:
            expr = adata.X
        X = torch.from_numpy(expr).to(torch.float32)
        
        self.X = X
        self.num_list = num_list
        self.adata = adata
        
        # use initial embedding to construct graph
        if  (check_detect and path is not None) and os.path.isfile(path+'edge_index.pt'):
            # graph_list = utils.construct_graph(adata.obsm['spatial'],num_list)
            edge_index = torch.load(path+'edge_index.pt')
        else:
            print('building graphs based on coordinate...')
            graph_list = utils.construct_graph(adata.obsm['spatial'],num_list)
            # graph = utils.merge_csr_graphs(graph_list)
            # edge_index = utils.adj_to_edge_index(graph)
            edge_index_list = [utils.adj_to_edge_index(i) for i in graph_list]
            edge_index = edge_index_list
            # edge_index = utils.adj_to_edge_index(graph)
            
            if check_detect and path is not None:
                torch.save(edge_index, path+'edge_index.pt')
                print('edge_index saved')

        
        if (check_detect and path is not None) and os.path.isfile(path+'edge_weight.pt'):
            edge_weight = torch.load(path+'edge_weight.pt')
            with open(path+'node_cs.pkl', 'rb') as f:
                node_cs = pickle.load(f)
            # node_cs = np.load(path+'node_cs.npy')
            print('load nodecs, edge_weight saved')
        else:
            edge_weight= []
            node_cs = []
            assert len(edge_index) == len(num_list)
            flag = 0
            for i in range(len(edge_index)):
                ew,nc = utils.community_augmentation(X[flag:flag+num_list[i],],edge_index[i],com_detection)
                edge_weight.append(ew)
                node_cs.append(nc)
                flag+=num_list[i]
            
            # edge_weight, node_cs = utils.community_augmentation(X, edge_index, com_detection)
            if check_detect and path is not None:
                torch.save(edge_weight, path+'edge_weight.pt')
                with open(path+'node_cs.pkl', 'wb') as f:
                    pickle.dump(node_cs, f)
                # np.save(path+'node_cs.npy', node_cs)
                print('node_cs, edge_weight saved')
        # if init_embedding:
        #     assert not init_args is None,'lack initial args'
        #     self.init_embedding,_ = self.runner(edge_index,edge_weight,node_cs,init_args,path)
            
        #     print('Building graphs based on initial embedding...')
        #     graph_list_ = utils.construct_graph(self.init_embedding,num_list)
        #     graph_ = utils.merge_csr_graphs(graph_list_)
        #     edge_index_ = utils.adj_to_edge_index(graph_).to(self.device)
        #     graph_weighted_ = graph_ + sp.eye(int(X.shape[0]))
        #     graph_weighted_ = utils.normalize_adj(graph_weighted_)  # coo
        #     sp_weighted_graph_ = utils.sparse_mx_to_torch_sparse_tensor(graph_weighted_).to(self.device)
        #     # self.graph  = sp_weighted_graph_
        #     edge_weight_, node_cs_ = utils.community_augmentation(self.init_embedding, edge_index_, com_detection)
        #     self.edge_index = edge_index_
        #     self.edge_weight = edge_weight_
        #     self.node_cs = node_cs_
            
        # else:
            
        self.init_embedding  = None
        self.edge_index = edge_index # device
        self.edge_weight = edge_weight
        self.node_cs = node_cs

        print('initial community finished!')
    
    def train(self,args,embedding_save_filepath="./data/08_75/results/"):
        self.embedding,self.model,self.gene = self.runner(self.edge_index,self.edge_weight,self.node_cs,args,embedding_save_filepath)
        param_str = f"csgcl{args['base_w']}spatial{args['spatial_w']}cross{args['cross_w']}_recon{args['recon_w']}/"
        save_dir = os.path.dirname(embedding_save_filepath+param_str)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        
        
        file_path = f'{param_str}'
        flag = 1
        
        while os.path.exists(save_dir+'/embedding_'+str(flag)+'.csv'):
            # file_path = file_path+str(flag)
            flag+=1
        np.savetxt(save_dir+'/embedding_'+str(flag)+'.csv', self.embedding[:, :], delimiter="\t")
        torch.save(self.model.state_dict(), save_dir+'/model_'+str(flag)+'.pth')
        print(f"Training complete!\nEmbedding is saved at {embedding_save_filepath}")
        return file_path+str(flag)

    
    def runner(self,edge_index,edge_weight,node_cs,args,path=None,):
        
        coor = torch.from_numpy(self.adata.obsm['spatial']).to(torch.float32).to(self.device)
        coor_tensors =  torch.split(coor, self.num_list)
        
        model = TacosModel(self.X.shape[1],self.latent_dim,self.device,base =args['base']).to(self.device)
        
        
            
        print(f"start training embedding for epoch:{args['epoch']}")
        
        self.X = self.X.to(self.device)
        edge_index = edge_index
        file_path = f'{param_str}'
        flag = 1
        
        while os.path.exists(save_dir+'/embedding_'+str(flag)+'.csv'):
            # file_path = file_path+str(flag)
            flag+=1
        np.savetxt(save_dir+'/embedding_'+str(flag)+'.csv', self.embedding[:, :], delimiter="\t")
        torch.save(self.model.state_dict(), save_dir+'/model_'+str(flag)+'.pth')
        print(f"Training complete!\nEmbedding is saved at {embedding_save_filepath}")
        return file_path+str(flag)

    
    def runner(self,edge_index,edge_weight,node_cs,args,path=None,):
        
        coor = torch.from_numpy(self.adata.obsm['spatial']).to(torch.float32).to(self.device)
        coor_tensors =  torch.split(coor, self.num_list)
        
        model = TacosModel(self.X.shape[1],self.latent_dim,self.device,base =args['base']).to(self.device)
        
        
            
        print(f"start training embedding for epoch:{args['epoch']}")
        
        self.X = self.X.to(self.device)
        edge_index = edge_index
        
        model.train()
        
        
        
        
        min_loss = np.inf
        min_epoch = 0
        patience = 0
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
        
        best_params = model.state_dict()
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
        
        best_params = model.state_dict()
        embedding = None
        
        if args['cross_w'] !=0:
        
        if args['cross_w'] !=0:
            print('start mnn calculating')
            mnn_dict = utils.update_mnn(self.adata,self.num_list,embedding,path,args['k'],cpu=args['cpu'])
            # sub_graph = torch.from_numpy(utils.update_mnn(self.adata,self.num_list,embedding,path,args['k'],cpu=args['cpu'])).to(torch.float32)
            # mnn_graph_list = torch.nonzero(sub_graph).to(self.device)
            a,p,n = utils.get_triplet(self.adata,mnn_dict=mnn_dict)
            mnn_dict = utils.update_mnn(self.adata,self.num_list,embedding,path,args['k'],cpu=args['cpu'])
            # sub_graph = torch.from_numpy(utils.update_mnn(self.adata,self.num_list,embedding,path,args['k'],cpu=args['cpu'])).to(torch.float32)
            # mnn_graph_list = torch.nonzero(sub_graph).to(self.device)
            a,p,n = utils.get_triplet(self.adata,mnn_dict=mnn_dict)
            print('mnn calculated!')
        
        if args['base']=='graphcl':
            b_xent = torch.nn.BCEWithLogitsLoss()
            flag = 0
            aug_feture_list=[]
            aug_adj_list=[]
            for i in range(len(self.num_list)):
                # print(i)
                aug_features1, aug_adj1 = utils.aug_drop_node(self.X[flag:flag+self.num_list[i],].cpu(),edge_index[i].cpu(), drop_percent=0.1)
                aug_features2, aug_adj2 = utils.aug_drop_node(self.X[flag:flag+self.num_list[i],].cpu(),edge_index[i].cpu(), drop_percent=0.1)
                aug_feture_list.append((aug_features1.to(self.device),aug_features2.to(self.device)))
                aug_adj_list.append((aug_adj1.to(self.device),aug_adj2.to(self.device)))
                flag+=self.num_list[i]
        
        time1 = time.time()
        for epoch in range(args['epoch']):
        
        if args['base']=='graphcl':
            b_xent = torch.nn.BCEWithLogitsLoss()
            flag = 0
            aug_feture_list=[]
            aug_adj_list=[]
            for i in range(len(self.num_list)):
                # print(i)
                aug_features1, aug_adj1 = utils.aug_drop_node(self.X[flag:flag+self.num_list[i],].cpu(),edge_index[i].cpu(), drop_percent=0.1)
                aug_features2, aug_adj2 = utils.aug_drop_node(self.X[flag:flag+self.num_list[i],].cpu(),edge_index[i].cpu(), drop_percent=0.1)
                aug_feture_list.append((aug_features1.to(self.device),aug_features2.to(self.device)))
                aug_adj_list.append((aug_adj1.to(self.device),aug_adj2.to(self.device)))
                flag+=self.num_list[i]
        
        time1 = time.time()
        for epoch in range(args['epoch']):
            train_loss = 0.0
            torch.set_grad_enabled(True)
            optimizer.zero_grad()
            temp_z = []
            flag = 0
            total_macs = 0
            total_params = 0
            if args['base']=='csgcl':
                for i in range(len(self.num_list)):
                    temp_z.append(model(self.X[flag:flag+self.num_list[i],],edge_index[i].to(self.device)))
                    flag+=self.num_list[i]
            elif args['base']=='graphcl':
                for i in range(len(self.num_list)):
                    temp_z.append(model(self.X[flag:flag+self.num_list[i],],edge_index[i].to(self.device)))
                    flag+=self.num_list[i]
            elif args['base']=='gcn':
                for i in range(len(self.num_list)):
                    temp_z.append(model(self.X[flag:flag+self.num_list[i],],edge_index[i].to(self.device)))
                    flag+=self.num_list[i]
            else:
                print('not implement')
                assert False
            loss = 0
            str_msg = ''
            
            if args['base_w'] != 0:
                if args['base']=='csgcl':
                    loss_csgcl = 0
                    flag = 0
                    for i in range(len(self.num_list)):
                        loss_csgcl+= model.csgcl_loss(self.X[flag:flag+self.num_list[i],],
                                                    edge_index[i].to(self.device),
                                                    edge_weight[i].to(self.device),
                                                    torch.from_numpy(node_cs[i]).to(self.device),
                                                    epoch, **args['csgcl_arg'])
                        flag+=self.num_list[i]
                    loss+=args['base_w']*loss_csgcl
                    str_msg+=f'csgcl_loss: {loss_csgcl:2f}, '
                elif args['base']=='graphcl':
                    loss_graphcl = 0
                    flag = 0
                    for i in range(len(self.num_list)):
                        features = self.X[flag:flag+self.num_list[i],]
                        nb_nodes = features.shape[0]
                        features = features.unsqueeze(0)
                        idx = np.random.permutation(nb_nodes)
                        shuf_fts = features[:,idx, :]
                        lbl_1 = torch.ones(1, nb_nodes)
                        lbl_2 = torch.zeros(1, nb_nodes)
                        lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)
                        logits = model.base_model.loss(features,shuf_fts,aug_feture_list[i][0],aug_feture_list[i][1],
                                                       edge_index[i].to(self.device),aug_adj_list[i][0],aug_adj_list[i][0])
                        loss_graphcl+=b_xent(logits, lbl)
                    loss+=args['base_w']*loss_graphcl
                    str_msg+=f'graphcl_loss: {loss_graphcl:2f},'

                elif args['base']=='gcn':
                    loss_gcn = 0
                else:
                    print('not implement')
                    assert False
                    
                    
            if args['cross_w'] != 0:
                if epoch%args['update_mnn']==0 and epoch!=0:
                    self.adata.obsm['Agg'] = embedding
                    time3 = time.time()
                    assert not embedding is None
                    mnn_dict = utils.update_mnn(self.adata,self.num_list,embedding,path,args['k'],cpu=args['cpu'])
                    a,p,n = utils.get_triplet(self.adata,mnn_dict=mnn_dict)

                    time4 = time.time()
                    print(f'update mnn! time:{str(time4-time3)}')
                loss_cross = model._cross_loss(temp_z,a,p,n,**args['cross_arg'])
                
                # loss_cross = model.cross_loss(temp_z,mnn_dict,**args['cross_arg'])
                # loss_cross = model.cross_loss(temp_z,mnn_dict,**args['cross_arg'])
                loss+= args['cross_w']*loss_cross
                str_msg+=f'cross_loss: {loss_cross:2f}, '
                
            if args['spatial_w'] !=0:
                loss_spatial = 0
                
                for i in range(len(temp_z)):
                    loss_spatial+= model.spatial_loss(temp_z[i],coor_tensors[i],**args['spatial_arg'])
                # loss_spatial = model.spatial_loss(z_tensors,coor_tensors,**args['spatial_arg'])
                loss+= args['spatial_w']*loss_spatial
                str_msg+=f'spatial_loss: {loss_spatial:2f}.'
            
            if args['recon_w'] !=0:
                loss_recon,self.recon_x = model.reconstr_loss(torch.cat(temp_z, dim=0).to(self.device),torch.cat(self.edge_index,dim=-1).to(self.device),self.X)
                loss += args['recon_w']*loss_recon
                str_msg+=f'recon_loss: {loss_recon:2f}.'
                
                
                
            temp_z = []
            flag = 0
            total_macs = 0
            total_params = 0
            if args['base']=='csgcl':
                for i in range(len(self.num_list)):
                    temp_z.append(model(self.X[flag:flag+self.num_list[i],],edge_index[i].to(self.device)))
                    flag+=self.num_list[i]
            elif args['base']=='graphcl':
                for i in range(len(self.num_list)):
                    temp_z.append(model(self.X[flag:flag+self.num_list[i],],edge_index[i].to(self.device)))
                    flag+=self.num_list[i]
            elif args['base']=='gcn':
                for i in range(len(self.num_list)):
                    temp_z.append(model(self.X[flag:flag+self.num_list[i],],edge_index[i].to(self.device)))
                    flag+=self.num_list[i]
            else:
                print('not implement')
                assert False
            loss = 0
            str_msg = ''
            
            if args['base_w'] != 0:
                if args['base']=='csgcl':
                    loss_csgcl = 0
                    flag = 0
                    for i in range(len(self.num_list)):
                        loss_csgcl+= model.csgcl_loss(self.X[flag:flag+self.num_list[i],],
                                                    edge_index[i].to(self.device),
                                                    edge_weight[i].to(self.device),
                                                    torch.from_numpy(node_cs[i]).to(self.device),
                                                    epoch, **args['csgcl_arg'])
                        flag+=self.num_list[i]
                    loss+=args['base_w']*loss_csgcl
                    str_msg+=f'csgcl_loss: {loss_csgcl:2f}, '
                elif args['base']=='graphcl':
                    loss_graphcl = 0
                    flag = 0
                    for i in range(len(self.num_list)):
                        features = self.X[flag:flag+self.num_list[i],]
                        nb_nodes = features.shape[0]
                        features = features.unsqueeze(0)
                        idx = np.random.permutation(nb_nodes)
                        shuf_fts = features[:,idx, :]
                        lbl_1 = torch.ones(1, nb_nodes)
                        lbl_2 = torch.zeros(1, nb_nodes)
                        lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)
                        logits = model.base_model.loss(features,shuf_fts,aug_feture_list[i][0],aug_feture_list[i][1],
                                                       edge_index[i].to(self.device),aug_adj_list[i][0],aug_adj_list[i][0])
                        loss_graphcl+=b_xent(logits, lbl)
                    loss+=args['base_w']*loss_graphcl
                    str_msg+=f'graphcl_loss: {loss_graphcl:2f},'

                elif args['base']=='gcn':
                    loss_gcn = 0
                else:
                    print('not implement')
                    assert False
                    
                    
            if args['cross_w'] != 0:
                if epoch%args['update_mnn']==0 and epoch!=0:
                    self.adata.obsm['Agg'] = embedding
                    time3 = time.time()
                    assert not embedding is None
                    mnn_dict = utils.update_mnn(self.adata,self.num_list,embedding,path,args['k'],cpu=args['cpu'])
                    a,p,n = utils.get_triplet(self.adata,mnn_dict=mnn_dict)

                    time4 = time.time()
                    print(f'update mnn! time:{str(time4-time3)}')
                loss_cross = model._cross_loss(temp_z,a,p,n,**args['cross_arg'])
                
                # loss_cross = model.cross_loss(temp_z,mnn_dict,**args['cross_arg'])
                # loss_cross = model.cross_loss(temp_z,mnn_dict,**args['cross_arg'])
                loss+= args['cross_w']*loss_cross
                str_msg+=f'cross_loss: {loss_cross:2f}, '
                
            if args['spatial_w'] !=0:
                loss_spatial = 0
                
                for i in range(len(temp_z)):
                    loss_spatial+= model.spatial_loss(temp_z[i],coor_tensors[i],**args['spatial_arg'])
                # loss_spatial = model.spatial_loss(z_tensors,coor_tensors,**args['spatial_arg'])
                loss+= args['spatial_w']*loss_spatial
                str_msg+=f'spatial_loss: {loss_spatial:2f}.'
            
            if args['recon_w'] !=0:
                loss_recon,self.recon_x = model.reconstr_loss(torch.cat(temp_z, dim=0).to(self.device),torch.cat(self.edge_index,dim=-1).to(self.device),self.X)
                loss += args['recon_w']*loss_recon
                str_msg+=f'recon_loss: {loss_recon:2f}.'
                
                
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            
            
            
            if train_loss > min_loss:
                patience += 1
            else:
                # final_z = temp_z
                embedding = torch.cat(temp_z, dim=0).cpu().detach().numpy()
                # embedding = embedding
                min_epoch = epoch
                patience = 0
                min_loss = train_loss
                best_params = model.state_dict()
            if (epoch+1) % args['save_inter'] == 0:
                torch.save(best_params, path+'/model_'+str(epoch+1)+'.pth')
            if (epoch+1) % args['save_inter'] == 0:
                torch.save(best_params, path+'/model_'+str(epoch+1)+'.pth')
            if (epoch+1) % 10 == 0:
                encoder_parms=0
                total_params = sum(p.numel() for p in model.parameters())

                # print("Number of encoder mlp parameters:",total_params)
                
                
                time2 = time.time()
                print(str_msg)
                print(f"Epoch {epoch + 1}/{args['epoch']}, Loss: {train_loss:2f}, from min_epoch:{epoch-min_epoch}-------")
                # print(f'FLOPs: {total_macs}; params:{total_params}; time:{str(time2-time1)}')
                # print('=============================================')
                time1 = time2
            if patience >args['max_patience'] and epoch > args['min_stop']:
                print(f'early stop!{patience}')
                encoder_parms=0
                total_params = sum(p.numel() for p in model.parameters())

                # print("Number of encoder mlp parameters:",total_params)
                
                
                time2 = time.time()
                print(str_msg)
                print(f"Epoch {epoch + 1}/{args['epoch']}, Loss: {train_loss:2f}, from min_epoch:{epoch-min_epoch}-------")
                # print(f'FLOPs: {total_macs}; params:{total_params}; time:{str(time2-time1)}')
                # print('=============================================')
                time1 = time2
            if patience >args['max_patience'] and epoch > args['min_stop']:
                print(f'early stop!{patience}')
                break
        
        # get reconstructed gene expression
        edge_index = torch.cat(self.edge_index,dim = 1).to(self.device)
        embed_tensor = torch.from_numpy(embedding).to(self.device)
        for i in range(100):
            # gene = []
            train_loss = 0.0
            torch.set_grad_enabled(True)
            optimizer.zero_grad()
            # edge_index = torch.cat(self.edge_index,dim = 0)
            # gene_expr = model.decoder(embedding,self.edge_index.to(self.device))
            loss,gene_expr = model.reconstr_loss(embed_tensor,edge_index,self.X)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.load_state_dict(best_params)
        return embedding,model,gene_expr.cpu().detach().numpy()