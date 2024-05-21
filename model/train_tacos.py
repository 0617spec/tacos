import os
import torch
import random
import numpy as np

from .utils import build_graph_G,get_node_cs_edge_weight,update_mnn
from .tacos import Tacos_model

class Tacos(object):
    def __init__(self,adata,expr_data,cor_cat,n_list,latent_dim,random_seed=42,regularization_acceleration=True,spatial_regularization_strength=1.0,lamb=1.0,gpu=0,epochs=500,edge_subset_sz=1000000,use_partialOT=False,use_CSGCL=True,init_epoch = 500,lr=1e-3,init_embedding = False):
        if not random_seed==None:
            self.random_seed = random_seed
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)
        self.adata = adata
        self.expr_data = expr_data
        # self.coord = coord
        self.n_list = n_list
        self.latent_dim = latent_dim
        # self.mnn_graph = mnn_graph
        self.coord = torch.from_numpy(cor_cat).to(torch.float32)
        #G,edge_list = build_graph_G(adata.obsm['X_pca'],n_list)
        
        
        
        device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
        if gpu is None:
            device = 'cpu'
        # device = 'cpu'
        print(device)
        self.device = device
        
        
        # preprocess data to get initial embedding (use spaceflow only?)
        if init_embedding:
            
            model = Tacos_model(self.expr_data,self.coord,self.n_list,self.expr_data.shape[1],regularization_acceleration,edge_subset_sz,spatial_regularization_strength,self.latent_dim,lamb,device,use_CSGCL=False)
            print(f'strat training init embedding for epoch:{init_epoch}')
            model.train()
            
            
            min_loss = np.inf
            min_epoch = 0
            patience = 0
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            best_params = model.state_dict()
            # torch.set_default_tensor_type(torch.DoubleTensor)
            
            embedding = None
            
            if lamb !=0:
                print('start mnn calculating')
                sub_graph = torch.from_numpy(update_mnn(self.adata,self.n_list,embedding,use_partialOT=False)).to(torch.float32).to(device)
                print('mnn calculated!')
            else:
                sub_graph=None
                
            for epoch in range(epochs):
                if not use_partialOT:
                    if epoch%100==0 and epoch!=0:
                        if lamb !=0:
                            sub_graph = torch.from_numpy(update_mnn(self.adata,self.n_list,embedding,use_partialOT=False)).to(torch.float32).to(device)
                            print('update mnn!')
                            if embedding is None:
                                print('there is something wrong')
                                
                train_loss = 0.0
                torch.set_grad_enabled(True)
                optimizer.zero_grad()
                # print(epoch)
                loss,temp_z = model(epoch,sub_graph)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if train_loss > min_loss:
                    patience += 1
                else:
                    embedding = temp_z.cpu().detach().numpy()
                    embedding = embedding
                    min_epoch = epoch
                    patience = 0
                    min_loss = train_loss
                    self.best_params = model.state_dict()
                if (epoch+1) % 10 == 0:
                    
                    print(f"InitEpoch {epoch + 1}/{epochs}, Loss: {str(train_loss)}, from min_epoch:{epoch-min_epoch}")
            
            G,edge_list = build_graph_G(embedding,n_list,use_cor = False)
        else:
            G,edge_list = build_graph_G(cor_cat,n_list)
            self.best_params = None
        edge_weight,node_cs = get_node_cs_edge_weight(G,edge_list)
        self.edge_weight = edge_weight
        self.node_cs = node_cs
        self.edge_list = edge_list
        

    def train(self, embedding_save_filepath="./data/08_75/results/",start_epoch = 0, spatial_regularization_strength=0.9,lamb=0.1,use_partialOT=False,lr=1e-3, epochs=500, max_patience=100, min_stop=100,check_inter=50, regularization_acceleration=True, edge_subset_sz=1000000,use_CSGCL=True):
        # if not random_seed==None:
        #     torch.manual_seed(random_seed)
        #     random.seed(random_seed)
        #     np.random.seed(random_seed)
        save_dir = os.path.dirname(embedding_save_filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        
        print(self.device)
        device = self.device
        model = Tacos_model(self.expr_data,self.coord,self.n_list,self.expr_data.shape[1],regularization_acceleration,edge_subset_sz,spatial_regularization_strength,self.latent_dim,lamb,device,use_CSGCL,edge_list = self.edge_list)
        model.init_CSGCL_communities(self.node_cs,self.edge_weight,self.edge_list)
        print(f'start epoch:{start_epoch}')
        if not start_epoch==0:
            model_path = f'{embedding_save_filepath}model_epoch{start_epoch}.pth'
            print(model_path)
            if os.path.exists(model_path):
                print('model_exist!')
                model.load_state_dict(torch.load(model_path))
                print('model loaded')
            else:
                print('Failed')
        
        
        model.train()
        min_loss = np.inf
        min_epoch = 0
        patience = 0
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if not self.best_params is None:
            best_params = self.best_params
        else:    
            best_params = model.state_dict()
        # torch.set_default_tensor_type(torch.DoubleTensor)
        model.load_state_dict(best_params)
        
        embedding = None
        if not start_epoch==0:
             z = model.base_model(model.Y, model.edge_list)
             embedding = z.cpu().detach().numpy()
        if lamb !=0: 
            print('start mnn calculating')
            sub_graph = torch.from_numpy(update_mnn(self.adata,self.n_list,embedding,use_partialOT=use_partialOT)).to(torch.float32).to(device)
            print('mnn calculated!')
        else:
            sub_graph=None
        
        
        for epoch in range(start_epoch,epochs):
            if not use_partialOT:
                if epoch%100==0 and epoch!=0:
                    if lamb !=0:
                        sub_graph = torch.from_numpy(update_mnn(self.adata,self.n_list,embedding,use_partialOT=use_partialOT)).to(torch.float32).to(device)
                        print('update mnn!')
                        if embedding is None:
                            print('there is something wrong')
                
            train_loss = 0.0
            torch.set_grad_enabled(True)
            optimizer.zero_grad()
            # print(epoch)
            loss,temp_z = model(epoch,sub_graph)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (epoch+1)%check_inter==0:
                embedding_temp=temp_z.cpu().detach().numpy()
                np.savetxt(f'{embedding_save_filepath}embedding_epoch{epoch+1}.csv', embedding_temp, delimiter="\t")
                torch.save(model.state_dict(), f'{embedding_save_filepath}model_epoch{epoch+1}.pth')
            if train_loss > min_loss:
                patience += 1
            else:
                # if (epoch+1)>100:
                #     embedding_temp=temp_z.cpu().detach().numpy()
                #     np.savetxt(f'{embedding_save_filepath}embedding_epoch{epoch+1}_temp_min.csv', embedding_temp, delimiter="\t")
                embedding = temp_z.cpu().detach().numpy()
                min_epoch = epoch
                patience = 0
                min_loss = train_loss
                best_params = model.state_dict()
            if (epoch+1) % 10 == 0:
                
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {str(train_loss)}, from min_epoch:{epoch-min_epoch}")
            if patience > max_patience and epoch > min_stop:
                break
                
        
        model.load_state_dict(best_params)
        z = model.base_model(model.Y, model.edge_list)
        embedding = z.cpu().detach().numpy()
        
        np.savetxt(f'{embedding_save_filepath}embedding_final_epoch{min_epoch+1}_seed{self.random_seed}.csv', embedding[:, :], delimiter="\t")
        print(f"Training complete!\nEmbedding is saved at {embedding_save_filepath}")

        self.embedding = embedding
        return embedding