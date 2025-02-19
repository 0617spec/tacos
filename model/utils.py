from sklearn.neighbors import NearestNeighbors,kneighbors_graph
import networkx as nx
import gudhi
from cdlib import algorithms
from cdlib.utils import convert_graph_formats
from typing import Sequence,Optional
import numpy as np
import torch
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import anndata as ad
import scanpy as sc
import pandas as pd
from annoy import AnnoyIndex
import hnswlib
import os
from joblib import Parallel, delayed
import itertools
import random
from typing import List

from sklearn.neighbors import NearestNeighbors,kneighbors_graph
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import anndata as ad
import scanpy as sc
import pandas as pd
from annoy import AnnoyIndex
import hnswlib
import os
from joblib import Parallel, delayed
import itertools
import random
from typing import List

from sklearn.neighbors import NearestNeighbors,kneighbors_graph
import scipy

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

def mclust_R(adata, num_cluster, modelNames='EII', used_obsm='spamc', random_seed=42):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    # import rpy2.robjects as robjects
    robjects.r.library("mclust")

    # import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    
def match_cluster_labels(true_labels,est_labels):
    true_labels_arr = np.array(list(true_labels))
    est_labels_arr = np.array(list(est_labels))
    org_cat = list(np.sort(list(pd.unique(true_labels))))
    est_cat = list(np.sort(list(pd.unique(est_labels))))
    B = nx.Graph()
    B.add_nodes_from([i+1 for i in range(len(org_cat))], bipartite=0)
    B.add_nodes_from([-j-1 for j in range(len(est_cat))], bipartite=1)
    for i in range(len(org_cat)):
        for j in range(len(est_cat)):
            weight = np.sum((true_labels_arr==org_cat[i])* (est_labels_arr==est_cat[j]))
            B.add_edge(i+1,-j-1, weight=-weight)
    match = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)
#     match = minimum_weight_full_matching(B)
    if len(org_cat)>=len(est_cat):
        return np.array([match[-est_cat.index(c)-1]-1 for c in est_labels_arr])
    else:
        unmatched = [c for c in est_cat if not (-est_cat.index(c)-1) in match.keys()]
        l = []
        for c in est_labels_arr:
            if (-est_cat.index(c)-1) in match: 
                l.append(match[-est_cat.index(c)-1]-1)
            else:
                l.append(len(org_cat)+unmatched.index(c))
        return np.array(l)

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

def mclust_R(adata, num_cluster, modelNames='EII', used_obsm='spamc', random_seed=42):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    # import rpy2.robjects as robjects
    robjects.r.library("mclust")

    # import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    
def match_cluster_labels(true_labels,est_labels):
    true_labels_arr = np.array(list(true_labels))
    est_labels_arr = np.array(list(est_labels))
    org_cat = list(np.sort(list(pd.unique(true_labels))))
    est_cat = list(np.sort(list(pd.unique(est_labels))))
    B = nx.Graph()
    B.add_nodes_from([i+1 for i in range(len(org_cat))], bipartite=0)
    B.add_nodes_from([-j-1 for j in range(len(est_cat))], bipartite=1)
    for i in range(len(org_cat)):
        for j in range(len(est_cat)):
            weight = np.sum((true_labels_arr==org_cat[i])* (est_labels_arr==est_cat[j]))
            B.add_edge(i+1,-j-1, weight=-weight)
    match = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)
#     match = minimum_weight_full_matching(B)
    if len(org_cat)>=len(est_cat):
        return np.array([match[-est_cat.index(c)-1]-1 for c in est_labels_arr])
    else:
        unmatched = [c for c in est_cat if not (-est_cat.index(c)-1) in match.keys()]
        l = []
        for c in est_labels_arr:
            if (-est_cat.index(c)-1) in match: 
                l.append(match[-est_cat.index(c)-1]-1)
            else:
                l.append(len(org_cat)+unmatched.index(c))
        return np.array(l)

def batch_entropy_mixing_score(data, batches, n_neighbors=100, n_pools=100, n_samples_per_pool=100):
    """
    Calculate batch entropy mixing score

    Algorithm
    ---------
         * 1. Calculate the regional mixing entropies at the location of 100 randomly chosen cells from all batches
         * 2. Define 100 nearest neighbors for each randomly chosen cell
         * 3. Calculate the mean mixing entropy as the mean of the regional entropies
         * 4. Repeat above procedure for 100 iterations with different randomly chosen cells.
     
     Parameters
    ----------
    data
        np.array of shape nsamples x nfeatures.
    batches
        batch labels of nsamples.
    n_neighbors
        The number of nearest neighbors for each randomly chosen cell. By default, n_neighbors=100.
    n_samples_per_pool
        The number of randomly chosen cells from all batches per iteration. By default, n_samples_per_pool=100.
    n_pools
        The number of iterations with different randomly chosen cells. By default, n_pools=100.

    Returns
    -------
    Batch entropy mixing score
    """
#     print("Start calculating Entropy mixing score")
    def entropy(batches):
        p = np.zeros(N_batches)
        adapt_p = np.zeros(N_batches)
        a = 0
        for i in range(N_batches):
            p[i] = np.mean(batches == batches_[i])
            a = a + p[i]/P[i]
        entropy = 0
        for i in range(N_batches):
            adapt_p[i] = (p[i]/P[i])/a
            entropy = entropy - adapt_p[i]*np.log(adapt_p[i]+10**-8)
        return entropy

    n_neighbors = min(n_neighbors, len(data) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])

    score = 0
    batches_ = np.unique(batches)
    N_batches = len(batches_)
    if N_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")
    P = np.zeros(N_batches)
    for i in range(N_batches):
            P[i] = np.mean(batches == batches_[i])
    for t in range(n_pools):
        indices = np.random.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
        score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]
                                          [kmatrix[indices].nonzero()[0] == i]])
                          for i in range(n_samples_per_pool)])
    Score = score / float(n_pools)
    return Score / float(np.log2(N_batches))


def calc_domainAveraged_FOSCTTM(x1_mat, x2_mat):
    fracs1,xs = calc_frac_idx(x1_mat, x2_mat)
    fracs2,xs = calc_frac_idx(x2_mat, x1_mat)
    fracs = []
    for i in range(len(fracs1)):
        fracs.append((fracs1[i]+fracs2[i])/2)
    return np.mean(fracs)

def calc_frac_idx(x1_mat,x2_mat):
    """
    Author Kai Cao and full documentation can be found at (https://github.com/caokai1073/Pamona)
    
    Returns fraction closer than true match for each sample (as an array)
    """
    fracs = []
    x = []
    nsamp = x1_mat.shape[0]
    rank=0
    for row_idx in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx,:], x2_mat)), axis=1))
        true_nbr = euc_dist[row_idx]
        sort_euc_dist = sorted(euc_dist)
        rank =sort_euc_dist.index(true_nbr)
        frac = float(rank)/(nsamp -1)

        fracs.append(frac)
        x.append(row_idx+1)

    return fracs,x

def visualize_embeddings(
    combined_adata: ad.AnnData,
    embedding: np.ndarray,
    reconstruct_gene:np.ndarray=None,
    true_label_key :str=None,
    embedding_key: str = "tacos",
    
    spatial_key: str = "spatial",

    random_state: int = 666,
    fontsize: int = 10
) -> ad.AnnData:
    """Create visualization-ready AnnData from embeddings
    
    Args:
        combined_adata: Original combined AnnData object with metadata
        embedding: 2D numpy array of embeddings (cells x dimensions)
        embedding_key: Key name for storing embeddings in obsm (default: 'tacos')
        spatial_key: Key name for spatial coordinates in obsm (default: 'spatial')
        random_state: Random seed for reproducibility (default: 666)

        
    
    Returns:
        New AnnData object containing embeddings and visualization results
    """
    # Create new AnnData for embeddings
    embedding_df = pd.DataFrame(embedding, index=combined_adata.obs.index)
    if reconstruct_gene is None:
        adata_emb = ad.AnnData(
            X=combined_adata.X,  # Store embeddings in X
            obs=combined_adata.obs.copy(),
            var = combined_adata.var.copy()
        )
    else:
        print('use reconstructed genes')
        adata_emb = ad.AnnData(
            X=reconstruct_gene,  # Store embeddings in X
            obs=combined_adata.obs.copy(),
            var = combined_adata.var.copy()
        )
    
    # Preserve spatial coordinates and store embeddings in obsm
    adata_emb.obsm[spatial_key] = combined_adata.obsm[spatial_key]
    adata_emb.obsm[embedding_key] = embedding_df.values
    
    # Compute neighborhood graph and UMAP
    sc.pp.neighbors(adata_emb, use_rep=embedding_key, random_state=random_state)
    sc.tl.umap(adata_emb, random_state=random_state)
    
    # # Configure plot settings

    sc.pl.umap(adata_emb, color=['batch'])
    if true_label_key is not None:
        sc.pl.umap(adata_emb, color=[true_label_key])

    # Compute PAGA analysis
    if true_label_key is not None:
        sc.tl.paga(adata_emb, groups=true_label_key)
        sc.pl.paga(
            adata_emb, 
            color=[true_label_key], 
            fontsize=fontsize,
            show=False
        )

    
    return adata_emb



def integrate_datasets(
    adata_list: List[sc.AnnData], 
    sample_ids: List[str]
) -> sc.AnnData:
    """Integrate multiple AnnData objects with common features
    
    Args:
        adata_list: List of processed AnnData objects
        sample_ids: Corresponding sample IDs for batch labeling
    
    Returns:
        Concatenated AnnData with aligned features
    """
    # Find common genes across all samples
    common_genes = set(adata_list[0].var_names)
    for adata in adata_list[1:]:
        common_genes.intersection_update(adata.var_names)
    print(f"Common genes across datasets: {len(common_genes)}")
    
    # Align and concatenate
    aligned_data = []
    for adata, sample in zip(adata_list, sample_ids):
        # Subset to common genes and clean metadata
        adata_ = adata[:, list(common_genes)].copy()
        adata_ = adata_[~adata_.obs.isna().any(axis=1)]  # Remove NA observations
        adata_.obs['batch'] = sample  # Add batch annotation
        aligned_data.append(adata_)
        print(f"{sample} aligned shape: {adata_.shape}")
    
    return ad.concat(aligned_data, join='inner', merge='same')



def process_adata(
    adata: sc.AnnData,
    marker_genes: List[str],
    min_genes: int = 100,
    min_cells: int = 50,
    n_top_genes: int = 6000
) -> sc.AnnData:
    """Process single AnnData object with quality control and normalization
    
    Args:
        adata: Raw AnnData object
        marker_genes: List of genes to retain regardless of filtering
        min_genes: Minimum genes per cell (default: 100)
        min_cells: Minimum cells expressing a gene (default: 50)
        n_top_genes: Number of highly variable genes to select (default: 6000)
    
    Returns:
        Processed AnnData object
    """
    # Quality control
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    
    # Gene filtering with marker protection
    initial_filter, _ = sc.pp.filter_genes(adata, min_cells=min_cells, inplace=False)
    adata.var['filter_bool'] = initial_filter
    adata.var.loc[marker_genes, 'filter_bool'] = True  # Override marker genes
    adata = adata[:, adata.var['filter_bool']]

    # Normalization and feature selection
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Ensure markers in HVG
    adata.var.highly_variable.loc[marker_genes] = True
    return adata[:, adata.var.highly_variable]

def aug_random_mask(input_feature, drop_percent=0.2):
    
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def aug_random_edge(input_adj, drop_percent=0.2):

    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    single_index_list = []
    for i in list(index_list):
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))
    
    
    edge_num = int(len(row_idx) / 2)      # 9228 / 2
    add_drop_num = int(edge_num * percent / 2) 
    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)

    
    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0
    
    '''
    above finish drop edges
    '''
    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:
        
        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1
    
    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj


def aug_drop_node(input_fea, input_adj, drop_percent=0.2):

    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)

    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)    # number of drop nodes
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj

def aug_subgraph(input_fea, input_edge_index, drop_percent=0.2):
    
    row = input_edge_index[0].numpy()
    col = input_edge_index[1].numpy()
    data = np.ones(len(row))  # 假设每条边的权重为 1

    # 创建 CSR 矩阵
    input_adj = sp.csr_matrix((data, (row, col)), shape=(input_fea.shape[0], input_fea.shape[0]))
    
    # edge_attr = torch.ones(input_edge_index.shape[1])
    # input_adj = torch.sparse_coo_tensor(input_edge_index, edge_attr, (input_fea.shape[0], input_fea.shape[0]))
    print(1001)
    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)
    node_num = input_fea.shape[0]

    all_node_list = [i for i in range(node_num)]
    s_node_num = int(node_num * (1 - drop_percent))
    center_node_id = random.randint(0, node_num - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []
    print(1002)

    for i in range(s_node_num - 1):
        
        all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()
        
        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break
        # print(10022)

    print(10022)
    drop_node_list = sorted([i for i in all_node_list if not i in sub_node_id_list])

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))
    print(1003)
    
    row, col = aug_input_adj.nonzero()  # 提取非零元素的行列索引
    edge_index_reconstructed = torch.tensor([row, col])
    # aug_input_adj = aug_input_adj.coalesce()  # 确保去重重复的边（如果有的话）
    # aug_input_adj = aug_input_adj.indices() 

    return aug_input_fea, edge_index_reconstructed


def aug_drop_node(input_fea, input_edge_index, drop_percent=0.2):

    row = input_edge_index[0].numpy()
    col = input_edge_index[1].numpy()
    data = np.ones(len(row))  # 假设每条边的权重为 1

    # 创建 CSR 矩阵
    input_adj = sp.csr_matrix((data, (row, col)), shape=(input_fea.shape[0], input_fea.shape[0]))

    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)

    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)    # number of drop nodes
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))
    
    row, col = aug_input_adj.nonzero()  # 提取非零元素的行列索引
    edge_index_reconstructed = torch.tensor([row, col])
    # aug_input_adj = aug_input_adj.coalesce()  # 确保去重重复的边（如果有的话）

    return aug_input_fea, edge_index_reconstructed



def delete_row_col(input_matrix, drop_list, only_row=False):

    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out



def nn_approx(ds1, ds2, names1, names2, knn=50):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M = 16)
    p.set_ef(10)
    p.add_items(ds2)
    ind,  distances = p.knn_query(ds1, k=knn)
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))
    return match

def nn(ds1, ds2, names1, names2, knn=50, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def nn_annoy(ds1, ds2, names1, names2, knn = 20, metric='euclidean', n_trees = 50, save_on_disk = True):
    """ Assumes that Y is zero-indexed. """
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    if(save_on_disk):
        a.on_disk_build('annoy.index')
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    # Match.
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match

def mnn_multi(ds1, ds2, names1, names2, knn=20, save_on_disk=True, approx=True, n_jobs=1):
    # print(ds1.shape)
    # print(ds2.shape)
    # for ds1_chunk in np.array_split(ds1, n_jobs):
    # #     print(ds1_chunk.shape)
    print(f'use {n_jobs} cpu')
    if approx:
        # 并行计算最近邻匹配
        match1 = Parallel(n_jobs=n_jobs)(
            delayed(nn_approx)(ds1_chunk, ds2, names1_chunk, names2, knn=knn)
            for ds1_chunk,names1_chunk in zip( np.array_split(ds1, n_jobs),np.array_split(names1, n_jobs))
        )
        match1 = set().union(*match1)  # 合并结果
        # print(match1[])

        match2 = Parallel(n_jobs=n_jobs)(
            delayed(nn_approx)(ds2_chunk, ds1, names2_chunk, names1, knn=knn)
            for ds2_chunk,names2_chunk in zip( np.array_split(ds2, n_jobs),np.array_split(names2, n_jobs))
        )
        match2 = set().union(*match2)  # 合并结果
    else:
        match1 = Parallel(n_jobs=n_jobs)(
            delayed(nn)(ds1_chunk, ds2, names1_chunk, names2, knn=knn)
            for ds1_chunk,names1_chunk in zip( np.array_split(ds1, n_jobs),np.array_split(names1, n_jobs))
        )
        match1 = set().union(*match1)  # 合并结果

        match2 = Parallel(n_jobs=n_jobs)(
            delayed(nn)(ds2_chunk, ds1, names2_chunk, names1, knn=knn)
            for ds2_chunk,names2_chunk in zip( np.array_split(ds2, n_jobs),np.array_split(names2, n_jobs))
        )
        match2 = set().union(*match2)  # 合并结果

    # Compute mutual nearest neighbors
    mutual = match1 & set([(b, a) for a, b in match2])

    return mutual
def mnn_multi(ds1, ds2, names1, names2, knn=20, save_on_disk=True, approx=True, n_jobs=1):
    # print(ds1.shape)
    # print(ds2.shape)
    # for ds1_chunk in np.array_split(ds1, n_jobs):
    # #     print(ds1_chunk.shape)
    print(f'use {n_jobs} cpu')
    if approx:
        # 并行计算最近邻匹配
        match1 = Parallel(n_jobs=n_jobs)(
            delayed(nn_approx)(ds1_chunk, ds2, names1_chunk, names2, knn=knn)
            for ds1_chunk,names1_chunk in zip( np.array_split(ds1, n_jobs),np.array_split(names1, n_jobs))
        )
        match1 = set().union(*match1)  # 合并结果
        # print(match1[])

        match2 = Parallel(n_jobs=n_jobs)(
            delayed(nn_approx)(ds2_chunk, ds1, names2_chunk, names1, knn=knn)
            for ds2_chunk,names2_chunk in zip( np.array_split(ds2, n_jobs),np.array_split(names2, n_jobs))
        )
        match2 = set().union(*match2)  # 合并结果
    else:
        match1 = Parallel(n_jobs=n_jobs)(
            delayed(nn)(ds1_chunk, ds2, names1_chunk, names2, knn=knn)
            for ds1_chunk,names1_chunk in zip( np.array_split(ds1, n_jobs),np.array_split(names1, n_jobs))
        )
        match1 = set().union(*match1)  # 合并结果

        match2 = Parallel(n_jobs=n_jobs)(
            delayed(nn)(ds2_chunk, ds1, names2_chunk, names1, knn=knn)
            for ds2_chunk,names2_chunk in zip( np.array_split(ds2, n_jobs),np.array_split(names2, n_jobs))
        )
        match2 = set().union(*match2)  # 合并结果

    # Compute mutual nearest neighbors
    mutual = match1 & set([(b, a) for a, b in match2])

    return mutual

def mnn(ds1, ds2, names1, names2, knn = 20, save_on_disk = True, approx = True):
    if approx:
        # Find nearest neighbors in first direction.
        # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)#, save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        print(type(match1))
        print(type(match1))
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)#, save_on_disk = save_on_disk)
    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn)
        match2 = nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual


def update_mnn(adata,n_list,embedding:Optional[np.ndarray] = None,path=None,k=50,cpu=1):
    

    if embedding is None:
        n1 = n_list[0]
        n_rest = 0
        for i in n_list[1:]:
            n_rest+=i
        n2 = n_rest
        if os.path.isfile(path+'mg_total.npy'):
            mg_total = np.load(path+'mg_total.npy')
            adata.obsm['Agg']= mg_total
            print('load agg')
        # adata_new = adata
        if adata.obsm.get('Agg') is None:
            sc.pp.neighbors(adata)  #计算观测值的邻域图
            snn1 = adata.obsp['connectivities'].todense()
            snn1_g = snn1[0:n1,0:n1]
            snn2_g = snn1[n1:(n1+n2),n1:(n1+n2)]
            # detect similarity corresponding spots with similar expression the same type use MNN.
            # MNN is computed on the spatially smoothed level
            # based on original gene expression
            # spatially smoothed gene expression
            # n_neighbor = 10
            if isinstance(adata.X, np.ndarray):
                embedding = adata.X
            else:
                embedding = adata.X.todense()
            X1 = embedding[0:n1,:].copy()
            X2 = embedding[n1:,:].copy()
            X1_mg = X1.copy()
            for i in range(n1):
                # detect non-zero of snn1_g
                index_i = snn1_g[i,:].argsort().A[0]
                index_i = index_i[(n1-10):n1]
                X1_mg[i,:] = X1[index_i,:].mean(0)

def update_mnn(adata,n_list,embedding:Optional[np.ndarray] = None,path=None,k=50,cpu=1):
    

    if embedding is None:
        n1 = n_list[0]
        n_rest = 0
        for i in n_list[1:]:
            n_rest+=i
        n2 = n_rest
        if os.path.isfile(path+'mg_total.npy'):
            mg_total = np.load(path+'mg_total.npy')
            adata.obsm['Agg']= mg_total
            print('load agg')
        # adata_new = adata
        if adata.obsm.get('Agg') is None:
            sc.pp.neighbors(adata)  #计算观测值的邻域图
            snn1 = adata.obsp['connectivities'].todense()
            snn1_g = snn1[0:n1,0:n1]
            snn2_g = snn1[n1:(n1+n2),n1:(n1+n2)]
            # detect similarity corresponding spots with similar expression the same type use MNN.
            # MNN is computed on the spatially smoothed level
            # based on original gene expression
            # spatially smoothed gene expression
            # n_neighbor = 10
            if isinstance(adata.X, np.ndarray):
                embedding = adata.X
            else:
                embedding = adata.X.todense()
            X1 = embedding[0:n1,:].copy()
            X2 = embedding[n1:,:].copy()
            X1_mg = X1.copy()
            for i in range(n1):
                # detect non-zero of snn1_g
                index_i = snn1_g[i,:].argsort().A[0]
                index_i = index_i[(n1-10):n1]
                X1_mg[i,:] = X1[index_i,:].mean(0)

            X2_mg = X2.copy()
            for i in range(n2):
                # detect non-zero of snn1_g
                index_i = snn2_g[i,:].argsort().A[0]
                index_i = index_i[(n2-10):n2]
                X2_mg[i,:] = X2_mg[index_i,:].mean(0)
            mg_total = np.concatenate([X1_mg,X2_mg],axis=0)
            np.save(path+'mg_total.npy', mg_total)
            print('agg saved')
            adata.obsm['Agg'] = mg_total
            
    # else:
        
    #     # 创建新的adata
    #     obs = adata.obs
    #     adata_new = ad.AnnData(embedding,obs=obs)
    #     adata_new.obsm['Agg'] = embedding
    # print(mg_total.shape)
    # 创建mnn
    # print("create_dictionary_mnn")
    # print(adata_new)
    mnn_dict = create_dictionary_mnn(adata, use_rep='Agg', batch_name='batch', k=k,cpu=cpu)
    # sub_graph = mnn_matrix[0:n1, n1:] #n1*n2

    return mnn_dict

def get_triplet(adata,mnn_dict,batch_name = 'batch'):
    anchor_ind = []
    positive_ind = []
    negative_ind = []
    section_ids = np.array(adata.obs['batch'].unique())
    for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
        batchname_list = adata.obs[batch_name][mnn_dict[batch_pair].keys()]
        #             print("before add KNN pairs, len(mnn_dict[batch_pair]):",
        #                   sum(adata_new.obs['batch_name'].isin(batchname_list.unique())), len(mnn_dict[batch_pair]))

        cellname_by_batch_dict = dict()
        for batch_id in range(len(section_ids)):
            cellname_by_batch_dict[section_ids[batch_id]] = adata.obs_names[
                adata.obs[batch_name] == section_ids[batch_id]].values

        anchor_list = []
        positive_list = []
        negative_list = []
        for anchor in mnn_dict[batch_pair].keys():
            anchor_list.append(anchor)
            ## np.random.choice(mnn_dict[batch_pair][anchor])
            positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
            positive_list.append(positive_spot)
            section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
            negative_list.append(
                cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

        batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
        anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
        positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
        negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))
    return anchor_ind,positive_ind,negative_ind



def create_dictionary_mnn(adata, use_rep, batch_name, k = 50, save_on_disk = True, approx = True, verbose = 1, iter_comb = None,cpu = 1):
    cell_names = adata.obs_names
    batch_list = adata.obs[batch_name]
    datasets = []
    datasets_pcs = []
    cells = []

    for i in batch_list.unique():
        datasets.append(adata[batch_list == i])
        datasets_pcs.append(adata[batch_list == i].obsm[use_rep])
        cells.append(cell_names[batch_list == i])


    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))

    mnns = dict()
    
    # iter_comb=[]
    # for i in range(1,len(cells)):
    #     iter_comb.append((0,i))
    if iter_comb is None:
        iter_comb = list(itertools.combinations(range(len(cells)), 2))
    cells_all = cells[0].tolist() # cells index of anchor slice


    for comb in iter_comb:

        i = comb[0]
        j = comb[1]

        key_name1 = batch_name_df.loc[comb[0]].values[0] + "_" + batch_name_df.loc[comb[1]].values[0]
        mnns[key_name1] = {} # for multiple-slice setting, the key_names1 can avoid the mnns replaced by previous slice-pair

        new = list(cells[j])
        ref = list(cells[i])
        
        cells_all.extend(list(cells[j]))
        
        
        # print(adata[new])
        ds1 = adata[new].obsm[use_rep]
        ds2 = adata[ref].obsm[use_rep]
        names1 = new
        names2 = ref
        # if k>1，one point in ds1 may have multiple MNN points in ds2.
        
        if cpu==1:
            match = mnn(ds1, ds2, names1, names2, knn=k, save_on_disk = save_on_disk, approx = approx)
        else:
            match = mnn_multi(ds1, ds2, names1, names2, knn=k, save_on_disk=save_on_disk, approx=approx, n_jobs=cpu)
        # print(len(match))

        G = nx.Graph()
        G.add_edges_from(match)
        node_names = np.array(G.nodes)
    
        anchors = list(node_names)
        adj = nx.adjacency_matrix(G)
        tmp = np.split(adj.indices, adj.indptr[1:-1])
        # print(len(anchors))

        for i in range(0, len(anchors)):
            key = anchors[i]
            i = tmp[i]
            names = list(node_names[i])
            mnns[key_name1][key]= names

    return mnns



def transition(communities: Sequence[Sequence[int]],
               num_nodes: int) -> np.ndarray:
    classes = np.full(num_nodes, -1)
    for i, node_list in enumerate(communities):
        classes[np.asarray(node_list)] = i
    return classes

def community_detection(name):
    algs = {
        # non-overlapping algorithms
        'louvain': algorithms.louvain,
        'combo': algorithms.pycombo,
        'leiden': algorithms.leiden,
        'ilouvain': algorithms.ilouvain,
        # 'edmot': algorithms.edmot,
        'eigenvector': algorithms.eigenvector,
        'girvan_newman': algorithms.girvan_newman,
        # overlapping algorithms
        'demon': algorithms.demon,
        'lemon': algorithms.lemon,
        # 'ego-splitting': algorithms.egonet_splitter,
        # 'nnsed': algorithms.nnsed,
        'lpanni': algorithms.lpanni,
    }
    return algs[name]

def community_strength(graph: nx.Graph,
                       communities: Sequence[Sequence[int]]) -> (np.ndarray, np.ndarray):
    graph = convert_graph_formats(graph, nx.Graph)
    coms = {}
    for cid, com in enumerate(communities):
        for node in com:
            coms[node] = cid
    inc, deg = {}, {}
    links = graph.size(weight="weight")
    assert links > 0, "A graph without link has no communities."
    for node in graph:
        try:
            com = coms[node]
            deg[com] = deg.get(com, 0.0) + graph.degree(node, weight="weight")
            for neighbor, dt in graph[node].items():
                weight = dt.get("weight", 1)
                if coms[neighbor] == com:
                    if neighbor == node:
                        inc[com] = inc.get(com, 0.0) + float(weight)
                    else:
                        inc[com] = inc.get(com, 0.0) + float(weight) / 2.0
        except:
            pass
    com_cs = []
    for idx, com in enumerate(set(coms.values())):
        com_cs.append((inc.get(com, 0.0) / links) - (deg.get(com, 0.0) / (2.0 * links)) ** 2)
    com_cs = np.asarray(com_cs)
    node_cs = np.zeros(graph.number_of_nodes(), dtype=np.float32)
    for i, w in enumerate(com_cs):
        for j in communities[i]:
            node_cs[j] = com_cs[i]
    return com_cs, node_cs

def get_edge_weight(edge_index: torch.Tensor,
                    com: np.ndarray,
                    com_cs: np.ndarray) -> torch.Tensor:
    edge_mod = lambda x: com_cs[x[0]] if x[0] == x[1] else -(float(com_cs[x[0]]) + float(com_cs[x[1]]))
    normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    edge_weight = np.asarray([edge_mod([com[u.item()], com[v.item()]]) for u, v in edge_index.T])
    edge_weight = normalize(edge_weight)
    return torch.from_numpy(edge_weight).to(edge_index.device)

def transition(communities: Sequence[Sequence[int]],
               num_nodes: int) -> np.ndarray:
    classes = np.full(num_nodes, -1)
    for i, node_list in enumerate(communities):
        classes[np.asarray(node_list)] = i
    return classes

def community_augmentation(expression, edge_index, detect_method):
    n_nodes = len(expression)
    data = Data(x=expression, edge_index=edge_index)
    g = to_networkx(data, to_undirected=True)
    print('detecting communities...')
    coms = community_detection(detect_method)(g).communities
    com_str, node_str = community_strength(g, coms)
    com_groups = transition(coms, n_nodes)
    com_sz = [len(com) for com in coms]
    print('{} communities found!'.format(len(com_sz)))
    edge_weight = get_edge_weight(edge_index, com_groups, com_str)
    
    return edge_weight, node_str



def merge_csr_graphs(csr_matrices):
    num_nodes = 0
    rows = []
    cols = []
    data = []
    
    for csr in csr_matrices:
        num_nodes_csr = csr.shape[0]
        rows_csr, cols_csr = csr.nonzero()
        data_csr = csr.data
        
        rows.extend(rows_csr + num_nodes)
        cols.extend(cols_csr + num_nodes)
        data.extend(data_csr)
        
        num_nodes += num_nodes_csr
    
    combined_csr = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    return combined_csr


def construct_graph(data,num_list,n_neighbors: int = 15):
    # print('construct graphs...') 
    # n_total = sum(num_list)
    # arr_total = np.zeros((n_total,n_total))
    flag = 0
    graph_list = []
    for n in num_list:
        data_slice = torch.from_numpy(data[flag:flag+n,])
        if data.shape[1]==2:
            spatial_graph = graph_alpha(data_slice,n_neighbors=n_neighbors)
            # print(type(spatial_graph))
        else:
            spatial_graph = kneighbors_graph(data_slice, n_neighbors=10, mode='distance')
            # print(type(spatial_graph))
        graph_list.append(spatial_graph) 
        
        # print(type(spatial_graph))
        flag+=n

    return graph_list

def normalize_adj(adj):
    # normalize adjacency matrix by D^-1/2 * A * D^-1/2,in order to apply to gcn conv

    if not np.array_equal(np.array(adj.todense()).transpose(), np.array(adj.todense())):
        raise AttributeError('adj matrix should be symmetrical!')
    adj = adj.tocoo()
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    # Convert a scipy sparse matrix to a torch sparse tensor.
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_to_edge_index(adj):
    n_nodes = int(adj.shape[0])
    # out = torch.tensor([i for i in range(n_nodes)]).unsqueeze(0)
    # self_loop = torch.cat([out,out],dim=0)
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj)
        edge_index = adj.nonzero().t()
        # edge_index = torch.cat([edge_index,self_loop],dim=1)

    elif isinstance(adj, sp.csr_matrix):
        adj = adj.tocoo().astype(np.float32)
        edge_index = torch.from_numpy(
            np.vstack((adj.row, adj.col)).astype(np.int64))
        # edge_index = torch.cat([edge_index, self_loop], dim=1)
    else:
        raise TypeError("unsupported adj type !")
    return edge_index


def graph_alpha(spatial_locs, n_neighbors=15):
        """
        Construct a geometry-aware spatial proximity graph of the spatial spots of cells by using alpha complex.
        :param adata: the annData object for spatial transcriptomics data with adata.obsm['spatial'] set to be the spatial locations.
        :type adata: class:`anndata.annData`
        :param n_neighbors: the number of nearest neighbors for building spatial neighbor graph based on Alpha Complex
        :type n_neighbors: int, optional, default: 10
        :return: a spatial neighbor graph
        :rtype: class:`scipy.sparse.csr_matrix`
        """
        A_knn = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode='distance')
        estimated_graph_cut = A_knn.sum() / float(A_knn.count_nonzero())
        spatial_locs_list = spatial_locs.tolist()
        n_node = len(spatial_locs_list)
        alpha_complex = gudhi.AlphaComplex(points=spatial_locs_list)
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=estimated_graph_cut ** 2)
        skeleton = simplex_tree.get_skeleton(1)
        initial_graph = nx.Graph()
        initial_graph.add_nodes_from([i for i in range(n_node)])
        for s in skeleton:
            if len(s[0]) == 2:
                initial_graph.add_edge(s[0][0], s[0][1])

        extended_graph = nx.Graph()
        extended_graph.add_nodes_from(initial_graph)
        extended_graph.add_edges_from(initial_graph.edges)

        # Remove self edges
        for i in range(n_node):
            try:
                extended_graph.remove_edge(i, i)
            except:
                pass
        
        edge_cnt = len(extended_graph.edges())
        nodes = np.ones(edge_cnt)
        rows = []
        cols = []
        for edge in extended_graph.edges():
            row, col = edge
            rows.append(row)
            cols.append(col)

        spa_graph_sp_mx = sp.csr_matrix((nodes, (rows, cols)), shape=(n_node, n_node), dtype=int)
        # symmetrical adj
        spa_graph_sp_mx += spa_graph_sp_mx.transpose()
        spa_graph_sp_mx.data[spa_graph_sp_mx.data == 2] = 1
        return spa_graph_sp_mx
        
        
        # return nx.to_scipy_sparse_array(extended_graph, format='csr')