from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.neighbors import NearestNeighbors,kneighbors_graph
from intervaltree import IntervalTree
import operator
import numpy as np
from annoy import AnnoyIndex
import itertools
import networkx as nx
import hnswlib
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from typing import Sequence
from cdlib.utils import convert_graph_formats
from cdlib import algorithms
import torch
import gudhi
from scipy.spatial import distance
from .glmpca import glmpca
import ot
from sklearn.preprocessing import MinMaxScaler
import scipy

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

def validate_sparse_labels(Y):
    if not zero_indexed(Y):
        raise ValueError('Ensure that your labels are zero-indexed')
    if not consecutive_indexed(Y):
        raise ValueError('Ensure that your labels are indexed consecutively')

def zero_indexed(Y):
    if min(abs(Y)) != 0:
        return False
    return True


def consecutive_indexed(Y):
    """ Assumes that Y is zero-indexed. """
    n_classes = len(np.unique(Y[Y != np.array(-1)]))
    if max(Y) >= n_classes:
        return False
    return True


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


def mnn(ds1, ds2, names1, names2, knn = 20, save_on_disk = True, approx = True):
    if approx:
        # Find nearest neighbors in first direction.
        # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)#, save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)#, save_on_disk = save_on_disk)
    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn)
        match2 = nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual

def create_dictionary_mnn(adata, use_rep, batch_name, k = 50, save_on_disk = True, approx = True, verbose = 1, iter_comb = None):
    cell_names = adata.obs_names
    batch_list = adata.obs[batch_name]
    datasets = []
    datasets_pcs = []
    cells = []
    for i in batch_list.unique():
        datasets.append(adata[batch_list == i])
        datasets_pcs.append(adata[batch_list == i].obsm[use_rep])
        cells.append(cell_names[batch_list == i])
    # print(cells)

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    # print(batch_name_df)
    mnns = dict()
    
    iter_comb=[]
    for i in range(1,len(cells)):
        iter_comb.append((0,i))
    # if iter_comb is None:
        # iter_comb = list(itertools.combinations(range(len(cells)), 2))
    cells_all = cells[0].tolist()
    # print(iter_comb)
    # print(iter_comb)
    for comb in iter_comb:
        # comb = iter_comb
        # print(comb)
        i = comb[0]
        j = comb[1]
        # print(i)
        # print(j)
        key_name1 = batch_name_df.loc[comb[0]].values[0] + "_" + batch_name_df.loc[comb[1]].values[0]
        mnns[key_name1] = {} # for multiple-slice setting, the key_names1 can avoid the mnns replaced by previous slice-pair
        # if(verbose > 0):
        #     print('Processing datasets {}'.format((i, j)))

        new = list(cells[j])
        ref = list(cells[i])
        
        cells_all.extend(list(cells[j]))
        
        

        ds1 = adata[new].obsm[use_rep]
        ds2 = adata[ref].obsm[use_rep]
        names1 = new
        names2 = ref
        # if k>1，one point in ds1 may have multiple MNN points in ds2.
        match = mnn(ds1, ds2, names1, names2, knn=k, save_on_disk = save_on_disk, approx = approx)
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
    # print(len(mnns))
    # print(mnns.keys())
    # print(mnns.values())
    
    # cells1 = cells[0].tolist()
    # cells2 = cells[1].tolist()
    # for i in cells2:
    #     cells1.append(i)
    # cells_all = cells1
    # len(cells_all)
    cross_adj = np.zeros((len(cells_all), len(cells_all)))
    key = list(mnns.keys())[0]
    val = list(mnns.values())[0]
    # build adj based on mnns n1*n2 (from n1 view)
    # cross_adj = np.zeros((len(cells_all), len(cells_all)))
    for i in range(len(mnns)):
        key = list(mnns.keys())[i]
        val = list(mnns.values())[i]
        # print(len(val))
        for s in range(len(val)):
            need_index1 = cells_all.index(list(val.keys())[s])
            need_index2 = []
            for j in range(len(list(val.values())[s])):
                need_index2.append(cells_all.index((list(val.values())[s])[j]))
            cross_adj[need_index1, need_index2] = 1
        cross_adj1 = cross_adj.transpose()
        np.array_equal(cross_adj, cross_adj1)
        
    # for s in range(0, len(anchors)):
    #     # detect the location of the anchors
    #     need_index1 = cells_all.index(list(val.keys())[s])
    #     need_index2 = []
    #     for j in range(len(list(val.values())[s])):
    #         need_index2.append(cells_all.index((list(val.values())[s])[j]))

    #     cross_adj[need_index1, need_index2] = 1
    # cross_adj1 = cross_adj.transpose()
    # np.array_equal(cross_adj, cross_adj1)
    
    return cross_adj

def update_mnn(adata,n_list,embedding=None,use_partialOT=False):
    n1 = n_list[0]
    n_rest = 0
    for i in n_list[1:]:
        n_rest+=i
    n2 = n_rest
    if not use_partialOT:
        if embedding is None:
            adata_new = adata
            if adata.obsm.get('Agg') is None:
                sc.pp.neighbors(adata_new)  #计算观测值的邻域图
                # sc.tl.umap(adata_new)  #使用UMAP嵌入邻域图
                # print(adata_new)
                snn1 = adata_new.obsp['connectivities'].todense()
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
                adata_new.obsm['Agg'] = mg_total
        else:
            
            # 创建新的adata
            obs = adata.obs
            adata_new = ad.AnnData(embedding,obs=obs)
            adata_new.obsm['Agg'] = embedding
        # print(mg_total.shape)
        # 创建mnn
        # print("create_dictionary_mnn")
    
        mnn_matrix = create_dictionary_mnn(adata_new, use_rep='Agg', batch_name='batch', k=50)
        sub_graph = mnn_matrix[0:n1, n1:] #n1*n2
    else:
        if embedding is None:
            adata_new = adata
            # if adata.obsm.get('Agg') is None:
            #     sc.pp.neighbors(adata)  #计算观测值的邻域图
            #     # sc.tl.umap(adata_new)  #使用UMAP嵌入邻域图
            #     # print(adata_new)
            #     snn1 = adata.obsp['connectivities'].todense()
            #     snn1_g = snn1[0:n1,0:n1]
            #     snn2_g = snn1[n1:(n1+n2),n1:(n1+n2)]
            #     # detect similarity corresponding spots with similar expression the same type use MNN.
            #     # MNN is computed on the spatially smoothed level
            #     # based on original gene expression
            #     # spatially smoothed gene expression
            #     # n_neighbor = 10
            #     if isinstance(adata.X, np.ndarray):
            #         embedding = adata.X
            #     else:
            #         embedding = adata.X.todense()
            #     X1 = embedding[0:n1,:].copy()
            #     X2 = embedding[n1:,:].copy()
            #     X1_mg = X1.copy()
            #     for i in range(n1):
            #         # detect non-zero of snn1_g
            #         index_i = snn1_g[i,:].argsort().A[0]
            #         index_i = index_i[(n1-10):n1]
            #         X1_mg[i,:] = X1[index_i,:].mean(0)

            #     X2_mg = X2.copy()
            #     for i in range(n2):
            #         # detect non-zero of snn1_g
            #         index_i = snn2_g[i,:].argsort().A[0]
            #         index_i = index_i[(n2-10):n2]
            #         X2_mg[i,:] = X2_mg[index_i,:].mean(0)
            #     mg_total = np.concatenate([X1_mg,X2_mg],axis=0)
            #     adata.obsm['Agg'] = mg_total
            # obs = adata.obs
            # adata_new = ad.AnnData(adata.obsm['Agg'],obs=obs)
            
        else:
            scaler = MinMaxScaler()
            normalized_embed = scaler.fit_transform(embedding)
            obs = adata.obs
            adata_new = ad.AnnData(normalized_embed,obs=obs)
            adata_new.obsm['Agg'] = normalized_embed
            
        adata_new.obsm = adata.obsm
        a_idx = adata_new.obs['batch'].cat.categories[0]
        b_idx = adata_new.obs['batch'].cat.categories[1]
        slice_a = adata_new[adata_new.obs['batch']==a_idx]
        slice_b = adata_new[adata_new.obs['batch']==b_idx]
        sub_graph = partial_pairwise_align(slice_a,slice_b,s=0.7)
    return sub_graph

## Covert a sparse matrix into a dense matrix
to_dense_array = lambda X: np.array(X.todense()) if isinstance(X,sparse.csr.spmatrix) else X

## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep] 

def generalized_kl_divergence(X, Y):
    """
    Returns pairwise generalized KL divergence (over all pairs of samples) of two matrices X and Y.

    param: X - np array with dim (n_samples by n_features)
    param: Y - np array with dim (m_samples by n_features)

    return: D - np array with dim (n_samples by m_samples). Pairwise generalized KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    log_X = np.log(X)
    log_Y = np.log(Y)
    X_log_X = np.matrix([np.dot(X[i], log_X[i].T) for i in range(X.shape[0])])
    D = X_log_X.T - np.dot(X, log_Y.T)
    sum_X = np.sum(X, axis=1)
    sum_Y = np.sum(Y, axis=1)
    D = (D.T - sum_X).T + sum_Y.T
    return np.asarray(D)

def kl_divergence(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.
    
    param: X - np array with dim (n_samples by n_features)
    param: Y - np array with dim (m_samples by n_features)
    
    return: D - np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    
    X = X/X.sum(axis=1, keepdims=True)
    Y = Y/Y.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    log_Y = np.log(Y)
    X_log_X = np.matrix([np.dot(X[i],log_X[i].T) for i in range(X.shape[0])])
    D = X_log_X.T - np.dot(X,log_Y.T)
    return np.asarray(D)

def high_umi_gene_distance(X, Y, n):
    """
    n: number of highest umi count genes to keep
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    joint_matrix = np.vstack((X, Y))
    gene_umi_counts = np.sum(joint_matrix, axis=0)
    top_indices = np.sort((-gene_umi_counts).argsort()[:n])
    X = X[:, top_indices]
    Y = Y[:, top_indices]
    X += np.tile(0.01 * (np.sum(X, axis=1) / X.shape[1]), (X.shape[1], 1)).T
    Y += np.tile(0.01 * (np.sum(Y, axis=1) / Y.shape[1]), (Y.shape[1], 1)).T
    return kl_divergence(X, Y)

def glmpca_distance(X, Y, latent_dim=30, filter=True, verbose=True):
    """
    param: X - np array with dim (n_samples by n_features)
    param: Y - np array with dim (m_samples by n_features)
    param: latent_dim - number of latent dimensions in glm-pca
    param: filter - whether to first select genes with highest UMI counts
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    joint_matrix = np.vstack((X, Y))
    if filter:
        gene_umi_counts = np.sum(joint_matrix, axis=0)
        top_indices = np.sort((-gene_umi_counts).argsort()[:2000])
        joint_matrix = joint_matrix[:, top_indices]

    print("Starting GLM-PCA...")
    res = glmpca(joint_matrix.T, latent_dim, penalty=1, verbose=verbose)
    #res = glmpca(joint_matrix.T, latent_dim, fam='nb', penalty=1, verbose=True)
    reduced_joint_matrix = res["factors"]
    # print("GLM-PCA finished with joint matrix shape " + str(reduced_joint_matrix.shape))
    print("GLM-PCA finished.")

    X = reduced_joint_matrix[:X.shape[0], :]
    Y = reduced_joint_matrix[X.shape[0]:, :]
    return distance.cdist(X, Y)



def gwgrad_partial(C1, C2, T, loss_fun="square_loss"):
    """Compute the GW gradient, as one term in the FGW gradient.

    Note: we can not use the trick in Peyre16 as the marginals may not sum to 1.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source cost matrix

    C2: array of shape (n_q,n_q)
        intra-target cost matrix

    T : array of shape(n_p, n_q)
        Transport matrix

    loss_fun

    Returns
    -------
    numpy.array of shape (n_p, n_q)
        gradient
    """
    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2)

        def f2(b):
            return (b**2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * np.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return np.log(b + 1e-15)

    #cC1 = np.dot(C1 ** 2 / 2, np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1)))
    A = np.dot(
        f1(C1),
        np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1))
    )

    #cC2 = np.dot(np.dot(np.ones(C1.shape[0]).reshape(1, -1), T), C2 ** 2 / 2)
    B = np.dot(
        np.dot(np.ones(C1.shape[0]).reshape(1, -1), T),
        f2(C2).T
    )  # does f2(C2) here need transpose?

    constC = A + B
    #C = -np.dot(C1, T).dot(C2.T)
    C = -np.dot(h1(C1), T).dot(h2(C2).T)
    tens = constC + C
    return tens * 2


def gwloss_partial(C1, C2, T, loss_fun='square_loss'):
    g = gwgrad_partial(C1, C2, T, loss_fun) * 0.5
    return np.sum(g * T)


def wloss(M, T):
    return np.sum(M * T)


def fgwloss_partial(alpha, M, C1, C2, T, loss_fun='square_loss'):
    return (1 - alpha) * wloss(M, T) + alpha * gwloss_partial(C1, C2, T, loss_fun)

def fgwgrad_partial(alpha, M, C1, C2, T, loss_fun='square_loss'):
    return (1 - alpha) * M + alpha * gwgrad_partial(C1, C2, T, loss_fun)




def partial_fused_gromov_wasserstein(M, C1, C2, p, q, alpha, m=None, G0=None, loss_fun='square_loss', armijo=False, log=False, verbose=False, numItermax=1000, tol=1e-7, stopThr=1e-9, stopThr2=1e-9):
    if m is None:
        # m = np.min((np.sum(p), np.sum(q)))
        raise ValueError("Parameter m is not provided.")
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    elif m > np.min((np.sum(p), np.sum(q))):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal to min(|p|_1, |q|_1).")

    if G0 is None:
        G0 = np.outer(p, q)

    nb_dummies = 1
    dim_G_extended = (len(p) + nb_dummies, len(q) + nb_dummies)
    q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)

    cpt = 0
    err = 1

    if log:
        log = {'err': [], 'loss': []}
    f_val = fgwloss_partial(alpha, M, C1, C2, G0, loss_fun)
    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(cpt, f_val, 0, 0))
        #print_fgwloss_partial(alpha, M, C1, C2, G0, loss_fun)

    # while err > tol and cpt < numItermax:
    while cpt < numItermax:
        Gprev = np.copy(G0)
        old_fval = f_val

        gradF = fgwgrad_partial(alpha, M, C1, C2, G0, loss_fun)
        gradF_emd = np.zeros(dim_G_extended)
        gradF_emd[:len(p), :len(q)] = gradF
        gradF_emd[-nb_dummies:, -nb_dummies:] = np.max(gradF) * 1e2
        gradF_emd = np.asarray(gradF_emd, dtype=np.float64)

        Gc, logemd = ot.lp.emd(p_extended, q_extended, gradF_emd, numItermax=1000000, log=True)
        if logemd['warning'] is not None:
            raise ValueError("Error in the EMD resolution: try to increase the"
                             " number of dummy points")

        G0 = Gc[:len(p), :len(q)]

        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log['err'].append(err)

        deltaG = G0 - Gprev

        if not armijo:
            a = alpha * gwloss_partial(C1, C2, deltaG, loss_fun)
            b = (1 - alpha) * wloss(M, deltaG) + 2 * alpha * np.sum(gwgrad_partial(C1, C2, deltaG, loss_fun) * 0.5 * Gprev)
            # c = (1 - alpha) * wloss(M, Gprev) + alpha * gwloss_partial(C1, C2, Gprev, loss_fun)
            c = fgwloss_partial(alpha, M, C1, C2, Gprev, loss_fun)

            gamma = ot.optim.solve_1d_linesearch_quad(a, b)
            # gamma = ot.optim.solve_1d_linesearch_quad(a, b, c)
            # f_val = a * gamma ** 2 + b * gamma + c
        else:
            def f(x, alpha, M, C1, C2, lossfunc):
                return fgwloss_partial(alpha, M, C1, C2, x, lossfunc)
            xk = Gprev
            pk = deltaG
            gfk = fgwgrad_partial(alpha, M, C1, C2, xk, loss_fun)
            old_val = fgwloss_partial(alpha, M, C1, C2, xk, loss_fun)
            args = (alpha, M, C1, C2, loss_fun)
            gamma, fc, fa = ot.optim.line_search_armijo(f, xk, pk, gfk, old_val, args)
            # f_val = f(xk + gamma * pk, alpha, M, C1, C2, loss_fun)

        if gamma == 0:
            cpt = numItermax
        G0 = Gprev + gamma * deltaG
        f_val = fgwloss_partial(alpha, M, C1, C2, G0, loss_fun)
        cpt += 1

        # TODO: better stopping criteria?
        abs_delta_fval = abs(f_val - old_fval)
        relative_delta_fval = abs_delta_fval / abs(f_val)
        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            cpt = numItermax
        if log:
            log['loss'].append(f_val)
        if verbose:
            # if cpt % 20 == 0:
            #     print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            #         'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(cpt, f_val, relative_delta_fval, abs_delta_fval))
            #print_fgwloss_partial(alpha, M, C1, C2, G0, loss_fun)

    if log:
        log['partial_fgw_cost'] = fgwloss_partial(alpha, M, C1, C2, G0, loss_fun)
        return G0[:len(p), :len(q)], log
    else:
        return G0[:len(p), :len(q)]

def partial_pairwise_align(sliceA, sliceB, s, alpha=0.1, armijo=False, dissimilarity='glmpca', use_rep=None, G_init=None, a_distribution=None,
                   b_distribution=None, norm=True, return_obj=False, verbose=True):
    """
    Calculates and returns optimal *partial* alignment of two slices.

    param: sliceA - AnnData object
    param: sliceB - AnnData object
    param: s - Amount of mass to transport; Overlap percentage between the two slices. Note: 0 ≤ s ≤ 1
    param: alpha - Alignment tuning parameter. Note: 0 ≤ alpha ≤ 1
    param: armijo - Whether or not to use armijo (approximate) line search during conditional gradient optimization of Partial-FGW. Default is to use exact line search.
    param: dissimilarity - Expression dissimilarity measure: 'kl' or 'euclidean' or 'glmpca'. Default is glmpca.
    param: use_rep - If none, uses slice.X to calculate dissimilarity between spots, otherwise uses the representation given by slice.obsm[use_rep]
    param: G_init - initial mapping to be used in Partial-FGW OT, otherwise default is uniform mapping
    param: a_distribution - distribution of sliceA spots (1-d numpy array), otherwise default is uniform
    param: b_distribution - distribution of sliceB spots (1-d numpy array), otherwise default is uniform
    param: norm - scales spatial distances such that maximum spatial distance is equal to maximum gene expression dissimilarity
    param: return_obj - returns objective function value if True, nothing if False

    return: pi - partial alignment of spots
    return: log['fgw_dist'] - objective function output of FGW-OT
    """
    m = s
    print("partial OT starts...")

    # subset for common genes
    # common_genes = intersect(sliceA.var.index, sliceB.var.index)
    # sliceA = sliceA[:, common_genes]
    # sliceB = sliceB[:, common_genes]
    # print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')

    # Calculate spatial distances
    D_A = distance.cdist(sliceA.obsm['spatial'], sliceA.obsm['spatial'])
    D_B = distance.cdist(sliceB.obsm['spatial'], sliceB.obsm['spatial'])

    # Calculate expression dissimilarity
    A_X, B_X = to_dense_array(extract_data_matrix(sliceA, use_rep)), to_dense_array(extract_data_matrix(sliceB, use_rep))
    if dissimilarity.lower() == 'euclidean' or dissimilarity.lower() == 'euc':
        M = distance.cdist(A_X, B_X)
    elif dissimilarity.lower() == 'gkl':
        s_A = A_X + 0.01
        s_B = B_X + 0.01
        M = generalized_kl_divergence(s_A, s_B)
        M /= M[M > 0].max()
        M *= 10
    elif dissimilarity.lower() == 'kl':
        s_A = A_X + 0.01
        s_B = B_X + 0.01
        M = kl_divergence(s_A, s_B)
    elif dissimilarity.lower() == 'selection_kl':
        M = high_umi_gene_distance(A_X, B_X, 2000)
    # elif dissimilarity.lower() == "pca":
    #     M = pca_distance(sliceA, sliceB, 2000, 20)
    elif dissimilarity.lower() == 'glmpca':
        M = glmpca_distance(A_X, B_X, latent_dim=30, filter=True, verbose=verbose)
    else:
        print("ERROR")
        exit(1)

    # init distributions
    if a_distribution is None:
        a = np.ones((sliceA.shape[0],)) / sliceA.shape[0]
    else:
        a = a_distribution

    if b_distribution is None:
        b = np.ones((sliceB.shape[0],)) / sliceB.shape[0]
    else:
        b = b_distribution

    if norm:
        D_A /= D_A[D_A > 0].min().min()
        D_B /= D_B[D_B > 0].min().min()

        """
        Code for normalizing distance matrix
        """
        D_A /= D_A[D_A>0].max()
        #D_A *= 10
        D_A *= M.max()
        D_B /= D_B[D_B>0].max()
        #D_B *= 10
        D_B *= M.max()
        """
        Code for normalizing distance matrix ends
        """
    pi, log = partial_fused_gromov_wasserstein(M, D_A, D_B, a, b, alpha=alpha, m=m, G0=G_init, loss_fun='square_loss', armijo=armijo, log=True, verbose=verbose,numItermax=50)

    if return_obj:
        return pi, log['partial_fgw_cost']
    return pi

def graph_alpha(spatial_locs, n_neighbors=10):
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

        return nx.to_scipy_sparse_array(extended_graph, format='csr')

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def build_graph_G(data,n_list,use_cor = True):
    n_total = sum(n_list)
    arr_total = np.zeros((n_total,n_total))
    
    flag = 0
    for n in n_list:
        data_slice = torch.from_numpy(data[flag:flag+n,])
        if use_cor:
            spatial_graph = graph_alpha(data_slice).toarray()
        else:
            spatial_graph = kneighbors_graph(data_slice, n_neighbors=10, mode='distance').toarray()
        #spatial_graph = graph_alpha(cor).toarray()
        
        arr_total[flag:flag+n,flag:flag+n] = spatial_graph[:n, :n]
        flag+=n
        
    
    # n1 = n_list[0]
    # n_rest = 0
    # for i in n_list[1:]:
    #     n_rest+=i
    # cor1 = torch.from_numpy(cor_cat[:n1,])
    # cor2 = torch.from_numpy(cor_cat[n1:,])
    # spatial_graph1 = graph_alpha(cor1)
    # spatial_graph2 = graph_alpha(cor2)
    # righttop = np.zeros((n1,n_rest))
    # leftbottem = np.zeros((n_rest,n1))
    # first_row = np.hstack((spatial_graph1.toarray(),righttop))  #横向合并
    # second_row = np.hstack((leftbottem,spatial_graph2.toarray()))  #横向合并
    # whole_graph = np.vstack((first_row,second_row))  #纵向合并
    # spatial_graph = sparse.csr_matrix(whole_graph)
    spatial_graph = sparse.csr_matrix(arr_total)
    
    edge_list_ = sparse_mx_to_torch_edge_list(spatial_graph)
    edge_list = edge_list_.numpy()
    #edge_list = sparse_mx_to_torch_edge_list(self.spatial_graph).to(device)
    edge_tuple_list = []
    for i in range(edge_list.shape[1]):
        edge_tuple_list.append((edge_list[0,i],edge_list[1,i]))
    G = nx.Graph()
    # G.add_nodes_from([i for i in range(whole_graph.shape[0])])
    G.add_nodes_from([i for i in range(arr_total.shape[0])])
    G.add_edges_from(edge_tuple_list)
    return G,edge_list_

def transition(communities: Sequence[Sequence[int]],
               num_nodes: int) -> np.ndarray:
    classes = np.full(num_nodes, -1)
    for i, node_list in enumerate(communities):
        classes[np.asarray(node_list)] = i
    return classes

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

def get_node_cs_edge_weight(G,edge_list):
    communities = algorithms.leiden(G).communities

    com = transition(communities, G.number_of_nodes())
    com_cs, node_cs = community_strength(G, communities)
    # edge_index = torch.from_numpy(edge_list)
    edge_weight = get_edge_weight(edge_list, com, com_cs)
    com_size = [len(c) for c in communities]
    print(f'Done! {len(com_size)} communities detected. \n')
    return edge_weight,node_cs