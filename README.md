# Tacos
![overview](https://github.com/user-attachments/assets/332799b5-57b2-4819-803c-a4cafd4c0437)
Here, we take two spatial transcriptomics data as an example. Tacos treats these two normalized gene expression matrices X1, X2 and their corresponding spatial coordinates Y1, Y2 as inputs. Tacos builds graphs G1 and G2 based on Y1 and Y2. Two augmented graph views are generated for G1 and G2. A GCN encoder with three different constraints (contrastive constraint, within-slice constraint and across-slice constraint) is used to learn the aligned low-dimensional embedding. Red edges and red nodes are masked ones.
# Installation
The Tacos is developed based on the Python libraries Scanpy, PyTorch and PyG (PyTorch Geometric) framework, and can be run on GPU (recommend) or CPU.

First, download the code to your local machine (you can download the zip file using the download button in the top-right corner, or use `git clone https://github.com/0617spec/tacos.git`.

After extracting the files, create a conda environment and install all necessary packages.

```
conda create -n env_tacos python=3.10
conda activate env_tacos
pip install -r requirement.txt
```

It is recommended to place the training `ipynb` file at the same level as the `model` folder.


# Tutorials
We provide a step-by-step tutorial using DLPFC data.

First, import the required packages:

```python
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata
from scanpy import read_10x_h5
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
```
Next, load the raw adata data and perform preprocessing.
We put all adata into one list.
```python
from model import process_adata,integrate_datasets
load_list=['151508','151675']
adata_list = []
MARKER_GENES = ['VIM','HPCAL1','CARTPT','RORB','PCP4','KRT17','MBP'] # optional
file_dir = './data/'

for section_id in load_list:
    ### read adata
    input_dir = file_dir+section_id
    adata= sc.read_visium(path=input_dir, count_file=section_id + '_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique(join="++")
    layer1=pd.read_csv(os.path.join(input_dir, 'metadata.tsv'),sep="\t",header=0)
    adata.obs['layer_guess_reordered'] = layer1.loc[:,'layer_guess_reordered']
    adata.obs_names = [x+'_'+section_id for x in adata.obs_names]
    processed = process_adata(
            adata,
            marker_genes=MARKER_GENES,
            min_genes=100,
            min_cells=50,
            n_top_genes=6000
        )
    print(f"Post-processed shape: {processed.shape}")
 
    adata_list.append(processed)
```
Then integrate all datasets into one concated adata:
```python
# Integrate all samples, get intersection gene of all slice
combined_adata = integrate_datasets(adata_list, load_list)
```
Start trainning:
```python
from model import Tacos
result_filepath = f'./results/{load_list[0]}_{load_list[-1]}/'
tacos = Tacos(combined_adata,latent_dim=50,gpu=0,check_detect=True,path=result_filepath)
train_args = { #default para
    'epoch': 1500,
    'base_w':1.0,
    'base':'csgcl',
    'spatial_w':1.5,
    'cross_w':1.0,
    'recon_w' :0.0,
    'lr':1e-3,
    'max_patience':100,
    'min_stop':500,
    'cpu':1,
    'k':50,
    'update_mnn':100,
    'save_inter':500,
    'csgcl_arg':{
        'ced_drop_rate_1' : 0.2,
        'ced_drop_rate_2' : 0.7,
        'cav_drop_rate_1' : 0.1,
        'cav_drop_rate_2' : 0.2,
        't0':5,
        'gamma':1
        },
    'cross_arg':{
        'alpha':1.0,
        # 'negative':True

    },
    'spatial_arg':{
        'regularization_acceleration':True,
        'edge_subset_sz':1000000
    }, 
    
}
path_str=tacos.train(train_args,result_filepath+'temp/')
```
The trained embedding are saved in ```tacos.embedding```
To visualize the result of Tacos, you can use the fuction ```visualize_embeddings```
It will return an adata with Tacos result in ```.obsm```
```python
from model import visualize_embeddings
adata_emb = visualize_embeddings(combined_adata, tacos.embedding,true_label_key='layer_guess_reordered')
```
The adata containing tacos result can be used for downstream analysis.
