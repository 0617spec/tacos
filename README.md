# Tacos
![overview](https://github.com/user-attachments/assets/332799b5-57b2-4819-803c-a4cafd4c0437)
Here, we take two spatial transcriptomics data as an example. Tacos treats these two normalized gene expression matrices X1, X2 and their corresponding spatial coordinates Y1, Y2 as inputs. Tacos builds graphs G1 and G2 based on Y1 and Y2. Two augmented graph views are generated for G1 and G2. A GCN encoder with three different constraints (contrastive constraint, within-slice constraint and across-slice constraint) is used to learn the aligned low-dimensional embedding. Red edges and red nodes are masked ones.
# Installation
The Tacos is developed based on the Python libraries Scanpy, PyTorch and PyG (PyTorch Geometric) framework, and can be run on GPU (recommend) or CPU.
First clone the  repository.

```
git clone https://github.com/0617spec/tacos.git
```
Then create a conda environment for Tacos and install all the required packages:
```
conda create -n env_tacos python=3.10
conda activate env_tacos
pip install -r requiement.txt
```

# Tutorials
We provide three step by step tutorals as follows:
Tutorial 1: Integrating two DLPFC slices (10x Visium)
Tutorial 2: Integrating two MOB slices across sequencing platforms (Slide-seqV2 and Stereo-seq)
Tutorial 3: Integrating two mouse embryo slices from different platforms.
