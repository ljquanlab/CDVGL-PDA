# CDVGL-PDA

## Contacts
Any more questions, please do not hesitate to contact me: 20245227066@stu.suda.edu.cn.

## Introduction
CDVGL-PDA is a contrastive dual-view graph learning framework for phosphorylation site--disease association prediction.

## Requirements

- python 3.10.16
- pytorch 2.4.0
- numpy 2.0.1
- scikit-learn 1.6.1
- dgl  2.4.0.th24.cu121
- cuda 12.9

## Quick start
1. cd src
2. train and test
- Train and test the model on low-similarity benchmarks
```
# protein-level split
python main_hard.py --generalization_benchmarks protein_split

# or psite-level split
python main_hard.py --generalization_benchmarks psite_split
```
- Perform 10-fold cross-validation on benchmarks with different positive-to-negative ratios (including both balanced and imbalanced settings)
```
# Balanced benchmark (1:1)
python main_10fold.py --baseline_benchmarks site_disease_1_1

# Imbalanced benchmarks (1:5 and 1:10)
python main_10fold.py --baseline_benchmarks site_disease_1_5 
python main_10fold.py --baseline_benchmarks site_disease_1_10
```
- Train and test the model once on benchmarks with different positive-to-negative ratios 
```
# Balanced benchmark (1:1)
python main_random.py --baseline_benchmarks site_disease_1_1

# Imbalanced benchmarks (1:5 and 1:10)
python main_random.py --baseline_benchmarks site_disease_1_5
python main_random.py --baseline_benchmarks site_disease_1_10
```
## Data description
***data/datasets***:
- different_radio: benchmarks with different positive-to-negative ratios
- protein_split: a low-similarity benchmark based on protein-level split
- psite_split: a low-similarity benchmark based on protein-level split

***data/graph_feature***:This folder contains both the node feature files and relationship matrices among nodes. These data can be used to construct a heterogeneous graph that connect phosphorylation sites, diseases, genes, proteins, GO terms, pathways, molecular functions, biological processes, and kinases for prediction tasks. 

***data/heterogeneous graph nodes.xlsx*** :This file records the nine types of nodes in the heterogeneous graph.

***data/heterogeneous graph edges.xlsx*** :This file records the ten types of edges in the heterogeneous graph.

***data/site_disease.csv***: This file contains phosphorylation site–disease associations collected and curated from the PTMD2.0 and PhosphoSitePlus (PSP) databases.

***data/site_disease_mat.txt***: This file is the association matrix between phosphorylation sites and diseases, with a shape of [2331, 300].





