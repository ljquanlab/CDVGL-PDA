# CDVGL-PDA

## Contacts
Any more questions, please do not hesitate to contact me: 20245227066@stu.suda.edu.cn.

## Introduction
CDVGL-PDA is a contrastive dual-view graph learning framework for phosphorylation site-disease association prediction.
![Overview of CDVGL-PDA framework](architecture.png)
## Requirements

- python 3.10.13
- pytorch 2.4.0
- numpy 2.0.1
- scikit-learn 1.6.1
- dgl  2.4.0.th24.cu121
- cuda 12.9
## Quick start

1. Unzip the zip files (71_psite_seq_blossum.zip, and site_site_NW_similarity.zip) in data/graph_feature to the current folder.
2. cd src
3. train and test
- Train and test the model on low-similarity benchmarks
```
# protein-level split
python main_hard.py --generalization_benchmarks protein_split

# or psite-level split
# python main_hard.py --generalization_benchmarks psite_split

# or pair-level split
# python main_hard.py --generalization_benchmarks pair_split
```
- Perform 10-fold cross-validation on benchmarks with different positive-to-negative ratios (including both balanced and imbalanced settings)
```
# Balanced benchmark (1:1)
python main_10fold.py --baseline_benchmarks site_disease_1_1

# or Imbalanced benchmarks (1:5 and 1:10)
# python main_10fold.py --baseline_benchmarks site_disease_1_5 
# python main_10fold.py --baseline_benchmarks site_disease_1_10
```
- Train and test the model once on benchmarks with different positive-to-negative ratios 
```
# Balanced benchmark (1:1)
python main_different_radio.py --baseline_benchmarks site_disease_1_1

# or Imbalanced benchmarks (1:5 and 1:10)
# python main_different_radio.py --baseline_benchmarks site_disease_1_5
# python main_different_radio.py --baseline_benchmarks site_disease_1_10
```
3. prediction scripts. 

These scripts support users to directly input phosphorylation sites and diseases, and predict their associations. 

- Download [BioBert](https://huggingface.co/dmis-lab/biobert-v1.1) and place it in src/tools/biobert-v1_1
- Download all pre-trained model weights and place them in src/model_weights


```
# predict_CDVGL-PDA.py : uses the full CDVGL-PDA model with heterogeneous graph information to predict associations for nodes already in the graph.

python predict_CDVGL-PDA.py

# predict_feature_only.py: uses a feature-only variant of CDVGL-PDA, ignoring graph structure, suitable for predicting associations for new nodes not present in the graph.

python predict_feature_only.py

```

## Data description
***data/datasets***:
- different_radio: benchmarks with different positive-to-negative ratios
- protein_split: a low-similarity benchmark based on protein-level split
- psite_split: a low-similarity benchmark based on protein-level split
- pair_split: a low-similarity benchmark based on pair-level split

***data/graph_feature***:This folder contains both the node feature files and relationship matrices among nodes. These data can be used to construct a heterogeneous graph that connect phosphorylation sites, diseases, genes, proteins, GO terms, pathways, molecular functions, biological processes, and kinases for prediction tasks. 

***data/example***:This folder contains the input data used by the prediction scripts, which are provided as illustrative examples.

***data/Table S30_heterogeneous graph nodes.xlsx*** :This file records the nine types of nodes in the heterogeneous graph.

***data/Table S31_heterogeneous graph edges.xlsx*** :This file records the ten types of edges in the heterogeneous graph.

***data/site_disease.csv***: This file contains phosphorylation site–disease associations collected and curated from the PTMD2.0 and PhosphoSitePlus (PSP) databases.

## Directory Description
***src/model_weights***: This folder contains the trained model weights under different experimental settings and benchmarks.

***src/tools***: This folder includes files used during model training and inference, such as data preprocessing scripts, heterogeneous graph construction code, and configuration files for training parameters.





