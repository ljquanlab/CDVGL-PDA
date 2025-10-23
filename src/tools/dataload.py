from random import random

import numpy
import numpy as np
import dgl
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.sparse import coo_matrix
import torch as th
import sys
import os

#10
PS_D_A = 'site_disease'

PS_PR_A = 'site_protein'
PS_K_A = 'site_kinase' 
PS_PC_A = 'site_process'
PS_F_A = 'site_function'

D_GO_A = 'disease_go'
D_GE_A = 'disease_gene'
D_PW_A = 'disease_pathway'


D_PS_A = 'disease_site'
PR_PS_A = 'protein_site'
K_PS_A = 'kinase_site' 
PC_PS_A = 'process_site'
F_PS_A = 'function_site'
GO_D_A = 'go_disease'
GE_D_A = 'gene_disease'
PW_D_A = 'pathway_disease'

PS_PS_I = 'site_site'
D_D_I = 'disease_disease'

psite = 'phosphorylation_site'
kinase = 'kinase'
protein = 'protein'
func = 'function'
process = 'process'
disease = 'disease'

go = 'GO'
gene = 'gene'
pathway = 'pathway'

#9
psite_len = 2331
kinas_len = 517
protein_len = 1236
func_len = 28
process_len = 47
disease_len = 300
go_len = 3658
gene_len = 5585
pathway_len =2261
g_path = "../data/graph_feature/" 

def load_node_feature():
    site_features = torch.tensor(np.loadtxt(g_path + "71mer_site_blossum.txt", delimiter="\t"), dtype=torch.float32) 
    disease_features = torch.tensor(np.loadtxt(g_path + "disease.txt", delimiter="\t"), dtype=torch.float32)
    
    kinase_features = torch.tensor(np.loadtxt(g_path + "kinase.txt", delimiter="\t"), dtype=torch.float32)
    protein_features = torch.tensor(np.loadtxt(g_path + "protein.txt", delimiter="\t"), dtype=torch.float32)
    func_features = torch.tensor(np.loadtxt(g_path + "function.txt", delimiter="\t"), dtype=torch.float32)
    process_features = torch.tensor(np.loadtxt(g_path + "process.txt", delimiter="\t"), dtype=torch.float32)
    
    go_features = torch.tensor(np.loadtxt(g_path + "GO.txt", delimiter="\t"), dtype=torch.float32)
    gene_features = torch.tensor(np.loadtxt(g_path + "gene.txt", delimiter="\t"), dtype=torch.float32)
    pathway_features = torch.tensor(np.loadtxt(g_path + "pathway.txt", delimiter="\t"), dtype=torch.float32)
    
    node_features = {psite: site_features, disease: disease_features, 
                     protein: protein_features,func: func_features,process: process_features, kinase:kinase_features,
                     go:go_features, gene:gene_features, pathway:pathway_features}
    return node_features  

def load_edge_feature(mat, weighted=False):
    src, dst = np.nonzero(mat) 
    
    weights = None
    if weighted:
        weights = mat[src, dst].astype(float)
        weights = torch.tensor(weights,dtype=torch.float32) 

    return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long), weights

def load_edge_matrix( ):
    site_kinase = np.loadtxt(g_path + "site_kinase_mat.txt")
    site_protein = np.loadtxt(g_path + "site_protein_mat.txt")
    site_function = np.loadtxt(g_path + "site_function_mat.txt")
    site_process = np.loadtxt(g_path + "site_process_mat.txt")
    
    disease_gene = np.loadtxt(g_path + "disease_gene_mat.txt")
    disease_go = np.loadtxt(g_path + "disease_GO_mat.txt")
    disease_pathway = np.loadtxt(g_path + "disease_pathway_mat.txt")
    
    protein_gene = np.loadtxt(g_path + "protein_gene_mat.txt")
    
    site_site = np.loadtxt(g_path + "site_site_NW_similarity.txt")  
    site_site[site_site < 0.5] = 0 
    num_site = len(site_site)
    site_site = site_site - np.identity(num_site)
    
    disease_disease = np.loadtxt(g_path + "disease_disease_similarity.txt", dtype=np.float32)
    disease_disease[disease_disease < 0.2] = 0  
    num_diease = len(disease_disease)
    disease_disease = disease_disease - np.identity(num_diease)

    return site_kinase, site_protein, site_function, site_process, site_site, disease_disease, disease_gene, disease_go, disease_pathway, protein_gene


def ConstructGraph_A(site_kinase, site_protein, site_function, site_process, site_site, disease_disease, disease_gene, disease_go, disease_pathway,protein_gene):
    
    site_kinase_src, site_kinase_dst, _ = load_edge_feature(site_kinase, weighted=False)
    site_protein_src, site_protein_dst, _ = load_edge_feature(site_protein, weighted=False)
    site_function_src, site_function_dst, _ = load_edge_feature(site_function, weighted=False)
    site_process_src, site_process_dst, _ = load_edge_feature(site_process, weighted=False)
    
    disease_gene_src, disease_gene_dst, _ = load_edge_feature(disease_gene, weighted=False)
    disease_go_src, disease_go_dst, _ = load_edge_feature(disease_go, weighted=False)   
    disease_pathway_src, disease_pathway_dst, _ = load_edge_feature(disease_pathway, weighted=False)
    
    protein_gene_src, protein_gene_dst, _ = load_edge_feature(protein_gene, weighted=False)

    site_site_src, site_site_dst, site_site_weights = load_edge_feature(site_site, weighted=True)
    disease_disease_src, disease_disease_dst, disease_disease_weights = load_edge_feature(disease_disease, weighted=True)
    
    
    PS_K_pair = (site_kinase_src, site_kinase_dst)
    PS_PR_pair = (site_protein_src, site_protein_dst)
    PS_F_pair = (site_function_src, site_function_dst)
    PS_PC_pair = (site_process_src, site_process_dst)
    
    D_GE_pair = (disease_gene_src, disease_gene_dst)
    D_GO_pair = (disease_go_src, disease_go_dst)
    D_PW_pair = (disease_pathway_src, disease_pathway_dst)    

    PS_PS_pair = (site_site_src, site_site_dst)
    D_D_pair = (disease_disease_src, disease_disease_dst)

    K_PS_pair = (site_kinase_dst, site_kinase_src)
    PR_PS_pair = (site_protein_dst, site_protein_src)
    F_PS_pair = (site_function_dst, site_function_src)
    PC_PS_pair = (site_process_dst, site_process_src)
    
    GE_D_pair = (disease_gene_dst, disease_gene_src)
    GO_D_pair = (disease_go_dst, disease_go_src)
    PW_D_pair = (disease_pathway_dst, disease_pathway_src)
    
    PR_GE_pair = (protein_gene_src, protein_gene_dst)
    GE_PR_pair = (protein_gene_dst, protein_gene_src)
    
    graph = {
        (psite,PS_K_A,kinase):PS_K_pair,
        (psite,PS_PR_A,protein):PS_PR_pair,
        (psite,PS_F_A,func):PS_F_pair,
        (psite,PS_PC_A,process):PS_PC_pair,
        
        (kinase, K_PS_A, psite): K_PS_pair,
        (protein, PR_PS_A, psite): PR_PS_pair,
        (func, F_PS_A, psite): F_PS_pair,
        (process, PC_PS_A, psite): PC_PS_pair,

        
        (psite, PS_PS_I, psite): PS_PS_pair,
        (disease, D_D_I, disease): D_D_pair,
        
        (disease, D_GE_A, gene): D_GE_pair,
        (disease, D_GO_A, go): D_GO_pair,
        (disease, D_PW_A, pathway): D_PW_pair,
        
        (gene, GE_D_A, disease): GE_D_pair,
        (go, GO_D_A, disease): GO_D_pair,
        (pathway, PW_D_A, disease): PW_D_pair,
        
        (protein, 'protein_gene', gene): PR_GE_pair,
        (gene, 'gene_protein', protein): GE_PR_pair,
        
    }
    
    G = dgl.heterograph(graph)
    
    
    node_features = load_node_feature()
    G.nodes[psite].data['inp'] = node_features[psite] 
    G.nodes[disease].data['inp'] = node_features[disease] 
    G.nodes[kinase].data['inp'] = node_features[kinase] 
    G.nodes[protein].data['inp'] = node_features[protein] 
    G.nodes[func].data['inp'] = node_features[func] 
    G.nodes[process].data['inp'] = node_features[process] 
    
    G.nodes[gene].data['inp'] = node_features[gene]
    G.nodes[go].data['inp'] = node_features[go]
    G.nodes[pathway].data['inp'] = node_features[pathway]

    if site_site_weights is not None:
        G.edges[PS_PS_I].data['weight'] = site_site_weights

    if disease_disease_weights is not None:
        G.edges[D_D_I].data['weight'] = disease_disease_weights
    
    return G


