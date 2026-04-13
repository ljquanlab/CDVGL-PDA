import os
import sys
import dgl
import copy
import random
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import  StratifiedKFold, train_test_split
from tools.parsers import parse_args
from tools.dataload_v1 import load_edge_matrix, ConstructGraph_A
from tools.tools import compute_auc_aupr, compute_metrics 
from model import CDVGL
import time

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
seed = 42 # 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

psite = 'phosphorylation_site'
disease = 'disease'

d_file = "../data/datasets/"

def get_feature(device):
    site_blossum = np.load('../data/node_feature/psite_blosum_residue_embeddings_71_21.npy') # b,71,21
    site_blossum = torch.tensor(site_blossum, dtype=torch.float32) 
    print("site_blossum.shape:",site_blossum.shape)
    features = site_blossum.to(device)
    return features



def train(model, graph, PD_train, PD_valid, PD_test, device, features, args, prefix):
    model = model.to(device)

    model_dir = "../saved_models/"
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = f"{model_dir}/{prefix}_best_model.pth"

    # =====================================================
    # ① 如果已存在最优模型，直接加载，跳过训练
    # =====================================================
    if os.path.exists(model_save_path):
        print(f"[INFO] Found saved model: {model_save_path}, skip training.")
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

    else:
        print(f"[INFO] No saved model found, start training.")
        optimizer = torch.optim.Adam( model.parameters(), lr=args.lr, weight_decay=1e-4)
        best_val_aupr, patience = 0.0, 0
        best_model_state = None

        for epoch in range(args.n_epochs):
            model.train()
            loss, train_probs, train_labels = model(graph, PD_train, features)
            train_auc, train_aupr = compute_auc_aupr(train_probs, train_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                *_, valid_probs, valid_labels = model(graph, PD_valid, features)
                val_auc, val_aupr = compute_auc_aupr(valid_probs, valid_labels)

            if val_aupr > best_val_aupr:
                best_val_aupr = val_aupr
                patience = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1
                if patience > args.patience:
                    print("Early Stopping")
                    break

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{args.n_epochs} | "
                    f"Loss: {loss:.4f} | "
                    f"Patience: {patience}"
                )

        # =====================================================
        # ② 保存最优模型
        # =====================================================
        torch.save(
            {
                "model_state_dict": best_model_state,
                "best_val_aupr": best_val_aupr,
                "args": vars(args),
            },
            model_save_path,
        )
        print(f"[INFO] Best model saved to {model_save_path}")

        model.load_state_dict(best_model_state)
        model.eval()

    # =====================================================
    # ③ 测试阶段（不管是加载的还是刚训练的）
    # =====================================================
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            assert module.training is False

    with torch.no_grad():
        *_, test_probs, test_labels = model(graph, PD_test, features)
        test_auc, test_aupr, acc, precision, recall, f1, mcc = compute_metrics(test_probs, test_labels)
        with open('test.txt', 'a') as f:
            f.write(
                f"data: {prefix} | ACC: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} |  MCC: {mcc:.4f} | " 
                f"F1: {f1:.4f} | AUC: {test_auc:.4f} | AUPR: {test_aupr:.4f} | way: mine \n" )
        
        
    print(f'Final Test Metrics:')
    print(f'AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}')
    return test_auc, test_aupr, acc, precision, recall, f1, mcc

def load_PD_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df[["psite_id", "disease_id", "label"]].values.astype(int)

def main_hard(prefix, train_txt, test_txt): 
    print(f"prefix: {prefix}")
    args = parse_args()
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    print("1.load datasets...") 
    site_kinase, site_protein, site_function, site_process, site_site, disease_disease, disease_gene, disease_go, disease_pathway, protein_gene = load_edge_matrix() 
    
    PD_train = load_PD_csv(f"{d_file}/{prefix}/{train_txt}.csv")
    PD_test  = load_PD_csv(f"{d_file}/{prefix}/{test_txt}.csv")
    
    PD_train, PD_valid = train_test_split(PD_train, test_size=0.1,  random_state=args.seed, stratify=PD_train[:, 2])
    
    print(f"PD_train shape: {PD_train.shape}, PD_valid shape: {PD_valid.shape}, PD_test shape: {PD_test.shape}")
    PD_train = torch.tensor(PD_train).to(device)
    PD_valid = torch.tensor(PD_valid).to(device)
    PD_test = torch.tensor(PD_test).to(device)
 
    print("2.Construct train Graph...")
    graph = ConstructGraph_A(site_kinase, site_protein, site_function, site_process, site_site, disease_disease, disease_gene, disease_go, disease_pathway, protein_gene).to(device)
    
    for ntype in graph.ntypes:
        graph.nodes[ntype].data['inp'] = graph.nodes[ntype].data['inp'].to(device)
        
    for etype in graph.etypes:
        print(f"{etype}: {graph.num_edges(etype)}")
    
    for ntype in graph.ntypes:
        graph.nodes[ntype].data['inp'] = graph.nodes[ntype].data['inp'].to(device)
    
        print(f"{ntype}: {graph.num_nodes(ntype)}")
        
    node_dict = {ntype: i for i, ntype in enumerate(graph.ntypes)}
    edge_dict = {etype: i for i, etype in enumerate(graph.etypes)}
    in_size = graph.nodes[psite].data['inp'].shape[1] 
    
    print("3.load features...")
    features = get_feature(device)
    
    print("4.model init...")
    model = CDVGL(graph,node_dict,edge_dict, in_size, args)
    print("5.start train...")
    test_auc, test_aupr, acc, precision, recall, f1, mcc = train(model, graph, PD_train,PD_valid,PD_test, device,features, args, prefix )    


if __name__ == '__main__':
    main_hard(prefix='1_psite_level',train_txt = 'psite_level_train_1',test_txt = 'psite_level_test_1')
    # python 1_main_hard.py
    
    # 1_psite_level, psite_level_train_1, psite_level_test_1
    # 2_protein_level, protein_level_train_1, protein_level_test_1
    # 3_pair_level, pair_level_train, pair_level_test
    # 4_disease_level , disease_level_train_13_Alzheimer , disease_level_test_13_Alzheimer
    # 4_disease_level , disease_level_train_68_Colorectal_Neoplasms , disease_level_test_68_Colorectal_Neoplasms

