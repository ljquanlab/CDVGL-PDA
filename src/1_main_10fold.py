import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) # 只是帮忙导入函数，对文件中的路径没用
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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

psite = 'phosphorylation_site'
disease = 'disease'

def get_feature(device):
    site_blossum = np.load('../data/node_feature/psite_blosum_residue_embeddings_71_21.npy') # b,71,21
    site_blossum = torch.tensor(site_blossum, dtype=torch.float32) 
    print("site_blossum.shape:",site_blossum.shape)
    features = site_blossum.to(device)
    return features

def train(model, graph, PD_train,PD_valid,PD_test, device,features, args, prefix):
    model = model.to(device)   
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val_aupr, patience = 0., 0
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
            print(f"patience:{patience}")
            print(f'Epoch {epoch+1}/{args.n_epochs}, a_loss: {loss:.4f} \n'
                  f'Train AUC: {train_auc:.4f}, Train AUPR: {train_aupr:.4f}, Valid AUC: {val_auc:.4f}, Valid AUPR: {val_aupr:.4f}')

    model.load_state_dict(best_model_state)
    model.eval() 
    
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            assert module.training == False

    with torch.no_grad():
        *_, test_probs, test_labels = model(graph, PD_test, features)
        test_auc, test_aupr, acc, precision, recall, f1, mcc = compute_metrics(test_probs, test_labels)
        with open('result.txt', 'a') as f:
            f.write(
                f"data: {prefix} | ACC: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | "
                f"F1: {f1:.4f} | MCC: {mcc:.4f} | AUC: {test_auc:.4f} | AUPR: {test_aupr:.4f} | way: mine \n" )
        
        
    print(f'Final Test Metrics:')
    print(f'AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}')
    return test_auc, test_aupr, acc, precision, recall, f1, mcc

def load_PD_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df[["psite_id", "disease_id", "label"]].values.astype(int)

def main_10fold_cross_validation(prefix,train_txt): 
    args = parse_args()
    
    print(f"prefix: {prefix}")
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    print("1.load datasets...")
    site_kinase, site_protein, site_function, site_process, site_site, disease_disease, disease_gene, disease_go, disease_pathway, protein_gene = load_edge_matrix()
    
    complete_data = load_PD_csv(f"{d_file}/{prefix}/{train_txt}.csv")
    
    print("2.Construct train Graph...")
    graph = ConstructGraph_A(site_kinase, site_protein, site_function, site_process, site_site, disease_disease, disease_gene, disease_go, disease_pathway, protein_gene).to(device)
    for ntype in graph.ntypes:
        graph.nodes[ntype].data['inp'] = graph.nodes[ntype].data['inp'].to(device)
        
    node_dict = {ntype: i for i, ntype in enumerate(graph.ntypes)}
    edge_dict = {etype: i for i, etype in enumerate(graph.etypes)}
    in_size = graph.nodes[psite].data['inp'].shape[1] 

    print("3.load features...")
    features = get_feature(device)
    
    auc_list, aupr_list = [], []
    acc_list, precision_list, recall_list, f1_list, mcc_list = [], [], [], [], []
    all_results=[]
    
    print("4.ten fold cross-validation...")
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(complete_data[:, :2],complete_data[:, 2])):
        print(f"\n=== Fold {fold + 1}/10 ===")
    
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1, random_state=args.seed,stratify=complete_data[train_val_idx, 2])
        PD_train = complete_data[train_idx]
        PD_valid = complete_data[val_idx]
        PD_test = complete_data[test_idx]
        
        print("model init...")
        model = CDVGL(graph,node_dict,edge_dict, in_size, args)
        
        PD_train = torch.tensor(PD_train).to(device)
        PD_valid = torch.tensor(PD_valid).to(device)
        PD_test = torch.tensor(PD_test).to(device)
        
        print("start train...")
        test_auc, test_aupr, acc, precision, recall, f1, mcc = train(model, graph, PD_train,PD_valid,PD_test, device,features, args, prefix)
        
        auc_list.append(test_auc)
        aupr_list.append(test_aupr)
        acc_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        mcc_list.append(mcc)
        
    
        all_results.append({
            'method': "CDGVL",
            'data': prefix,
            'fold': fold + 1,
            'acc': round(acc, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'mcc': round(mcc, 4),
            'auc': round(test_auc, 4),
            'aupr': round(test_aupr, 4)
        })
    
    print("\n=== Cross-Validation Results ===")
    print(f"F1-score : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print(f"AUC      : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
    print(f"AUPR     : {np.mean(aupr_list):.4f} ± {np.std(aupr_list):.4f}")

    # (1)拎出来合并：
    mean_std_row = {
        'method': "CDGVL",
        'data': prefix,
        'fold': 'mean±std',
        'acc': f"{np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}",
        'precision': f"{np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}",
        'recall': f"{np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}",
        'f1': f"{np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}",
        'mcc': f"{np.mean(mcc_list):.4f} ± {np.std(mcc_list):.4f}",
        'auc': f"{np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}",
        'aupr': f"{np.mean(aupr_list):.4f} ± {np.std(aupr_list):.4f}"
    }

    df_mean_std = pd.DataFrame([mean_std_row])
    df_mean_std.to_csv(
        csv_path_2,
        mode='a',
        header=not os.path.exists(csv_path_2),
        index=False
    )
    
    # (2)全部：
    all_results.extend([mean_std_row])
    df = pd.DataFrame(all_results)
    df.to_csv(
        csv_path,
        mode='a',
        header=not os.path.exists(csv_path),
        index=False
    )
    print(f"\nResults saved to {csv_path}")
    print(f"Mean ± Std results saved to {csv_path_2}")
    
    with open('test_10fold.txt', 'a') as f:
        f.write(
            f"datasets: {prefix}  | "
            f"way: CDVGL | "
            f"ACC: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f} | "
            f"Precision: {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f} | "
            f"Recall: {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f} | "
            f"F1: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f} | "
            f"MCC: {np.mean(mcc_list):.4f} ± {np.std(mcc_list):.4f} | "
            f"AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f} | "
            f"AUPR: {np.mean(aupr_list):.4f} ± {np.std(aupr_list):.4f}\n"
        )
    
d_file = "../data/datasets/"
datasets = "5_random"  
train_name = "PDA_1_all"
csv_path = "results/1_cv_10fold_all.csv"
csv_path_2 = "results/1_cv_10fold_all_mean_std.csv"

# PDA_1_1  PDA_1_5  PDA_1_all
# 1_cv_10fold | 1_cv_10fold_5 | 1_cv_10fold_all
# 1_cv_10fold_mean_std | 1_cv_10fold_5_mean_std | 1_cv_10fold_all_mean_std

if __name__ == '__main__':
    main_10fold_cross_validation(prefix=datasets,train_txt=train_name) 
    
    # python 1_main_10fold.py

