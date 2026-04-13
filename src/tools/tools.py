from sklearn.model_selection import train_test_split
import torch
import dgl
import pandas as pd
import numpy as np
import sys
import random
from sklearn.metrics import ( roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef )
import os
# from tools.dataload_v1 import load_edge_matrix

sys.path.append('../')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def load_dataset_random(args):
    site_disease = np.loadtxt("../data/site_disease_mat.txt")
    print("neg_ratio",args.neg_ratio)
    positive_index = []
    negative_index = []

    for i in range(np.shape(site_disease)[0]):
        for j in range(np.shape(site_disease)[1]):
            if int(site_disease[i][j]) == 1:
                positive_index.append([i, j])
            elif int(site_disease[i][j]) == 0:
                negative_index.append([i, j])

    n_pos = len(positive_index)
    n_neg = min(int(args.neg_ratio * n_pos), len(negative_index))
    negative_sample_index = np.random.choice(len(negative_index), size=n_neg, replace=False)
    
    data_set = np.zeros((n_pos + n_neg, 3), dtype=int)
    data_set[:n_pos, :2] = positive_index
    data_set[:n_pos, 2] = 1
    data_set[n_pos:, :2] = np.array(negative_index)[negative_sample_index]
    data_set[n_pos:, 2] = 0
        
    np.random.shuffle(data_set)
    print(data_set.shape)
    # save_dataset_all(data_set)
    return data_set


def save_dataset_all(dataset, prefix="site_disease_dataset"):
    np.savetxt(f"{prefix}.txt", dataset, fmt="%d", delimiter="\t")

    
def compute_auc_aupr(probs, label):
    probs = torch.sigmoid(probs)
    auc = roc_auc_score(label.detach().cpu().numpy(), probs.detach().cpu().numpy())
    aupr = average_precision_score(label.detach().cpu().numpy(), probs.detach().cpu().numpy())
    return auc, aupr

#计算指标
def compute_metrics(probs, labels, threshold=0.5):
    probs = torch.sigmoid(probs)
    
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    preds_np = (probs_np >= threshold).astype(int)

    auc = roc_auc_score(labels_np, probs_np)
    aupr = average_precision_score(labels_np, probs_np)
    
    acc = accuracy_score(labels_np, preds_np)
    precision = precision_score(labels_np, preds_np, zero_division=0)
    recall = recall_score(labels_np, preds_np, zero_division=0)
    f1 = f1_score(labels_np, preds_np, zero_division=0)
    mcc = matthews_corrcoef(labels_np, preds_np)

    return auc,aupr,acc,precision,recall,f1,mcc
    
def save_logits_by_np(test_probs,test_labels,prefix):
    # 转成 numpy
    test_probs = torch.sigmoid(test_probs)
    
    test_probs_np = test_probs.detach().cpu().numpy()
    test_labels_np = test_labels.detach().cpu().numpy()

    # 保存
    np.save(f'../analysis_data_v1/probs_labels/{prefix}_CDVGL_test_probs.npy', test_probs_np)
    np.save(f'../analysis_data_v1/probs_labels/{prefix}_CDVGL_test_labels.npy', test_labels_np)