import argparse

# original:

def parse_args():
    parser = argparse.ArgumentParser(description='Phosphorylation Site-Disease Prediction')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of epochs')  
    parser.add_argument('--n_hid', type=int, default=512, help='Hidden layer dimension 512') 
    
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate 1e-3')
    parser.add_argument('--use_norm', action='store_true',default=False, help='Use layer norm  True') 

    parser.add_argument('--neg_ratio', type=int, default=1, help='Negative sample ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validate set ratio')
    
    parser.add_argument('--tau', type=float, default=0.5, help='Temperature parameter for contrastive loss')
    
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--patience", type=int, default=35,help='35') 
    # new:
    parser.add_argument('--n_layer', type=int, default=2, help=' HGT layer number')
    parser.add_argument('--n_head', type=int, default=8, help=' GNN head number')
    parser.add_argument('--atten_drop', type=float, default=0.2, help=' attention dropout rate ')
    
    parser.add_argument("--w1", type=float, default=1, help = 'the weight of loss1: 1')
    parser.add_argument("--w2", type=float, default=1, help = 'the weight of loss2: 1') 
    parser.add_argument("--w3", type=float, default=10, help = 'the weight of loss3 :10') 
    
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--baseline_benchmarks', type=str, default='site_disease_1_1',help='site_disease_1_1,site_disease_1_5,site_disease_1_10')
    parser.add_argument('--generalization_benchmarks', type=str, default='psite_split',help=' protein_split, psite_split, pair_split')
    parser.add_argument('--feature_only', type=str, default='protein_split',help='protein_split')
    
    
    return parser.parse_args()
