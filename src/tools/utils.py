import os
import sys
import torch
import numpy as np
import dgl
from tools.dataload import load_edge_matrix, ConstructGraph_A 
from model import CDVGL
import pickle

class PDA_Predictor:
    def __init__(self, model_path, device=None):
        """
        Initialize predictor
        Args:
            model_path: path to saved model file
            device: specify device (cuda/cpu)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Reconstruct graph (same as during training)
        print("Constructing heterogeneous graph...")
        site_kinase, site_protein, site_function, site_process, site_site, \
        disease_disease, disease_gene, disease_go, disease_pathway, protein_gene = load_edge_matrix()
        
        self.graph = ConstructGraph_A(
            site_kinase, site_protein, site_function, site_process, 
            site_site, disease_disease, disease_gene, disease_go, 
            disease_pathway, protein_gene
        ).to(self.device)
        
        for ntype in self.graph.ntypes:
            self.graph.nodes[ntype].data['inp'] = self.graph.nodes[ntype].data['inp'].to(self.device)
        
        # Create node and edge dictionaries
        self.node_dict = {ntype: i for i, ntype in enumerate(self.graph.ntypes)}
        self.edge_dict = {etype: i for i, etype in enumerate(self.graph.etypes)}
        
        # Initialize model
        args = checkpoint['args']
        args = type('Args', (), args)()  # Convert dictionary to object
        
        in_size = self.graph.nodes['phosphorylation_site'].data['inp'].shape[1]
        
        self.model = CDVGL(
            self.graph,
            self.node_dict,
            self.edge_dict,
            in_size,
            config=args
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load saved sequence features
        self.site_features = np.load('../data/graph_feature/71_psite_seq_blossum.npy')
        print(f"Predictor initialized, using device: {self.device}")
    
    
    def predict_pair(self, site_id, disease_id):
        """
        Predict association probability for single phosphorylation site-disease pair
        Args:
            site_id: phosphorylation site ID (0-2330)
            disease_id: disease ID (0-299)
        Returns:
            score: association score (logit)
            probability: association probability (0-1)
        """
        # Prepare all site features
        all_features = torch.tensor(self.site_features, dtype=torch.float32).to(self.device)
        
        # Create data tensor
        data = torch.tensor([[site_id, disease_id, 0]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            _, score, _ = self.model(self.graph, data, all_features)
            probability = torch.sigmoid(score).item()  # Convert logit to probability
        
        return {
            'site_id': site_id,
            'disease_id': disease_id,
            'score': score.item(),
            'probability': probability,
            'prediction': 'Associated' if probability > 0.5 else 'Not Associated'
        }
    
    def predict_batch(self, pairs):
        """
        Batch prediction
        Args:
            pairs: list, each element is (site_id, disease_id)
        Returns:
            list of prediction results
        """
        results = []
        for site_id, disease_id in pairs:
            result = self.predict_pair(site_id, disease_id)
            results.append(result)
        return results