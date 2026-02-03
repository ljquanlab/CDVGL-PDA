# predict.py
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add project path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from tools.feature_process import get_psite_embedding_by_blosum62_coding, get_disease_embedding_by_name
import dgl
import torch.nn as nn

# Import model classes (adjust according to your project structure)
sys.path.append(str(current_dir / ".."))
from model_feature_only import CDVGL, LinkPredictor, GRU_Model


class PDAPredictor:
    def __init__(self, model_path, device='cpu'):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the saved model
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = None
        self.load_model()
        
        # Define node types
        self.psite_node_type = 'phosphorylation_site'
        self.disease_node_type = 'disease'
        
    def load_model(self):
        """Load the trained model"""
        # print(f"Loading model: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model arguments
        args = checkpoint['args']
        
        # Since we don't need the full graph structure during prediction, we create simple placeholders
        class MockGraph:
            def __init__(self):
                self.ntypes = ['phosphorylation_site', 'disease']
                self.etypes = []
                
            def nodes(self, ntype):
                class NodeData:
                    def __init__(self):
                        self.data = {'inp': torch.zeros(1, 128).to(self.device)}
                return NodeData()
        
        # Create mock graph and dictionaries
        mock_graph = MockGraph()
        node_dict = {ntype: i for i, ntype in enumerate(mock_graph.ntypes)}
        edge_dict = {etype: i for i, etype in enumerate(mock_graph.etypes)}
        
        # Initialize model
        in_size = 128  # Adjust according to your feature dimension
        self.model = CDVGL(mock_graph, node_dict, edge_dict, in_size, type('args', (), args)())
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        # print("Model loaded successfully")
    
    def extract_sequence(self, protein_seq, site_position, window_size=35):
        """
        Extract sequence around phosphorylation site from protein sequence
        
        Args:
            protein_seq: Protein sequence string
            site_position: Phosphorylation site position (1-based index)
            window_size: Length to extract on each side
            
        Returns:
            71-length sequence string
        """
        # Convert to 0-based index
        site_idx = site_position - 1
        
        center_residue = protein_seq[site_position - 1]
        # print("Center residue (original sequence):", center_residue)
        
        # Calculate start and end positions
        start = max(0, site_idx - window_size)
        end = min(len(protein_seq), site_idx + window_size + 1)
        
        # Extract sequence
        extracted_seq = protein_seq[start:end]
        
        # Calculate padding lengths
        total_len = 2 * window_size + 1  # 71
        padding_left = window_size - (site_idx - start)
        padding_right = total_len - len(extracted_seq) - padding_left
        
        # Pad with 'X'
        padded_seq = 'X' * padding_left + extracted_seq + 'X' * padding_right
        
        return padded_seq
    
    def get_psite_features(self, short_sequence):
        """
        Get features for phosphorylation site
        
        Args:
            short_sequence: 71-length sequence string
            
        Returns:
            Sequence feature tensor [1, 71, 21]
        """
        # Use BLOSUM62 encoding
        features = get_psite_embedding_by_blosum62_coding([short_sequence])
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        return features_tensor
    
    def get_disease_features(self, disease_name):
        """
        Get features for disease
        
        Args:
            disease_name: Disease name
            
        Returns:
            Disease feature tensor [1, 128]
        """
        # Get disease features
        _, embedding = get_disease_embedding_by_name(
            disease_name=disease_name,
            model_name="tools/biobert_v1_1",
            save_to_file=True,
            output_file="tools/disease_features.txt"
        )
        
        # Convert to tensor
        features_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        
        return features_tensor
    
    def prepare_input_data(self, psite_features, disease_features):
        """
        Prepare model input data
        
        Args:
            psite_features: Phosphorylation site features [1, 71, 21]
            disease_features: Disease features [1, 128]
            
        Returns:
            Dictionary containing all necessary inputs
        """
        # Create mock data tensor [batch_size, 3]
        # Columns: site_idx, disease_idx, label
        # For prediction, label can be set to 0
        data_tensor = torch.tensor([[0, 0, 0]], dtype=torch.float32)
        
        return {
            'psite_features': psite_features.to(self.device),
            'disease_features': disease_features.to(self.device),
            'data': data_tensor.to(self.device)
        }
    
    def predict(self, protein_seq, site_position, disease_name):
        """
        Perform prediction
        
        Args:
            protein_seq: Protein sequence
            site_position: Phosphorylation site position
            disease_name: Disease name
            
        Returns:
            Prediction score and probability
        """
        # print(f"Starting prediction:")
        # print(f"  Protein sequence: {protein_seq}")
        # print(f"  Site position: {site_position}")
        # print(f"  Disease name: {disease_name}")
        
        # 1. Extract 71-length sequence
        short_seq = self.extract_sequence(protein_seq, site_position)
        # print(f"Extracted 71-length sequence: {short_seq}")
        
        # 2. Get features
        psite_features = self.get_psite_features(short_seq)
        disease_features = self.get_disease_features(disease_name)
        
        # 3. Prepare input
        input_data = self.prepare_input_data(psite_features, disease_features)
        
        # 4. Perform prediction
        with torch.no_grad():
            site_embeds = self.model.seq_encoder(input_data['psite_features'])
            site_embeds = self.model.linear(site_embeds)
            disease_embeds = self.model.linear(input_data['disease_features'])
            
            # Use link predictor
            scores = self.model.linkpredictor(
                site_embeds, 
                disease_embeds, 
                input_data['data'][:, 0].long(), 
                input_data['data'][:, 1].long()
            )
            
            # Calculate probability
            probabilities = torch.sigmoid(scores)
        
        # 5. Prediction completed! Return results
        score = scores.item()
        probability = probabilities.item()
    
        return {
            'score': score,
            'probability': probability,
            'prediction': 'Associated' if probability > 0.5 else 'Not Associated',
            'short_sequence': short_seq
        }
    
    
def main():
    """Interactive prediction mode"""
    print("Phosphorylation Site - Disease Association Prediction System")
    print("=" * 50)
    print("data/example/new_PDA.csv contains some new phosphorylation site-disease association pairs to choose from.")
    
    # Input parameters
    protein_seq = input("Please enter protein sequence: ").strip()
    site_position = int(input("Please enter phosphorylation site position (starting from 1): ").strip())
    disease_name = input("Please enter disease name: ").strip()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "model_weights/feature_only.pth"
    
    # Create predictor
    predictor = PDAPredictor(model_path=model_path, device=device)
    
    # Perform prediction
    result = predictor.predict(protein_seq, site_position, disease_name)
    
    # Display results
    print("\n" + "=" * 50)
    print("Prediction Results:")
    print("=" * 50)
    print(f"Extracted sequence: {result['short_sequence']}")
    print(f"Disease name: {disease_name}")
    print(f"Association probability: {result['probability']:.4f}")
    print(f"Prediction: {result['prediction']}")
    

if __name__ == "__main__":
    main()
    
    # python predict_feature_only.py
    
"""
Phosphorylation Site - Disease Association Prediction System
==================================================
data/new_PDA.csv contains some new PDAs to choose from.
Please enter protein sequence: MDPGAALQRRAGGGGGLGAGSPALSGGQGRRRKQPPRPADFKLQVIIIGSRGVGKTSLMERFTDDTFCEACKSTVGVDFKIKTVELRGKKIRLQIWDTAGQERFNSITSAYYRSAKGIILVYDITKKETFDDLPKWMKMIDKYASEDAELLLVGNKLDCETDREITRQQGEKFAQQITGMRFCEASAKDNFNVDEIFLKLVDDILKKMPLDILRNELSNSILSLQPEPEIPPELPPPRPHVRCC
Please enter phosphorylation site position (starting from 1): 106
Please enter disease name: Parkinson Disease

==================================================
Prediction Results:
==================================================
Extracted sequence: CKSTVGVDFKIKTVELRGKKIRLQIWDTAGQERFNSITSAYYRSAKGIILVYDITKKETFDDLPKWMKMID
Disease name: Parkinson Disease
Association probability: 0.6923
Prediction: Associated

"""
    
    
