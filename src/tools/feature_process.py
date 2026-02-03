import pandas as pd
import torch
import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import os
import torch
import torch.nn as nn

def get_psite_embedding_by_blosum62_coding(short_seqs):
    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0],
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, 0],
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, 3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 0],
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 0],
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],
        'Q': [-1, 1, 0, 0, 3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0],
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, 2, 0],
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, 0],
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0],
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, 0],
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, 0],
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0],
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, 0],
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, 0],
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, 0],
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0],
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, 0],
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, 0],
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, 0],
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, 0],
        'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
    Matr = np.array([[blosum62[aa] for aa in seq] for seq in short_seqs])
    return Matr

def get_disease_embedding_by_name(
    disease_name="Heart Diseases", 
    model_name="tools/biobert_v1_1",
    pca_components=128,
    save_to_file=True,
    output_file="disease_feature.txt"
):
    """
    Get embedding representation for a specified disease name and apply PCA dimensionality reduction.
    If the disease already exists in the feature file, read it directly; otherwise, generate using model.
    
    Returns:
    --------
    tuple: (disease_name, embedding_128d)
        Disease name and corresponding 128-dimensional embedding
    """
    
    # 1. Process disease name
    disease_name = disease_name.strip()
    # print(f"Processing disease: {disease_name}")
    
    # 2. Check if disease already exists in feature file
    if save_to_file and os.path.exists(output_file):
        try:
            # Read entire feature file
            data = np.loadtxt(output_file, delimiter='\t', dtype=str, encoding='utf-8')
            
            # Handle single-line and multi-line cases
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            # Extract disease name column
            disease_names_in_file = data[:, 0]
            
            # Check if current disease exists
            indices = np.where(disease_names_in_file == disease_name)[0]
            
            if len(indices) > 0:
                # Disease exists, read feature directly
                idx = indices[0]
                embedding_128d = data[idx, 1:].astype(np.float32)
                # print(f"Found disease '{disease_name}' in file {output_file}, reading feature directly")

                return disease_name, embedding_128d
                
        except Exception as e:
            print(f"Error reading feature file, will regenerate: {e}")
    
    # 3. If disease doesn't exist, load Biobert model and generate feature
    # print(f"Disease '{disease_name}' not found in feature file, generating...")
    # print(f"Loading model: {model_name}")
    
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Get disease embedding
    with torch.no_grad():
        # Encode text
        inputs = tokenizer(disease_name, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        # Take [CLS] token embedding
        cls_embedding_768d = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    projector = nn.Linear(768, 128, bias=False)
    embedding_128d = projector(torch.from_numpy(cls_embedding_768d).float()).detach().numpy()
    
    # Save to file
    if save_to_file:
        save_disease_feature_to_file(disease_name, embedding_128d, output_file)
        # print(f"Disease feature saved to: {output_file}")
        
    return disease_name, embedding_128d


def save_disease_feature_to_file(disease_name, embedding, output_file):
    """
    Save disease feature to file, avoiding duplicate entries
    """
    try:
        # Check if file exists
        if os.path.exists(output_file):
            # Read existing data
            existing_data = np.loadtxt(output_file, delimiter='\t', dtype=str, encoding='utf-8')
            
            # Handle data dimensions
            if len(existing_data.shape) == 1:
                existing_data = existing_data.reshape(1, -1)
            
            # Check if disease already exists (prevent duplicate saving)
            disease_names_in_file = existing_data[:, 0]
            if disease_name in disease_names_in_file:
                print(f"Warning: Disease '{disease_name}' already exists in file, skipping save")
                return
            
            # Create new data row
            new_row = np.concatenate([[disease_name], embedding.astype(str)])
            
            # Merge data
            all_data = np.vstack([existing_data, new_row])
            
            print(f"Appending data to existing file {output_file}")
            
        else:
            # File doesn't exist, create new file
            all_data = np.concatenate([[disease_name], embedding.astype(str)]).reshape(1, -1)
            # print(f"Creating new file {output_file}")
        
        # Save data
        np.savetxt(output_file, all_data, delimiter='\t', fmt='%s', encoding='utf-8')
        
    except Exception as e:
        print(f"Error saving file: {e}")
        # Try simple save method
        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                line = disease_name + '\t' + '\t'.join(map(str, embedding)) + '\n'
                f.write(line)
        except Exception as e2:
            print(f"Simple save also failed: {e2}")


