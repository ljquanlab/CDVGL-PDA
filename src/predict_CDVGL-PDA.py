import torch
import pandas as pd
import os
from tools.utils import PDA_Predictor
import numpy as np


def load_mapping_tables():
    # Load phosphorylation site mapping table
    site_file = "../data/example/site_with_idx_with_71psite.csv"
    disease_file = "../data/example/disease_with_idx.csv"
    
    site_df = pd.read_csv(site_file)
    disease_df = pd.read_csv(disease_file)
    
    # Create mapping dictionaries
    site_to_idx = dict(zip(site_df['Site'], site_df['site_idx']))
    disease_to_idx = dict(zip(disease_df['Mesh_name'], disease_df['disease_idx']))
    
    return site_to_idx, disease_to_idx, site_df, disease_df

def validate_site_id(site_name, site_to_idx):
    if site_name not in site_to_idx:
        raise ValueError(f"Invalid phosphorylation site name: {site_name}")
    return site_to_idx[site_name]

def validate_disease_id(disease_name, disease_to_idx):
    if disease_name not in disease_to_idx:
        raise ValueError(f"Invalid disease name: {disease_name}")
    return disease_to_idx[disease_name]

def main():
    print("="*60)
    print("Phosphorylation Site-Disease Association Prediction System")
    print("="*60)
    
    # Load mapping tables
    try:
        site_to_idx, disease_to_idx, site_df, disease_df = load_mapping_tables()
        print(f"Mapping tables loaded successfully: {len(site_to_idx)} phosphorylation sites, {len(disease_to_idx)} diseases")
    except Exception as e:
        print(f"Failed to load mapping tables: {e}")
        return
    
    # Load model
    model_path = "model_weights/psite_split.pth"
    print(f"Loading model: {model_path}")
    
    try:
        predictor = PDA_Predictor(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    while True:
        print("\n" + "-"*40)
        print("1. Single pair prediction")
        print("2. Batch prediction")
        print("3. View available phosphorylation sites and diseases")
        print("4. Exit")
        print("-"*40)
        
        choice = input("Please select an option (1-4): ").strip()
        
        if choice == '1':
            try:
                print(f"\nAvailable phosphorylation site examples: P60484_S179 | Q15116_S261 | P49768_S310 ...")
                print(f"Available disease examples: Ovarian Neoplasms | Carcinoma, Non-Small-Cell Lung | Mouth Neoplasms...")
                
                site_name = input("Please enter phosphorylation site name (For example, input : P60484_S179): ").strip()
                disease_name = input("Please enter disease name (For example, input : Ovarian Neoplasms): ").strip()
                
                # Validate and get indices
                site_idx = validate_site_id(site_name, site_to_idx)
                disease_idx = validate_disease_id(disease_name, disease_to_idx)
                
                result = predictor.predict_pair(site_idx, disease_idx)
                
                print("\n" + "="*50)
                print("Prediction Result")
                print("="*50)
                print(f"Phosphorylation site: {site_name} (ID: {site_idx})")
                print(f"Disease: {disease_name} (ID: {disease_idx})")
                # print(f"Association score: {result['score']:.4f}")
                print(f"Association probability: {result['probability']:.4%}")
                print(f"Prediction: {result['prediction']}")
                
                # Interpret probability
                prob = result['probability']
                if prob > 0.8:
                    confidence = "High confidence"
                elif prob > 0.6:
                    confidence = "Medium confidence"
                elif prob > 0.4:
                    confidence = "Low confidence"
                else:
                    confidence = "Very low confidence"
                
                print(f"Confidence: {confidence}")
                print("="*50)
                
            except ValueError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Error during prediction: {e}")
        
        elif choice == '2':
            print("\nBatch prediction function")
            print("Please prepare a CSV file with the following format:")
            print("Site,Mesh_name")
            print("Q02548_Y102, Choriocarcinoma")
            print("...")
            print("\nNote: Column names must be 'Site' and 'Mesh_name'")
            
            file_path = input("Please enter CSV file path (For example, input file name: ../data/example/test_file.csv): ").strip()
            
            if not file_path:
                print("Operation cancelled")
                continue
            
            try:
                df = pd.read_csv(file_path)
                
                if 'Site' not in df.columns or 'Mesh_name' not in df.columns:
                    print("Error: CSV file must contain 'Site' and 'Mesh_name' columns!")
                    continue
                
                # Validate and convert names to indices
                pairs = []
                valid_pairs = []
                invalid_pairs = []
                
                for i, row in df.iterrows():
                    site_name = str(row['Site']).strip()
                    disease_name = str(row['Mesh_name']).strip()
                    
                    try:
                        site_idx = validate_site_id(site_name, site_to_idx)
                        disease_idx = validate_disease_id(disease_name, disease_to_idx)
                        pairs.append((site_idx, disease_idx))
                        valid_pairs.append((site_name, disease_name))
                    except ValueError as e:
                        invalid_pairs.append((site_name, disease_name, str(e)))
                
                if invalid_pairs:
                    print(f"Warning: Found {len(invalid_pairs)} invalid pairs:")
                    for site, disease, error in invalid_pairs[:5]:  # Show only first 5
                        print(f"  {site} - {disease}: {error}")
                    if len(invalid_pairs) > 5:
                        print(f"  ... and {len(invalid_pairs)-5} more invalid pairs")
                    
                    proceed = input(f"Continue predicting {len(pairs)} valid pairs? (y/n): ").strip().lower()
                    if proceed != 'y':
                        continue
                
                if not pairs:
                    print("Error: No valid pairs for prediction!")
                    continue
                
                print(f"Starting prediction for {len(pairs)} pairs...")
                
                results = []
                for i, (site_idx, disease_idx) in enumerate(pairs, 1):
                    try:
                        result = predictor.predict_pair(site_idx, disease_idx)
                        # Add original name information
                        site_name, disease_name = valid_pairs[i-1]
                        result['Site'] = site_name
                        result['Mesh_name'] = disease_name
                        results.append(result)
                        print(f"Progress: {i}/{len(pairs)}", end='\r')
                    except Exception as e:
                        print(f"\nSkipping invalid pair ({site_idx}, {disease_idx}): {e}")
                
                # Save results
                output_df = pd.DataFrame(results)
                
                # Keep 3 decimal places for probability
                if 'probability' in output_df.columns:
                    output_df['probability'] = output_df['probability'].round(3)
                
                # Rearrange column order
                cols = ['Site', 'Mesh_name', 'probability', 'prediction']
                # Keep only existing columns
                cols = [col for col in cols if col in output_df.columns]
                output_df = output_df[cols]
                
                output_file = file_path.replace('.csv', '_predictions.csv')
                if output_file == file_path:  # If no .csv extension
                    output_file = file_path + '_predictions.csv'
                
                output_df.to_csv(output_file, index=False)
                
                print(f"\nPrediction completed! Results saved to: {output_file}")
                
                # Statistics
                pos_count = sum(1 for r in results if r['probability'] > 0.5)
                print(f"Predicted associations: {pos_count} pairs")
                print(f"Predicted non-associations: {len(results) - pos_count} pairs")
                
            except FileNotFoundError:
                print(f"Error: File {file_path} does not exist!")
            except Exception as e:
                print(f"Error during batch prediction: {e}")
        
        elif choice == '3':
            print("\n" + "="*50)
            print("Available Phosphorylation Sites and Diseases")
            print("="*50)
            print(f"Total phosphorylation sites: {len(site_df)}")
            print("\nFirst 10 phosphorylation sites:")
            for i, row in site_df.head(10).iterrows():
                print(f"  {row['Site']} -> Index: {row['site_idx']}")
            
            print(f"\nTotal diseases: {len(disease_df)}")
            print("\nFirst 10 diseases:")
            for i, row in disease_df.head(10).iterrows():
                print(f"  {row['Mesh_name']} -> Index: {row['disease_idx']}")
            
            print(f"\nComplete lists are saved in CSV files:")
            print(f"  Phosphorylation sites: ../data/site_with_idx_with_71psite.csv")
            print(f"  Diseases: ../data/disease_with_idx.csv")
            print("="*50)
        
        elif choice == '4':
            print("Thank you for using, goodbye!")
            break
        
        else:
            print("Invalid choice, please re-enter!")

if __name__ == '__main__':
    main()
    
# python predict_CDVGL-PDA.py


"""
an example:
 
1. Single pair prediction
2. Batch prediction
3. View available phosphorylation sites and diseases
4. Exit
----------------------------------------
Please select an option (1-4): 1

Available phosphorylation site examples: P60484_S179 | Q15116_S261 | P49768_S310 ...
Available disease examples: Ovarian Neoplasms | Carcinoma, Non-Small-Cell Lung | Mouth Neoplasms...
Please enter phosphorylation site name (For example, input : P60484_S179): P60484_S179
Please enter disease name (For example, input : Ovarian Neoplasms): Ovarian Neoplasms

==================================================
Prediction Result
==================================================
Phosphorylation site: P60484_S179 (ID: 1989)
Disease: Ovarian Neoplasms (ID: 142)
Association probability: 97.2247%
Prediction: Associated
Confidence: High confidence
==================================================

"""
