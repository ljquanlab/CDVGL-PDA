import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def blosum62_coding(short_seqs):
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


def process_blossum(csv_file, output_file):
    df = pd.read_csv(csv_file)
    short_seqs = df['seq'].tolist()
    window_size = len(short_seqs[0])

    load_blossum = blosum62_coding(short_seqs)#(2331, 15, 21)
    print(load_blossum.shape)
    np.save('site_blosum_features.npy', load_blossum)  # loaded_features = np.load('blosum_features.npy')
    
    blosum_features = load_blossum.reshape(len(short_seqs), -1)
    print(blosum_features.shape)#(2331, 315)


    if blosum_features.shape[1] > 128:
        pca = PCA(n_components=128)
        reduced_features = pca.fit_transform(blosum_features)
    else:
        reduced_features = blosum_features

    print(reduced_features.shape)#(2331, 21)
    with open(output_file, 'w') as f:
        for site, feature in zip(df['Site'], reduced_features):
            feature_str = "\t".join(map(str, feature))
            f.write(f"{feature_str}\n")

def get_site_by_blossum(csv_file):
    df = pd.read_csv('psite_hand.csv')
    short_seqs = df[csv_file].tolist()
    load_blossum = blosum62_coding(short_seqs)#(2331, 15, 21)
    print(load_blossum.shape)
    np.save(f'{csv_file}_blossum.npy', load_blossum)  # loaded_features = np.load('blosum_features.npy')


if __name__ == '__main__':
    input_name = '71_psite_seq'
    get_site_by_blossum(input_name)

    # python cal_feature_for_site.py