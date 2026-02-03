import pandas as pd
import torch
import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
def PCA_normal():
    input_file = "/data/xiang/KDSGNN/PSite_Disease/feature/disease_biobert_768.txt"  # 输入的 txt 文件路径
    output_file = "disease.txt"  # 输出的降维后 txt 文件路径

    data = np.loadtxt(input_file, delimiter='\t')  # 假设数据是 '\t' 分隔

    print(f"原始数据形状: {data.shape}")  # (1236, 1280) (517, 1280)

    pca = PCA(n_components=128)
    data_reduced = pca.fit_transform(data)

    print(f"降维后数据形状: {data_reduced.shape}")  # (1236, 128) (517, 128)

    np.savetxt(output_file, data_reduced, delimiter='\t')

    print(f"降维后的数据已保存至 {output_file}")

def get_disease_embeddings_from_similarity_by_biobert(input_csv="disease.csv", output_txt="disease_biobert.txt", model_name="dmis-lab/biobert-base-cased-v1.1"):

    df = pd.read_csv(input_csv)
    diseases = df['Mesh_name'].tolist()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval() 

    with open(output_txt, 'w', encoding='utf-8') as f:
        for disease in diseases:
            # 处理空白字符
            disease = disease.strip()
            if not disease:
                continue
            # 编码文本
            inputs = tokenizer(disease, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # 取 [CLS] token 的嵌入
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()  # shape: (768,)
            # 写入文件：疾病名 \t 768维向量
            line = '\t'.join(map(str, cls_embedding)) + '\n'
            f.write(line)

    print(f"已保存 {len(diseases)} 条嵌入到 {output_txt}")


# python cal_feature_for_disease.py