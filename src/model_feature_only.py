import copy
import os
import sys
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import apply_each
import numpy as np
torch.manual_seed(42)

import math
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from collections import defaultdict

class LinkPredictor(nn.Module):
    def __init__(self, emb_dim, model = "complex"):
        """
        emb_dim: 512
        model: "complex"
        """
        super(LinkPredictor, self).__init__()
        self.model = model.lower()
        self.emb_dim = emb_dim

        if self.model == "complex":
            assert emb_dim % 2 == 0, "The embedding dim of ComplEx must be even"
            d = emb_dim // 2
            self.rel_re = nn.Parameter(torch.randn(d))
            self.rel_im = nn.Parameter(torch.randn(d))

        else:
            raise ValueError("model must be complex")

    def forward(self, site_embeds, disease_embeds, site_idx, disease_idx):
        h = site_embeds[site_idx]      # [batch, d]
        t = disease_embeds[disease_idx]

        d = self.emb_dim // 2
        h_re, h_im = h[:, :d], h[:, d:]
        t_re, t_im = t[:, :d], t[:, d:]
        r_re, r_im = self.rel_re.unsqueeze(0), self.rel_im.unsqueeze(0)

        # ComplEx score function
        score = torch.sum(
            h_re * r_re * t_re
          + h_re * r_im * t_im
          + h_im * r_re * t_im
          - h_im * r_im * t_re,
            dim=1
        )

        return score


class GRU_Model(nn.Module): 
    def __init__(self, input_dim=21, hidden_dim=128, max_len=71):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.pos_embedding = nn.Embedding(max_len, input_dim)
        self.linear_proj = nn.Linear(input_dim, hidden_dim)

        # BiGRU
        self.gru_left = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gru_right = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.fusion = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x): 
        B, L, D = x.size()
        device = x.device

        # 位置编码
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        pos_emb = self.pos_embedding(pos_ids)  # [B, L, 21]
        x = x + pos_emb  # [B, 71, 21]

        x = self.linear_proj(x)  # [B, 71, 128]
        
        center = (L + 1) // 2

        # 划分左右片段
        left = x[:, :center, :]   # [B, 36, 128] 36
        right = x[:, center-1:, :]  # [B, 36, 128] 35
        
        out_left, h_left = self.gru_left(left)     # out_left: [B, 36, 256], h_left: [1, B, 256]
        out_right, h_right = self.gru_right(right) # out_right: [B, 36, 256], h_right: [1, B, 256]
        
        out_left = out_left.mean(dim=1)     # [B, 256]
        out_right = out_right.mean(dim=1)   # [B, 256]

        # 融合
        combined = torch.cat([out_left, out_right], dim=1)  # [B, 512]
        out = self.fusion(combined)                     # [B, 128]

        return out

class CDVGL(nn.Module):
    def __init__(self, graph,node_dict,edge_dict, in_size, config):
        super(CDVGL, self).__init__()
        self.config = config
        
        self.seq_encoder = GRU_Model(hidden_dim=in_size) 
        
        self.linear = nn.Linear(in_size, config.n_hid)
        
        self.linkpredictor = LinkPredictor(emb_dim=config.n_hid, model="complex") 
        
        
    def forward(self, graph, data, features):
        
        site_blossum = features  
        
        node1 = 'phosphorylation_site'
        node2 = 'disease'
        
        graph.nodes[node1].data["inp"] = self.seq_encoder(site_blossum)
        
        p_express = self.linear(graph.nodes[node1].data["inp"])
        d_express = self.linear(graph.nodes[node2].data["inp"])
        
        site_idx = data[:, 0].long()
        disease_idx = data[:, 1].long()
        labels = data[:, 2].float()  

        scores = self.linkpredictor(p_express, d_express, site_idx, disease_idx)
        
        criterion = nn.BCEWithLogitsLoss()
        loss3 = criterion(scores, labels)
        
        loss =  self.config.w3 * loss3 
    
        
        return loss, scores , labels 

    







