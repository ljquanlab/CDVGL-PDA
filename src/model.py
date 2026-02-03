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


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau=0.5):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        """cosine similarity"""
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())                     # [N, N]
        dot_denominator = torch.mm(z1_norm, z2_norm.t()) + 1e-8  # [N, N]
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z1, z2):
        """
        z1: RGAT embedding [N, d]
        z2: HGT embedding [N, d]
        """
        z1_proj = self.proj(z1)
        z2_proj = self.proj(z2)

        # (N x N)
        sim_matrix = self.sim(z1_proj, z2_proj)   # [N, N]
        sim_matrix_t = sim_matrix.t()             # [N, N]

        pos_mask = torch.eye(z1.size(0), device=z1.device)

        # mp -> sc
        sim_matrix = sim_matrix / (sim_matrix.sum(dim=1, keepdim=True) + 1e-8)
        loss1 = -torch.log((sim_matrix * pos_mask).sum(dim=-1)).mean()

        # sc -> mp
        sim_matrix_t = sim_matrix_t / (sim_matrix_t.sum(dim=1, keepdim=True) + 1e-8)
        loss2 = -torch.log((sim_matrix_t * pos_mask).sum(dim=-1)).mean()

        # final loss
        return 0.5 * (loss1 + loss2)
    
    
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
    

class HGTLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        node_dict,
        edge_dict,
        n_heads,
        dropout=0.2,
        use_norm=False,
    ):
        
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.node_dict = node_dict
        self.edge_dict = edge_dict

        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)

        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        
        self.o_linears = nn.ModuleList()# new add
        
        self.norms = nn.ModuleList()

        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            
            self.o_linears.append(nn.Linear(in_dim, out_dim))# new add
            
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(
            torch.ones(self.num_relations, self.n_heads)
        )
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )

        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q
                sub_graph.srcdata["v_%d" % e_id] = v

                sub_graph.apply_edges(fn.u_dot_v("k", "q", "t"))
                attn_score = (
                    sub_graph.edata.pop("t").sum(-1)
                    * relation_pri
                    / self.sqrt_dk
                )
                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")

                sub_graph.edata["t"] = attn_score.unsqueeze(-1)

            G.multi_update_all(
                {
                    etype: (
                        fn.u_mul_e("v_%d" % e_id, "t", "m"),
                        fn.sum("m", "t"),
                    )
                    for etype, e_id in edge_dict.items()
                },
                cross_reducer="mean",
            )

            new_h = {}
            for ntype in G.ntypes:
                """
                Step 3: Target-specific Aggregation
                x = norm( W[node_type] * gelu( Agg(x) ) + x )
                """
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data["t"].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                
                h[ntype] = self.o_linears[n_id](h[ntype])# new add, ensure that the feature dimension of h [type] is consistent with trans_out
                
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class HGT(nn.Module):
    def __init__(
        self,
        G,
        node_dict,
        edge_dict,
        n_inp,
        n_hid,
        n_out,
        n_layers,
        n_heads,
        use_norm=False,
        dropout=0.5,
    ):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        
        self.adapt_ws = nn.ModuleDict({
            ntype: nn.Linear(G.nodes[ntype].data["inp"].shape[-1], n_inp)
            for ntype in node_dict
        })
        
        for i in range(n_layers):
            if i == 0:  # first
                if n_layers == 1: 
                    self.gcs.append(
                        HGTLayer(
                            in_dim=n_inp,out_dim=n_out,
                            node_dict=node_dict,edge_dict=edge_dict,
                            n_heads=n_heads,dropout=dropout,use_norm=use_norm))
                else:  # When there are multiple layers, the first layer is mapped to the hidden layer dimension
                    self.gcs.append(
                        HGTLayer(
                            in_dim=n_inp,out_dim=n_hid,
                            node_dict=node_dict,edge_dict=edge_dict,
                            n_heads=n_heads,dropout=dropout,use_norm=use_norm))
            elif i == n_layers - 1:  # last layer is mapped to the output dimension
                self.gcs.append(
                    HGTLayer(
                        in_dim=n_hid,out_dim=n_out,
                        node_dict=node_dict,edge_dict=edge_dict,
                        n_heads=n_heads,dropout=dropout,use_norm=use_norm))
            else:  # hidden layer
                self.gcs.append(
                    HGTLayer(
                        in_dim=n_hid,out_dim=n_hid,
                        node_dict=node_dict,edge_dict=edge_dict,
                        n_heads=n_heads,dropout=dropout,use_norm=use_norm))
        
   
    def forward(self, G, out_key=None):
        device = next(self.parameters()).device
        
        h = {}
        for ntype in G.ntypes:
            feat = G.nodes[ntype].data["inp"].to(device)
            h[ntype] = F.gelu(self.adapt_ws[ntype](feat))  
        
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
        
        return h if out_key is None else h[out_key]
    
    
    
class RGAT(nn.Module):
    def __init__(self, etypes, in_size, hid_size, out_size, n_heads=8):
        super().__init__()
        # Graph Attention Network part
        self.gat_layers = nn.ModuleList()
        # First layer
        self.gat_layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.GATConv(in_size, hid_size // n_heads, n_heads,allow_zero_in_degree=True )  
                    for etype in etypes
                },
                aggregate='mean'
            )
        )
        # Second layer
        self.gat_layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.GATConv(hid_size, hid_size // n_heads, n_heads,allow_zero_in_degree=True)
                    for etype in etypes
                },
                aggregate='mean'
            )
        )
        
        self.linear = nn.Linear(hid_size, out_size)       
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, graph):
        # print("graph.ntypes",graph.ntypes)
        x = {
            ntype: graph.nodes[ntype].data['inp']  
            for ntype in graph.ntypes
        }
        
        # GAT forward pass
        h = x
        for l, layer in enumerate(self.gat_layers):
            h = layer(graph, h) # [num_nodes, num_heads, hid_size_per_head]
            h = apply_each(
                h, lambda x: x.view(x.shape[0], -1) if len(x.shape) > 2 else x
            ) # [num_nodes, hid_size]
            if l != len(self.gat_layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
                
        h = apply_each(h, lambda x: self.linear(x))
        
        return h


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(-1)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
            
        self.last_beta = None 

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        
        self.last_beta = beta.detach().cpu()
        
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i] * beta[i]
        return z_mp



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

    def forward(self, x):  # x shape: [B, 71, 21]
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
        
        self.RGAT = RGAT(etypes=graph.etypes, in_size=in_size, hid_size=config.n_hid, out_size=config.n_hid, n_heads=config.n_head)
        
        self.HGT = HGT(graph, node_dict, edge_dict, config.n_hid, config.n_hid, config.n_hid, n_layers=config.n_layer, n_heads= config.n_head)
        
        self.contrast = Contrast(config.n_hid, tau= config.tau)
        
        self.semantic_attention = Attention(hidden_dim=config.n_hid, attn_drop= config.atten_drop) 
        
        self.linkpredictor = LinkPredictor(emb_dim=config.n_hid, model="complex") 
        
        
    def forward(self, graph, data, features):
        
        site_blossum = features # [batch,71,21] 
        
        node1 = 'phosphorylation_site'
        node2 = 'disease'
        
        graph.nodes[node1].data["inp"] = self.seq_encoder(site_blossum)
        
        RGAT_feat = self.RGAT(graph)
        HGT_feat = self.HGT(graph)
        
        loss1 = self.contrast(RGAT_feat[node1], HGT_feat[node1])
        loss2 = self.contrast(RGAT_feat[node2], HGT_feat[node2])
        
        node_fused = {}
        
        for i in [node1,node2]:
            node_fused[i] = self.semantic_attention([RGAT_feat[i],HGT_feat[i]])
        
        p_express = node_fused[node1] 
        d_express = node_fused[node2]
        
        site_idx = data[:, 0].long()
        disease_idx = data[:, 1].long()
        labels = data[:, 2].float()  

        scores = self.linkpredictor(p_express, d_express, site_idx, disease_idx)
        
        criterion = nn.BCEWithLogitsLoss()
        loss3 = criterion(scores, labels)
        
        loss =  self.config.w1 * loss1 + self.config.w2 * loss2  +  self.config.w3 * loss3 
    
        
        return loss, scores , labels 

    
     

    

    

        
