import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import os


class SimpleGCN(nn.Module):
    def __init__(self, num_items, hidden_size, adj_path):
        super(SimpleGCN, self).__init__()
        
        if os.path.exists(adj_path):
            print(f"[Encoder] Loading global graph from {adj_path}...")
            adj_mat = sp.load_npz(adj_path)

          
            adj_mat = adj_mat.tocoo()
            
            values = adj_mat.data
            indices = np.vstack((adj_mat.row, adj_mat.col))

            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = adj_mat.shape

           
            self.register_buffer('adj', torch.sparse_coo_tensor(i, v, torch.Size(shape)))
        else:
            raise FileNotFoundError(f"找不到全局图文件: {adj_path}")

        self.W = nn.Linear(hidden_size, hidden_size)

    def forward(self, item_embedding_table):
        

        
        support = self.W(item_embedding_table)

        
        output = torch.sparse.mm(self.adj, support)

        return output


class TriViewEncoder(nn.Module):
    def __init__(self, num_items, hidden_size, adj_path):
        super(TriViewEncoder, self).__init__()
        self.hidden_size = hidden_size

       
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

       
        self.pop_mlp = nn.Linear(1, hidden_size)
       
        self.trend_attn_W = nn.Linear(hidden_size, hidden_size)

        
        self.global_gcn = SimpleGCN(num_items, hidden_size, adj_path)

      
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3), 
            nn.Softmax(dim=-1)
        )

    def forward(self, hybrid_emb, pop_seq, item_seq, base_item_table):
   
        batch_size, seq_len, _ = hybrid_emb.shape

      
        _, h_personal = self.gru(hybrid_emb)
        h_personal = h_personal.squeeze(0)  # [Batch, Hidden]

       
        pop_emb = self.pop_mlp(pop_seq)  # [Batch, Seq, Hidden]
        trend_input = hybrid_emb + pop_emb  

       
        scores = torch.matmul(h_personal.unsqueeze(1), trend_input.transpose(1, 2))  # [Batch, 1, Seq]
        scores = scores / (self.hidden_size ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

       
        h_trend = torch.matmul(attn_weights, trend_input).squeeze(1)  # [Batch, Hidden]

        
        global_item_embs = self.global_gcn(base_item_table)  # [Num_Items, Hidden]

      
        batch_global_embs = F.embedding(item_seq, global_item_embs)

       
        mask = (item_seq != 0).float().unsqueeze(-1)
        h_global = (batch_global_embs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)  # [Batch, Hidden]

     
        concat_state = torch.cat([h_personal, h_trend, h_global], dim=1)

      
        gates = self.fusion_gate(concat_state)

        h_final = (gates[:, 0:1] * h_personal +
                   gates[:, 1:2] * h_trend +
                   gates[:, 2:3] * h_global)

        return h_final, gates




    