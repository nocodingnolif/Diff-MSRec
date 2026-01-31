import torch
import torch.nn as nn
import numpy as np
import os


class HybridEmbedding(nn.Module):
    def __init__(self, num_items, hidden_size, freq_table_path, dropout=0.1):
        super(HybridEmbedding, self).__init__()

        self.num_items = num_items
        self.hidden_size = hidden_size

       
        self.id_embedding = nn.Embedding(num_items, hidden_size, padding_idx=0)

      
        if os.path.exists(freq_table_path):
            print(f"[HybridEmbedding] Loading freq table from {freq_table_path}...")
            freq_data = np.load(freq_table_path)  # Shape: [N_items, 2]

           
            freq_tensor = torch.FloatTensor(freq_data)

           
            self.freq_lookup = nn.Embedding.from_pretrained(freq_tensor, freeze=True)
        else:
            raise FileNotFoundError(f"找不到频域特征表: {freq_table_path}")

       
        self.freq_mlp = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),  # 激活函数
            nn.Linear(hidden_size, hidden_size)  
        )

       
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, item_seq):
       

        
        id_emb = self.id_embedding(item_seq)

      
        freq_raw = self.freq_lookup(item_seq)

       
        freq_emb = self.freq_mlp(freq_raw)

       
        hybrid_emb = id_emb + freq_emb

     
        hybrid_emb = self.layer_norm(hybrid_emb)
        hybrid_emb = self.dropout(hybrid_emb)

        return hybrid_emb



if __name__ == "__main__":
    # 配置
    NUM_ITEMS = 100
    HIDDEN_SIZE = 64
    FREQ_PATH = 'data/processed/item_freq_table.npy'

  
    dummy_input = torch.tensor([
        [1, 5, 10, 0, 0],  
        [99, 50, 2, 8, 7]  
    ], dtype=torch.long)

    try:
        # 实例化模型
        model = HybridEmbedding(NUM_ITEMS, HIDDEN_SIZE, FREQ_PATH)

        # 前向传播
        output = model(dummy_input)

     
    except Exception as e:
        print(f"❌ 出错了: {e}")