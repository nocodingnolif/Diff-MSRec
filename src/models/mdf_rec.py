import torch
import torch.nn as nn
from src.models.embeddings import HybridEmbedding
from src.models.encoders import TriViewEncoder
from src.models.diffusion import DiffusionGenerator


class MDFRec(nn.Module):
    def __init__(self, config):
        super(MDFRec, self).__init__()

        
        self.num_items = config['num_items']
        self.hidden_size = config['hidden_size']

       
        self.embedding_layer = HybridEmbedding(
            num_items=self.num_items,
            hidden_size=self.hidden_size,
            freq_table_path=config['freq_path']
        )

       
        self.encoder = TriViewEncoder(
            num_items=self.num_items,
            hidden_size=self.hidden_size,
            adj_path=config['adj_path']
        )

        
        self.diffusion = DiffusionGenerator(
            hidden_size=self.hidden_size,
            num_steps=config['diff_steps']
        )

    
        self.output_bias = nn.Parameter(torch.zeros(self.num_items))

    def forward(self, item_seq, pop_seq):
        

        
        hybrid_emb = self.embedding_layer(item_seq)

        
        base_item_table = self.embedding_layer.id_embedding.weight

        # h_final: [Batch, Hidden]
        h_final, gates = self.encoder(
            hybrid_emb=hybrid_emb,
            pop_seq=pop_seq,
            item_seq=item_seq,
            base_item_table=base_item_table
        )

       
        scores = torch.matmul(h_final, base_item_table.transpose(0, 1)) + self.output_bias

        return scores, h_final, gates

    def calculate_losses(self, batch_data):
        
        
        item_seq = batch_data['item_seq']  
        pop_seq = batch_data['pop_seq']  
        target_item = batch_data['target_item']  

       
        scores, h_final, gates = self.forward(item_seq, pop_seq)

       
        main_loss = nn.CrossEntropyLoss()(scores, target_item)

      
        target_emb = self.embedding_layer.id_embedding(target_item)

        
        diff_loss = self.diffusion.calc_loss(target_emb, h_final)

        return main_loss, diff_loss, h_final, gates


if __name__ == "__main__":
    
    config = {
        'num_items': 100,
        'hidden_size': 64,
        'freq_path': 'data/processed/item_freq_table.npy',
        'adj_path': 'data/processed/global_graph.npz',
        'diff_steps': 50
    }

    # 模拟数据 Batch=2, Seq=5
    batch_data = {
        'item_seq': torch.randint(1, 100, (2, 5)),
        'pop_seq': torch.rand(2, 5, 1) * 10,
        'target_item': torch.randint(1, 100, (2,))  
    }

    try:
        
        model = MDFRec(config)

        
        main_loss, diff_loss, h_final, _ = model.calculate_losses(batch_data)

        print("-" * 30)
        print(f"主任务 Loss: {main_loss.item():.4f}")
        print(f"扩散 Loss:   {diff_loss.item():.4f}")
        print(f"意图向量:    {h_final.shape}")
        print("-" * 30)

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"❌ 出错了: {e}")