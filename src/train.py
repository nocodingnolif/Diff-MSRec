import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import time
import os
from sklearn.mixture import GaussianMixture

from src.models.mdf_rec import MDFRec
from src.dataset import AmazonDataset


DATASET_NAME = 'CellPhones'  


BASE_PATH = f'data/processed/{DATASET_NAME}'


with open(f'{BASE_PATH}/meta_data.pkl', 'rb') as f:
    meta = pickle.load(f)
    NUM_ITEMS = meta['num_items']

CONFIG = {
    'num_items': NUM_ITEMS,
    'hidden_size': 128,
    
    'freq_path': f'{BASE_PATH}/item_freq_table.npy',
    'adj_path': f'{BASE_PATH}/global_graph.npz',
    'diff_steps': 50,
    'lr': 0.001,
    'batch_size': 32,
    'epochs': 20,
    'max_seq_len': 10,
    'lambda_main': 1.0,
    'lambda_diff': 0.2,
    'lambda_cl': 0.1,
    'lambda_distill': 0.1
}


def calc_gmm_contrastive_loss(h_final, i_gen, neg_items_emb):
    batch_size, num_negs, _ = neg_items_emb.shape
    distill_loss = nn.MSELoss()(h_final, i_gen.detach())

    i_gen_norm = torch.nn.functional.normalize(i_gen, dim=-1)
    neg_emb_norm = torch.nn.functional.normalize(neg_items_emb, dim=-1)
    sim_matrix = torch.matmul(i_gen_norm.unsqueeze(1), neg_emb_norm.transpose(1, 2)).squeeze(1)

    sim_values = sim_matrix.detach().cpu().numpy().reshape(-1, 1)

    try:
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(sim_values)
        means = gmm.means_.flatten()
        hard_component_idx = np.argmax(means)
        probs = gmm.predict_proba(sim_values)
        hard_probs = probs[:, hard_component_idx]
        weight_mask = torch.tensor(hard_probs, device=h_final.device).reshape(batch_size, num_negs)

       
        fn_mask = (sim_matrix > 0.95).float()
        final_weights = weight_mask * (1 - fn_mask)

    except Exception:
        final_weights = torch.ones_like(sim_matrix)

    pos_score = torch.sum(h_final * i_gen.detach(), dim=-1)
    neg_scores = torch.matmul(h_final.unsqueeze(1), neg_emb_norm.transpose(1, 2)).squeeze(1)
    loss_per_pair = torch.nn.functional.softplus(neg_scores - pos_score.unsqueeze(1))
    cl_loss = (loss_per_pair * final_weights).mean()

    return cl_loss, distill_loss



if __name__ == "__main__":
    
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备数据
    dataset = AmazonDataset(
        seq_path=f'{BASE_PATH}/train_sequences.npy',  
        max_len=CONFIG['max_seq_len'],
        num_items=CONFIG['num_items']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        drop_last=True  
    )

    
    model = MDFRec(CONFIG).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

  
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0

        
        for batch_idx, batch_data in enumerate(dataloader):
            
            item_seq = batch_data['item_seq'].to(device)
            pop_seq = batch_data['pop_seq'].to(device)
            target_item = batch_data['target_item'].to(device)
            neg_items = batch_data['neg_items'].to(device)

            
            main_loss, diff_loss, h_final, _ = model.calculate_losses({
                'item_seq': item_seq,
                'pop_seq': pop_seq,
                'target_item': target_item
            })

           
            i_gen = model.diffusion.generate(h_final)

            
            neg_emb = model.embedding_layer.id_embedding(neg_items)
            cl_loss, distill_loss = calc_gmm_contrastive_loss(h_final, i_gen, neg_emb)

            
            loss = (CONFIG['lambda_main'] * main_loss +
                    CONFIG['lambda_diff'] * diff_loss +
                    CONFIG['lambda_cl'] * cl_loss +
                    CONFIG['lambda_distill'] * distill_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch + 1} | Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} (Main:{main_loss:.2f} Diff:{diff_loss:.2f})")

        avg_loss = total_loss / len(dataloader)
        print(f"🏁 Epoch {epoch + 1} 完成 | Avg Loss: {avg_loss:.4f}")

    
    SAVE_DIR = f'saved/checkpoints/{DATASET_NAME}'
    os.makedirs(SAVE_DIR, exist_ok=True)  

    save_path = f'{SAVE_DIR}/mdf_rec_final.pth'
    torch.save(model.state_dict(), save_path)
    print(f"💾 模型已保存至 {save_path}")