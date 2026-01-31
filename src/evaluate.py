import torch
import numpy as np
from torch.utils.data import DataLoader
import pickle
import os

from src.models.mdf_rec import MDFRec
from src.dataset import AmazonDataset


DATASET_NAME = 'CellPhones'  


BASE_PATH = f'data/processed/{DATASET_NAME}'
CHECKPOINT_PATH = f'saved/checkpoints/{DATASET_NAME}/mdf_rec_final.pth'  

BATCH_SIZE = 32
Ks = [10, 20]


def evaluate():
    

    
    with open(f'{BASE_PATH}/meta_data.pkl', 'rb') as f:
        meta = pickle.load(f)
        num_items = meta['num_items']

    
    config = {
        'num_items': num_items,
        'hidden_size': 128,
        'freq_path': f'{BASE_PATH}/item_freq_table.npy',  
        'adj_path': f'{BASE_PATH}/global_graph.npz',  
        'diff_steps': 50
    }

   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDFRec(config).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
    else:
        raise FileNotFoundError("❌ 找不到模型权重文件！请先运行 train.py")

    model.eval()

    
    test_path = f'{BASE_PATH}/test_sequences.npy' # <---
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试集: {test_path}")

    dataset = AmazonDataset(
        seq_path=test_path,
        max_len=10,
        num_items=num_items
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    
    metrics = {
        10: {'recall': 0.0, 'mrr': 0.0},
        20: {'recall': 0.0, 'mrr': 0.0}
    }
    total_samples = 0

    

    with torch.no_grad():
        for batch in dataloader:
            item_seq = batch['item_seq'].to(device)
            pop_seq = batch['pop_seq'].to(device)
            target = batch['target_item'].to(device)

            
            scores, _, _ = model(item_seq, pop_seq)

            
            scores[:, 0] = -np.inf

            
            max_k = max(Ks)
            _, topk_indices = torch.topk(scores, max_k, dim=-1)  # [Batch, 20]

            
            topk_indices = topk_indices.cpu().numpy()
            targets = target.cpu().numpy()

            for i in range(len(targets)):
                gt_item = targets[i]
                pred_items = topk_indices[i]  

                
                if gt_item in pred_items:
                    
                    rank_idx = np.where(pred_items == gt_item)[0][0]
                    rank = rank_idx + 1  

                    
                    for k in Ks:
                        if rank <= k:
                            
                            metrics[k]['recall'] += 1.0

                           
                            metrics[k]['mrr'] += 1.0 / rank

            total_samples += len(targets)

  
    print("\n" + "=" * 40)
  
    print("=" * 40)

    for k in sorted(Ks):
        avg_recall = metrics[k]['recall'] / total_samples
        avg_mrr = metrics[k]['mrr'] / total_samples

        print(f"📌 Top-{k} Metrics:")
        print(f"   Recall@{k:<2}: {avg_recall:.6f}")
        print(f"   MRR@{k:<2}:    {avg_mrr:.6f}")
        print("-" * 20)
    print("=" * 40)


if __name__ == "__main__":
    evaluate()