import torch
from torch.utils.data import Dataset
import numpy as np


class AmazonDataset(Dataset):
    def __init__(self, seq_path, max_len=20, num_items=0):

        
        self.data = np.load(seq_path, allow_pickle=True)
        self.max_len = max_len
        self.num_items = num_items
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        seq = self.data[idx]['item_seq']


        target_item = seq[-1]
        input_ids = seq[:-1]

        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
        else:
           
            padding_len = self.max_len - len(input_ids)
            input_ids = [0] * padding_len + input_ids

       
        pop_seq = [1.0] * self.max_len

        
        neg_items = []
        while len(neg_items) < 10:
           
            n = np.random.randint(1, self.num_items)  
            
            if n != target_item and n not in seq:
                neg_items.append(n)

        return {
            'item_seq': torch.tensor(input_ids, dtype=torch.long),
            'pop_seq': torch.tensor(pop_seq, dtype=torch.float).unsqueeze(-1),  # [Seq, 1]
            'target_item': torch.tensor(target_item, dtype=torch.long),
            'neg_items': torch.tensor(neg_items, dtype=torch.long)
        }