import sys
import os
import pickle
import numpy as np
import pandas as pd
import gzip
import json
from scipy.fft import fft


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DATASET_NAME = 'Sports'  
RAW_FILE = 'Sports_and_Outdoors_5.json.gz'  
RAW_PATH = os.path.join('data/raw', RAW_FILE)
PROCESSED_DIR = f'data/processed/{DATASET_NAME}'


TIME_VARIANTS = {
    '1_Day': 1,
    '3_Days': 3,
    '1_Week': 7,
    '2_Weeks': 14,
    '1_Month': 30
}


MIN_ITEM_COUNT = 70  
MIN_SESSION_LEN = 5


def generate_variants():
    

    data = []
    
    with gzip.open(RAW_PATH, 'rb') as g:
        for l in g:
            entry = json.loads(l)
            data.append([entry['reviewerID'], entry['asin'], entry['unixReviewTime']])

    df = pd.DataFrame(data, columns=['user_str', 'item_str', 'time'])

   
    item_counts = df['item_str'].value_counts()
    valid_items = item_counts[item_counts >= MIN_ITEM_COUNT].index
    df = df[df['item_str'].isin(valid_items)].copy()

  
    df['date'] = pd.to_datetime(df['time'], unit='s').dt.date
    df = df.sort_values(['user_str', 'time'])
    sessions = df.groupby(['user_str', 'date'])['item_str'].apply(list)
    valid_sessions = sessions[sessions.apply(len) >= MIN_SESSION_LEN]


    final_items = set()
    for seq in valid_sessions:
        for item in seq:
            final_items.add(item)
    item_unique = sorted(list(final_items))
    item2id = {item: i + 1 for i, item in enumerate(item_unique)}
    num_items = len(item_unique) + 1




    valid_df = df[df['item_str'].isin(item2id.keys())].copy()
    valid_df['item_id'] = valid_df['item_str'].map(item2id)
    min_time = valid_df['time'].min()

    output_dir = f'{PROCESSED_DIR}/fft_variants'
    os.makedirs(output_dir, exist_ok=True)

    for name, days in TIME_VARIANTS.items():
  


        bucket_size = days * 24 * 3600
        valid_df['bucket_id'] = ((valid_df['time'] - min_time) // bucket_size).astype(int)


        pivot = valid_df.pivot_table(index='item_id', columns='bucket_id', aggfunc='size', fill_value=0)

        freq_features = np.zeros((num_items, 2))
        signals = pivot.values
        item_indices = pivot.index.tolist()

        for i, item_id in enumerate(item_indices):
            signal = signals[i]
            if len(signal) < 2: continue

            fft_coeffs = np.abs(fft(signal))
            n = len(signal)
            magnitudes = fft_coeffs[:n // 2]

            if len(magnitudes) > 3:
                s_score = np.sum(magnitudes[:3])
                d_score = np.sum(magnitudes[3:])
            else:
                s_score = np.sum(magnitudes)
                d_score = 0
            freq_features[item_id] = [s_score, d_score]

        # 归一化
        for col in range(2):
            den = freq_features[:, col].max() - freq_features[:, col].min()
            if den > 0:
                freq_features[:, col] = (freq_features[:, col] - freq_features[:, col].min()) / den

       
        save_path = f'{output_dir}/freq_table_{name}.npy'
        np.save(save_path, freq_features)
       


if __name__ == "__main__":
    generate_variants()