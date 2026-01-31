import gzip
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.fft import fft
import os
import pickle
import random


DATASET_NAME = 'Grocery'


RAW_FILE = 'Grocery_and_Gourmet_Food_5.json.gz'


RAW_PATH = os.path.join('data/raw', RAW_FILE)
PROCESSED_DIR = f'data/processed/{DATASET_NAME}'


MIN_ITEM_COUNT = 20  
MIN_SESSION_LEN = 5 
FFT_WEEK_BUCKETS = 52  
SPLIT_RATIO = 0.8  


ENABLE_AUGMENTATION = True  
MIN_AUGMENT_LEN = 3  

os.makedirs(PROCESSED_DIR, exist_ok=True)



def parse_json(path):
    
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def preprocess():
  
    data = []
   
    for i, entry in enumerate(parse_json(RAW_PATH)):
        data.append([entry['reviewerID'], entry['asin'], entry['unixReviewTime']])
        if (i + 1) % 100000 == 0:
            print(f"  已扫描 {i + 1} 行...")

    df = pd.DataFrame(data, columns=['user_str', 'item_str', 'time'])
    print(f"📦 数据行数: {len(df)}")

    
    
    item_counts = df['item_str'].value_counts()
    valid_items = item_counts[item_counts >= MIN_ITEM_COUNT].index

    # 保留有效商品
    df = df[df['item_str'].isin(valid_items)].copy()


    
    print("✂️ 正在按 [用户, 日期] 切分会话...")

    
    df['date'] = pd.to_datetime(df['time'], unit='s').dt.date

  
    df = df.sort_values(['user_str', 'time'])

    
    sessions_series = df.groupby(['user_str', 'date'])['item_str'].apply(list)
    print(f"  初步切分出会话数: {len(sessions_series)}")


    valid_sessions_series = sessions_series[sessions_series.apply(len) >= MIN_SESSION_LEN]

    raw_sessions = valid_sessions_series.tolist()
   

    if len(raw_sessions) == 0:
        raise ValueError(
            f"❌ 过滤后没有剩余会话！请降低阈值 MIN_ITEM_COUNT ({MIN_ITEM_COUNT}) 或 MIN_SESSION_LEN ({MIN_SESSION_LEN})")

    

    final_items = set()
    for seq in raw_sessions:
        for item in seq:
            final_items.add(item)

    item_unique = sorted(list(final_items))
   
    item2id = {item: i + 1 for i, item in enumerate(item_unique)}
    num_items = len(item_unique) + 1

    full_sessions = []
    for seq in raw_sessions:
        int_seq = [item2id[s] for s in seq]
        full_sessions.append({'item_seq': int_seq})

    random.seed(42)
    random.shuffle(full_sessions)

    split_idx = int(len(full_sessions) * SPLIT_RATIO)
    train_sessions_raw = full_sessions[:split_idx]
    test_sessions_raw = full_sessions[split_idx:]

    # [训练集] -> 应用滑动窗口增强
    train_data_list = []
    if ENABLE_AUGMENTATION:
        for sess in train_sessions_raw:
            seq = sess['item_seq']
            if len(seq) > 1:
                for i in range(1, len(seq)):
                    sub_seq = seq[:i + 1]
                    
                    if len(sub_seq) >= MIN_AUGMENT_LEN:
                        train_data_list.append({'item_seq': sub_seq})
    else:
        
        train_data_list = train_sessions_raw

  
    test_data_list = []
    for sess in test_sessions_raw:
        if len(sess['item_seq']) > 1:
            test_data_list.append(sess)


    # 保存序列
    np.save(f'{PROCESSED_DIR}/train_sequences.npy', train_data_list)
    np.save(f'{PROCESSED_DIR}/test_sequences.npy', test_data_list)

    # 保存元数据
    with open(f'{PROCESSED_DIR}/meta_data.pkl', 'wb') as f:
        pickle.dump({'num_items': num_items}, f)

    return df, item2id, num_items, train_data_list


def build_freq_table(original_df, item2id, num_items):
    

    # 筛选有效数据
    valid_df = original_df[original_df['item_str'].isin(item2id.keys())].copy()
    valid_df['item_id'] = valid_df['item_str'].map(item2id)

    
    min_time = valid_df['time'].min()
    valid_df['week_id'] = ((valid_df['time'] - min_time) // (7 * 24 * 3600)).astype(int)

    
    pivot = valid_df.pivot_table(index='item_id', columns='week_id', aggfunc='size', fill_value=0)

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
            s_score = np.sum(magnitudes[:3])  # Static (低频)
            d_score = np.sum(magnitudes[3:])  # Dynamic (高频)
        else:
            s_score = np.sum(magnitudes)
            d_score = 0

        freq_features[item_id] = [s_score, d_score]

    # 归一化
    for col in range(2):
        den = freq_features[:, col].max() - freq_features[:, col].min()
        if den > 0:
            freq_features[:, col] = (freq_features[:, col] - freq_features[:, col].min()) / den

    np.save(f'{PROCESSED_DIR}/item_freq_table.npy', freq_features)
    


def build_global_graph(train_data_list, num_items):
   

    adj = sp.dok_matrix((num_items, num_items), dtype=np.float32)
    count = 0

    for sess_data in train_data_list:
        seq = sess_data['item_seq']
        
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                u, v = seq[i], seq[j]
                if u == v: continue
                adj[u, v] += 1
                adj[v, u] += 1
                count += 1

  

    adj = adj.tocsr()
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)

    norm_adj = d_mat.dot(adj)

    sp.save_npz(f'{PROCESSED_DIR}/global_graph.npz', norm_adj)
   



if __name__ == "__main__":
    if not os.path.exists(RAW_PATH):
        print(f"❌ 错误: 找不到文件 {RAW_PATH}")
        print("请检查 data/raw/ 目录下是否存在对应的数据集文件。")
    else:
        
        df_clean, item2id, num_items, train_seqs = preprocess()

        
        build_freq_table(df_clean, item2id, num_items)

       
        build_global_graph(train_seqs, num_items)

        print("\n" + "=" * 50)
        print(f"🎉 数据集 {DATASET_NAME} 预处理全部完成！")
        print(f"📁 输出目录: {PROCESSED_DIR}")
        print("=" * 50)