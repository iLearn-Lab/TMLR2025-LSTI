import os.path
import pandas as pd
import numpy as np
import random
import math


def makeMissingData(path, missing_ratio, seed, miss_len, max_miss_len = 1e8, seq_len = 96, type='Blackout'):
    df = pd.read_csv(path)
    return makeMissingDataByDf(df=df, missing_ratio=missing_ratio, seed=seed,
                               miss_len=miss_len, seq_len=seq_len, max_miss_len=max_miss_len, type=type)



def makeMissingDataByDf(df, missing_ratio, seed, miss_len, max_miss_len = 1e8, seq_len = 96, type='Blackout'):
    np.random.seed(seed)
    random.seed(seed)

    print("Get CNT Random:  seed={}, rate={}, len={}, type={}".format(
        seed, missing_ratio, miss_len, type))
    
    mask = np.ones(df.shape)
    if type == 'MCAR':
        mask = np.random.rand(*df.shape)
        mask[mask <= missing_ratio] = 0 
        mask[mask != 0] = 1     
    df2 = df.copy()

    num_train = int(len(df) * 0.7)
    num_test = int(len(df) * 0.2)
    num_vali = len(df) - num_train - num_test
    border1s = [0, num_train, len(df) - num_test]
    border2s = [num_train, num_train + num_vali, len(df)]
  
    for o in range(len(border1s)):
        blen = border2s[o] - border1s[o]                    # 当前区间长度
        tot_miss_len = (int)(blen * missing_ratio * (df.shape[1]-1))          # 总共要挖这么多个点
        # tot_miss_len = min(tot_miss_len, blen - 2*seq_len)  # 保证至少留2个观测区间
        # tot_miss_len = min(tot_miss_len, max_miss_len)
        l = (int) (seq_len)
        r = (int) (blen - seq_len - miss_len)

        fh = 0
        maxfx = 2
        if type == 'Disjoint':
            while np.sum(mask[border1s[o]:border2s[o]] == 0) < tot_miss_len:
                i = random.randint(l, r) + border1s[o]
                submask = mask[i:i+miss_len, :]
                if np.sum(submask == 0) > miss_len and fh < maxfx: 
                    fh += 1
                    continue
                fh = 0
                j = np.random.randint(1, df.shape[1])
                mask[i:i+miss_len, j] = 0
        elif type == 'Overlap':
            for k in range(100):
                i = random.randint(l, r) + border1s[o]
                j = np.random.randint(1, df.shape[1])
                mask[i:i+miss_len, j] = 0
            while np.sum(mask[border1s[o]:border2s[o]] == 0) < tot_miss_len:
                i = random.randint(l, r) + border1s[o]
                j = np.random.randint(1, df.shape[1])
                if (not np.any(mask[i:i+miss_len, :] == 0)) and fh < maxfx: 
                    fh += 1
                    continue
                fh = 0
                mask[i:i+miss_len, j] = 0
        elif type == 'MCAR_B':
            while np.sum(mask[border1s[o]:border2s[o]] == 0) < tot_miss_len:
                i = random.randint(l, r) + border1s[o]
                j = np.random.randint(1, df.shape[1])
                mask[i:i+miss_len, j] = 0
        elif type == 'Blackout':
            while np.sum(mask[border1s[o]:border2s[o]] == 0) < tot_miss_len:
                begin = random.randint(l, r) + border1s[o]
                for i in range(df.shape[1]):
                    mask[int(begin) : int(begin + miss_len), i] = 0
        elif type == 'MCAR':
            mask[border1s[o] : border1s[o]+seq_len, :] = 1
            mask[border2s[o]-seq_len : border2s[o], :] = 1
        
    
    mask[:, 0] = 1
    df2[mask == 0] = np.nan
    print(df.shape[0]*(df.shape[1]-1), (1-mask).sum())

    return df, df2





