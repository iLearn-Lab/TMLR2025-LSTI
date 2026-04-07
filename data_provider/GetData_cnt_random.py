import os.path
import pandas as pd
import numpy as np
import random
import math


def makeMissingData(path, missing_ratio, seed, miss_len, max_miss_len = 1e8, seq_len = 96):
    df = pd.read_csv(path)
    return makeMissingDataByDf(df=df, missing_ratio=missing_ratio, seed=seed,
                               miss_len=miss_len, seq_len=seq_len, max_miss_len=max_miss_len)



def makeMissingDataByDf(df, missing_ratio, seed, miss_len, max_miss_len = 1e8, seq_len = 96):
    np.random.seed(seed)
    random.seed(seed)

    print("Get CNT Random:  seed={}, rate={}, len={}, max_len={}".format(
        seed, missing_ratio, miss_len, max_miss_len))
    
    mask = np.ones(df.shape)
    df2 = df.copy()

    num_train = int(len(df) * 0.7)
    num_test = int(len(df) * 0.2)
    num_vali = len(df) - num_train - num_test
    border1s = [0, num_train, len(df) - num_test]
    border2s = [num_train, num_train + num_vali, len(df)]

    for j in range(len(border1s)):
        blen = border2s[j] - border1s[j]                    # 当前区间长度
        tot_miss_len = (int)(blen * missing_ratio)          # 总共要挖这么多个点
        tot_miss_len = min(tot_miss_len, blen - 2*seq_len)  # 保证至少留2个观测区间
        tot_miss_len = min(tot_miss_len, max_miss_len)
        l = (int) (seq_len)
        r = (int) (blen - seq_len - miss_len)
        while np.sum(mask[border1s[j]:border2s[j], 1] == 0) < tot_miss_len:
            begin = random.randint(l, r) + border1s[j]
            for i in range(df.shape[1]):
                mask[int(begin) : int(begin + miss_len), i] = 0
    
    mask[:, 0] = 1
    df2[mask == 0] = np.nan

    return df, df2





