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

    print("Get CNT:  seed={}, rate={}, len={}, max_len={}".format(
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
        tot_section_cnt = math.ceil(tot_miss_len / miss_len)   # 总共需要挖的区间个数
        
        
        # 两个区间的最长间隔，保证所有区间都间隔最长不会超过限制
        max_interval_len = (int) ((blen - tot_miss_len - 2*seq_len) / tot_section_cnt)
        l = (int) (seq_len)                 # 左侧保证留有1个seq_len长度
        while tot_miss_len > 0:             # 在挖完缺失点前一直挖
            r = l + max_interval_len + 1
            interval = random.randint(0, max_interval_len)
            begin = l + interval + border1s[j]                      # 当前挖取区间的起始位置
            mlen = (int) (min(tot_miss_len, miss_len))              # 当前挖取区间的长度
            l += interval + mlen
            tot_miss_len -= mlen
            for i in range(df.shape[1]):
                mask[int(begin) : int(begin + mlen), i] = 0
            
        # print((mask[:, 1]==0).sum())
    
    mask[:, 0] = 1
    df2[mask == 0] = np.nan

    return df, df2





