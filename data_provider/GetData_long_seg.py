import os.path
import pandas as pd
import numpy as np
import random


def makeMissingData(path, missing_ratio, seed):
    df = pd.read_csv(path)
    return makeMissingDataByDf(df=df, missing_ratio=missing_ratio, seed=seed)



def makeMissingDataByDf(df, missing_ratio, seed):
    np.random.seed(seed)
    random.seed(seed)

    print("LONG:  seed and rate:", seed, missing_ratio)
    
    
    mask = np.ones(df.shape)
    df2 = df.copy()

    num_train = int(len(df) * 0.7)
    num_test = int(len(df) * 0.2)
    num_vali = len(df) - num_train - num_test
    border1s = [0, num_train, len(df) - num_test]
    border2s = [num_train, num_train + num_vali, len(df)]

    l_rate = 0.1
    r_rate = (1 - missing_ratio) - 0.1
    cnt_len = 1     # 挖取2个不重叠的区间，这两个区间的长度之和为missing_ratio
    for j in range(len(border1s)):
        blen = border2s[j] - border1s[j]
        miss_len = int(missing_ratio * blen / cnt_len)
        begin_now = 0
        for k in range(cnt_len):
            low = begin_now + miss_len + 0.2 * blen
            high = blen - ((cnt_len - k - 1) * (miss_len + 0.2 * blen)) - 0.2 * blen
            begin_now = random.randint(int(low), int(high))
            begin = begin_now + border1s[j]
            # begin = np.random.uniform(low=l_rate, high=r_rate, size=1) * blen + border1s[j]
            # print(begin_now, begin_now + miss_len)
            for i in range(df.shape[1]):
                mask[int(begin) : int(begin + miss_len), i] = 0
    
    mask[:, 0] = 1
    df2[mask == 0] = np.nan

    return df, df2





