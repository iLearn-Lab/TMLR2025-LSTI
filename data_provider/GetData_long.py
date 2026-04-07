import os.path
import pandas as pd
import numpy as np
import random


def makeMissingData(path, missing_ratio, seed):
    df = pd.read_csv(path)
    return makeMissingDataByDf(df=df, missing_ratio=missing_ratio, seed=seed)



def makeMissingDataByDf(df, missing_ratio, seed):
    np.random.seed(seed)

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
    for j in range(len(border1s)):
        blen = border2s[j] - border1s[j]
        # begin = np.random.uniform(low=0.1, high=0.8, size=df.shape[1]) * blen + border1s[j]
        begin = np.random.uniform(low=l_rate, high=r_rate, size=1) * blen + border1s[j]
        for i in range(df.shape[1]):
            mask[int(begin) : int(begin + missing_ratio * blen), i] = 0
    
    mask[:, 0] = 1
    df2[mask == 0] = np.nan

    return df, df2





