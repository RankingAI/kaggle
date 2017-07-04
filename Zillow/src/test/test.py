import pandas as pd
import numpy as np
import os
import time
import numba
import math

@numba.jit
def ApplyOHE(ColumnValues,headers):
    n = len(ColumnValues)
    result = np.zeros((n, len(headers)), dtype='int32')
    for i in range(n):
        v = ColumnValues[i]
        if(math.isnan(v)):
            result[i,headers['missing']] = 1
        else:
            result[i, headers[str(int(v))]] = 1

    return result

def func1(df):

    NewCols = [str(int(v)) for v in df['col1'].value_counts().index.values]
    NewCols.append('missing')
    d_col = dict((k, v) for (v, k) in enumerate(NewCols, start=0))
    oh = ApplyOHE(df['col1'],d_col)
    headers = dict((('col1_%s' % k),d_col[k]) for k in d_col)
    tmp = [v[0] for v in sorted(headers.items(), key=lambda x: x[1])]
    df_oh = pd.DataFrame(oh,index = df.index,columns=tmp)
    df = pd.concat([df, df_oh], axis=1)

    return df

def func2(df):

    NewCols = [v for v in df['col1'].value_counts().index.values]
    #NewCols = np.append(NewCols, ['missing'])
    for k in NewCols:
        df['col1_%d' % int(k)] = 0
        df.loc[df['col1'] == k, 'col1_%d' % (int(k))] = 1
    df['col1_missing'] = 0
    df.loc[df['col1'].isnull() == True, 'col1_missing'] = 1

    return df

if __name__ == '__main__':

    # out = '/Users/yuanpingzhou/project/workspace/python/kaggle/Zillow/data/SingleModel'
    #
    # if (os.path.exists(out) == False):
    #     print(out)
    #     os.makedirs(out)
    #
    # df = pd.DataFrame(data = [[20.0,'301'],[np.nan,'6'],[20.0,'50'],[np.nan,'70'],[2.0,'15']],index = range(5),columns=['col1','col2'])
    # df['missing'] = 0
    # df.loc[df['col1'].isnull() == True,'missing'] = 1
    # print(df)

    N = 300
    M = 5

    values = [v*1.0 for v in np.random.choice(M, N)]
    df = pd.DataFrame(data = values,index = range(N),columns= ['col1'])
    df['col1'] = df['col1'].astype(np.float32)
    for idx in df.index:
        if(idx % 10 == 0):
            df.loc[idx,'col1'] = np.nan

    start = time.time()

    df = func1(df)
    print(df.head())

    end = time.time()

    print('time elapsed %ds' % (end - start))

