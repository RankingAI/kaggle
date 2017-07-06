import pandas as pd
import numpy as np
import dill as pickle
import time
import h5py

if __name__ == '__main__':

    # df = pd.DataFrame(data = [[20.0,'301'],[np.nan,'6'],[20.0,None],[np.nan,None],[2.0,'15']],index = range(5),columns=['col1','col2'])
    # print(df['col2'].isnull())
    # for idx in df.index:
    #     if(df.loc[idx,'col2'] == None):
    #         print('------')
    #
    # print('%s\n' % (','.join(list(df.columns))))
    # for idx in df.index:
    #     print(','.join([str(v) for v in df.ix[idx].values]))
    #     break
    #
    #
    # import os.path
    #
    # N = 1000000
    # M = 400
    # file_path = "/Users/yuanpingzhou/project/workspace/python/kaggle/Zillow/data/test.pkl"
    # data = pd.DataFrame(data = np.random.sample(N),index = range(N),columns=['col'])
    # for c in range(M):
    #     data[c] = data['col']
    # n_bytes = data.values.nbytes + data.index.nbytes + data.columns.nbytes
    # print(n_bytes)
    # max_bytes = 2 ** 31 - 1

    # start = time.time()
    # #data.to_hdf(path_or_buf=file_path,key='test',mode = 'w',complib='blosc')
    # data = pd.read_hdf(path_or_buf= file_path,key = 'test')
    # print(data.head())
    # end = time.time()
    # print('time elapsed %ds' % (end - start))

    # n_bytes = 2 ** 31 + 1000
    # data = bytearray(n_bytes)

    ## write
    # with open(file_path,'wb') as f_out:
    #     pickle.dump(data,f_out,protocol=2)

    # bytes_out = pickle.dumps(data,protocol=2)
    # print(len(bytes_out))
    # with open(file_path, 'wb') as f_out:
    #     for idx in range(0, n_bytes, max_bytes):
    #         f_out.write(bytes_out[idx:idx + max_bytes])
    # f_out.close()

    # ## read
    # bytes_in = bytearray(0)
    # input_size = os.path.getsize(file_path)
    # with open(file_path, 'rb') as f_in:
    #     for _ in range(0, input_size, max_bytes):
    #         bytes_in += f_in.read(max_bytes)
    # data2 = pickle.loads(bytes_in)
    #
    # assert (data == data2)

    df = pd.DataFrame(np.random.randn(100, 2))

    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    print(train.index)
