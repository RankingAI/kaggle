import dill as pickle
import pandas as pd
import numpy as np
import os

class DataIO:

    ## class method, load data with pkl format
    @staticmethod
    def LoadFromPklFile(InputDir):

        with open('%s/train.pkl' % InputDir, 'rb') as i_file:
            TrainData = pickle.load(i_file)
        i_file.close()

        with open('%s/test.pkl' % InputDir, 'rb') as i_file:
            TestData = pickle.load(i_file)
        i_file.close()

        return TrainData,TestData

    ## class method, load data with text format
    @staticmethod
    def LoadFromTextFile(InputDir):

        ## raw data
        TrainData = pd.read_csv('%s/train_2016_v2.csv' % InputDir, parse_dates=['transactiondate'], header=0)
        TestData = pd.read_csv('%s/sample_submission.csv' % InputDir, header=0)
        TestData['parcelid'] = TestData['ParcelId']
        TestData.drop('ParcelId', axis=1, inplace=True)
        PropertyData = pd.read_csv('%s/properties_2016.csv' % InputDir,header=0)
        for c, dtype in zip(PropertyData.columns, PropertyData.dtypes):
            if dtype == np.float64:
                PropertyData[c] = PropertyData[c].astype(np.float32)

        ## join dynamic data with static data
        TrainData = pd.merge(TrainData, PropertyData, how='left', on='parcelid')
        TestData = pd.merge(TestData, PropertyData, how='left', on='parcelid')

        return TrainData,TestData

    ## class method, save data with pkl format
    @staticmethod
    def SaveToPklFile(Data,OutputDir):

        df_train,df_test = Data

        if(os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)

        with open('%s/train.pkl' % OutputDir, 'wb') as o_file:
            pickle.dump(df_train, o_file, -1)
        o_file.close()

        max_bytes = 2 ** 31 - 1
        bytes_out = pickle.dumps(df_test)
        n_bytes = len(bytes_out)
        with open('%s/test.pkl' % OutputDir, 'wb') as o_file:
            for idx in range(0, n_bytes, max_bytes):
                o_file.write(bytes_out[idx:idx + max_bytes])
                # too big for pickle
                #pickle.dump(df_test, o_file, -1)
        o_file.close()

        # with open('%s/test.csv' % OutputDir, 'w') as o_file:
        #     o_file.write('%s\n' % (','.join(list(df_test.columns))))
        #     for idx in df_test.index:
        #         rec = [str(v) for v in df_test.ix[idx].values]
        #         o_file.write('%s\n' % (','.join(rec)))
        # o_file.close()

    @staticmethod
    def SaveToHdfFile(Data,OutputDir):

        df_train,df_test = Data

        if(os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)

        df_train.to_hdf(path_or_buf='%s/train.hdf' % OutputDir,key='test',mode = 'w',complib='blosc')
        df_test.to_hdf(path_or_buf='%s/test.hdf' % OutputDir,key='test',mode = 'w',complib='blosc')


