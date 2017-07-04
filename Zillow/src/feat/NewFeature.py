import datetime
import pandas as pd
import numpy as np
import gc

class NewFeature:

    @classmethod
    def create(cls,data):

        ## last gap
        data = cls.__LastGap(data)
        print('lastgap was added')

        ## month of year
        data = cls.__MonthYear(data)
        print('monthyear was added')

        ##
        data = cls.__BuildingAge(data)
        print('buildingage was added')

        return data

    @classmethod
    def __BuildingAge(cls,data):

        df_train,df_test = data
        ##
        df_train['buildingage'] = 2016 - df_train['yearbuilt']

        ##
        l_test_transdate = ['201610', '201611', '201612', '201710', '201711', '201712']
        for d in l_test_transdate:
            df_test['buildingage%s' % d] = int(d[:4]) - df_test['yearbuilt']

        return (df_train,df_test)

    @classmethod
    def __MonthYear(cls,data):

        df_train, df_test = data
        ## month year for train
        df_train['monthyear'] = df_train['transactiondate'].dt.month

        ## month year for test
        l_test_transdate = ['201610', '201611', '201612', '201710', '201711', '201712']
        for d in l_test_transdate:
            df_test['monthyear%s' % d] = int(d[-2:])

        return (df_train,df_test)

    @classmethod
    def __LastGap(cls,data):

        df_train,df_test = data
        ## last gap for train

        ## add index
        df_train_lastgap = df_train.sort_values(['parcelid', 'transactiondate'], inplace=False)
        df_train_lastgap['idx'] = range(len(df_train_lastgap))
        df_tmp = df_train_lastgap[['idx', 'parcelid', 'transactiondate']].set_index(['idx'])
        ## join with train
        df_lastgap = pd.DataFrame(cls.__ComputeGap(df_tmp), columns=['lastgap']).reset_index()
        df_train_lastgap = df_train_lastgap.merge(df_lastgap, on='idx', how='left')
        ## clean
        df_train_lastgap.drop(['idx'], inplace=True, axis=1)
        df_train = df_train_lastgap.copy()
        del df_train_lastgap, df_tmp, df_lastgap
        gc.collect()

        ## last gap for test

        df_latest_transaction = df_train.groupby(['parcelid'])['transactiondate'].max().to_frame().reset_index()
        df_latest_transaction.columns = ['parcelid', 'lasttransactiondate']
        df_test_lastgap = df_test.merge(df_latest_transaction, on='parcelid', how='left')
        l_test_transdate = ['201610', '201611', '201612', '201710', '201711', '201712']
        for d in l_test_transdate:
            mt = datetime.datetime.strptime(d, '%Y%m')
            df_test_lastgap['lastgap%s' % d] = df_test_lastgap['lasttransactiondate'].apply(
                lambda v: 0 if (pd.isnull(v)) else (mt.month - v.month) if(mt.year == 2016) else (mt.month - v.month + 12))
        #df_test_lastgap[df_test_lastgap['lastgap201610'] > 0].head(20)
        df_test_lastgap.drop(['lasttransactiondate'], axis= 1, inplace= True)
        df_test = df_test_lastgap.copy()
        del df_latest_transaction, df_test_lastgap
        gc.collect()

        return (df_train,df_test)

    @staticmethod
    def __ComputeGap(data):

        ## compute gap
        result = pd.Series(index=data.index)
        for i in data.index:
            if ((i == 0) or (data.at[i, 'parcelid'] != data.at[i - 1, 'parcelid'])):
                result[i] = 0
            else:
                result[i] = data.at[i, 'transactiondate'].month - data.at[i - 1, 'transactiondate'].month

        return result
