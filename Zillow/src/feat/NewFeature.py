import datetime
import pandas as pd
import numpy as np
import gc

class NewFeature:

    _l_valid_transdate = ['201607', '201608', '201609', '201610', '201611', '201612']
    _l_test_transdate = ['201610', '201611', '201612', '201710', '201711', '201712']

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

        ##
        data = cls.__NullCount(data)
        print('nullcount was added')

        return data

    @classmethod
    def __NullCount(cls,data):
        """"""
        df_train, df_valid, df_test = data

        ##
        df_train["nullcount"] = df_train.isnull().sum(axis=1)
        df_valid["nullcount"] = df_valid.isnull().sum(axis=1)
        df_test["nullcount"] = df_test.isnull().sum(axis=1)

        return (df_train, df_valid, df_test)

    @classmethod
    def __BuildingAge(cls,data):

        df_train, df_valid, df_test = data
        ## for train
        df_train['buildingage'] = 2016 - df_train['yearbuilt']

        ##  for valid
        df_valid['buildingage'] = 2016 - df_valid['yearbuilt']

        for d in cls._l_valid_transdate:
            df_valid['buildingage%s' % d] = int(d[:4]) - df_valid['yearbuilt']

        ## for test

        for d in cls._l_test_transdate:
            df_test['buildingage%s' % d] = int(d[:4]) - df_test['yearbuilt']

        return (df_train,df_valid,df_test)

    @classmethod
    def __MonthYear(cls,data):

        df_train, df_valid, df_test = data
        ## month year for train
        df_train['monthyear'] = df_train['transactiondate'].dt.month

        ## month year for
        df_valid['monthyear'] = df_valid['transactiondate'].dt.month
        for d in cls._l_valid_transdate:
            df_valid['monthyear%s' % d] = int(d[-2:])

        ## month year for test
        for d in cls._l_test_transdate:
            df_test['monthyear%s' % d] = int(d[-2:])

        return (df_train,df_valid,df_test)

    @classmethod
    def __LastGap(cls,data):

        df_train,df_valid,df_test = data

        df_train_valid = pd.concat([df_train,df_valid],ignore_index= True)
        df_train_valid = df_train_valid.reset_index(drop = True)

        ## last gap for train
        df_train = cls.__LastGap1(df_train)
        df_valid_tmp = cls.__LastGap1(df_train_valid)[['parcelid','transactiondate','lastgap']]

        df_valid = df_valid.merge(df_valid_tmp,on = ['parcelid','transactiondate'],how = 'left')
        ## TODO
        #print(df_valid['lastgap'].isnull().sum())

        ## last gap for valid
        df_valid = cls.__LastGap2(df_train,df_valid,cls._l_valid_transdate)

        ## last gap for test
        df_test = cls.__LastGap2(df_train_valid,df_test,cls._l_test_transdate)

        return (df_train,df_valid,df_test)

    @classmethod
    def __LastGap1(cls,df_train):
        """"""
        df_train_lastgap = df_train.sort_values(['parcelid', 'transactiondate'], inplace=False)
        df_train_lastgap['idx'] = range(len(df_train_lastgap))
        df_tmp = df_train_lastgap[['idx', 'parcelid', 'transactiondate']].set_index(['idx'])
        ## join with train
        df_lastgap = pd.DataFrame(cls.__ComputeGap(df_tmp), columns=['lastgap']).reset_index()
        df_train_lastgap = df_train_lastgap.merge(df_lastgap, on='idx', how='left')
        ## clean
        df_train_lastgap.drop(['idx'], inplace=True, axis=1)

        return  df_train_lastgap

    @staticmethod
    def __LastGap2(df1,df2,PredCols):
        """"""
        df_latest_transaction = df1.groupby(['parcelid'])['transactiondate'].max().to_frame().reset_index()
        df_latest_transaction.columns = ['parcelid', 'lasttransactiondate']

        df_lastgap = df2.merge(df_latest_transaction,on = 'parcelid',how = 'left')
        for d in PredCols:
            mt = datetime.datetime.strptime(d, '%Y%m')
            df_lastgap['lastgap%s' % d] = df_lastgap['lasttransactiondate'].apply(
                lambda v: 0 if (pd.isnull(v)) else (mt.month - v.month) if(mt.year == 2016) else (mt.month - v.month + 12)
            )
        df_lastgap.drop(['lasttransactiondate'],axis= 1,inplace= True)

        return df_lastgap

    @staticmethod
    def __ComputeGap(data):

        ## compute gap
        result = pd.Series(index=data.index, dtype= 'int8')
        for i in data.index:
            if ((i == 0) or (data.at[i, 'parcelid'] != data.at[i - 1, 'parcelid'])):
                result[i] = 0
            else:
                result[i] = data.at[i, 'transactiondate'].month - data.at[i - 1, 'transactiondate'].month

        return result
