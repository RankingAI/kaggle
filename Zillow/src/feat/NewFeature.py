import datetime
import pandas as pd

class NewFeature:

    @classmethod
    def create(cls,data, PredCols):

        ## last gap
        data = cls.__LastGap(data, PredCols)
        print('lastgap was added')

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
        df_train, df_test = data

        ##
        df_train["nullcount"] = df_train.isnull().sum(axis=1)
        df_test["nullcount"] = df_test.isnull().sum(axis=1)

        return (df_train, df_test)

    @classmethod
    def __BuildingAge(cls,data):

        df_train, df_test = data
        ## for train
        df_train['buildingage'] = 2016 - df_train['yearbuilt']

        ## for test
        df_test['buildingage'] = 2016 - df_test['yearbuilt']

        return (df_train, df_test)

    @classmethod
    def __LastGap(cls,data, PredCols):

        df_train, df_test = data

        ## last gap for train
        df_train = cls.__LastGap1(df_train)

        ## last gap for test
        df_test = cls.__LastGap2(df_train, df_test, PredCols)

        return (df_train, df_test)

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

    @classmethod
    def __LastGap2(cls, df1, df2, PredCols):
        """"""
        df_latest_transaction = df1.groupby(['parcelid'])['transactiondate'].max().to_frame().reset_index()
        df_latest_transaction.columns = ['parcelid', 'lasttransactiondate']

        df_lastgap = df2.merge(df_latest_transaction,on = 'parcelid',how = 'left')
        for mth in PredCols:
            #mt = datetime.datetime.strptime(d, '%Y%m')
            df_lastgap['lastgap%s' % mth] = df_lastgap['lasttransactiondate'].apply(
                lambda v: 0 if (pd.isnull(v)) else (mth - v.month)
            )
        df_lastgap.drop(['lasttransactiondate'],axis= 1,inplace= True)

        return df_lastgap

    @classmethod
    def __ComputeGap(cls, data):

        ## compute gap
        result = pd.Series(index=data.index, dtype= 'int8')
        for i in data.index:
            if ((i == 0) or (data.at[i, 'parcelid'] != data.at[i - 1, 'parcelid'])):
                result[i] = 0
            else:
                result[i] = data.at[i, 'transactiondate'].month - data.at[i - 1, 'transactiondate'].month

        return result
