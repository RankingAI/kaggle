import numba
import numpy as np
import pandas as pd
import time
import math

class FeatureEncoding:
    """"""
    CategoryCols = ['hashottuborspa','taxdelinquencyflag','airconditioningtypeid','architecturalstyletypeid',
                    'buildingqualitytypeid','decktypeid','heatingorsystemtypeid','pooltypeid10','pooltypeid2','pooltypeid7',
                    'propertylandusetypeid', 'regionidcity','regionidcounty','regionidneighborhood','regionidzip', 'fipsid',
                    'tractid', 'blockid']

    MinThresholdFeat = 20

    @classmethod
    def ordinal(cls, data, d_feat):
        """"""
        ## filter low ratio for feature values
        n = len(d_feat)
        s_feat = set()
        s_tmp = set()
        for k in d_feat:
            if(d_feat[k] > cls.MinThresholdFeat):
                s_feat.add(k)
            else:
                s_tmp.add(k.split(':')[0])
        for k in s_tmp:
            s_feat.add('%s:less' % k)
        d_feat = dict((k, v) for (v, k) in enumerate(list(s_feat), start = 0))
        nn = len(d_feat)
        print('original size of feature space %d, current size of feature space %d' % (n, nn))
        #print(d_feat)

        ## update category columns
        #cls.CategoryCols = [col for col in cls.CategoryCols if(col not in ['rawcensustractandblock', 'fips'])]
        #cls.CategoryCols.extend(['fipsid', 'tractid', 'blockid'])

        df_train, df_test = data

        ## one-hot encode
        df_train = cls.__OHE(df_train, d_feat)
        df_test = cls.__OHE(df_test, d_feat)
        #print('test : %d' % (len(df_test)))
        #print('train : %d' % (len(df_train)))

        df_train.drop(cls.CategoryCols, axis = 1, inplace = True)
        df_test.drop(cls.CategoryCols, axis = 1, inplace = True)

        return (df_train, df_test)

    @classmethod
    def __OHE(cls, data, d_feat):
        """"""
        headers = [v[0] for v in sorted(d_feat.items(), key=lambda x: x[1])]

        ohe = cls.__ApplyOHE(data, d_feat)
        df_ohe = pd.DataFrame(ohe, index = data.index, columns= headers)
        data = pd.concat([data, df_ohe], axis=1)

        return data

    @classmethod
    @numba.jit
    def __ApplyOHE(cls, data, d_feat):
        """"""
        n = len(data)
        result = np.zeros((n, len(d_feat)), dtype='int8')
        ##
        d_stat = {}
        for i in range(n):
            for col in cls.CategoryCols:
                v = data.ix[i, col]
                if(col not in d_stat):
                    d_stat[col] = {}
                if(pd.isnull(v)):
                    result[i, d_feat['%s:missing' % col]] = 1
                    if('missing' in d_stat[col]):
                        d_stat[col]['missing'] += 1
                    else:
                        d_stat[col]['missing'] = 1
                elif('%s:%s' % (col, v) in d_feat):
                    result[i, d_feat['%s:%s' % (col, v)]] = 1
                    if('hit' in d_stat[col]):
                        d_stat[col]['hit'] += 1
                    else:
                        d_stat[col]['hit'] = 1
                else:
                    result[i, d_feat['%s:less' % col]] = 1
                    if('less' in d_stat[col]):
                        d_stat[col]['less'] += 1
                    else:
                        d_stat[col]['less'] = 1

        ## check
        for col in d_stat:
            if(np.sum(list(d_stat[col].values())) != n):
                print('Encoding for column %s error, %d : %d. ' % (col, np.sum(list(d_stat[col].values())),n))

        return result
