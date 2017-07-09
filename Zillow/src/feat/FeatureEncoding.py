import numba
import numpy as np
import pandas as pd
import time
import math

class FeatureEncoding:

    @classmethod
    def ordinal(cls,data):

        ##
        #data = cls.__SimpleOrdinal(data)

        ##
        data = cls.__OHE(data)

        return data

    @classmethod
    def __OHE(cls,data):

        df_train, df_valid, df_test = data

        CategoryCols = ['hashottuborspa','taxdelinquencyflag','airconditioningtypeid','architecturalstyletypeid',
                        'buildingqualitytypeid','decktypeid','fips','heatingorsystemtypeid','pooltypeid10','pooltypeid2','pooltypeid7',
                        'propertylandusetypeid','rawcensustractandblock','regionidcity','regionidcounty','regionidneighborhood','regionidzip']

        for cc in CategoryCols:

            dt = df_train[cc].dtype.name

            start0 = time.time()
            ValueCounts = [str(int(v)) if(dt != 'object') else v for v in df_train[cc].value_counts().index.values]
            ValueCounts.append('missing')
            SelectedValues = dict((k, v) for (v, k) in enumerate(ValueCounts, start=0))
            OHTr = cls.__ApplyOH(df_train[cc], SelectedValues,dt)
            OHVa = cls.__ApplyOH(df_valid[cc],SelectedValues,dt)
            OHTe = cls.__ApplyOH(df_test[cc],SelectedValues,dt)

            headers = dict((('%s_%s' % (cc,k)),SelectedValues[k]) for k in SelectedValues)
            tmp = [v[0] for v in sorted(headers.items(), key=lambda x: x[1])]
            OHDFTr = pd.DataFrame(OHTr, index= df_train.index, columns= tmp)
            OHDFVa = pd.DataFrame(OHVa, index= df_valid.index, columns= tmp)
            OHDFTe = pd.DataFrame(OHTe, index= df_test.index, columns= tmp)
            end0 = time.time()
            print('ohe, time elapsed %ds' % (end0 - start0))

            start1 = time.time()
            df_train = pd.concat([df_train, OHDFTr], axis= 1)
            df_valid = pd.concat([df_valid, OHDFVa], axis= 1)
            df_test = pd.concat([df_test, OHDFTe], axis= 1)
            end1 = time.time()
            print('concat, time elapsed %ds' % (end1 - start1))

            df_train.drop(cc, axis= 1, inplace= True)
            df_valid.drop(cc, axis= 1, inplace= True)
            df_test.drop(cc, axis=1, inplace=True)
            print('Column %s was encoded.' % cc)

        return (df_train, df_valid, df_test)

    @classmethod
    def __SimpleOrdinal(cls,data):

        df_train, df_test = data

        for c in df_train.dtypes[df_train.dtypes == object].index.values:
            df_train[c] = (df_train[c] == True)
        for c in df_test.dtypes[df_test.dtypes == object].index.values:
            df_test[c] = (df_test[c] == True)

        return (df_train,df_test)

    ## speed-up version of apply function
    @classmethod
    @numba.jit
    def __ApplyOH(cls,ColumnValues, headers,dt):

        n = len(ColumnValues)
        result = np.zeros((n, len(headers)), dtype='int8')
        if(dt == 'object'):
            for i in range(n):
                v = ColumnValues[i]
                if(pd.isnull(v)):
                    result[i,headers['missing']] = 1
                elif(v in headers):
                     result[i,headers[v]] = 1
        else:
            for i in range(n):
                v = ColumnValues[i]
                if(math.isnan(v)):
                    result[i,headers['missing']] = 1
                elif(('%d' % int(v)) in headers):
                    result[i,headers['%d' % int(v)]] = 1

        return result
