import pandas as pd
import numpy as np
import sys, os, time
import dill as pickle
import numba
import math

class CountFeature:
    """"""
    @classmethod
    @numba.jit
    def __ApplyCateRatio(cls, ColumnValues, d, dt):
        """"""
        n = len(ColumnValues)
        result = np.zeros((n), dtype='float32')
        if(dt == 'object'):
            for i in range(n):
                v = ColumnValues[i]
                if (pd.isnull(v)):
                    result[i] = d['missing']
                else:
                    result[i] = d[v]
        else:
            for i in range(n):
                v = ColumnValues[i]
                if(math.isnan(v)):
                    result[i] = d['missing']
                else:
                    result[i] = d[v]

        return result

    @classmethod
    def GenerateCountFeature(cls, InputFile, OutputFile):
        """"""
        start = time.time()

        prop = pd.read_csv(InputFile)
        #prop = prop.sample(frac = 0.01)

        CategoryCols = ['hashottuborspa', 'taxdelinquencyflag', 'airconditioningtypeid', 'architecturalstyletypeid',
                        'buildingqualitytypeid', 'decktypeid', 'heatingorsystemtypeid', 'pooltypeid10', 'pooltypeid2',
                        'pooltypeid7', 'propertylandusetypeid', 'regionidcity', 'regionidcounty','regionidneighborhood','regionidzip']

        df_cf = pd.DataFrame(index= prop.index)
        df_cf['parcelid'] = prop['parcelid']

        for col in CategoryCols:
            start0 = time.time()

            N = len(prop)
            vcs = prop[col].value_counts()
            total = np.sum(vcs)
            d_vcs = dict(vcs)
            for vc in d_vcs:
                d_vcs[vc] = d_vcs[vc] / N
            if (total < N):
                d_vcs['missing'] = (N - total) / N

            dt = prop[col].dtype.name
            CateRatio = cls.__ApplyCateRatio(prop[col].values, d_vcs, dt)
            df_tmp = pd.DataFrame(data=CateRatio, index= df_cf.index, columns=['%sratio' % col])
            df_cf = pd.concat([df_cf, df_tmp], axis=1)

            end0 = time.time()
            print('%s was added, time consumed %ds.' % (col,(end0 - start0)))

        with open(OutputFile, 'wb') as o_file:
            pickle.dump(df_cf, o_file, -1)
        o_file.close()

        end = time.time()
        print('Add count features done, time consumed %ds' % (end - start))
