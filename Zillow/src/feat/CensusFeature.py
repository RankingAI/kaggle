import pandas as pd
import numpy as np
import sys, os, time
import dill as pickle
import numba
import math

class CensusFeature:
    """"""
    @classmethod
    #@numba.jit
    def __ApplyParseCensus(cls, ColumnValues1, ColumnValues2):
        """"""
        n = len(ColumnValues1)
        result = np.zeros((n, 3), dtype= 'int32')
        for i in range(n):
            v = ColumnValues1[i]
            if(math.isnan(v)):
                if(math.isnan(ColumnValues2[i])):
                    result[i, 0] = -1
                else:
                    result[i, 0] = int(ColumnValues2[i])
                result[i, 1] = -1
                result[i, 2] = -1
            else:
                #s = str(int(v * 1000))
                s = str(int(v))
                try:
                    result[i, 0] = int(s[:5])
                except:
                    result[i, 0] = -1
                try:
                    result[i, 1] = int(s[5:10])
                except:
                    result[i, 1] = -1
                try:
                    result[i, 2] = int(s[10:])
                except:
                    result[i, 2] = -1

        return result

    @classmethod
    def GenerateCensusFeature(cls, InputFile, OutputFile):
        """"""
        start = time.time()

        prop = pd.read_csv(InputFile)
        #prop = prop.sample(frac = 0.1)

        df_cf = pd.DataFrame(index= prop.index)
        df_cf['parcelid'] = prop['parcelid']

        parsed = cls.__ApplyParseCensus(prop['censustractandblock'].values, prop['fips'].values)
        df_tmp = pd.DataFrame(data= parsed, index= df_cf.index, columns=['fipscode', 'tractcode', 'blockcode'])
        df_cf = pd.concat([df_cf, df_tmp], axis=1)

        with open(OutputFile, 'wb') as o_file:
            pickle.dump(df_cf, o_file, -1)
        o_file.close()

        end = time.time()
        print('Add census features done, time consumed %ds' % (end - start))
