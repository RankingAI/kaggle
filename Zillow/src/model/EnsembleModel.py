import time
import numba
import pandas as pd
import numpy as np
import os

class EnsembleModel:

    @classmethod
    @numba.jit
    def __ApplyEnsemble(cls,LgbCol, XgbCol, weight):

        n = len(LgbCol)
        result = np.empty((n), dtype='float32')
        for i in range(n):
            result[i] = weight * LgbCol[i] + (1.0 - weight) * XgbCol[i]

        return result

    @classmethod
    def SimpleEnsemble(cls,InputDir,OutputDir):

        if(os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)

        start = time.time()

        ## ensemble the best ones of lgb and xgb
        lgb_result = pd.read_csv('%s/lgb_646244.csv' % InputDir)
        xgb_result = pd.read_csv('%s/xgb_645968.csv' % InputDir)

        ensembled_result = pd.DataFrame(index=lgb_result.index)
        ensembled_result['ParcelId'] = lgb_result['ParcelId']

        ensembled_result.set_index('ParcelId', inplace=True)
        lgb_result.set_index('ParcelId', inplace=True)
        xgb_result.set_index('ParcelId', inplace=True)

        ## determined by MAE value of public score
        lgb_weight = 0.20

        for col in lgb_result.columns:
            tmp = cls.__ApplyEnsemble(lgb_result[col].values, xgb_result[col].values, lgb_weight)
            df_tmp = pd.DataFrame(tmp, index=lgb_result.index, columns=[col])
            ensembled_result = pd.concat([ensembled_result, df_tmp], axis=1)
            print('Enssemble for column %s is done.' % col)

        ensembled_result = ensembled_result.reset_index()
        print('Examples of lgb: ')
        print(lgb_result.head())
        print('Examples of xgb: ')
        print(xgb_result.head())
        print('Examples of ensemble(lgb:xgb=%d:%d)' % (int(lgb_weight*100), int(100 - lgb_weight*100)))
        print(ensembled_result.head())

        ensemble_sub = '%s/lgb_xgb_%d.csv' % (OutputDir,int(lgb_weight * 100))
        ensembled_result.to_csv(ensemble_sub, index=False, float_format='%.4f')

        end = time.time()
        print('\nEnsemble of lgb and xgb is done, time consumed %ds' % (end - start))