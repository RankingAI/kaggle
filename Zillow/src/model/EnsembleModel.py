import time
import numba
import pandas as pd
import numpy as np
import os
from model.ModelBase import  ModelBase
import dill as pickle
import xgboost
import sys

class EnsembleModel(ModelBase):

    d_weight = {
        'lgb': 0.3,
        'xgb': 0.25,
        'rf': 0.1,
        'en': 0.15,
        'gbr': 0.2
    }
    bias_weight = 0.01
    bias = 0.011
    l_drop_columns = ['logerror', 'parcelid', 'transactiondate', 'index', 'nullcount']

    @numba.jit
    def __ApplyEnsemble(self,LgbCol, XgbCol, RfCol, EnCol, GbrCol):
        """"""
        n = len(LgbCol)
        result = np.empty((n), dtype='float32')
        for i in range(n):
            result[i] = (self.d_weight['lgb'] * LgbCol[i] +
                         self.d_weight['xgb'] * XgbCol[i] +
                         self.d_weight['rf'] * RfCol[i] +
                         self.d_weight['en'] * EnCol[i] +
                         self.d_weight['gbr'] * GbrCol[i]) * (1.0 - self.bias_weight) + self.bias * self.bias_weight

        return result

    ## evaluate ensemble model with local MAE
    def EvaluateEnsembleModel(self,InputDir):
        """"""
        start = time.time()

        lgb_file = '%s/LGB_20170713-12:54:46.pkl' % InputDir
        xgb_file = '%s/XGB_20170713-13:57:47.pkl' % InputDir
        rf_file = '%s/RF_20170713-15:00:17.pkl' % InputDir
        en_file = '%s/EN_20170713-15:23:53.pkl' % InputDir
        gbr_file = '%s/GBR_20170713-17:02:00.pkl' % InputDir

        with open(lgb_file,'rb') as i_file:
            lgb = pickle.load(i_file)
        i_file.close()
        print('Load lgb model done.')
        with open(xgb_file,'rb') as i_file:
            xgb = pickle.load(i_file)
        i_file.close()
        print('Load xgb model done.')
        with open(rf_file,'rb') as i_file:
            rf = pickle.load(i_file)
        i_file.close()
        print('Load rf model done.')
        with open(en_file,'rb') as i_file:
            en = pickle.load(i_file)
        i_file.close()
        print('Load en model done.')
        with open(gbr_file,'rb') as i_file:
            gbr = pickle.load(i_file)
        i_file.close()
        print('Load en model done.')

        mean_logerror = np.mean(self.TrainData['logerror'])
        print('Mean logerror %.4f' % mean_logerror)

        x_train = self.TrainData.drop(self.l_drop_columns, axis= 1)
        self._l_train_columns = x_train.columns

        pred_ensemble = pd.DataFrame(index=self.ValidData.index)
        pred_ensemble['parcelid'] = self.ValidData['parcelid']

        pred_lgb = pd.DataFrame(index=self.ValidData.index)
        pred_lgb['parcelid'] = self.ValidData['parcelid']

        pred_xgb = pd.DataFrame(index=self.ValidData.index)
        pred_xgb['parcelid'] = self.ValidData['parcelid']

        pred_rf = pd.DataFrame(index=self.ValidData.index)
        pred_rf['parcelid'] = self.ValidData['parcelid']

        pred_en = pd.DataFrame(index=self.ValidData.index)
        pred_en['parcelid'] = self.ValidData['parcelid']

        pred_gbr = pd.DataFrame(index=self.ValidData.index)
        pred_gbr['parcelid'] = self.ValidData['parcelid']

        truth_valid = pd.DataFrame(index=self.ValidData.index)
        truth_valid['parcelid'] = self.ValidData['parcelid']

        for d in self._l_valid_predict_columns:
            l_valid_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in self._l_train_columns]
            x_valid = self.ValidData[l_valid_columns]

            ## for lgb
            x_valid_common = x_valid.values.astype(np.float32, copy=False)
            ## for xgb
            x_valid.columns = ['lastgap' if('lastgap' in col) else 'monthyear' if('monthyear' in col) else 'buildingage' if('buildingage' in col) else col for col in x_valid.columns]

            ## add new feature nullcount for lgb, so need to be excluded fo xgb, rf, and en
            dvalid = xgboost.DMatrix(x_valid)
            ## predict
            pred_lgb_slice = lgb.predict(x_valid_common)
            pred_xgb_slice = xgb.predict(dvalid)
            pred_rf_slice = rf.predict(x_valid_common)
            pred_en_slice = en.predict(x_valid_common)
            pred_gbr_slice = gbr.predict(x_valid_common)
            ## ensemble
            pred_lgb[d] = pred_lgb_slice
            pred_xgb[d]= pred_xgb_slice
            pred_rf[d] = pred_rf_slice #* (1.0 - bias_weight ) + bias_weight * bias
            pred_en[d] = pred_en_slice #* (1.0 - bias_weight ) + bias_weight * bias
            pred_gbr[d] = pred_gbr_slice #* (1.0 - bias_weight ) + bias_weight * bias
            score = pred_lgb_slice * self.d_weight['lgb'] + \
                    pred_xgb_slice * self.d_weight['xgb'] + \
                    pred_rf_slice * self.d_weight['rf'] + \
                    pred_en_slice * self.d_weight['en'] + \
                    pred_gbr_slice * self.d_weight['gbr']

            pred_ensemble[d] = (1.0 - self.bias_weight) * score + self.bias_weight * self.bias
            df_tmp = self.ValidData[self.ValidData['transactiondate'].dt.month == int(d[-2:])]
            truth_valid.loc[df_tmp.index,d] = df_tmp['logerror']

        lgb_score = 0.0
        lgb_ae = np.abs(pred_lgb - truth_valid)
        for col in lgb_ae.columns:
            lgb_score += np.sum(lgb_ae[col])
        lgb_score /= len(pred_lgb)

        xgb_score = 0.0
        xgb_ae = np.abs(pred_xgb - truth_valid)
        for col in xgb_ae.columns:
            xgb_score += np.sum(xgb_ae[col])
        xgb_score /= len(pred_xgb)

        rf_score = 0.0
        rf_ae = np.abs(pred_rf - truth_valid)
        for col in rf_ae.columns:
            rf_score += np.sum(rf_ae[col])
        rf_score /= len(pred_rf)

        en_score = 0.0
        en_ae = np.abs(pred_en - truth_valid)
        for col in en_ae.columns:
            en_score += np.sum(en_ae[col])
        en_score /= len(pred_en)

        gbr_score = 0.0
        gbr_ae = np.abs(pred_gbr - truth_valid)
        for col in gbr_ae.columns:
            gbr_score += np.sum(gbr_ae[col])
        gbr_score /= len(pred_gbr)

        ensemble_score = 0.0
        ensemble_ae = np.abs(pred_ensemble - truth_valid)
        for col in ensemble_ae.columns:
            ensemble_score += np.sum(ensemble_ae[col])
        ensemble_score /= len(pred_ensemble)  ##!! divided by number of instances, not the number of 'cells'
        print('=============================')
        print('Local MAE is %.6f(ensemble), %.6f(lgb), %.6f(xgb), %.6f(rf), %.6f(en), %.6f(gbr).' % (ensemble_score, lgb_score, xgb_score, rf_score, en_score, gbr_score))
        print('=============================')

        end = time.time()
        print('time elapsed %ds' % (end - start))

    ## predict for the test data with optimized ensemble model in LOCAL mode
    def SimpleEnsemble(self,InputDir,OutputDir):

        if(os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)

        start = time.time()

        ## ensemble the best ones of lgb and xgb
        lgb_result = pd.read_csv('%s/lgb_418_biased.csv' % InputDir)
        xgb_result = pd.read_csv('%s/xgb_418_biased.csv' % InputDir)  # parameter base_score equals the mean of target
        rf_result = pd.read_csv('%s/rf_418.csv' % InputDir)
        en_result = pd.read_csv('%s/en_418.csv' % InputDir)
        gbr_result = pd.read_csv('%s/gbr_418.csv' % InputDir)

        ensembled_result = pd.DataFrame(index=lgb_result.index)
        ensembled_result['ParcelId'] = lgb_result['ParcelId']

        ensembled_result.set_index('ParcelId', inplace=True)
        lgb_result.set_index('ParcelId', inplace=True)
        xgb_result.set_index('ParcelId', inplace=True)
        rf_result.set_index('ParcelId', inplace=True)
        en_result.set_index('ParcelId', inplace=True)
        gbr_result.set_index('ParcelId', inplace=True)

        ## determined by MAE value of public score

        up_threshold = 0.418

        for col in lgb_result.columns:
            tmp = self.__ApplyEnsemble(lgb_result[col].values,
                                      xgb_result[col].values,
                                      rf_result[col].values,
                                      en_result[col].values,
                                      gbr_result[col].values)
            df_tmp = pd.DataFrame(tmp, index=lgb_result.index, columns=[col])
            ensembled_result = pd.concat([ensembled_result, df_tmp], axis=1)
            print('Enssemble for column %s is done.' % col)

        ensembled_result = ensembled_result.reset_index()
        print('Examples of lgb: ')
        print(lgb_result.head())
        print('Examples of xgb: ')
        print(xgb_result.head())
        print('Examples of ensemble(lgb:xgb:rf:en:gbr=%d:%d:%d:%d:%d), bias weight %.4f' % (int(self.d_weight['lgb']*100),
                                                                               int(self.d_weight['xgb']*100),
                                                                               int(self.d_weight['rf'] * 100),
                                                                               int(self.d_weight['en'] * 100),
                                                                               int(self.d_weight['gbr'] * 100),
                                                                               self.bias_weight)
              )
        print(ensembled_result.head())

        ensemble_sub = '%s/lgb_xgb_rf_en_gbr_%d_%d_%d_%d_%d_%d.csv' % (OutputDir,int(up_threshold * 1000),
                                                          int(self.d_weight['lgb'] * 100),
                                                          int(self.d_weight['xgb'] * 100),
                                                          int(self.d_weight['rf'] * 100),
                                                          int(self.d_weight['en'] * 100),
                                                          int(self.d_weight['gbr'] * 100),
                                                          )
        ensembled_result.to_csv(ensemble_sub, index=False, float_format='%.4f')

        end = time.time()
        print('\nEnsemble of lgb and xgb is done, time consumed %ds' % (end - start))