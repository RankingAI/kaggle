import time
import numba
import pandas as pd
import numpy as np
import os
from model.ModelBase import  ModelBase
import dill as pickle
import xgboost

class EnsembleModel(ModelBase):

    @classmethod
    @numba.jit
    def __ApplyEnsemble(cls,LgbCol, XgbCol, RfCol, EnCol, weights, bias, bias_weight):

        n = len(LgbCol)
        result = np.empty((n), dtype='float32')
        for i in range(n):
            result[i] = (weights['lgb'] * LgbCol[i] +
                         weights['xgb'] * XgbCol[i] +
                         weights['rf'] * RfCol[i] +
                         weights['en'] * EnCol[i]) * (1.0 - bias_weight) + bias * bias_weight

        return result

    ## evaluate ensemble model with local MAE
    def EvaluateEnsembleModel(self,InputDir):
        """"""
        d_weight = {
            'lgb': 0.4,
            'xgb': 0.2,
            'rf': 0.2,
            'en': 0.2
        }
        bias_weight = 0.01
        bias = 0.011

        start = time.time()

        lgb_file = '%s/LGB_20170709-10:49:41.pkl' % InputDir
        xgb_file = '%s/XGB_20170709-11:45:22.pkl' % InputDir
        rf_file = '%s/RF_20170711-19:02:26.pkl' % InputDir
        en_file = '%s/EN_20170712-17:33:36.pkl' % InputDir

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

        mean_logerror = np.mean(self.TrainData['logerror'])
        print('Mean logerror %.4f' % mean_logerror)

        x_train = self.TrainData.drop(['logerror','parcelid','transactiondate'],axis= 1)
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

        truth_valid = pd.DataFrame(index=self.ValidData.index)
        truth_valid['parcelid'] = self.ValidData['parcelid']

        for d in self._l_valid_predict_columns:
            l_valid_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in self._l_train_columns]
            x_valid = self.ValidData[l_valid_columns]

            ## for lgb
            x_valid_common = x_valid.values.astype(np.float32, copy=False)
            ## for xgb
            x_valid.columns = ['lastgap' if('lastgap' in col) else 'monthyear' if('monthyear' in col) else 'buildingage' if('buildingage' in col) else col for col in x_valid.columns]
            dvalid = xgboost.DMatrix(x_valid)
            ## predict
            pred_lgb_slice = lgb.predict(x_valid_common)
            pred_xgb_slice = xgb.predict(dvalid)
            pred_rf_slice = rf.predict(x_valid_common)
            pred_en_slice = en.predict(x_valid_common)
            ## ensemble
            pred_lgb[d] = pred_lgb_slice
            pred_xgb[d]= pred_xgb_slice
            pred_rf[d] = pred_rf_slice #* (1.0 - bias_weight ) + bias_weight * bias
            pred_en[d] = pred_en_slice #* (1.0 - bias_weight ) + bias_weight * bias
            score = pred_lgb_slice * d_weight['lgb'] + pred_xgb_slice * d_weight['xgb'] + pred_rf_slice * d_weight['rf'] + pred_en_slice * d_weight['en']
            #score = pred_lgb_slice * d_weight['lgb'] + pred_rf_slice * d_weight['rf'] + pred_en_slice * d_weight['en']
            pred_ensemble[d] = (1.0 - bias_weight) * score + bias_weight * bias

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

        ensemble_score = 0.0
        ensemble_ae = np.abs(pred_ensemble - truth_valid)
        for col in ensemble_ae.columns:
            ensemble_score += np.sum(ensemble_ae[col])
        ensemble_score /= len(pred_ensemble)  ##!! divided by number of instances, not the number of 'cells'
        print('============================= ')
        print('Local MAE is %.6f(ensemble), %.6f(lgb), %.6f(xgb), %.6f(rf), %.6f(en).' % (ensemble_score, lgb_score, xgb_score, rf_score, en_score))
        print('=============================')

        end = time.time()
        print('time elapsed %ds' % (end - start))

    ## predict for the test data with optimized ensemble model in LOCAL mode
    @classmethod
    def SimpleEnsemble(cls,InputDir,OutputDir):

        if(os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)

        start = time.time()

        ## ensemble the best ones of lgb and xgb
        lgb_result = pd.read_csv('%s/lgb_418_bias.csv' % InputDir)
        xgb_result = pd.read_csv('%s/xgb_418_bias.csv' % InputDir)  # parameter base_score equals the mean of target
        rf_result = pd.read_csv('%s/rf_418.csv' % InputDir)
        en_result = pd.read_csv('%s/en_418.csv' % InputDir)

        ensembled_result = pd.DataFrame(index=lgb_result.index)
        ensembled_result['ParcelId'] = lgb_result['ParcelId']

        ensembled_result.set_index('ParcelId', inplace=True)
        lgb_result.set_index('ParcelId', inplace=True)
        xgb_result.set_index('ParcelId', inplace=True)
        rf_result.set_index('ParcelId', inplace=True)
        en_result.set_index('ParcelId', inplace=True)

        ## determined by MAE value of public score
        d_weight = {
            'lgb': 0.4,
            'xgb': 0.2,
            'rf': 0.2,
            'en': 0.2
        }
        bias_weight = 0.01
        bias = 0.011

        up_threshold = 0.418

        for col in lgb_result.columns:
            tmp = cls.__ApplyEnsemble(lgb_result[col].values,
                                      xgb_result[col].values,
                                      rf_result[col].values,
                                      en_result[col].values,
                                      d_weight, bias, bias_weight)
            df_tmp = pd.DataFrame(tmp, index=lgb_result.index, columns=[col])
            ensembled_result = pd.concat([ensembled_result, df_tmp], axis=1)
            print('Enssemble for column %s is done.' % col)

        ensembled_result = ensembled_result.reset_index()
        print('Examples of lgb: ')
        print(lgb_result.head())
        print('Examples of xgb: ')
        print(xgb_result.head())
        print('Examples of ensemble(lgb:xgb:rf:en=%d:%d:%d:%d), bias weight %.4f' % (int(d_weight['lgb']*100),
                                                                               int(d_weight['xgb']*100),
                                                                               int(d_weight['rf'] * 100),
                                                                               int(d_weight['en'] * 100),
                                                                               bias_weight)
              )
        print(ensembled_result.head())

        ensemble_sub = '%s/lgb_xgb_rf_en_%d_%d_%d_%d_%d.csv' % (OutputDir,int(up_threshold * 1000),
                                                          int(d_weight['lgb'] * 100),
                                                          int(d_weight['xgb'] * 100),
                                                          int(d_weight['rf'] * 100),
                                                          int(d_weight['en'] * 100)
                                                          )
        ensembled_result.to_csv(ensemble_sub, index=False, float_format='%.4f')

        end = time.time()
        print('\nEnsemble of lgb and xgb is done, time consumed %ds' % (end - start))