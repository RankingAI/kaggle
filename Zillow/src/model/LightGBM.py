from model.ModelBase import ModelBase
import numpy as np
import pandas as pd
import lightgbm
import gc
import os
import time
import math
from datetime import datetime
import numba
import dill as pickle

class LGB(ModelBase):

    ## rewritten method
    def train(self):

        start = time.time()

        print('size before truncated outliers is %d ' % len(self.TrainData))
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print('size after truncated outliers is %d ' % len(self.TrainData))

        X = self.TrainData.drop(['logerror','parcelid','transactiondate'],axis= 1)
        Y = self.TrainData['logerror']
        self._l_train_columns = X.columns

        X = X.values.astype(np.float32, copy=False)
        d_cv = lightgbm.Dataset(X,label=Y)

        ## one-hold mode for parameter tuning
        # msk = np.random.rand(len(self.TrainData)) < 0.9
        # train = self.TrainData[msk]
        # valid = self.TrainData[~msk]
        # x_train = train.drop(['logerror','parcelid','transactiondate'],axis= 1)
        # y_train = train['logerror']
        # self._l_train_columns = x_train.columns

        # x_valid = valid.drop(['logerror','parcelid','transactiondate'],axis= 1)
        # y_valid = valid['logerror']
        #
        # x_train = x_train.values.astype(np.float32, copy=False)
        # x_valid = x_valid.values.astype(np.float32, copy=False)
        #
        # d_train = lightgbm.Dataset(x_train,label=y_train)
        # d_valid = lightgbm.Dataset(x_valid,label=y_valid)
        # params['learning_rate'] = 0.026
        # params['bagging_freq'] = 20
        # self._model = lightgbm.train(params,d_train,100,verbose_eval= True,valid_sets=[d_valid])

        params = {}
        params['max_bin'] = 8
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'regression'
        params['metric'] = 'mae'
        params['sub_feature'] = 0.8
        params['bagging_fraction'] = 0.85  # sub_row
        params['num_leaves'] = 128
        params['min_data'] = 300
        params['min_hessian'] = 0.01

        ## cv mode for parameter tuning
        # l_learning_rate = [0.014 + 0.002*i for i in range(5)]
        # l_bagging_freq = [10 + i*10 for i in range(5)]
        #
        # BestParams = {'learning_rate':0.0,'bagging_freq':0}
        # BestMAE = 1.0
        # for lr in l_learning_rate:
        #     for bf in l_bagging_freq:
        #         params['learning_rate'] = lr
        #         params['bagging_freq'] = bf
        #
        #         self._model = lightgbm.cv(params, d_cv, 100, nfold=5,verbose_eval= True)
        #         if(self._model.get('l1-mean')[-1] < BestMAE):
        #             BestMAE = self._model.get('l1-mean')[-1]
        #             BestParams['learning_rate'] = lr
        #             BestParams['bagging_freq'] = bf
        # print(BestParams)
        # params['learning_rate'] = BestParams['learning_rate']
        # params['bagging_freq'] = BestParams['bagging_freq']

        params['learning_rate'] = 0.02
        params['bagging_freq'] = 20
        self._model = lightgbm.train(params,d_cv,100,verbose_eval= True)

        self._f_eval_train_model = '{0}/{1}_{2}.pkl'.format(self.OutputDir, self.__class__.__name__,datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        with open(self._f_eval_train_model,'wb') as o_file:
            pickle.dump(self._model,o_file,-1)
        o_file.close()

        self.TrainData = pd.concat([self.TrainData,self.ValidData[self.TrainData.columns]],ignore_index= True) ## ignore_index will reset the index or index will be overlaped

        end = time.time()
        print('Training is done. Time elapsed %ds' % (end - start))

    ## evaluate on valid data
    def evaluate(self):
        """"""
        pred_valid = pd.DataFrame(index = self.ValidData.index)
        pred_valid['parcelid'] = self.ValidData['parcelid']

        truth_valid = pd.DataFrame(index = self.ValidData.index)
        truth_valid['parcelid'] = self.ValidData['parcelid']

        start = time.time()

        for d in self._l_valid_predict_columns:
            l_valid_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in self._l_train_columns]
            x_valid = self.ValidData[l_valid_columns]
            x_valid = x_valid.values.astype(np.float32, copy=False)
            pred_valid[d] = self._model.predict(x_valid)# * 0.97 + 0.011 * 0.03
            df_tmp = self.ValidData[self.ValidData['transactiondate'].dt.month == int(d[-2:])]
            truth_valid.loc[df_tmp.index,d] = df_tmp['logerror']

        score = 0.0
        ae = np.abs(pred_valid - truth_valid)
        for col in ae.columns:
            score += np.sum(ae[col])
        score /= len(pred_valid)  ##!! divided by number of instances, not the number of 'cells'
        print('============================= ')
        print('Local MAE is %.6f' % score)
        print('=============================')

        end = time.time()

        del self.ValidData
        gc.collect()

        print('time elapsed %ds' % (end - start))

    @numba.jit
    def __ApplyAE(self,PredColumn,TruthColumn):

        n = len(PredColumn)
        result = np.empty(n,dtype= 'float32')
        for i in range(n):
            v = TruthColumn[i]
            if(math.isnan(v) == False):
                result[i] = np.abs(v - PredColumn[i])

        return result

    def __ComputeMAE(self,df_pred,df_truth):
        """"""
        mae = pd.DataFrame(index = df_pred)
        for col in df_pred.columns:
            if(col == 'parcelid'):
                continue
            ret = self.__ApplyAE(df_pred[col],df_truth[col])
            mae[col] = pd.Series(ret,index= df_pred.index)

    ## predict on test data
    def submit(self):

        ## retrain with the whole training data
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]

        X = self.TrainData.drop(['logerror', 'parcelid', 'transactiondate'], axis=1)
        Y = self.TrainData['logerror']

        X = X.values.astype(np.float32, copy=False)
        d_train = lightgbm.Dataset(X, label=Y)

        params = {}
        params['max_bin'] = 8
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'regression'
        params['metric'] = 'mae'
        params['sub_feature'] = 0.8
        params['bagging_fraction'] = 0.85  # sub_row
        params['num_leaves'] = 128
        params['min_data'] = 300
        params['min_hessian'] = 0.01
        params['learning_rate'] = 0.02
        params['bagging_freq'] = 20
        self._model = lightgbm.train(params, d_train, 100, verbose_eval=True)

        del self.TrainData, X, Y, d_train
        gc.collect()

        self.TestData = self._data.LoadFromHdfFile(self.InputDir,'test')

        self._sub = pd.DataFrame(index = self.TestData.index)
        self._sub['ParcelId'] = self.TestData['parcelid']

        N = 200000
        start = time.time()
        for d in self._l_test_predict_columns:
            s0 = time.time()

            print('Prediction for column %s ' % d)
            l_test_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in self._l_train_columns]
            x_test = self.TestData[l_test_columns]

            for idx in range(0, len(x_test), N):
                x_test_block = x_test[idx:idx + N].values.astype(np.float32, copy=False)
                self._model.reset_parameter({"num_threads": 4})
                ret = self._model.predict(x_test_block)# * 0.93 + 0.012 * 0.07
                self._sub.loc[x_test[idx:idx + N].index, d] = ret
                print(np.mean(np.abs(ret)))

            e0 = time.time()
            print('Prediction for column %s is done. time elapsed %ds' % (d, (e0 - s0)))

        ## clean
        del self.TestData
        gc.collect()

        end = time.time()
        print('Prediction is done. time elapsed %ds' % (end - start))

        if (os.path.exists(self.OutputDir) == False):
            os.makedirs(self.OutputDir)

        self._sub.to_csv('{0}/{1}_{2}.csv'.format(self.OutputDir, self.__class__.__name__,datetime.now().strftime('%Y%m%d-%H:%M:%S')),
                         index=False, float_format='%.4f')
