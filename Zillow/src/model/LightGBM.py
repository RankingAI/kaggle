from model.ModelBase import ModelBase
import numpy as np
import pandas as pd
import lightgbm
import gc
import os
import time
import math
from datetime import datetime

class LGB(ModelBase):

    ## rewritten method
    def train(self):

        start = time.time()

        print('size before truncated outliers is %d ' % len(self.TrainData))
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > -0.4) & (self.TrainData['logerror'] < 0.4)]
        print('size after truncated outliers is %d ' % len(self.TrainData))

        x_train = self.TrainData.drop(['logerror','parcelid'],axis= 1)
        y_train = self.TrainData['logerror']
        self._l_train_columns = x_train.columns

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

        ## for validation in parameter tuning
        # split = 80000
        # x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
        # x_train = x_train.values.astype(np.float32, copy=False)
        # x_valid = x_valid.values.astype(np.float32, copy=False)
        #
        # d_train = lightgbm.Dataset(x_train, label=y_train)
        # d_valid = lightgbm.Dataset(x_valid, label=y_valid)

        # x_train = x_train.values.astype(np.float32, copy=False)
        # d_train = lightgbm.Dataset(x_train, label=y_train)
        #l_learning_rate = [0.001*math.pow(2.0,i) for i in range(7)]
        # l_learning_rate = [0.018 + 0.002*i for i in range(5)]
        # l_bagging_freq = [30+i*10 for i in range(1)]
        #
        # BestParams = {'learning_rate':0.0,'bagging_freq':0}
        # BestMAE = 1.0
        # for lr in l_learning_rate:
        #     for bf in l_bagging_freq:
        #         params['learning_rate'] = lr
        #         params['bagging_freq'] = bf
        #
        #         self._model = lightgbm.cv(params, d_train, 200, nfold=5,verbose_eval= True)
        #         if(self._model.get('l1-mean')[-1] < BestMAE):
        #             BestMAE = self._model.get('l1-mean')[-1]
        #             BestParams['learning_rate'] = lr
        #             BestParams['bagging_freq'] = bf
        # print(BestParams)

        d_train = lightgbm.Dataset(x_train,label=y_train)
        params['learning_rate'] = 0.020
        params['bagging_freq'] = 30
        self._model = lightgbm.train(params,d_train,100,verbose_eval= True)

        del self.TrainData
        gc.collect()

        end = time.time()
        print('Training is done. Time elapsed %ds' % (end - start))

    ## rewritten method
    def predict(self):

        # CalibrationFactor = 0.03
        # cal = CalibrationFactor * 0.013
        self.TestData = self._data.LoadFromHdfFile(self.InputDir,'test')
        #self.TestData = self.TestData.sample(frac = 0.2)

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
                ret = self._model.predict(x_test_block)
                self._sub.loc[x_test[idx:idx + N].index, d] = ret
                print(np.mean(np.abs(ret)))

            e0 = time.time()
            print('Prediction for column %s is done. time elapsed %ds' % (d, (e0 - s0)))

        ## clean
        del self.TestData
        gc.collect()
        end = time.time()
        print('Prediction is done. time elapsed %ds' % (end - start))


