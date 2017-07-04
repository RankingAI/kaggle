from model.ModelBase import ModelBase
import numpy as np
import pandas as pd
import lightgbm
import gc
import os
import time
import math

class LGB(ModelBase):

    ## rewritten method
    def train(self):

        print(len(self.TrainData))
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > -0.4) & (self.TrainData['logerror'] < 0.4)]
        print(len(self.TrainData))

        x_train = self.TrainData.drop(['logerror','parcelid'],axis= 1)
        y_train = self.TrainData['logerror']
        self._l_train_columns = x_train.columns

        split = 80000
        x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
        x_train = x_train.values.astype(np.float32, copy=False)
        x_valid = x_valid.values.astype(np.float32, copy=False)

        d_train = lightgbm.Dataset(x_train, label=y_train)
        d_valid = lightgbm.Dataset(x_valid, label=y_valid)

        params = {}
        params['max_bin'] = 8
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'regression'
        params['metric'] = 'mae'
        params['sub_feature'] = 0.8
        params['bagging_fraction'] = 0.85  # sub_row
        params['num_leaves'] = 64
        params['min_data'] = 300
        params['min_hessian'] = 0.01

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
        params['learning_rate'] = 0.020
        params['bagging_freq'] = 30
        self._model = lightgbm.train(params,d_train,200,verbose_eval= True,valid_sets= [d_valid])

        del self.TrainData
        gc.collect()

    ## rewritten method
    def predict(self):

        # CalibrationFactor = 0.03
        # cal = CalibrationFactor * 0.013
        self._sub = pd.DataFrame(index = self.TestData.index)
        self._sub['ParcelId'] = self.TestData['parcelid']

        for d in self._l_test_predict_columns:
            start = time.time()

            l_test_columns = ['%s%s' % (col,d) if(col in ['lastgap','monthyear']) else col for col in self._l_train_columns]

            x_test = self.TestData[l_test_columns]
            for c in x_test.dtypes[x_test.dtypes == object].index.values:
                x_test[c] = (x_test[c] == True)
            x_test = x_test.values.astype(np.float32, copy=False)

            print("Start prediction ...")
            self._model.reset_parameter({"num_threads": 4})
            # sub[d] = (1 - CalibrationFactor) * clf.predict(x_test) + cal
            self._sub[d] = self._model.predict(x_test)

            end = time.time()
            print('%s done. time elapsed %ds' % (d,(end - start)))

        ## clean
        del self.TestData
        gc.collect()

    def submmit(self):

        if(os.path.exists(self.OutputDir) == False):
            os.makedirs(self.OutputDir)

        m = np.mean(self._sub[self._l_test_predict_columns].mean())
        with open('%s/eval.log' % self.OutputDir,'a+') as o_file:
            o_file.write('%.4f\n' % m)
        o_file.close()
        self._sub.to_csv('%s/lgb.csv' % self.OutputDir, index=False, float_format='%.4f')
