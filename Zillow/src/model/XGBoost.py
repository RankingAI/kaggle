from model.ModelBase import ModelBase
import numpy as np
import pandas as pd
import xgboost
import gc
import math
import time
import dill as pickle

class XGB(ModelBase):

    def train(self):
        """"""
        print(len(self.TrainData))
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > -0.4) & (self.TrainData['logerror'] < 0.4)]
        print(len(self.TrainData))

        x_train = self.TrainData.drop(['logerror','parcelid'],axis= 1)
        y_train = self.TrainData["logerror"].values.astype(np.float32)
        y_mean = np.mean(y_train)

        self._l_train_columns = x_train.columns
        params = {
            'subsample': 0.80,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'base_score': y_mean,
            'silent': 0,
            'npthread': 2
        }
        dtrain = xgboost.DMatrix(x_train, y_train)

        ## parameter tuning with CV
        # BestParams = {'eta':0.0,'max_depth':0}
        # BestMAE = 1.0
        # l_eta = [0.03*math.pow(2.0,i) for i in range(1)]
        # l_max_depth = [(6 + v) for v in range(5)]
        # for eta in l_eta:
        #     for depth in l_max_depth:
        #         params['eta'] = eta
        #         params['max_depth'] = depth
        #         print( "Running XGBoost CV ..." )
        #         cv_result = xgboost.cv(params,
        #                   dtrain,
        #                   nfold=5,
        #                   num_boost_round=100,
        #                   early_stopping_rounds=50,
        #                   verbose_eval=10,
        #                   show_stdv=True
        #                  )
        #         if(cv_result.get('test-mae')[-1] < BestMAE):
        #             BestMAE = cv_result.get('test-mae')[-1]
        #             BestParams['eta'] = eta
        #             BestParams['max_depth'] = depth
        # print(BestParams)

        params['eta'] = 0.04
        params['max_depth'] = 10

        # # train model
        print("\nTraining XGBoost ...")
        self._model = xgboost.train(params, dtrain, num_boost_round= 100,early_stopping_rounds= 80, verbose_eval= True)

        del self.TrainData
        gc.collect()

    def predict(self):
        """"""
        self.TestData = self._data.LoadFromHdfFile(self.InputDir, 'test')
        #self.TestData = self.TestData.sample(frac = 0.2)

        self._sub = pd.DataFrame(index=self.TestData.index)
        self._sub['ParcelId'] = self.TestData['parcelid']

        N = 200000
        start = time.time()
        for d in self._l_test_predict_columns:
            s0 = time.time()

            print("Start prediction ...")
            l_test_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in
                              self._l_train_columns]
            x_test = self.TestData[l_test_columns]

            for idx in range(0, len(x_test), N):
                x_test_block = x_test[idx:idx + N].values.astype(np.float32, copy=False)
                dtest = xgboost.DMatrix(x_test_block)
                #self._model.reset_parameter({"num_threads": 4})
                ret = self._model.predict(dtest)
                self._sub.loc[x_test[idx:idx + N].index, d] = ret
                print(np.mean(np.abs(ret)))

            e0 = time.time()
            print('%s done. time elapsed %ds' % (d, (e0 - s0)))

        ## clean
        del self.TestData
        gc.collect()
        end = time.time()
        print('time elapsed %ds' % (end - start))

