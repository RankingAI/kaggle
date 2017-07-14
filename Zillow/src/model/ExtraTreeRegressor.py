from model.ModelBase import ModelBase
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
import numpy as np
import sys
import time
from datetime import datetime
import dill as pickle
import math
import gc
import os

class ETR(ModelBase):
    """"""
    _l_drop_cols = ['logerror', 'parcelid', 'transactiondate','index','nullcount']
    _iter = 120
    _depth= 20

    def train(self):
        """"""
        print('size before truncated outliers is %d ' % len(self.TrainData))
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print('size after truncated outliers is %d ' % len(self.TrainData))

        X = self.TrainData.drop(self._l_drop_cols, axis=1)
        Y = self.TrainData['logerror']
        self._l_train_columns = X.columns
        FeatCols = list(self._l_train_columns)

        etr = ExtraTreesRegressor(
            n_estimators= self._iter,
            criterion= 'mse',
            max_features= int(math.sqrt(len(FeatCols))),
            max_depth = self._depth,
            n_jobs= 2,
            random_state= 2017,
            verbose= True
        )

        self._model = etr.fit(X, Y)

        ## evaluate on valid data
        self._f_eval_train_model = '{0}/{1}_{2}.pkl'.format(self.OutputDir, self.__class__.__name__,
                                                            datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        with open(self._f_eval_train_model, 'wb') as o_file:
            pickle.dump(self._model, o_file, -1)
        o_file.close()

        self.TrainData = pd.concat([self.TrainData, self.ValidData[self.TrainData.columns]],
                                   ignore_index=True)  ## ignore_index will reset the index or index will be overlaped

        return

    def evaluate(self):
        """"""
        pred_valid = pd.DataFrame(index=self.ValidData.index)
        pred_valid['parcelid'] = self.ValidData['parcelid']

        truth_valid = pd.DataFrame(index=self.ValidData.index)
        truth_valid['parcelid'] = self.ValidData['parcelid']

        start = time.time()

        for d in self._l_valid_predict_columns:
            l_valid_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in
                               self._l_train_columns]
            x_valid = self.ValidData[l_valid_columns]
            x_valid = x_valid.values.astype(np.float32, copy=False)
            pred_valid[d] = self._model.predict(x_valid)
            df_tmp = self.ValidData[self.ValidData['transactiondate'].dt.month == int(d[-2:])]
            truth_valid.loc[df_tmp.index, d] = df_tmp['logerror']

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

    def submit(self):
        """"""
        ## retrain with the whole training data
        self.TrainData = self.TrainData[
            (self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]

        X = self.TrainData.drop(self._l_drop_cols, axis=1)
        Y = self.TrainData['logerror']

        FeatCols = list(self._l_train_columns)

        etr = ExtraTreesRegressor(
            n_estimators=self._iter,
            criterion='mse',
            max_features=int(math.sqrt(len(FeatCols))),
            max_depth=self._depth,
            n_jobs=2,
            random_state=2017,
            verbose=True
        )

        self._model = etr.fit(X, Y)

        del self.TrainData, X, Y
        gc.collect()

        self.TestData = self._data.LoadFromHdfFile(self.InputDir, 'test')
        # self.TestData = self.TestData.sample(frac=0.01)

        self._sub = pd.DataFrame(index=self.TestData.index)
        self._sub['ParcelId'] = self.TestData['parcelid']

        N = 200000
        start = time.time()
        for d in self._l_test_predict_columns:
            s0 = time.time()

            print('Prediction for column %s ' % d)
            l_test_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in
                              self._l_train_columns]
            x_test = self.TestData[l_test_columns]

            for idx in range(0, len(x_test), N):
                x_test_block = x_test[idx:idx + N]  # .values.astype(np.float32, copy=False)
                ret = self._model.predict(x_test_block)  # * 0.93 + 0.012 * 0.07
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

        self._sub.to_csv('{0}/{1}_{2}.csv'.format(self.OutputDir, self.__class__.__name__,
                                                  datetime.now().strftime('%Y%m%d-%H:%M:%S')),
                         index=False, float_format='%.4f')
