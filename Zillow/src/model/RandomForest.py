from model.ModelBase import ModelBase
from sklearn.ensemble import RandomForestRegressor
import math
import numpy as np
import pandas as pd
import time

class RF(ModelBase):
    """"""
    _l_drop_cols = ['logerror', 'parcelid', 'transactiondate']

    _iter = 180
    _depth = 40

    def train(self):
        """"""
        self.__fit()

        return self.__predict()

    def __fit(self):
        """"""
        ## truncate outliers
        print('initial data shape : ', self.TrainData.shape)
        TrainData = self.TrainData[(self.TrainData['logerror'] > self.MinLogError) & (self.TrainData['logerror'] < self.MaxLogError)]
        print('data shape truncated: ', TrainData.shape)

        TrainData['longitude'] -= -118600000
        TrainData['latitude'] -= 34220000

        X = TrainData.drop(self._l_drop_cols, axis=1)
        Y = TrainData['logerror']
        self._l_train_columns = X.columns

        RF = RandomForestRegressor(random_state=2017, criterion='mse',
                                 n_estimators= self._iter, n_jobs=2,
                                 max_depth= self._depth,
                                 max_features=int(math.sqrt(len(self._l_train_columns))), verbose= True)
        self._model = RF.fit(X, Y)

        return

    def __predict(self):
        """"""
        TestData = self.TestData

        TestData['longitude'] -= -118600000
        TestData['latitude'] -= 34220000

        pred_test = pd.DataFrame(index= TestData.index)
        pred_test['parcelid'] = TestData['parcelid']

        truth_test = pd.DataFrame(index= TestData.index)
        truth_test['parcelid'] = TestData['parcelid']

        start = time.time()

        for mth in self.l_test_predict_columns:
            l_test_columns = ['%s%s' % (c, mth) if (c in ['lastgap']) else c for c in self._l_train_columns]
            x_test = TestData[l_test_columns]
            x_test = x_test.values.astype(np.float32, copy=False)
            pred_test['%s' % mth] = self._model.predict(x_test)
            df_tmp = TestData[TestData['transactiondate'].dt.month == mth]
            truth_test.loc[df_tmp.index, '%s' % mth] = df_tmp['logerror']

        score = 0.0
        ae = np.abs(pred_test - truth_test)
        for col in ae.columns:
            score += np.sum(ae[col])
        score /= len(pred_test)  ##!! divided by number of instances, not the number of 'cells'
        print('============================= ')
        print('Local MAE is %.6f' % score)
        print('=============================')

        end = time.time()
        print('time elapsed %ds' % (end - start))

        return score
