import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import time
import math
import gc

from model.ModelBase import ModelBase

class KR(ModelBase):
    """"""
    _l_drop_cols = ['logerror', 'parcelid', 'transactiondate','index']
    _iter = 20
    _batch_size = 5

    # define base model
    def baseline_model(self):
        # create model
        model = Sequential()
        model.add(Dense(int(math.sqrt(len(self._l_train_columns))),
                        input_dim= len(self._l_train_columns),
                        kernel_initializer='normal',
                        activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_absolute_error', optimizer='adam')

        return model

    def train(self):
        """"""
        start = time.time()

        print('size before truncated outliers is %d ' % len(self.TrainData))
        TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print('size after truncated outliers is %d ' % len(TrainData))

        X = TrainData.drop(self._l_drop_cols, axis=1)
        Y = TrainData['logerror']

        self._l_train_columns = X.columns

        X = X.values
        Y = Y.values

        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)
        # evaluate model with standardized dataset
        self._model = KerasRegressor(build_fn= self.baseline_model, nb_epoch= self._iter, batch_size= self._batch_size, verbose= False)
        self._model.fit(X, Y)

        return

    def evaluate(self):
        """"""
        ValidData = self.ValidData
        #
        # ValidData['structuretaxvalueratio'] = ValidData['structuretaxvaluedollarcnt'] / ValidData['taxvaluedollarcnt']
        # ValidData.loc[ValidData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1
        #
        # ValidData['landtaxvalueratio'] = ValidData['landtaxvaluedollarcnt'] / ValidData['taxvaluedollarcnt']
        # ValidData.loc[ValidData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

        pred_valid = pd.DataFrame(index=ValidData.index)
        pred_valid['parcelid'] = ValidData['parcelid']

        truth_valid = pd.DataFrame(index=ValidData.index)
        truth_valid['parcelid'] = ValidData['parcelid']

        start = time.time()

        for d in self._l_valid_predict_columns:
            l_valid_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in
                               self._l_train_columns]
            x_valid = ValidData[l_valid_columns]
            x_valid = x_valid.values.astype(np.float32, copy=False)
            pred_valid[d] = self._model.predict(x_valid)  # * 0.95 + 0.011 * 0.05
            df_tmp = ValidData[ValidData['transactiondate'].dt.month == int(d[-2:])]
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


