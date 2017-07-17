from model.ModelBase import ModelBase
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import dill as pickle
import time,os
from datetime import datetime
from sklearn.cross_validation import StratifiedKFold,cross_val_score
import gc

class SVR_(ModelBase):
    """"""
    _l_drop_cols = ['logerror', 'parcelid', 'transactiondate', 'index', 'nullcount']
    _C = 0.8
    _epsilon = 0.1

    def train(self):
        """"""
        start = time.time()

        print('size before truncated outliers is %d ' % len(self.TrainData))
        self.TrainData = self.TrainData[
            (self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print('size after truncated outliers is %d ' % len(self.TrainData))

        X = self.TrainData.drop(self._l_drop_cols, axis=1)
        Y = self.TrainData['logerror']
        self._l_train_columns = X.columns
        X = X.values.astype(np.float32, copy=False)

        svr = SVR(C = self._C, epsilon= self._epsilon, tol= 1e-3, kernel= 'linear',max_iter= 100, verbose= True)

        self._model = svr.fit(X, Y)
        end = time.time()

        print('time consumed %d ' % ((end - start)))

        self._f_eval_train_model = '{0}/{1}_{2}.pkl'.format(self.OutputDir, self.__class__.__name__,
                                                            datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        # with open(self._f_eval_train_model, 'wb') as o_file:
        #   pickle.dump(self._model, o_file, -1)
        # o_file.close()

        self.TrainData = pd.concat([self.TrainData, self.ValidData[self.TrainData.columns]],
                                   ignore_index=True)  ## ignore_index will reset the index or index will be overlaped

        return

    def evaluate(self):
        """"""
        ## not truncate outliers
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
            pred_valid[d] = self._model.predict(x_valid)  # * 0.99 + 0.011 * 0.01
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
