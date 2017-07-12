from model.ModelBase import ModelBase
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
import pandas as pd
import numpy as np
import dill as pickle
import time,os
from datetime import datetime
from sklearn.cross_validation import StratifiedKFold,cross_val_score
import gc

class EN(ModelBase):
   """"""
   def train(self):
      """"""
      start = time.time()

      print('size before truncated outliers is %d ' % len(self.TrainData))
      self.TrainData = self.TrainData[
         (self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
      print('size after truncated outliers is %d ' % len(self.TrainData))

      X = self.TrainData.drop(['logerror','parcelid', 'transactiondate'], axis=1)
      Y = self.TrainData['logerror']

      self._l_train_columns = X.columns
      X = X.values.astype(np.float32, copy=False)

      al = 0.1
      ratio = 0.1

      en = ElasticNet(alpha= 0.1, l1_ratio = 0.1, max_iter= 800, tol= 1e-4, selection= 'random', random_state= 2017)
      self._model = en.fit(X, Y)
      end = time.time()

      print('Training iterates %d, time consumed %d ' % (self._model.n_iter_,(end - start)))

      self._f_eval_train_model = '{0}/{1}_{2}.pkl'.format(self.OutputDir, self.__class__.__name__,
                                                          datetime.now().strftime('%Y%m%d-%H:%M:%S'))
      with open(self._f_eval_train_model, 'wb') as o_file:
         pickle.dump(self._model, o_file, -1)
      o_file.close()

      self.TrainData = pd.concat([self.TrainData, self.ValidData[self.TrainData.columns]],ignore_index=True)  ## ignore_index will reset the index or index will be overlaped

      return

      #x_train = X.values.astype(np.float32, copy = False)
      #y_train = Y.values.astype(np.float32, copy = False)

      #l_l1_ratio = [.1, .5, .7, .9, .95, .99, 1]
      #l_alpha = [1e-3,1e-2,1e-1,1.0,10.0]
      # l_l1_ratio = [0.4, 0.6, 0.8]
      # l_alpha = [.1]
      # nfolds = 3
      #
      # BestRatio = 0.0
      # BestAlpha = .0
      # MinMeanMetrics = 100
      # for ratio in l_l1_ratio:
      #    for al in l_alpha:
      #       s0 = time.time()
      #       KFold = StratifiedKFold(Y, nfolds, shuffle=True, random_state=2017)
      #       metrics = np.zeros((len(KFold)), dtype=float)
      #       for fold, (tr, va) in enumerate(KFold, start=0):
      #
      #           x_train = X[X.index.isin(tr)].copy()
      #           y_train = Y[Y.index.isin(tr)].copy()
      #           x_valid = X[X.index.isin(va)].copy()
      #           y_valid = Y[Y.index.isin(va)].copy()
      #
      #           en = ElasticNet(alpha= al, l1_ratio = ratio, max_iter= 800, tol= 1e-4, selection= 'random', random_state= 2017)
      #           model = en.fit(x_train, y_train)
      #
      #           pred = model.predict(x_valid)
      #           mae = np.sum(np.abs(pred - y_valid))/len(pred)
      #           metrics[fold] = mae
      #           print('%d - %d - %.6f' % (fold, model.n_iter_, mae))
      #
      #       MeanMetrics = np.mean(metrics)
      #       e0 = time.time()
      #       print('ratio %.6f, alpha %.6f, mean mae %.6f, time consumed %ds' % (ratio, al, MeanMetrics,(e0 - s0)))
      #       if(MeanMetrics < MinMeanMetrics):
      #           MinMeanMetrics = MeanMetrics
      #           BestRatio = ratio
      #           BestAlpha = al
      #
      # print('Best alpha %.6f, best ratio %.6f' % (BestAlpha, BestRatio))


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
         pred_valid[d] = self._model.predict(x_valid)# * 0.99 + 0.011 * 0.01
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

      # del self.ValidData
      # gc.collect()

      print('time elapsed %ds' % (end - start))

   def submit(self):
      """"""
      ## retrain with the whole training data
      self.TrainData = self.TrainData[
         (self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]

      X = self.TrainData.drop(['logerror', 'parcelid', 'transactiondate'], axis=1)
      Y = self.TrainData['logerror']

      X = X.values.astype(np.float32, copy=False)

      al = 0.1
      ratio = 0.1

      en = ElasticNet(alpha= 0.1, l1_ratio = 0.1, max_iter= 800, tol= 1e-4, selection= 'random', random_state= 2017)
      self._model = en.fit(X, Y)

      del self.TrainData, X, Y
      gc.collect()

      self.TestData = self._data.LoadFromHdfFile(self.InputDir, 'test')

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
            x_test_block = x_test[idx:idx + N].values.astype(np.float32, copy=False)
            ret = self._model.predict(x_test_block)# * 0.99 + 0.011 * 0.01
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

      self._sub.to_csv(
         '{0}/{1}_{2}.csv'.format(self.OutputDir, self.__class__.__name__, datetime.now().strftime('%Y%m%d-%H:%M:%S')),
         index=False, float_format='%.4f')