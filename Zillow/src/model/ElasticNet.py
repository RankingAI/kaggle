from model.ModelBase import ModelBase
from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np
import time
from datetime import datetime
import dill as pickle

class EN(ModelBase):

   _l_drop_cols = ['logerror', 'parcelid', 'transactiondate']
   _alpha = 0.001
   _ratio = 0.02
   _iter = 10
   _sel = 'random'

   def train(self):
      """"""
      self.__fit()

      return self.__predict()

   """"""
   def __fit(self):
      """"""
      start = time.time()

      ## truncate outliers
      print('initial data shape : ', self.TrainData.shape)
      TrainData = self.TrainData[(self.TrainData['logerror'] > self.MinLogError) & (self.TrainData['logerror'] < self.MaxLogError)]
      print('data shape truncated: ', TrainData.shape)

      ## lon/lat transformed
      TrainData['longitude'] -= -118600000
      #TrainData['lat1'] = TrainData['latitude'] - 34500000
      TrainData['latitude'] -= 34220000
      ## add structure tax ratio
      TrainData['structuretaxvalueratio'] = TrainData['structuretaxvaluedollarcnt'] / TrainData['taxvaluedollarcnt']
      TrainData.loc[TrainData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1
      ## add land tax ratio
      TrainData['landtaxvalueratio'] = TrainData['landtaxvaluedollarcnt'] / TrainData['taxvaluedollarcnt']
      TrainData.loc[TrainData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

      ##
      #TrainData['N-TaxScore'] = TrainData['taxvaluedollarcnt'] * TrainData['taxamount']
      #TrainData['N-ValueProp'] = TrainData['structuretaxvaluedollarcnt'] / TrainData['landtaxvaluedollarcnt']

      print('data shape after feature space being extended : ', TrainData.shape)

      X = TrainData.drop(self._l_drop_cols, axis=1)
      Y = TrainData['logerror']
      self._l_train_columns = X.columns
      X = X.values.astype(np.float32, copy=False)

      en = ElasticNet(alpha= self._alpha, l1_ratio= self._ratio, max_iter= self._iter, tol= 1e-4, selection= self._sel, random_state= 2017)
      self._model = en.fit(X, Y)

      end = time.time()

      print('Training iterates %d, time consumed %d ' % (self._model.n_iter_,(end - start)))
      return

   def __predict(self):
      """"""
      TestData = self.TestData

      ## lon/lat transformed
      TestData['longitude'] -= -118600000
      #TestData['lat1'] = TestData['latitude'] - 34500000
      TestData['latitude'] -= 34220000
      ## add structure tax ratio
      TestData['structuretaxvalueratio'] = TestData['structuretaxvaluedollarcnt'] / TestData['taxvaluedollarcnt']
      TestData.loc[TestData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1
      ## add land tax ratio
      TestData['landtaxvalueratio'] = TestData['landtaxvaluedollarcnt'] / TestData['taxvaluedollarcnt']
      TestData.loc[TestData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

      ##
      #TestData['N-TaxScore'] = TestData['taxvaluedollarcnt'] * TestData['taxamount']
      #TestData['N-ValueProp'] = TestData['structuretaxvaluedollarcnt'] / TestData['landtaxvaluedollarcnt']

      ## not truncate outliers
      pred_test = pd.DataFrame(index= TestData.index)
      pred_test['parcelid'] = TestData['parcelid']

      truth_test = pd.DataFrame(index= TestData.index)
      truth_test['parcelid'] = TestData['parcelid']

      start = time.time()

      for mth in self.l_test_predict_columns:
         l_test_columns = ['%s%s' % (c, mth) if (c in ['lastgap']) else c for c in self._l_train_columns]
         x_test = TestData[l_test_columns]
         x_test = x_test.values.astype(np.float32, copy= False)
         pred_test['%s' % mth] = self._model.predict(x_test)
         df_tmp = TestData[TestData['transactiondate'].dt.month == mth]
         truth_test.loc[df_tmp.index, '%s' % mth] = df_tmp['logerror']

      # ## save
      # OutputFile = '{0}/{1}_{2}.pkl'.format(self.OutputDir, self.__class__.__name__, datetime.now().strftime('%Y%m%d-%H:%M:%S'))
      # with open(OutputFile, 'wb') as o_file:
      #    pickle.dump(pred_test, o_file, -1)
      # o_file.close()
      #
      # ## save for truth
      # with open('%s/truth.pkl' % self.OutputDir, 'wb') as o_file:
      #    pickle.dump(truth_test, o_file, -1)
      # o_file.close()

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

