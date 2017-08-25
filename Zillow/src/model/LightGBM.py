import time
import lightgbm
import pandas as pd
import numpy as np
import dill as pickle
from datetime import datetime
from model.ModelBase import ModelBase

class LGB(ModelBase):
    """"""
    _params = {
        'max_bin': 8,
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'lambda_l1': 2.0,
        'sub_feature': 0.80,
        'bagging_fraction':  0.85,
        'num_leaves': 32,
        'min_data':  100,
        'min_hessian':  0.01,
        'learning_rate': 0.02,
        'bagging_freq': 10
    }

    _iter = 1400

    _l_drop_cols = ['logerror', 'parcelid', 'transactiondate']

    def train(self):
        """"""
        self.__fit()

        return self.__predict()

    def __fit(self):
        """"""
        start = time.time()

        ## truncate outliers
        print('initial data shape : ', self.TrainData.shape)
        TrainData = self.TrainData[(self.TrainData['logerror'] > self.MinLogError) & (self.TrainData['logerror'] < self.MaxLogError)]
        print('data shape truncated: ', TrainData.shape)

        ## ---- new features -----
        ## lon/lat transformed
        TrainData['longitude'] -= -118600000
        TrainData['lat1'] = TrainData['latitude'] - 34500000
        TrainData['latitude'] -= 34220000

        ## add structure tax ratio
        TrainData['structuretaxvalueratio'] = TrainData['structuretaxvaluedollarcnt'] / TrainData['taxvaluedollarcnt']
        TrainData.loc[TrainData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1
        ## add land tax ratio
        TrainData['landtaxvalueratio'] = TrainData['landtaxvaluedollarcnt'] / TrainData['taxvaluedollarcnt']
        TrainData.loc[TrainData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

        ##

        ##
        #TrainData['N-ValueRatio'] = TrainData['taxvaluedollarcnt'] / TrainData['taxamount']
        #TrainData['N-LivingAreaProp'] = TrainData['calculatedfinishedsquarefeet'] / TrainData['lotsizesquarefeet']
        #TrainData['N-ValueProp'] = TrainData['structuretaxvaluedollarcnt'] / TrainData['landtaxvaluedollarcnt']
        #TrainData['N-TaxScore'] = TrainData['taxvaluedollarcnt'] * TrainData['taxamount']

        print('data shape after feature space being extended : ', TrainData.shape)

        X = TrainData.drop(self._l_drop_cols,axis= 1)
        Y = TrainData['logerror']

        self._l_train_columns = X.columns

        X = X.values.astype(np.float32, copy=False)
        d_cv = lightgbm.Dataset(X,label=Y)

        self._model = lightgbm.train(self._params, d_cv, self._iter, verbose_eval= 100)

        end = time.time()

        print('Training is done. Time elapsed %ds' % (end - start))

        return

    def __predict(self):
        """"""
        TestData = self.TestData
        print('initial data shape: ', self.TestData.shape)

        ## ---- new features ----
        ## lon/lat transformed
        TestData['longitude'] -= -118600000
        TestData['lat1'] = TestData['latitude'] - 34500000
        TestData['latitude'] -= 34220000

        ## add structure tax ratio
        TestData['structuretaxvalueratio'] = TestData['structuretaxvaluedollarcnt'] / TestData['taxvaluedollarcnt']
        TestData.loc[TestData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1
        ## add land tax ratio
        TestData['landtaxvalueratio'] = TestData['landtaxvaluedollarcnt'] / TestData['taxvaluedollarcnt']
        TestData.loc[TestData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

        ##

        ##
        #TestData['N-ValueRatio'] = TestData['taxvaluedollarcnt'] / TestData['taxamount']
        #TestData['N-LivingAreaProp'] = TestData['calculatedfinishedsquarefeet'] / TestData['lotsizesquarefeet']
        #TestData['N-ValueProp'] = TestData['structuretaxvaluedollarcnt'] / TestData['landtaxvaluedollarcnt']
        #TestData['N-TaxScore'] = TestData['taxvaluedollarcnt'] * TestData['taxamount']

        print('data shape after feature space being extended : ', TestData.shape)

        pred_test = pd.DataFrame(index = TestData.index)
        pred_test['parcelid'] = TestData['parcelid']

        truth_test = pd.DataFrame(index = TestData.index)
        truth_test['parcelid'] = TestData['parcelid']

        start = time.time()

        for mth in self.l_test_predict_columns: ## objective columns need to be predicted
            FixedTestPredCols = ['%s%s' % (c, mth) if (c in ['lastgap']) else c for c in self._l_train_columns]
            x_test = TestData[FixedTestPredCols]
            x_test = x_test.values.astype(np.float32, copy=False)
            ## fill prediction
            pred_test['%s' % mth] = self._model.predict(x_test)
            ## fill truth
            df_tmp = TestData[TestData['transactiondate'].dt.month == mth]
            truth_test.loc[df_tmp.index, '%s' % mth] = df_tmp['logerror']

        # ## save for predict
        # OutputFile = '{0}/{1}_{2}.pkl'.format(self.OutputDir, self.__class__.__name__, datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        # with open(OutputFile, 'wb') as o_file:
        #    pickle.dump(pred_test, o_file, -1)
        # o_file.close()
        # ## save for truth
        # with open('%s/truth.pkl' % self.OutputDir, 'wb') as o_file:
        #     pickle.dump(truth_test, o_file, -1)
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