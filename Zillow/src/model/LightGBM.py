from model.ModelBase import ModelBase
import numpy as np
import pandas as pd
import lightgbm
import gc
import os
import time
import math
from datetime import datetime
import numba
import dill as pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import StratifiedKFold,cross_val_score
from sklearn.grid_search import GridSearchCV
import sys

class LGB(ModelBase):

    _params = {
        'max_bin': 8,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'sub_feature': 0.8,
        'bagging_fraction':  0.85,
        'num_leaves': 128,
        'min_data':  200,
        'min_hessian':  0.01,
        'learning_rate': 0.02,
        'bagging_freq': 20
    }

    _iter = 120

    _l_drop_cols = ['logerror', 'parcelid', 'transactiondate','index']
    #_l_drop_cols = ['logerror', 'parcelid', 'transactiondate','index','taxdelinquencyyear', 'finishedsquarefeet15', 'finishedsquarefeet6', 'yardbuildingsqft17']

    def selection(self):
        """"""
        print('size before truncated outliers is %d ' % len(self.TrainData))
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print('size after truncated outliers is %d ' % len(self.TrainData))

        X = self.TrainData.drop(self._l_drop_cols, axis=1)
        Y = self.TrainData['logerror']
        self._l_train_columns = X.columns
        nfolds = 5
        FeatCols = list(self._l_train_columns)

        # rfr = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
        #                   max_features= 40, max_leaf_nodes=None,
        #                   min_samples_leaf= 20,
        #                   min_samples_split= 100, min_weight_fraction_leaf=0.2,
        #                   n_estimators= 10, n_jobs=1, oob_score=False, random_state=None,
        #                   verbose=1, warm_start=False)

        # BestTreeDepth = 0
        # BestBaggingFeat = 0
        # BestTreeNum = 0
        # MinMeanMetrics = 1000
        # for TreeNum in range(30, 33):
        #     for TreeDepth in list(range(200, 201)):
        #         # for BaggingFeat in range(int(math.sqrt(len(FeatCols))) - 2,int(math.sqrt(len(FeatCols))) + 3,1):
        #         for BaggingFeat in [int(math.sqrt(len(FeatCols)))]:
        #             KFold = StratifiedKFold(self.TrainData['logerror'], nfolds, shuffle=True, random_state=2017)
        #             metrics = np.zeros((len(KFold)), dtype=float)
        #             for fold, (tr, va) in enumerate(KFold, start=0):
        #                 print(len(tr),len(va))
        #                 TrainData = self.TrainData[self.TrainData.index.isin(tr)].copy()
        #                 ValidData = self.TrainData[self.TrainData.index.isin(va)].copy()
        #
        #                 RF = RandomForestRegressor(random_state=2017 * (fold + 1),criterion= 'mse',
        #                                             n_estimators=TreeNum, n_jobs= 2,
        #                                             max_depth=TreeDepth,
        #                                             max_features=BaggingFeat)
        #                 RF.fit(TrainData[FeatCols], TrainData['logerror'])
        #
        #                 ValidData['predict'] = RF.predict(ValidData[FeatCols])
        #                 #accuracy = 1.0 * len(ValidData[ValidData['predict'] == ValidData['y']]) / len(ValidData)
        #                 mae = np.mean(np.abs(ValidData['predict'] - ValidData['logerror']))
        #                 metrics[fold] = mae
        #             MeanMetrics = np.mean(metrics)
        #             print("TreeNum %d, TreeDepth %d, Mean mae %f" % (TreeNum, TreeDepth, MeanMetrics))
        #             if (MeanMetrics < MinMeanMetrics):
        #                 MinMeanMetrics = MeanMetrics
        #                 BestTreeNum = TreeNum
        #                 BestTreeDepth = TreeDepth
        #                 BestBaggingFeat = BaggingFeat

        RF = RandomForestRegressor(random_state=2017, criterion= 'mse',
                                    n_estimators= 50, n_jobs= 2,
                                    max_depth= 200,
                                    max_features= int(math.sqrt(len(FeatCols))))

        self._model = RF.fit(X,Y)
        importances = RF.feature_importances_
        #std = np.std([tree.feature_importances_ for tree in RF.estimators_],axis=0)
        self._l_selected_features = [FeatCols[i] for i in np.argsort(importances)[:50]]
        #self._model = RF.fit(X[self._l_train_columns],Y)
        #print(self._l_train_columns)

    def retrain(self):

        X = self.TrainData.drop(self._l_drop_cols, axis=1)
        Y = self.TrainData['logerror']

        RF = RandomForestRegressor(random_state=2017, criterion= 'mse',
                                    n_estimators= 50, n_jobs= 2,
                                    max_depth= 200,
                                    max_features= int(math.sqrt(len(self._l_selected_features))))
        self._model = RF.fit(X[self._l_selected_features],Y)
        self._l_train_columns = self._l_selected_features

    ## rewritten method
    def train(self):

        start = time.time()

        print('size before truncated outliers is %d ' % len(self.TrainData))
        TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print('size after truncated outliers is %d ' % len(TrainData))

        #TrainData['bathroomratio'] = TrainData['bathroomcnt'] / TrainData['calculatedbathnbr']
        #TrainData.loc[TrainData['bathroomratio'] < 0, 'bathroomratio'] = -1

        #TrainData['calculatedfinishedsquarefeatratio'] = TrainData['calculatedfinishedsquarefeet'] / TrainData['lotsizesquarefeet']
        #TrainData.loc[TrainData['calculatedfinishedsquarefeatratio'] > 1, 'calculatedfinishedsquarefeatratio'] = 1
        #TrainData.loc[TrainData['calculatedfinishedsquarefeatratio'] < 0, 'calculatedfinishedsquarefeatratio'] = -1

        # TrainData['finishedsquarefeatratio'] = TrainData['finishedfloor1squarefeet'] / TrainData['lotsizesquarefeet']
        # TrainData.loc[TrainData['finishedsquarefeatratio'] > 1, 'finishedsquarefeatratio'] = 1
        # TrainData.loc[TrainData['finishedsquarefeatratio'] < 0, 'finishedsquarefeatratio'] = -1

        TrainData['structuretaxvalueratio'] = TrainData['structuretaxvaluedollarcnt'] / TrainData['taxvaluedollarcnt']
        TrainData.loc[TrainData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1

        TrainData['landtaxvalueratio'] = TrainData['landtaxvaluedollarcnt'] / TrainData['taxvaluedollarcnt']
        TrainData.loc[TrainData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

        #TrainData['structurelandtaxvalueratio'] = TrainData['structuretaxvalueratio'] / TrainData['landtaxvalueratio']
        #TrainData['theothertaxvalueratio'] = 1.0 - TrainData['structuretaxvalueratio'] - TrainData['landtaxvalueratio']

        #TrainData['propertytaxratio'] = TrainData['taxamount'] / TrainData['taxvaluedollarcnt']
        #TrainData.loc[TrainData['propertytaxratio'] < 0, 'propertytaxratio'] = -1
        #TrainData.loc[TrainData['propertytaxratio'] > 1, 'propertytaxratio'] = 1

        X = TrainData.drop(self._l_drop_cols,axis= 1)
        Y = TrainData['logerror']
        ## features not been selected yet
        if(len(self._l_selected_features) == 0):
            self._l_train_columns = X.columns
        else:
            self._l_train_columns = self._l_selected_features

        X = X.values.astype(np.float32, copy=False)
        d_cv = lightgbm.Dataset(X,label=Y)

        ## one-hold mode for parameter tuning
        # msk = np.random.rand(len(self.TrainData)) < 0.9
        # train = self.TrainData[msk]
        # valid = self.TrainData[~msk]
        # x_train = train.drop(['logerror','parcelid','transactiondate'],axis= 1)
        # y_train = train['logerror']
        # self._l_train_columns = x_train.columns

        # x_valid = valid.drop(['logerror','parcelid','transactiondate'],axis= 1)
        # y_valid = valid['logerror']
        #
        # x_train = x_train.values.astype(np.float32, copy=False)
        # x_valid = x_valid.values.astype(np.float32, copy=False)
        #
        # d_train = lightgbm.Dataset(x_train,label=y_train)
        # d_valid = lightgbm.Dataset(x_valid,label=y_valid)
        # params['learning_rate'] = 0.026
        # params['bagging_freq'] = 20
        # self._model = lightgbm.train(params,d_train,100,verbose_eval= True,valid_sets=[d_valid])

        ## cv mode for parameter tuning
        # l_learning_rate = [0.014 + 0.002*i for i in range(5)]
        # l_bagging_freq = [10 + i*10 for i in range(5)]
        #
        # BestParams = {'learning_rate':0.0,'bagging_freq':0}
        # BestMAE = 1.0
        # for lr in l_learning_rate:
        #     for bf in l_bagging_freq:
        #         params['learning_rate'] = lr
        #         params['bagging_freq'] = bf
        #
        #         self._model = lightgbm.cv(params, d_cv, 100, nfold=5,verbose_eval= True)
        #         if(self._model.get('l1-mean')[-1] < BestMAE):
        #             BestMAE = self._model.get('l1-mean')[-1]
        #             BestParams['learning_rate'] = lr
        #             BestParams['bagging_freq'] = bf
        # print(BestParams)
        # params['learning_rate'] = BestParams['learning_rate']
        # params['bagging_freq'] = BestParams['bagging_freq']

        self._model = lightgbm.train(self._params, d_cv, self._iter, verbose_eval= True)

        self._f_eval_train_model = '{0}/{1}_{2}.pkl'.format(self.OutputDir, self.__class__.__name__,datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        #with open(self._f_eval_train_model,'wb') as o_file:
        #    pickle.dump(self._model,o_file,-1)
        #o_file.close()

        self.TrainData = pd.concat([self.TrainData,self.ValidData[self.TrainData.columns]],ignore_index= True) ## ignore_index will reset the index or index will be overlaped

        end = time.time()
        print('Training is done. Time elapsed %ds' % (end - start))

    ## evaluate on valid data
    def evaluate(self):
        """"""
        ValidData = self.ValidData

        #ValidData['bathroomratio'] = ValidData['bathroomcnt'] / ValidData['calculatedbathnbr']
        #ValidData.loc[ValidData['bathroomratio'] < 0, 'bathroomratio'] = -1

        #ValidData['calculatedfinishedsquarefeatratio'] = ValidData['calculatedfinishedsquarefeet'] / ValidData['lotsizesquarefeet']
        #ValidData.loc[ValidData['calculatedfinishedsquarefeatratio'] > 1, 'calculatedfinishedsquarefeatratio'] = 1
        #ValidData.loc[ValidData['calculatedfinishedsquarefeatratio'] < 0, 'calculatedfinishedsquarefeatratio'] = -1

        # ValidData['finishedsquarefeatratio'] = ValidData['finishedfloor1squarefeet'] / ValidData['lotsizesquarefeet']
        # ValidData.loc[ValidData['finishedsquarefeatratio'] > 1, 'finishedsquarefeatratio'] = 1
        # ValidData.loc[ValidData['finishedsquarefeatratio'] < 0, 'finishedsquarefeatratio'] = -1

        ValidData['structuretaxvalueratio'] = ValidData['structuretaxvaluedollarcnt'] / ValidData['taxvaluedollarcnt']
        ValidData.loc[ValidData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1

        ValidData['landtaxvalueratio'] = ValidData['landtaxvaluedollarcnt'] / ValidData['taxvaluedollarcnt']
        ValidData.loc[ValidData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

        #ValidData['structurelandtaxvalueratio'] = ValidData['structuretaxvalueratio'] / ValidData['landtaxvalueratio']
        #ValidData['theothertaxvalueratio'] = 1.0 - ValidData['structuretaxvalueratio'] - ValidData['landtaxvalueratio']

        #ValidData['propertytaxratio'] = ValidData['taxamount'] / ValidData['taxvaluedollarcnt']
        #ValidData.loc[ValidData['propertytaxratio'] < 0, 'propertytaxratio'] = -1
        #ValidData.loc[ValidData['propertytaxratio'] > 1, 'propertytaxratio'] = 1

        pred_valid = pd.DataFrame(index = ValidData.index)
        pred_valid['parcelid'] = ValidData['parcelid']

        truth_valid = pd.DataFrame(index = ValidData.index)
        truth_valid['parcelid'] = ValidData['parcelid']

        start = time.time()

        for d in self._l_valid_predict_columns:
            l_valid_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in self._l_train_columns]
            x_valid = ValidData[l_valid_columns]
            x_valid = x_valid.values.astype(np.float32, copy=False)
            pred_valid[d] = self._model.predict(x_valid) * 0.99 + 0.011 * 0.01
            df_tmp = ValidData[ValidData['transactiondate'].dt.month == int(d[-2:])]
            truth_valid.loc[df_tmp.index,d] = df_tmp['logerror']

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

    @numba.jit
    def __ApplyAE(self,PredColumn,TruthColumn):

        n = len(PredColumn)
        result = np.empty(n,dtype= 'float32')
        for i in range(n):
            v = TruthColumn[i]
            if(math.isnan(v) == False):
                result[i] = np.abs(v - PredColumn[i])

        return result

    def __ComputeMAE(self,df_pred,df_truth):
        """"""
        mae = pd.DataFrame(index = df_pred)
        for col in df_pred.columns:
            if(col == 'parcelid'):
                continue
            ret = self.__ApplyAE(df_pred[col],df_truth[col])
            mae[col] = pd.Series(ret,index= df_pred.index)

    ## predict on test data
    def submit(self):

        ## retrain with the whole training data
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]

        self.TrainData['structuretaxvalueratio'] = self.TrainData['structuretaxvaluedollarcnt'] / self.TrainData['taxvaluedollarcnt']
        self.TrainData['landtaxvalueratio'] = self.TrainData['landtaxvaluedollarcnt'] / self.TrainData['taxvaluedollarcnt']
        self.TrainData.loc[self.TrainData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1
        self.TrainData.loc[self.TrainData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

        X = self.TrainData.drop(self._l_drop_cols, axis=1)
        Y = self.TrainData['logerror']

        self._l_train_columns = X.columns

        X = X.values.astype(np.float32, copy=False)
        d_train = lightgbm.Dataset(X, label=Y)

        self._model = lightgbm.train(self._params, d_train, self._iter, verbose_eval=True)

        del self.TrainData, X, Y, d_train
        gc.collect()

        self.TestData = self._data.LoadFromHdfFile(self.InputDir,'test')
        #self.TestData = self.TestData.sample(frac = 0.01)

        self._sub = pd.DataFrame(index = self.TestData.index)
        self._sub['ParcelId'] = self.TestData['parcelid']

        N = 200000
        start = time.time()
        for d in self._l_test_predict_columns:
            s0 = time.time()

            print('Prediction for column %s ' % d)
            self.TestData['structuretaxvalueratio'] = self.TestData['structuretaxvaluedollarcnt'] / self.TestData['taxvaluedollarcnt']
            self.TestData['landtaxvalueratio'] = self.TestData['landtaxvaluedollarcnt'] / self.TestData['taxvaluedollarcnt']
            self.TestData.loc[self.TestData['structuretaxvalueratio'] < 0, 'structuretaxvalueratio'] = -1
            self.TestData.loc[self.TestData['landtaxvalueratio'] < 0, 'landtaxvalueratio'] = -1

            l_test_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in self._l_train_columns]
            x_test = self.TestData[l_test_columns]

            for idx in range(0, len(x_test), N):
                x_test_block = x_test[idx:idx + N].values.astype(np.float32, copy=False)
                self._model.reset_parameter({"num_threads": 4})
                ret = self._model.predict(x_test_block) * 0.99 + 0.011 * 0.01
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

        self._sub.to_csv('{0}/{1}_{2}.csv'.format(self.OutputDir, self.__class__.__name__,datetime.now().strftime('%Y%m%d-%H:%M:%S')),
                         index=False, float_format='%.4f')
