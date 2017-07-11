from model.ModelBase import ModelBase
import numpy as np
import pandas as pd
import xgboost
import gc
import math
import time
import dill as pickle
import os
from datetime import datetime

class XGB(ModelBase):

    ## compute MAE of single column
    def ComputeMAE(self,y_pred,y_truth):
        """"""
        return np.sum(np.abs(y_pred - y_truth))/len(y_pred)

    ## feature selection with importance
    def selection(self):
        """"""
        # split data into train and valid sets
        msk = np.random.rand(len(self.TrainData)) < 0.1
        valid = self.TrainData[msk]
        train = self.TrainData[~msk]
        print('Length of train for selection is %d, while that of valid is %d' % (len(train),len(valid)))

        print('Before truncating outliers(train), %d ' % len(train))
        train = train[(train['logerror'] > self._low) & (train['logerror'] < self._up)]
        print('After truncating outliers(train), %d' % len(train))

        x_train = train.drop(['logerror','parcelid','transactiondate'],axis= 1)
        y_train = train['logerror']#.values.astype(np.float32)
        x_valid = valid.drop(['logerror','parcelid','transactiondate'],axis= 1)
        y_valid = valid['logerror']#.values.astype(np.float32)
        y_mean = np.mean(y_train)

        #
        params = {
            'subsample': 0.80,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'base_score': y_mean,
            'silent': 0,
            'npthread': 4,
            'lambda': 0.8,
            'alpha': 0.3995
        }
        TreeNum = 120
        params['eta'] = 0.04
        params['max_depth'] = 10

        # fit model on all training data
        dtrain = xgboost.DMatrix(x_train,y_train)
        model = xgboost.train(params,dtrain,num_boost_round= TreeNum)
        print('Fitting is done ...')

        # make predictions for test data and evaluate
        dvalid = xgboost.DMatrix(x_valid)
        y_pred = model.predict(dvalid)
        mae = self.ComputeMAE(y_pred, y_valid)
        print("MAE : %.6f" % mae)

        # Fit model using each importance as a threshold
        scores = model.get_score(importance_type='weight')
        SortedFeats = sorted(scores.items(), key=lambda x: x[1], reverse= True)
        thresholds = [0.1,0.2,0.3,0.4]

        MinMAE = 100
        BestThreshold = 0.0
        for thresh in thresholds:
            # select features using threshold
            feats = [feat[0] for feat in SortedFeats[:int(len(SortedFeats)*thresh)]]
            select_x_train = xgboost.DMatrix(x_train[feats],y_train)
            # train model
            params['max_depth'] = 6
            params['eta'] = 0.02
            selection_model = xgboost.train(params, select_x_train, num_boost_round= 60)
            # eval model
            select_x_valid = xgboost.DMatrix(x_valid[feats])
            y_pred = selection_model.predict(select_x_valid)
            mae = self.ComputeMAE(y_pred, y_valid)
            print("Thresh=%.3f, n=%d, MAE: %.6f" % (thresh, x_train[feats].shape[1], mae))
            if(mae < MinMAE):
                MinMAE = mae
                BestThreshold = thresh

        self._l_selected_features = [feat[0] for feat in SortedFeats[:int(len(SortedFeats) * BestThreshold)]]

        with open('%s/selected.dat' % self.OutputDir,'w') as o_file:
            for col in self._l_selected_features:
                o_file.write('%s\n' % col)
        o_file.close()

        return

    def train(self):
        """"""
        start = time.time()

        print(len(self.TrainData))
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print(len(self.TrainData))

        x_train = self.TrainData.drop(['logerror','parcelid','transactiondate'],axis= 1)
        y_train = self.TrainData['logerror'].values.astype(np.float32)
        # Judge if feature selection has been done.
        if(len(self._l_selected_features) == 0):
            print('Full featureed ...')
            self._l_train_columns = x_train.columns
        else:
            print('Selected featured ...')
            self._l_train_columns = self._l_selected_features
        x_train = x_train[self._l_train_columns]

        #x_valid = x_train.values.astype(np.float32, copy=False)
        y_mean = np.mean(y_train)

        dtrain = xgboost.DMatrix(x_train, y_train)
        params = {
            'subsample': 0.80,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'base_score': y_mean,
            'silent': 0,
            'npthread': 4,
            'lambda': 0.8,
            'alpha': 0.3995
        }

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
        self._model = xgboost.train(params, dtrain, num_boost_round= 120)

        self._f_eval_train_model = '{0}/{1}_{2}.pkl'.format(self.OutputDir, self.__class__.__name__,datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        with open(self._f_eval_train_model,'wb') as o_file:
            pickle.dump(self._model,o_file,-1)
        o_file.close()

        self.TrainData = pd.concat([self.TrainData,self.ValidData[self.TrainData.columns]],ignore_index= True) ## ignore_index will reset the index or index will be overlaped

        end = time.time()
        print('Training is done. Time elapsed %ds' % (end - start))

    def evaluate(self):
        """"""
        pred_valid = pd.DataFrame(index=self.ValidData.index)
        pred_valid['parcelid'] = self.ValidData['parcelid']

        truth_valid = pd.DataFrame(index=self.ValidData.index)
        truth_valid['parcelid'] = self.ValidData['parcelid']

        start = time.time()

        for d in self._l_valid_predict_columns:
            l_valid_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in self._l_train_columns]
            x_valid = self.ValidData[l_valid_columns]
            x_valid.columns = ['lastgap' if('lastgap' in col) else 'monthyear' if('monthyear' in col) else 'buildingage' if('buildingage' in col) else col for col in x_valid.columns]
            #x_valid = x_valid.values.astype(np.float32, copy=False)
            dvalid = xgboost.DMatrix(x_valid)
            pred_valid[d] = self._model.predict(dvalid)  # * 0.97 + 0.011 * 0.03
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
        start = time.time()

        ## retrain model
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        X = self.TrainData.drop(['logerror', 'parcelid', 'transactiondate'], axis=1)
        Y = self.TrainData['logerror'].values.astype(np.float32)
        y_mean = np.mean(Y)

        dtrain = xgboost.DMatrix(X,Y)

        params = {
            'subsample': 0.80,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'base_score': y_mean,
            'silent': 0,
            'npthread': 4,
            'lambda': 0.8,
            'alpha': 0.3995
        }

        params['eta'] = 0.04
        params['max_depth'] = 10

        print('\n Retraining XGBoost ...')
        self._model = xgboost.train(params, dtrain, num_boost_round= 80)
        print('\n Retraining done.')

        del X,Y,dtrain,self.TrainData
        gc.collect()

        ## for test
        self.TestData = self._data.LoadFromHdfFile(self.InputDir, 'test')

        self._sub = pd.DataFrame(index=self.TestData.index)
        self._sub['ParcelId'] = self.TestData['parcelid']

        N = 200000
        for d in self._l_test_predict_columns:
            s0 = time.time()

            print("Start prediction ...")
            l_test_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in
                              self._l_train_columns]
            x_test = self.TestData[l_test_columns]
            x_test.columns = ['lastgap' if('lastgap' in col) else 'monthyear' if('monthyear' in col) else 'buildingage' if('buildingage' in col) else col for col in x_test.columns]

            for idx in range(0, len(x_test), N):
                x_test_block = x_test[idx:idx + N]
                dtest = xgboost.DMatrix(x_test_block)
                ret = self._model.predict(dtest)
                self._sub.loc[x_test[idx:idx + N].index, d] = ret
                print(np.mean(np.abs(ret)))

            e0 = time.time()
            print('Prediction for column %s is done, time elapsed %ds' % (d, (e0 - s0)))

        ## clean
        del self.TestData
        gc.collect()

        end = time.time()
        print('Prediction is done, time elapsed %ds' % (end - start))

        if (os.path.exists(self.OutputDir) == False):
            os.makedirs(self.OutputDir)

        self._sub.to_csv('{0}/{1}_{2}.csv'.format(self.OutputDir, self.__class__.__name__,datetime.now().strftime('%Y%m%d-%H:%M:%S')),
                         index=False, float_format='%.4f')
        print('Submit is done.')
