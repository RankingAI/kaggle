from model.ModelBase import ModelBase
from sklearn.linear_model import LassoLars
import pandas as pd
import numpy as np
import dill as pickle
import time,os
from datetime import datetime
from model.GBREncoding import GBRE
import gc

class LLR(ModelBase):
    """"""
    _l_drop_cols = ['logerror', 'parcelid', 'transactiondate', 'index', 'nullcount']
    _lr_alpha = 0.001
    _lr_iter = 200

    #_encode = GBRE()
    #
    # def __transform(self):
    #     """"""
    #     start = time.time()
    #
    #     slice = 20
    #
    #     TrainData = self.TrainData
    #
    #     l_drop_cols_en = [col for col in self._l_drop_cols if((col != 'logerror') & (col != 'parcelid'))]
    #     TrainData = TrainData.drop(l_drop_cols_en, axis= 1)
    #     EncodeModel = self._encode.train(TrainData)
    #     XTrain = TrainData.drop(['logerror'], axis= 1)
    #     l_train_columns = XTrain.columns
    #     IndiceTrain = EncodeModel.apply(XTrain)[..., :slice]
    #
    #     print('Tree number %d, slice %d' % (EncodeModel.estimators_.shape[0], slice))
    #
    #     cols = ['tree%d' % i for i in range(slice)]
    #     TransformedTrain = pd.DataFrame(data= IndiceTrain, index= XTrain.index, columns= cols)
    #     TransformedTrain['logerror'] = TrainData['logerror']
    #     TransformedTrain['parcelid'] = TrainData['parcelid']
    #
    #     d_transformed_valid = {}
    #     for d in self._l_valid_predict_columns:
    #         l_valid_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in l_train_columns]
    #         XValid = self.ValidData[l_valid_columns]
    #         IndiceValid = EncodeModel.apply(XValid)[..., :slice]
    #         TransformedValid = pd.DataFrame(data= IndiceValid, index= XValid.index, columns= cols)
    #         TransformedValid['logerror'] = self.ValidData['logerror']
    #         TransformedValid['parcelid'] = self.ValidData['parcelid']
    #         TransformedValid['transactiondate'] = self.ValidData['transactiondate']
    #         d_transformed_valid[d] = TransformedValid
    #
    #     return TransformedTrain, d_transformed_valid
    #
    # def EncodeEvaluate(self):
    #
    #
    #     print('size before truncated outliers is %d ' % len(self.TrainData))
    #     self.TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
    #     print('size after truncated outliers is %d ' % len(self.TrainData))
    #
    #     start = time.time()
    #     print('Transform begins...')
    #     train, valid = self.__transform()
    #     print('Transform done.')
    #
    #     XTrain = train.drop(['logerror', 'parcelid'], axis=1)
    #     YTrain = train['logerror']
    #
    #     XTrain, EncodeDict = self._encode.OHETr(XTrain)
    #     XTrain = pd.concat([self.TrainData.drop(self._l_drop_cols, axis= 1), XTrain], axis = 1)
    #     l_train_columns = self.TrainData.drop(self._l_drop_cols, axis= 1).columns
    #     print('Encoding for train is done.')
    #
    #     print('Lasso training begins...')
    #     lr = Lasso(alpha= self._lr_alpha, max_iter= self._lr_iter, tol=1e-4, random_state=2017, selection= self._lr_sel)
    #     model = lr.fit(XTrain, YTrain)
    #     print('Lasso training done.')
    #     del XTrain, YTrain, train
    #     gc.collect()
    #
    #     tmp_va = valid[list(valid.keys())[0]]
    #     pred_valid = pd.DataFrame(index= tmp_va.index)
    #     pred_valid['parcelid'] = tmp_va['parcelid']
    #
    #     truth_valid = pd.DataFrame(index= tmp_va.index)
    #     truth_valid['parcelid'] = tmp_va['parcelid']
    #
    #     del tmp_va
    #     gc.collect()
    #
    #     for d in self._l_valid_predict_columns:
    #         XValid = valid[d].drop(['logerror', 'parcelid', 'transactiondate'], axis= 1)
    #         XValid = self._encode.OHEVa(EncodeDict, XValid)
    #
    #         l_valid_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in l_train_columns]
    #
    #         XValid = pd.concat([self.ValidData[l_valid_columns], XValid], axis= 1)
    #         print('Encoding for valid is done.')
    #         pred_valid[d] = model.predict(XValid)  # * 0.99 + 0.011 * 0.01
    #         df_tmp = valid[d][valid[d]['transactiondate'].dt.month == int(d[-2:])]
    #         truth_valid.loc[df_tmp.index, d] = df_tmp['logerror']
    #
    #     score = 0.0
    #     ae = np.abs(pred_valid - truth_valid)
    #     for col in ae.columns:
    #         score += np.sum(ae[col])
    #     score /= len(pred_valid)  ##!! divided by number of instances, not the number of 'cells'
    #     print('============================= ')
    #     print('Local MAE is %.6f' % score)
    #     print('=============================')
    #
    #     end = time.time()
    #
    #     del self.ValidData
    #     gc.collect()
    #
    #     print('time elapsed %ds' % (end - start))

    def train(self):
        """"""
        start = time.time()

        print('size before truncated outliers is %d ' % len(self.TrainData))
        TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print('size after truncated outliers is %d ' % len(self.TrainData))

        TrainData['longitude'] -= -118600000
        TrainData['latitude'] -= 34220000
        #extra_tr = pd.read_hdf(path_or_buf='%s/p21/eval_train.hdf' % self.InputDir, key='train')
        #self.TrainData = pd.concat([self.TrainData, extra_tr.drop('parcelid', axis= 1)], axis = 1)

        X = self.TrainData.drop(self._l_drop_cols, axis=1)
        Y = self.TrainData['logerror']
        self._l_train_columns = X.columns
        X = X.values.astype(np.float32, copy=False)

        lr = LassoLars(alpha= self._lr_alpha, max_iter= self._lr_iter, verbose= True)
        self._model = lr.fit(X, Y)
        end = time.time()

        print('Training iterates %d, time consumed %d ' % (self._model.n_iter_, (end - start)))

        self._f_eval_train_model = '{0}/{1}_{2}.pkl'.format(self.OutputDir, self.__class__.__name__,
                                                            datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        #with open(self._f_eval_train_model, 'wb') as o_file:
        #    pickle.dump(self._model, o_file, -1)
        #o_file.close()

        #self.TrainData = pd.concat([self.TrainData, self.ValidData[self.TrainData.columns]],
        #                           ignore_index=True)  ## ignore_index will reset the index or index will be overlaped

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

            extra_va = pd.read_hdf(path_or_buf='%s/p21/eval_valid_%s.hdf' % (self.InputDir, d), key='valid')
            #ValidData = self.ValidData.join(extra_va, on= 'parcelid', how= 'left')
            ValidData = pd.concat([self.ValidData, extra_va.drop('parcelid', axis= 1)], axis= 1)

            x_valid = ValidData[l_valid_columns]
            x_valid = x_valid.values.astype(np.float32, copy=False)
            pred_valid[d] = self._model.predict(x_valid)  # * 0.99 + 0.011 * 0.01
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

    def submit(self):
        """"""
        ## retrain with the whole training data
        self.TrainData = self.TrainData[
            (self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]

        X = self.TrainData.drop(self._l_drop_cols, axis=1)
        Y = self.TrainData['logerror']
        X = X.values.astype(np.float32, copy=False)

        lr = Lasso(alpha= self._alpha, max_iter= self._iter, tol= 1e-4, random_state= 2017, selection= self._sel)
        self._model = lr.fit(X, Y)

        del self.TrainData, X, Y
        gc.collect()

        self.TestData = self._data.LoadFromHdfFile(self.InputDir, 'test')
        # self.TestData = self.TestData.sample(frac = 0.01)

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
                ret = self._model.predict(x_test_block)  # * 0.99 + 0.011 * 0.01
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
            '{0}/{1}_{2}.csv'.format(self.OutputDir, self.__class__.__name__,
                                     datetime.now().strftime('%Y%m%d-%H:%M:%S')),
            index=False, float_format='%.4f')
