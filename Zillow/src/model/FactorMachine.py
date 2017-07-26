from model.ModelBase import ModelBase
import pandas as pd
import numpy as np
import sys,os,time
import numba
import math
from fastFM import als
import scipy.sparse
import gc
import dill as pickle
from datetime import datetime

class FM(ModelBase):

    _l_drop_cols = ['logerror', 'parcelid', 'transactiondate', 'index']

    _l_cate_cols = ['hashottuborspa', 'taxdelinquencyflag', 'airconditioningtypeid', 'architecturalstyletypeid',
                    'buildingqualitytypeid', 'decktypeid', 'heatingorsystemtypeid', 'pooltypeid10', 'pooltypeid2',
                    'pooltypeid7', 'propertylandusetypeid', 'regionidcity', 'regionidcounty', 'regionidneighborhood',
                    'regionidzip', 'fips']  # , 'nullcount', 'lastgap', 'buildingage', 'monthyear']

    _iter = 100
    _reg_w = 20000
    _reg_v = 20000
    _rank = 1

    """"""
    @numba.jit
    def __ApplyOH(ColumnValues, headers):

        n = len(ColumnValues)
        result = np.zeros((n, len(headers)), dtype='int8')
        for i in range(n):
            v = ColumnValues[i]
            if (math.isnan(v)):
                result[i, headers['missing']] = 1
            elif (('%d' % int(v)) in headers):
                result[i, headers['%d' % int(v)]] = 1

        return result

    def __ExtraEncode(self):
        """"""
        EncodeCols = ['nullcount']#, 'lastgap', 'buildingage', 'monthyear']
        for ec in EncodeCols:

            ValueCounts = [str(int(v)) for v in self.TrainData[ec].value_counts().index.values]
            ValueCounts.append('missing')
            SelectedValues = dict((k, v) for (v, k) in enumerate(ValueCounts, start=0))

            ## for train
            OHTr = self.__ApplyOH(self.TrainData[ec].values, SelectedValues)
            headers = dict((('%s_%s' % (ec, k)), SelectedValues[k]) for k in SelectedValues)
            tmp = [v[0] for v in sorted(headers.items(), key=lambda x: x[1])]
            OHDFTr = pd.DataFrame(OHTr, index= self.TrainData.index, columns=tmp)
            self.TrainData = pd.concat([self.TrainData, OHDFTr], axis=1)

            ## for valid
            if(ec == 'nullcount'):
                cname = ec
                OHVa = self.__ApplyOH(self.ValidData[cname].values, SelectedValues)
                headers = dict((('%s_%s' % (cname, k)), SelectedValues[k]) for k in SelectedValues)
                tmp = [v[0] for v in sorted(headers.items(), key=lambda x: x[1])]
                OHDFVa = pd.DataFrame(OHVa, index=self.ValidData.index, columns=tmp)
                self.ValidData = pd.concat([self.ValidData, OHDFVa], axis=1)
            else:
                for d in self._l_valid_predict_columns:
                    cname = '%s%s' % (ec, d)
                    OHVa = self.__ApplyOH(self.ValidData[cname].values, SelectedValues)
                    headers = dict((('%s_%s' % (cname, k)), SelectedValues[k]) for k in SelectedValues)
                    tmp = [v[0] for v in sorted(headers.items(), key=lambda x: x[1])]
                    OHDFVa = pd.DataFrame(OHVa, index=self.ValidData.index, columns=tmp)
                    self.ValidData = pd.concat([self.ValidData, OHDFVa], axis=1)

        return

    def train(self):
        """"""
        print('size before truncated outliers is %d ' % len(self.TrainData))
        #TrainData = self.TrainData[(self.TrainData['logerror'] > -0.4) & (self.TrainData['logerror'] < 0.418)]
        TrainData = self.TrainData
        print('size after truncated outliers is %d ' % len(TrainData))
        print('train data size %d' % len(TrainData))

        #self.__ExtraEncode()

        X = TrainData.drop(self._l_drop_cols, axis=1)
        Y = TrainData['logerror']
        l_train_columns = X.columns

        cols = []
        for col in l_train_columns:
            for cc in self._l_cate_cols:
                if (col.startswith('%s_' % cc)):
                    cols.append(col)
                    break

        tmp_cols = set(cols)
        if(len(tmp_cols) != len(cols)):
            print('!!!! cols duplicated .')

        self._l_train_columns = list(tmp_cols)

        X = scipy.sparse.csr_matrix(X[self._l_train_columns])
        self._model = als.FMRegression(n_iter= self._iter, init_stdev=0.1, rank= self._rank, l2_reg_w= self._reg_w, l2_reg_V= self._reg_v)
        self._model.fit(X, Y)

        print('training done.')

        self._f_eval_train_model = '{0}/{1}_{2}.pkl'.format(self.OutputDir, self.__class__.__name__,datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        with open(self._f_eval_train_model,'wb') as o_file:
            pickle.dump(self._model,o_file,-1)
        o_file.close()

        self.TrainData = pd.concat([self.TrainData,self.ValidData[self.TrainData.columns]],ignore_index= True) ## ignore_index will reset the index or index will be overlaped

        return

    def evaluate(self):
        """"""
        ValidData = self.ValidData

        pred_valid = pd.DataFrame(index = ValidData.index)
        pred_valid['parcelid'] = ValidData['parcelid']

        truth_valid = pd.DataFrame(index = ValidData.index)
        truth_valid['parcelid'] = ValidData['parcelid']

        start = time.time()

        for d in self._l_valid_predict_columns:
            ## no target-dependent columns
            l_valid_columns = self._l_train_columns
            x_valid = scipy.sparse.csr_matrix(self.ValidData[l_valid_columns])

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
        ## retrain
        TrainData = self.TrainData
        print('train data size %d' % len(TrainData))

        X = TrainData.drop(self._l_drop_cols, axis=1)
        Y = TrainData['logerror']
        l_train_columns = X.columns

        cols = []
        for col in l_train_columns:
            for cc in self._l_cate_cols:
                if (col.startswith('%s_' % cc)):
                    cols.append(col)
                    break

        tmp_cols = set(cols)
        if(len(tmp_cols) != len(cols)):
            print('!!!! cols duplicated .')

        self._l_train_columns = list(tmp_cols)

        X = scipy.sparse.csr_matrix(X[self._l_train_columns])
        self._model = als.FMRegression(n_iter= self._iter, init_stdev=0.1, rank= self._rank, l2_reg_w= self._reg_w, l2_reg_V= self._reg_v)
        self._model.fit(X, Y)

        print('training done.')

        del self.TrainData, X, Y
        gc.collect()

        ## predict for test
        self.TestData = self._data.LoadFromHdfFile(self.InputDir,'test')
        #self.TestData = self.TestData.sample(frac = 0.01)

        self._sub = pd.DataFrame(index = self.TestData.index)
        self._sub['ParcelId'] = self.TestData['parcelid']

        N = 200000
        start = time.time()
        for d in self._l_test_predict_columns:
            s0 = time.time()

            print('Prediction for column %s ' % d)

            l_test_columns = self._l_train_columns
            #x_test = scipy.sparse.csr_matrix(self.TestData[l_test_columns])
            x_test = self.TestData[l_test_columns]

            for idx in range(0, len(x_test), N):
                x_test_block = x_test[idx:idx + N]
                x_test_block = scipy.sparse.csr_matrix(x_test_block)

                ret = self._model.predict(x_test_block)
                self._sub.loc[x_test[idx:idx + N].index, d] = ret
                print(np.mean(np.abs(ret)))

            e0 = time.time()
            print('Prediction for column %s is done. time elapsed %ds' % (d, (e0 - s0)))

        ## clean
        del self.TestData
        gc.collect()

        end = time.time()
        print('Prediction is done. time elapsed %ds' % (end - start))

        if(os.path.exists(self.OutputDir) == False):
            os.makedirs(self.OutputDir)

        self._sub.to_csv('{0}/{1}_{2}.csv'.format(self.OutputDir,
                                                  self.__class__.__name__,
                                                  datetime.now().strftime('%Y%m%d-%H:%M:%S')),
                        index=False, float_format='%.4f')
