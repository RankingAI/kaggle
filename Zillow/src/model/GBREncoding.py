from sklearn.ensemble import GradientBoostingRegressor
from model.ModelBase import ModelBase
import pandas as pd
import time,os
import numba
import math
import numpy as np
import gc

class GBRE(ModelBase):
    """"""
    _l_drop_cols = ['logerror', 'parcelid', 'transactiondate', 'index', 'nullcount']

    _encode_dict = {}

    _slice = 10

    _gbr_iter = 50
    _gbr_learning_rate = 0.4
    _gbr_depth = 10
    _gbr_loss = 'lad'

    @numba.jit
    def __ApplyOH(self, ColumnValues, headers):

        n = len(ColumnValues)
        result = np.zeros((n, len(headers)), dtype='int8')
        for i in range(n):
            v = ColumnValues[i]
            if (str(int(v)) in headers):
                result[i, headers[str(int(v))]] = 1
            else:
                result[i, headers['missing']] = 1

        return result

    def OHEVa(self, OHEDict, va):
        """"""
        for col in OHEDict:
            OHVa = self.__ApplyOH(va[col].values, OHEDict[col])

            headers = dict((('%s_%s' % (col, k)), OHEDict[col][k]) for k in OHEDict[col])
            tmp = [v[0] for v in sorted(headers.items(), key=lambda x: x[1])]
            OHDFVa = pd.DataFrame(OHVa, index=va.index, columns=tmp)

            va = pd.concat([va, OHDFVa], axis=1)

            va.drop(col, axis=1, inplace=True)
            #print('Column %s was encoded.' % col)

        return va

    def OHETr(self, tr):
        """"""
        OHEDict = {}
        for col in tr.columns:
            ValueCounts = [str(int(v)) for v in tr[col].value_counts().index.values]
            ValueCounts.append('missing')
            SelectedValues = dict((k, v) for (v, k) in enumerate(ValueCounts, start=0))
            OHTr = self.__ApplyOH(tr[col].values, SelectedValues)

            headers = dict((('%s_%s' % (col, k)), SelectedValues[k]) for k in SelectedValues)
            tmp = [v[0] for v in sorted(headers.items(), key=lambda x: x[1])]
            OHDFTr = pd.DataFrame(OHTr, index=tr.index, columns=tmp)

            tr = pd.concat([tr, OHDFTr], axis=1)

            tr.drop(col, axis=1, inplace=True)
            OHEDict[col] = SelectedValues
            #print('Column %s was encoded.' % col)

        return tr, OHEDict

    def train(self):
        """"""
        print('size before truncated outliers is %d ' % len(self.TrainData))
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print('size after truncated outliers is %d ' % len(self.TrainData))

        X_Train = self.TrainData.drop(self._l_drop_cols, axis=1)
        Y_Train = self.TrainData['logerror']
        l_train_columns = X_Train.columns

        FeatCols = list(X_Train.columns)
        gbr = GradientBoostingRegressor(n_estimators=self._gbr_iter,
                                        learning_rate=self._gbr_learning_rate,
                                        max_depth=self._gbr_depth,
                                        random_state=2017,
                                        subsample=0.8,
                                        loss=self._gbr_loss,
                                        max_features=int(math.sqrt(len(FeatCols))),
                                        verbose=True)
        self._model = gbr.fit(X_Train, Y_Train)

        cols = ['tree%d' % i for i in range(self._slice)]

        IndiceTrain = self._model.apply(X_Train)[..., :self._slice]
        TransformedTrain = pd.DataFrame(data= IndiceTrain, index= X_Train.index, columns= cols)
        EncodedTrain, EncodeDict = self.OHETr(TransformedTrain[cols])
        EncodedTrain['parcelid'] = self.TrainData['parcelid']

        if(os.path.exists(self.OutputDir) == False):
            os.makedirs(self.OutputDir)
        EncodedTrain.to_hdf(path_or_buf='%s/eval_train.hdf' % self.OutputDir,key='train',mode = 'w',complib='blosc')

        del TransformedTrain, IndiceTrain, X_Train, Y_Train, EncodedTrain
        gc.collect()

        print('For valid data...')
        for d in self._l_valid_predict_columns:
            l_valid_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in l_train_columns]
            XValid = self.ValidData[l_valid_columns]
            IndiceValid = self._model.apply(XValid)[..., :self._slice]
            TransformedValid = pd.DataFrame(data= IndiceValid, index= XValid.index, columns= cols)
            EncodedValid = self.OHEVa(EncodeDict, TransformedValid[cols])
            EncodedValid['parcelid'] = self.ValidData['parcelid']
            EncodedValid.to_hdf(path_or_buf='%s/eval_valid_%s.hdf' % (self.OutputDir, d), key='valid', mode='w', complib='blosc')
            print('column %s done.' % d)

        ### for test
        self.TrainData = pd.concat([self.TrainData, self.ValidData[self.TrainData.columns]],ignore_index=True)  ## ignore_index will reset the index or index will be overlaped
        print('size before truncated outliers is %d ' % len(self.TrainData))
        self.TrainData = self.TrainData[(self.TrainData['logerror'] > self._low) & (self.TrainData['logerror'] < self._up)]
        print('size after truncated outliers is %d ' % len(self.TrainData))

        X_Train = self.TrainData.drop(self._l_drop_cols, axis=1)
        Y_Train = self.TrainData['logerror']
        self._model = gbr.fit(X_Train,Y_Train)

        IndiceTrain = self._model.apply(X_Train)[..., :self._slice]
        TransformedTrain = pd.DataFrame(data= IndiceTrain, index= X_Train.index, columns= cols)
        EncodedTrain, EncodeDict = self.OHETr(TransformedTrain[cols])
        EncodedTrain['parcelid'] = self.TrainData['parcelid']

        EncodedTrain.to_hdf(path_or_buf='%s/train.hdf' % self.OutputDir,key='train',mode = 'w',complib='blosc')
        del TransformedTrain, EncodedValid, IndiceTrain, X_Train, Y_Train, self.ValidData, self.TrainData
        gc.collect()

        N = 200000
        print('For test data...')
        TestData = pd.read_hdf(path_or_buf= '%s/test.hdf' % self.InputDir, key='test')
        #TestData = TestData.sample(frac = 0.01)

        for d in self._l_test_predict_columns:
            s0 = time.time()

            l_test_columns = ['%s%s' % (c, d) if (c in ['lastgap', 'monthyear', 'buildingage']) else c for c in l_train_columns]
            XTest = TestData[l_test_columns]

            TransformedTest = pd.DataFrame(index= XTest.index, columns= cols)
            #TransformedTest['parcelid'] = TestData['parcelid']

            for idx in range(0, len(XTest), N):
                XTestBlock = XTest[idx:idx + N]
                IndiceTest = self._model.apply(XTestBlock)[..., :self._slice]
                TransformedTest.loc[XTestBlock.index, cols] = IndiceTest
                print('------- %d' % idx)

            s1 = time.time()
            EncodedTest = self.OHEVa(EncodeDict, TransformedTest[cols])
            EncodedTest['parcelid'] = TestData['parcelid']
            e1 = time.time()
            print('Encoding done, time consumed %ds' % (e1 - s1))

            EncodedTest.to_hdf(path_or_buf='%s/test_%s.hdf' % (self.OutputDir, d), key='test', mode='w', complib='blosc')
            e0 = time.time()

            print('column %s done, time consumed %ds.' % (d, e0-s0))
