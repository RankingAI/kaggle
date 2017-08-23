import dill as pickle
import pandas as pd
import time
import os

from feat.NewFeature import NewFeature
from feat.FeatureSelection import FeatureSelection
from feat.MissingValue import MissingValue
from feat.FeatureEncoding import FeatureEncoding

class FeatureEngineering:
    """"""
    _kfold = 0

    TrainData = pd.DataFrame()
    TestData = pd.DataFrame()

    _InputDir = ''
    _OutputDir = ''

    ## composite classes
    _newfeat = NewFeature
    _missing = MissingValue
    _encoding = FeatureEncoding
    _select = FeatureSelection

    def __init__(self, InputDir, OutputDir, kfold = 4):
        """"""
        self._kfold = kfold
        self._InputDir = InputDir
        self._OutputDir = OutputDir

    def __LaunchTask(self, task, PredCols):
        """"""
        print('\n---- Begin to deal with %s' % (task))
        start = time.time()

        if(task == 'MissingValue'):
            self.TrainData, self.TestData = self._missing.impute((self.TrainData, self.TestData))
        elif(task == 'NewFeature'):
            self.TrainData, self.TestData = self._newfeat.create((self.TrainData, self.TestData), PredCols)
        elif(task == 'FeatureEncoding'):
            with open('%s/featmap.pkl' % self._InputDir, 'rb') as i_file:
                d_feat = pickle.load(i_file)
            i_file.close()
            self.TrainData, self.TestData = self._encoding.ordinal((self.TrainData, self.TestData), d_feat)
        elif(task == 'FeatureSelection'):
            self.TrainData, self.TestData = self._select.select((self.TrainData, self.TestData))

        end = time.time()
        print('---- Task %s done, time consumed %ds' % (task, (end - start)))

    def run(self, tasks, MonthsOfTest):
        """"""
        print('\nLoad data ...')
        start = time.time()
        ## load train
        with open('%s/1.pkl' % self._InputDir, 'rb') as i_file:
            self.TrainData = pickle.load(i_file)
        i_file.close()
        for i in range(2,MonthsOfTest[0]):
            with open('%s/%s.pkl' % (self._InputDir, i), 'rb') as i_file:
                df_tmp = pickle.load(i_file)
                self.TrainData = pd.concat([self.TrainData, df_tmp], ignore_index = True)
            i_file.close()
        ## load test
        with open('%s/%s.pkl' % (self._InputDir, MonthsOfTest[0]), 'rb') as i_file:
            self.TestData = pickle.load(i_file)
        i_file.close()
        for i in MonthsOfTest[1: ]:
            with open('%s/%s.pkl' % (self._InputDir, i), 'rb') as i_file:
                df_tmp = pickle.load(i_file)
                self.TestData = pd.concat([self.TestData, df_tmp], ignore_index = True)
            i_file.close()
        end = time.time()
        print('Load data done, time consumed %ds ...' % (end - start))

        ## tasks for l2 test
        print('\nLaunch task ...')
        start = time.time()
        for task in tasks:
            self.__LaunchTask(task, MonthsOfTest)
        end = time.time()
        if (os.path.exists(self._OutputDir) == False):
            os.makedirs(self._OutputDir)
        with open('%s/train.pkl' % self._OutputDir, 'wb') as o_file:
            pickle.dump(self.TrainData, o_file, -1)
        o_file.close()
        with open('%s/test.pkl' % self._OutputDir, 'wb') as o_file:
            pickle.dump(self.TestData, o_file, -1)
        o_file.close()
        print('All tasks done, time consumed %ds ...' % (end - start))
