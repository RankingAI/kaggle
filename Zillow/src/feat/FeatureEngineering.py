import pandas as pd
import numpy as np

import time
from feat.MissingValue import MissingValue
from feat.NewFeature import NewFeature
from feat.FeatureEncoding import FeatureEncoding
from feat.FeatureSelection import FeatureSelection
from utils.DataIO import DataIO

class Preprocessing:

    ## private member variables
    _InputDir = ''
    _OutputDir = ''

    ## public member variables
    TrainData = pd.DataFrame()
    ValidData = pd.DataFrame()
    TestData = pd.DataFrame()

    ## composite classes
    _data = DataIO
    _newfeat = NewFeature
    _missing = MissingValue
    _encoding = FeatureEncoding
    _select = FeatureSelection

    ## constructor,
    def __init__(self,InputDir,Outputdir,Mode = 'text'):

        self._InputDir = InputDir
        self._OutputDir = Outputdir

        if(Mode == 'pkl'):
            self.TrainData, self.TestData = self._data.LoadFromPklFile(InputDir)
        else:
            self.TrainData, self.TestData = self._data.LoadFromTextFile(InputDir)

        self.TrainData = self.TrainData.reset_index() ## additional columns 'index' will be newed

        df_tmp = self.TrainData[self.TrainData['transactiondate'].dt.month > 6]
        np.random.seed(2017)
        msk = np.random.rand(len(df_tmp)) < 0.25
        self.ValidData = df_tmp[msk]
        self.TrainData = pd.concat([self.TrainData[self.TrainData['transactiondate'].dt.month <= 6],df_tmp[~msk]],ignore_index= True)
        #self.TrainData = self.TrainData.reset_index(drop = True)
        #self.TestData = self.TestData.sample(frac = 1.00)

    ## launch one task
    def __LaunchTask(self,task):

        start = time.time()
        print('\n============== Begin to deal with %s' % (task))

        if (task == 'MissingValue'):
            self.TrainData,self.ValidData,self.TestData = self._missing.impute((self.TrainData,self.ValidData,self.TestData))
        elif (task == 'NewFeature'):
            self.TrainData,self.ValidData,self.TestData = self._newfeat.create((self.TrainData,self.ValidData,self.TestData))
        elif (task == 'FeatureEncoding'):
            self.TrainData,self.ValidData,self.TestData = self._encoding.ordinal((self.TrainData,self.ValidData,self.TestData))
        elif (task == 'FeatureSelection'):
            self.TrainData,self.ValidData,self.TestData = self._select.select((self.TrainData,self.ValidData,self.TestData))

        end = time.time()
        print('============= Task %s done, time consumed %ds' % (task, (end - start)))

    ## run all tasks one-by-one
    def run(self,tasks):

        start = time.time()
        for task in tasks:
            self.__LaunchTask(task)
        DataIO.SaveToHdfFile((self.TrainData,self.ValidData,self.TestData),self._OutputDir)
        end = time.time()
        print('\nAll tasks done, time consumed %ds' % (end - start))
