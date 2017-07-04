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

        self.TestData = self.TestData.sample(frac = 1.0)

    ## launch one task
    def __LaunchTask(self,task):

        start = time.time()
        print('\n============== Begin to deal with %s' % (task))

        if (task == 'MissingValue'):
            (self.TrainData,self.TestData) = self._missing.impute((self.TrainData,self.TestData))
        elif (task == 'NewFeature'):
            self.TrainData,self.TestData = self._newfeat.create((self.TrainData,self.TestData))
        elif (task == 'FeatureEncoding'):
            self.TrainData,self.TestData = self._encoding.ordinal((self.TrainData,self.TestData))
        elif (task == 'FeatureSelection'):
            self.TrainData,self.TestData = self._select.select((self.TrainData,self.TestData))

        end = time.time()
        print('============= Task %s done, time consumed %ds' % (task, (end - start)))

    ## run all tasks one-by-one
    def run(self,tasks):

        start = time.time()
        for task in tasks:
            self.__LaunchTask(task)
        #d = self.TestData
        #print(d[(d['lastgap201610'] > 0) | (d['lastgap201611'] > 0)].head())
        #print(d[[col for col in d.columns if 'monthyear' in col]].head())
        #print(self.TrainData.columns)
        #print('------------------------------')
        #print(self.TestData.columns)
        # print(self.TrainData.dtypes.value_counts())
        # for col,dt in zip(self.TrainData.columns,self.TrainData.dtypes):
        #     if(dt != np.int32):
        #         print(col,dt)
        DataIO.SaveToHdfFile((self.TrainData,self.TestData),self._OutputDir)
        end = time.time()
        print('\nAll tasks done, time consumed %ds' % (end - start))
