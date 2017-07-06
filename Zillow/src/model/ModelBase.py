import pandas as pd
import numpy as np
import abc
import time
import os
from datetime import datetime

from utils.DataIO import DataIO

class ModelBase(object):

    __metaclass__  = abc.ABCMeta

    TrainData = pd.DataFrame()
    TestData = pd.DataFrame()

    InputDir = ''
    OutputDir = ''

    _data = DataIO
    _l_train_columns = []
    _model = ''
    _sub = pd.DataFrame()
    _l_test_predict_columns = ['201610', '201611', '201612', '201710', '201711', '201712']

    def __init__(self,InputDir,OutputDir):

        self.InputDir = InputDir
        self.OutputDir = OutputDir

        #self.TrainData, self.TestData = self._data.LoadFromPklFile(self.InputDir)
        self.TrainData = self._data.LoadFromHdfFile(self.InputDir)

    def submmit(self):
        if (os.path.exists(self.OutputDir) == False):
            os.makedirs(self.OutputDir)

        # m = np.mean(self._sub[self._l_test_predict_columns].mean())
        # with open('%s/eval.log' % self.OutputDir, 'a+') as o_file:
        #     o_file.write('%.4f\n' % m)
        # o_file.close()
        self._sub.to_csv('{0}/{1}_{2}.csv'.format(self.OutputDir, self.__class__.__name__,datetime.now().strftime('%Y%m%d-%H:%M:%S')),
                         index=False, float_format='%.4f')

    @abc.abstractmethod
    def evaluate(self):
        """"""

    @abc.abstractmethod
    def predict(self):
        """"""

    @abc.abstractmethod
    def train(self):
        """"""

    @abc.abstractmethod
    def save(self):
        """"""
