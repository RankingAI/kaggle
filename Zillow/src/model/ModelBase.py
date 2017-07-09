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
    ValidData = pd.DataFrame()
    TestData = pd.DataFrame()

    InputDir = ''
    OutputDir = ''
    _f_eval_train_model = ''

    _data = DataIO
    _l_train_columns = []
    _model = ''
    _sub = pd.DataFrame()

    _l_test_predict_columns = ['201610', '201611', '201612', '201710', '201711', '201712']
    _l_valid_predict_columns = ['201607', '201608', '201609', '201610', '201611', '201612']

    _low = -0.4
    _up = 0.418

    def __init__(self,InputDir,OutputDir):

        self.InputDir = InputDir
        self.OutputDir = OutputDir

        self.TrainData = self._data.LoadFromHdfFile(self.InputDir,'train')
        self.ValidData = self._data.LoadFromHdfFile(self.InputDir,'valid')

    @abc.abstractmethod
    def evaluate(self):
        """"""

    @abc.abstractmethod
    def train(self):
        """"""

    @abc.abstractmethod
    def submit(self):
        """"""
