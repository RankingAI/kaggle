import pandas as pd
import numpy as np
import abc
import time
import os
from datetime import datetime
import dill as pickle

from util.DataIO import DataIO

class ModelBase(object):

    __metaclass__  = abc.ABCMeta

    TrainData = pd.DataFrame()
    ValidData = pd.DataFrame()
    TestData = pd.DataFrame()

    InputDir = ''
    OutputDir = ''

    l_test_predict_columns = []

    ## outliers
    MinLogError = -0.4
    MaxLogError = 0.418

    def __init__(self, PredCols, InputDir,OutputDir):

        self.l_test_predict_columns = PredCols

        self.InputDir = InputDir
        self.OutputDir = OutputDir

        if(os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)

        with open('%s/train.pkl' % self.InputDir, 'rb') as i_file:
            self.TrainData = pickle.load(i_file)
        i_file.close()
        with open('%s/test.pkl' % self.InputDir, 'rb') as i_file:
            self.TestData = pickle.load(i_file)
        i_file.close()

    @abc.abstractmethod
    def __fit(self):
        """"""

    @abc.abstractmethod
    def __predict(self):
        """"""

    @abc.abstractmethod
    def train(self):
        """"""

    @abc.abstractmethod
    def submit(self):
        """"""
