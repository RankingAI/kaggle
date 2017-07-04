import pandas as pd
import abc

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
    _sub = ''
    _l_test_predict_columns = ['201610', '201611', '201612', '201710', '201711', '201712']

    def __init__(self,InputDir,OutputDir):

        self.InputDir = InputDir
        self.OutputDir = OutputDir

        self.TrainData, self.TestData = self._data.LoadFromPklFile(self.InputDir)

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
