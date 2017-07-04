from feat.FeatureEngineering import Preprocessing
from model.SingleModel import SingleModel

if __name__ == '__main__':

    DataDir = '/Users/yuanpingzhou/project/workspace/python/kaggle/Zillow/data'

    ## merge
    # InputDir = DataDir
    # OutputDir = '%s/p0' % (DataDir)

    ## p0
    InputDir = '%s/p0' % DataDir
    OutputDir = '%s/p1' % DataDir
    #
    tasks = ['NewFeature','FeatureSelection','FeatureEncoding','MissingValue']
    #tasks = []
    process = Preprocessing(InputDir,OutputDir,Mode= 'pkl')
    process.run(tasks)

    ## single model
    # InputDir = '%s/p1' % DataDir
    # OutputDir = '%s/SingleModel' % DataDir
    #
    # strategies = ['lgb']
    # SingleModel.run(strategies,InputDir,OutputDir)
