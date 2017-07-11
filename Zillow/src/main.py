from feat.FeatureEngineering import Preprocessing
from model.SingleModel import SingleModel
from model.EnsembleModel import EnsembleModel

if __name__ == '__main__':

    DataDir = '/Users/yuanpingzhou/project/workspace/python/kaggle/Zillow/data'

    ## merge
    # InputDir = DataDir
    # OutputDir = '%s/p0' % (DataDir)

    ## process
    # InputDir = '%s/p0' % DataDir
    # OutputDir = '%s/p1' % DataDir
    # tasks = ['NewFeature','FeatureSelection','FeatureEncoding','MissingValue']
    # process = Preprocessing(InputDir,OutputDir,Mode= 'pkl')
    # process.run(tasks)

    ## single model
    # InputDir = '%s/p1' % DataDir
    # OutputDir = '%s/SingleModel' % DataDir
    # strategies = ['rf']
    # SingleModel.run(strategies,InputDir,OutputDir)

    ## ensemble model
    InputDir = '%s/SingleModel' % DataDir
    OutputDir = '%s/EnsembleModel' % DataDir

    ## evaluation ensemble model
    #em = EnsembleModel('%s/p1' % DataDir,OutputDir)
    #em.EvaluateEnsembleModel(InputDir)
    ## predict test data with ensemble model
    EnsembleModel.SimpleEnsemble(InputDir,OutputDir)
