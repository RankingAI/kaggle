from feat.FeatureEngineering import Preprocessing
from model.SingleModel import SingleModel
from model.EnsembleModel import EnsembleModel
from feat.CountFeature import CountFeature
from feat.CensusFeature import CensusFeature
import pandas as pd
import numpy as np
import dill as pickle
import os

if __name__ == '__main__':

    DataDir = '/Users/yuanpingzhou/project/workspace/python/kaggle/Zillow/data'

    ## merge
    # OutputDir = '%s/p00' % DataDir
    # train = pd.read_csv('%s/train_2016_v2.csv' % DataDir, parse_dates=['transactiondate'])
    # test = pd.read_csv('%s/sample_submission.csv' % DataDir)
    # prop = pd.read_csv('%s/properties_2016.csv' % DataDir)
    # df_train = train.merge(prop, how='left', on='parcelid')
    # tmp_test = pd.DataFrame(index= range(len(test)))
    # tmp_test['parcelid'] = test['ParcelId']
    # df_test = tmp_test.merge(prop, how='left', on='parcelid')
    # print(df_test.head(50))
    # if (os.path.exists(OutputDir) == False):
    #     os.makedirs(OutputDir)
    # with open('%s/train.pkl' % OutputDir, 'wb') as o_file:
    #     pickle.dump(df_train, o_file, -1)
    # o_file.close()
    # with open('%s/test.pkl' % OutputDir, 'wb') as o_file:
    #     pickle.dump(df_test, o_file, -1)
    # o_file.close()

    ## generate count features
    #CountFeature.GenerateCountFeature('%s/properties_2016.csv' % DataDir, '%s/CountFeat.pkl' % DataDir)

    ## generate census features
    #CensusFeature.GenerateCensusFeature('%s/properties_2016.csv' % DataDir, '%s/CensusFeat.pkl' % DataDir)

    ## process
    # InputDir = '%s/p00' % DataDir
    # OutputDir = '%s/p3' % DataDir ## add null_count feature
    # tasks = ['NewFeature','FeatureSelection','FeatureEncoding','MissingValue']
    # process = Preprocessing(InputDir,OutputDir,Mode= 'pkl')
    # process.run(tasks)

    # ## single model
    # InputDir = '%s/p2' % DataDir
    # OutputDir = '%s/SingleModel2' % DataDir
    # strategies = ['lgb']
    # SingleModel.run(strategies,InputDir,OutputDir)

    ## ensemble model
    InputDir = '%s/SingleModel2' % DataDir
    OutputDir = '%s/EnsembleModel2' % DataDir

    ## evaluation ensemble model
    em = EnsembleModel('%s/p2' % DataDir,OutputDir)
    #em.EvaluateEnsembleModel(InputDir)
    # predict test data with ensemble model
    em.SimpleEnsemble(InputDir,OutputDir)
