from feat.Preprocess import Preprocess
from feat.FeatureEngineering import FeatureEngineering
import gc

if __name__ == '__main__':

    kfold = 3

    DataDir = '/Users/yuanpingzhou/project/workspace/python/kaggle/Zillow-dev/data'

    ## preprocess, something to be done before feature engineering
    OutputDir = '%s/feat/Preprocess' % DataDir
    tasks = ['MergeData', 'ParseCTB', 'CollectCategories', 'SplitData']
    Preprocess.run(tasks, 'train', DataDir, OutputDir)

    ## feature engineering
    tasks = ['NewFeature', 'FeatureSelection', 'FeatureEncoding','MissingValue']

    ########## for ensemble model #######
    #### tasks for l2 test
    print('\n==== Tasks for l2 test ...')
    InputDir = '%s/feat/Preprocess' % DataDir
    OutputDir = '%s/feat/FeatureEngineering/ensemble/test' % DataDir
    MonthsOfTest = [10,11,12]
    fe = FeatureEngineering('%s/SplitData' % InputDir, OutputDir)
    fe.run(tasks, MonthsOfTest)
    del fe
    gc.collect()
    print('\n==== Tasks for l2 test done.')

    # #### tasks for l2 valid
    # print('\n==== Tasks for l2 valid ...')
    # InputDir = '%s/feat/Preprocess' % DataDir
    # OutputDir = '%s/feat/FeatureEngineering/ensemble/valid' % DataDir
    # MonthsOfTest = [9]
    # fe = FeatureEngineering('%s/SplitData' % InputDir, OutputDir)
    # fe.run(tasks, MonthsOfTest)
    # del fe
    # gc.collect()
    # print('\n==== Tasks for l2 valid done.')

    #### tasks for l2 train on the data set l2 valid used
    print('\n==== Tasks for l2 train ...')
    InputDir = '%s/feat/Preprocess' % DataDir
    OutputDir = '%s/feat/FeatureEngineering/ensemble/train' % DataDir
    MonthsOfTest = [9]
    fe = FeatureEngineering('%s/SplitData' % InputDir, OutputDir)
    fe.run(tasks, MonthsOfTest)
    del fe
    gc.collect()
    print('\n==== Tasks for l2 train done.')

    #### tasks for l1 test
    print('\n==== Tasks for l1 test ...')
    InputDir = '%s/feat/Preprocess' % DataDir
    OutputDir = '%s/feat/FeatureEngineering/single/test' % DataDir
    MonthsOfTest = [8]
    fe = FeatureEngineering('%s/SplitData' % InputDir, OutputDir)
    fe.run(tasks, MonthsOfTest)
    del fe
    gc.collect()
    print('\n==== Tasks for l1 test done.')

    ######### for single model ########
    ## tasks for l1 train, kfold mode
    for fold in range(kfold):
        print('\n==== Tasks for l1 train, fold %d ...' % fold)
        InputDir = '%s/feat/Preprocess' % DataDir
        OutputDir = '%s/feat/FeatureEngineering/single/train/%s' % (DataDir, fold)
        MonthsOfTest = [7 - (kfold - fold) + 1]
        fe = FeatureEngineering('%s/SplitData' % InputDir, OutputDir)
        fe.run(tasks, MonthsOfTest)
        print('\n==== Tasks for l1 train, fold %d done.' % fold)
