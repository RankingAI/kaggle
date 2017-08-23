import pandas as pd
import os
import dill as pickle
import numpy as np
import time

class Preprocess:
    """"""
    CategoryCols = ['rawcensustractandblock', 'hashottuborspa', 'taxdelinquencyflag', 'airconditioningtypeid', 'architecturalstyletypeid',
                    'buildingqualitytypeid', 'decktypeid', 'fips', 'heatingorsystemtypeid', 'pooltypeid10',
                    'pooltypeid2', 'pooltypeid7',
                    'propertylandusetypeid', 'rawcensustractandblock', 'regionidcity', 'regionidcounty',
                    'regionidneighborhood', 'regionidzip']

    @classmethod
    def __MergeData(cls, InputDir, OutputDir, mode):
        """"""
        if(mode == 'train'):
            ActionDataFile = '%s/train_2016_v2.csv' % InputDir
            OutputFile = '%s/train.pkl' % OutputDir
        else:
            ActionDataFile = '%s/sample_submission.csv' % InputDir
            OutputFile = '%s/test.pkl' % OutputDir

        print(OutputFile)

        PropertyDataFile = '%s/properties_2016.csv' % InputDir

        ## load
        ActionData = pd.read_csv(ActionDataFile, parse_dates=['transactiondate'])
        PropertyData = pd.read_csv(PropertyDataFile)

        ## left join
        MergedData = ActionData.merge(PropertyData, how='left', on='parcelid')

        ## output into pkl file
        if (os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)
        with open(OutputFile, 'wb') as o_file:
            pickle.dump(MergedData, o_file, -1)
        o_file.close()

        return

    ## split rawcensustractandblock into census, tract and block
    @classmethod
    def __ParseCTB(cls, InputDir, OutputDir, mode):
        """"""
        if(mode == 'train'):
            InputFile = '%s/train.pkl' % InputDir
            OutputFile = '%s/train.pkl' % OutputDir
        else:
            InputFile = '%s/test.pkl' % InputDir
            OutputFile = '%s/test.pkl' % OutputDir

        ## load
        with open(InputFile, 'rb') as i_file:
            df_data = pickle.load(i_file)
        i_file.close()

        ## extract census, tract and block identifies
        df_data['rawcensustractandblock'] = (df_data['rawcensustractandblock'] * 1000).astype(np.float64).astype(np.int64)
        df_data['fipsid'] = ((df_data['rawcensustractandblock'] / 10000000).astype(np.int64)).astype(str)
        df_data['tractandblock'] = df_data['rawcensustractandblock'] % 10000000
        df_data['tractid'] = ((df_data['tractandblock'] / 10).astype(np.int64)).astype(str)
        df_data['blockid'] = ((df_data['tractandblock'] % 10).astype(np.int64)).astype(str)
        df_data.drop(['fips', 'rawcensustractandblock', 'tractandblock'], axis = 1, inplace = True)

        ## output into pkl file
        if (os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)
        with open(OutputFile, 'wb') as o_file:
            pickle.dump(df_data, o_file, -1)
        o_file.close()

        return

    @classmethod
    def __CollectCategories(cls, InputDir, OutputDir, mode):
        """"""
        if(mode == 'train'):
            InputFile = '%s/train.pkl' % InputDir
        else:
            InputFile = '%s/test.pkl' % InputDir

        OutputFileFeatMap = '%s/featmap.pkl' % OutputDir
        OutputFileData = '%s/train.pkl' % OutputDir

        ## load
        with open(InputFile, 'rb') as i_file:
            df_data = pickle.load(i_file)
        i_file.close()

        ## update category columns
        cls.CategoryCols = [col for col in cls.CategoryCols if(col not in ['rawcensustractandblock', 'fips'])]
        cls.CategoryCols.extend(['fipsid', 'tractid', 'blockid'])

        ## collect value counts for each of categories
        d_values = {}
        for col in cls.CategoryCols:
            dt = df_data[col].dtype.name
            if(dt != 'object'): ## not string
                df_data[col] = df_data[col].astype(np.float64).astype(str)

            ValueCounts = df_data[col].value_counts()
            NullValueCounts = df_data[col].isnull().value_counts()
            if(True in NullValueCounts.index.values):
                NullCount = NullValueCounts[True]
            else:
                NullCount = 0
            for v in ValueCounts.index.values:
                d_values['%s:%s' % (col, v)] = ValueCounts[v]
            d_values['%s:%s' % (col, 'missing')] = NullCount
            ## check
            print('column %s null count %d' % (col, NullCount))

        ## output into pkl file
        if(os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)
        with open(OutputFileFeatMap, 'wb') as o_file:
            pickle.dump(d_values, o_file, -1)
        o_file.close()
        with open(OutputFileData, 'wb') as o_file:
            pickle.dump(df_data, o_file, -1)
        o_file.close()

        return

    ## split data with month
    @classmethod
    def __SplitData(cls, InputDir, OutputDir, mode):
        """"""
        if(mode == 'train'):
            InputFileData = '%s/train.pkl' % InputDir
        else:
            InputFileData = '%s/test.pkl' % InputDir

        InputFileFeatMap = '%s/featmap.pkl' % InputDir

        ## load
        with open(InputFileData, 'rb') as i_file:
            df_data = pickle.load(i_file)
        i_file.close()
        with open(InputFileFeatMap, 'rb') as i_file:
            d_feat = pickle.load(i_file)
        i_file.close()

        if (os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)
        with open('%s/featmap.pkl' % OutputDir, 'wb') as o_file:
            pickle.dump(d_feat, o_file, -1)
        o_file.close()

        ## output into individual pkl files
        for i in range(12):
            month = i + 1
            df_MonthData = df_data[(df_data['transactiondate'].dt.month == month)]
            with open('%s/%s.pkl'% (OutputDir, month), 'wb') as o_file:
                pickle.dump(df_MonthData, o_file, -1)
            o_file.close()

        return

    ## launch single task
    @classmethod
    def __LaunchTask(cls, task, mode, InputDir, OutputDir):
        """"""
        print('----- Task %s begins ...' % task)
        start = time.time()

        if(task == 'MergeData'):
            cls.__MergeData(InputDir, OutputDir, mode)
        elif(task == 'ParseCTB'):
            cls.__ParseCTB(InputDir, OutputDir, mode)
        elif(task == 'CollectCategories'):
            cls.__CollectCategories(InputDir, OutputDir, mode)
        elif(task == 'SplitData'):
            cls.__SplitData(InputDir, OutputDir, mode)

        end = time.time()
        print('----- Task %s done, time consumed %ds' % (task, (end - start)))

        return

    ## run all tasks one-by-one
    @classmethod
    def run(cls, tasks, mode, Input, Output):

        start = time.time()
        if (os.path.exists(Output) == False):
            os.makedirs(Output)

        for i in range(len(tasks)):
            task = tasks[i]
            if(i == 0):
                InputDir = Input
            else:
                InputDir = '%s/%s' % (Output, tasks[i - 1])
            OutputDir = '%s/%s' % (Output, tasks[i])

            cls.__LaunchTask(task, mode, InputDir, OutputDir)

        end = time.time()
        print('\nAll tasks done, time consumed %ds' % (end - start))

        return
