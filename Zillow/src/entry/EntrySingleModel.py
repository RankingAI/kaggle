from model.SingleModel import SingleModel

if __name__ == '__main__':
    """"""
    InputDir = '/Users/yuanpingzhou/project/workspace/python/kaggle/Zillow-dev/data/feat/featureengineering/single'
    OutputDir = '/Users/yuanpingzhou/project/workspace/python/kaggle/Zillow-dev/data/SingleModel'

    strategies = ['lgb']
    SingleModel.run(strategies, 3, InputDir, OutputDir)