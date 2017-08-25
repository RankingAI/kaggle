from model.EnsembleModel import EnsembleModel

if __name__ == '__main__':
    """"""
    InputDir = '/Users/yuanpingzhou/project/workspace/python/kaggle/Zillow-dev/data/SingleModel'
    OutputDir = '/Users/yuanpingzhou/project/workspace/python/kaggle/Zillow-dev/data/EnsembleModel'
    PredCols = [10, 11, 12]

    en = EnsembleModel(InputDir, OutputDir, PredCols)
    en.evaluate()
