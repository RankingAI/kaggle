import time
from model.LightGBM import LGB
from model.XGBoost import XGB

class SingleModel:

    @staticmethod
    def __LaunchTraining(task,InputDir,OutputDir):

        d_model = {'lgb': LGB,
                   'xgb': XGB
                   }

        start = time.time()

        model = d_model[task](InputDir,OutputDir)

        print('Training begins ...')
        model.train()
        print('Prediction begins ...')
        model.predict()
        print('Submmit begins ...')
        model.submmit()

        end = time.time()
        print('%s done, time elapsed %ds' % (task,(end - start)))

    @classmethod
    def run(cls,strategies,InputDir,OutputDir):

        start = time.time()
        for s in strategies:
            cls.__LaunchTraining(s,InputDir,OutputDir)
        end = time.time()
        print('\nAll tasks done, time consumed %ds' % (end - start))