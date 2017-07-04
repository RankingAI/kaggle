import time
from model.LightGBM import LGB

class SingleModel:

    @staticmethod
    def __LaunchTraining(task,InputDir,OutputDir):

        start = time.time()
        if(task == 'lgb'):
            lgb = LGB(InputDir,OutputDir)
            lgb.train()
            #lgb.predict()
            #lgb.submmit()
        end = time.time()
        print('%s done, time elapsed %ds' % (task,(end - start)))

    @classmethod
    def run(cls,strategies,InputDir,OutputDir):

        start = time.time()
        for s in strategies:
            cls.__LaunchTraining(s,InputDir,OutputDir)
        end = time.time()
        print('\nAll tasks done, time consumed %ds' % (end - start))