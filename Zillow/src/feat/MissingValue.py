import numpy as np

class MissingValue:

    @classmethod
    def impute(cls,data):

        df_train, df_test = data
        #
        # for c, dtype in zip(df_train.columns, df_train.dtypes):
        #     if((dtype == np.float64) | (dtype == np.int32) | (dtype == np.int64)):
        #         df_train[c] = df_train[c].astype(np.float32)

        df_train.fillna(-1,inplace = True)
        df_test.fillna(-1,inplace = True)

        return (df_train,df_test)
