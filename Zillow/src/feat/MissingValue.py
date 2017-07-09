import numpy as np

class MissingValue:

    @classmethod
    def impute(cls,data):

        df_train, df_valid, df_test = data

        df_dict = {
            'df_train': df_train,
            'df_valid': df_valid,
            'df_test': df_test
        }

        ## checking
        # print('Checking begins ...')
        # for name in df_dict:
        #     df = df_dict[name]
        #     for nc in ['lastgap','monthyear','buildingage']:
        #         for c in [col for col in df.columns if nc in col]:
        #             print(name,c,df[c].isnull().sum())

        for name in df_dict:
            df_dict[name].fillna(-1, inplace = True)

        return (df_train,df_valid,df_test)
