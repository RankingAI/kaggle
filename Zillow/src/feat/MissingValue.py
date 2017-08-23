class MissingValue:

    @classmethod
    def impute(cls,data):

        df_train,  df_test = data

        df_dict = {
            'df_train': df_train,
            'df_test': df_test
        }

        for name in df_dict:
            df_dict[name].fillna(-1, inplace = True)

        return (df_train, df_test)
