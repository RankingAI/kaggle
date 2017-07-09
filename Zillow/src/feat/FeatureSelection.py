class FeatureSelection:

    @classmethod
    def select(cls,data):

        df_train, df_valid, df_test = data
        ## drop certain columns, useless and the nearly null ones
        l_drop_columns = ['propertyzoningdesc','propertycountylandusecode','basementsqft','buildingclasstypeid',
                          'finishedsquarefeet13','storytypeid','assessmentyear','censustractandblock','typeconstructiontypeid',
                          'yardbuildingsqft26','fireplaceflag']
        df_train.drop(l_drop_columns, axis= 1, inplace = True)
        df_valid.drop(l_drop_columns, axis= 1, inplace= True)
        df_test.drop(l_drop_columns, axis= 1, inplace= True)

        return (df_train,df_valid,df_test)
