class FeatureSelection:

    @classmethod
    def select(cls,data):

        df_train, df_test = data
        ## drop certain columns, useless and the nearly null ones
        l_drop_columns = ['transactiondate','propertyzoningdesc','propertycountylandusecode','basementsqft','buildingclasstypeid',
                          'finishedsquarefeet13','storytypeid','assessmentyear','censustractandblock','typeconstructiontypeid',
                          'yardbuildingsqft26','fireplaceflag']
        df_train.drop(l_drop_columns, axis=1, inplace = True)

        l_drop_columns.remove('transactiondate')
        df_test.drop(l_drop_columns,axis= 1,inplace= True)

        return (df_train,df_test)
