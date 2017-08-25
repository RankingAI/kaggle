import dill as pickle
import numba
import numpy as np
import pandas as pd

class EnsembleModel:
    """"""
    d_weight = {
        'lgb': 0.75,
        'en': 0.25
    }

    def __init__(self, InputDir, OutputDir, PredCols):
        """"""
        self._InputDir = InputDir
        self._OutputDir = OutputDir
        self._PredCols = PredCols

    @numba.jit
    def __ApplyEnsemble(self,LgbCol, EnCol):
        """"""
        n = len(LgbCol)
        result = np.empty((n), dtype='float32')
        for i in range(n):
            result[i] = self.d_weight['lgb'] * LgbCol[i] + \
                         self.d_weight['en'] * EnCol[i]

        return result

    def evaluate(self):
        """"""
        truth_file = '%s/truth.pkl' % self._InputDir

        #lgb_file = '%s/LGB_20170825-10:59:15.pkl' % self._InputDir
        lgb_file = '%s/LGB_20170825-13:13:33.pkl' % self._InputDir ## add lat1
        en_file = '%s/EN_20170825-11:36:45.pkl' % self._InputDir

        with open(truth_file, 'rb') as i_truth, open(lgb_file, 'rb') as i_lgb, open(en_file, 'rb') as i_en:
            truth = pickle.load(i_truth)
            pred_lgb = pickle.load(i_lgb)
            pred_en = pickle.load(i_en)
        i_truth.close()
        i_lgb.close()
        i_en.close()

        ## for lgb
        lgb_score = 0.0
        ae = np.abs(pred_lgb - truth)
        for col in ae.columns:
            lgb_score += np.sum(ae[col])
        lgb_score /= len(truth)  ##!! divided by number of instances, not the number of 'cells'

        ## for en
        en_score = 0.0
        ae = np.abs(pred_en - truth)
        for col in ae.columns:
            en_score += np.sum(ae[col])
        en_score /= len(truth)  ##!! divided by number of instances, not the number of 'cells'

        ## for ensemble
        ensembled = pd.DataFrame(index= truth.index)
        for col in self._PredCols:
            tmp = self.__ApplyEnsemble(pred_lgb['%s' % col].values,pred_en['%s' % col].values)
            df_tmp = pd.DataFrame(tmp, index= pred_lgb.index, columns= ['%s' % col])
            ensembled = pd.concat([ensembled, df_tmp], axis=1)
            print('Enssemble for column %s is done.' % col)

        ensemble_score = 0.0
        ensemble_ae = np.abs(ensembled - truth[['%s' % col for col in self._PredCols]])
        for col in ensemble_ae.columns:
            ensemble_score += np.sum(ensemble_ae[col])
        ensemble_score /= len(ensembled)  ##!! divided by number of instances, not the number of 'cells'
        print('lgb %.6f, en %.6f, ensemble %.6f' % (lgb_score, en_score, ensemble_score))
