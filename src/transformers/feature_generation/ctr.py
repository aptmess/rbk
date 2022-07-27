import pandas as pd

from src.transformers.base import Transformer


class CTRTransformer(Transformer):
    def transform(self, input_data: pd.DataFrame, mode='train') -> pd.DataFrame:
        X = input_data['ctr']
        input_data['is_na_ctr'] = X.isna()
        # input_data['ctr_normal_distribution'] = (X - np.mean(X)) / np.std(X)
        # input_data['ctr_log'] = np.where(X == 0, mean_ctr_log, ctr_log)
        # input_data['ctr'] = input_data['ctr'] / 100
        # start = 0
        # end = 1
        # eps = 0.1
        # for i, j in zip(
        #     np.arange(start, end, eps), np.arange(start + eps, end + eps, eps)
        # ):
        #     input_data[f'ctr_{i}_{j}'] = X.apply(
        #         lambda x: (x >= i) & (x < j)
        #     )
        #
        # ctr_log = np.log(X)
        # mean_ctr_log = np.mean(ctr_log.values, where=(ctr_log != -np.inf))
        #
        # FEATURE
        #

        # eps = self.kwargs.get('eps') or 0.5

        #

        return input_data
