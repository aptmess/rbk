import pandas as pd

from src.constants import FeaturesConstants
from src.transformers.base import Transformer


class FeatureSelector(Transformer):
    def transform(
        self, input_data: pd.DataFrame, mode: str = 'train'
    ) -> pd.DataFrame:
        features = input_data.select_dtypes(
            include=[
                'int',
                'float',
                'bool',
                'int64',
                'float64',
                'int32',
                'float32',
                'category',
            ]
        )
        if mode == 'train':
            features = features.drop(FeaturesConstants.target, axis=1)
        drop_columns = self.kwargs.get('drop_columns')
        if drop_columns is not None:
            features = features.drop(drop_columns, axis=1)
        return features
