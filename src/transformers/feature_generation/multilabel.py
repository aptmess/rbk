import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


class MultiLabelTransformer:
    def __init__(
        self, col: str = 'authors_parsed', fill_value: str = 'unknown'
    ):
        self.mlb = MultiLabelBinarizer()
        self.col = col
        self.fill_value = [fill_value]

    def fit(self, X):
        unique_values = X[self.col].apply(
            lambda x: x if isinstance(x, list) else self.fill_value
        )
        self.mlb.fit(unique_values)

    def transform(self, X):
        data = self.mlb.transform(
            X[self.col].apply(
                lambda x: x if isinstance(x, list) else self.fill_value
            )
        )
        return pd.DataFrame(
            data=data,
            columns=[f'{self.col}_{x}' for x in self.mlb.classes_],
            index=X.index,
        )
