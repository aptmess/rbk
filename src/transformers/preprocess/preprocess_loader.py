import pandas as pd

from definitions import INPUT_DATA_DIR, PARSED_DATA_DIR
from src.transformers.base import Transformer


class LoaderMergePreprocess(Transformer):
    def transform(self, input_data: pd.DataFrame, mode='train') -> pd.DataFrame:
        data = pd.read_csv(INPUT_DATA_DIR / f'{mode}_dataset_{mode}.csv')
        data_parsed = pd.read_json(
            PARSED_DATA_DIR / f'{mode}_parsed.json',
            orient='records',
            lines=True,
        )

        for i in [data, data_parsed]:
            i['page_id'] = i['document_id'].str[:24]

        data_merge = (
            pd.merge(data, data_parsed, on='page_id')
            .rename({'document_id_x': 'document_id'}, axis=1)
            .drop('document_id_y', axis=1)
        )
        return data_merge
