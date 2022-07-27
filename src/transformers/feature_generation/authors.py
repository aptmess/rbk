import pandas as pd
from src.transformers.base import Transformer


class AuthorsTransformer(Transformer):
    def transform(self, input_data: pd.DataFrame, mode='train') -> pd.DataFrame:
        if self.kwargs.get('target_variable') != 'depth':
            input_data['amount_of_authors'] = input_data[
                'authors_parsed'
            ].apply(lambda x: len(x) if isinstance(x, list) else 0)

        if self.kwargs.get('target_variable') == 'full_reads_percent':
            input_data['is_author_na'] = input_data['authors_parsed'].isna()
        return input_data
