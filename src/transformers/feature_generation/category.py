import pandas as pd

from src.transformers.base import Transformer


class CategoryTransformer(Transformer):
    def transform(self, input_data: pd.DataFrame, mode='train') -> pd.DataFrame:
        input_data['category'] = input_data['category'].astype('category')
        input_data['category_parsed'] = input_data['category'].astype(
            'category'
        )
        input_data['category_from_title'] = input_data[
            'category_from_title'
        ].isna()

        # for start, end in [[9, 15]]:
        #     input_data[f'{start}_{end}'] = input_data['document_id'].apply(lambda x: x[start:end]).astype('category')
        return input_data
