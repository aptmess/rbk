import pandas as pd

from src.transformers.base import Transformer


class TextTransformer(Transformer):
    def transform(self, input_data: pd.DataFrame, mode='train') -> pd.DataFrame:
        input_data['num_words_text'] = (
            input_data['news_text_parsed'].str.split().apply(lambda x: len(x))
        )
        # input_data['num_words_overview'] = (
        #     input_data['news_text_overview_parsed']
        #     .str.split()
        #     .apply(lambda x: len(x))
        # )
        # input_data['num_of_sentence_text'] = input_data[
        #     'news_text_parsed'
        # ].apply(lambda x: len(nltk.sent_tokenize(x)))

        # input_data['num_of_sentence'] = input_data[
        #     'news_text_overview_parsed'
        # ].apply(lambda x: len(nltk.sent_tokenize(x)))
        # input_data['DIVIDE'] = input_data['num_words_text'] / input_data['num_of_sentence_text']
        # input_data['CTR'] = input_data['ctr'] * input_data['DIVIDE']
        # input_data['CTR'] = input_data['ctr'] / input_data['DIVIDE']
        # input_data['is_na_overwiew'] = input_data['news_text_overview_parsed'].apply(lambda x: x if len(x) == 0 else None).isna()

        # input_data['num_words_text_2'] = input_data['news_text_parsed'].apply(
        #     lambda x: len(x)
        # )
        # input_data['num_words_overview_2'] = input_data[
        #     'news_text_overview_parsed'
        # ].apply(lambda x: len(x))
        # if self.kwargs.get('use_text_transformer') is not None:

        # input_data['WOW_PERCENT_WORKS_2'] = input_data['num_words_overview'] / input_data['num_words_text']
        # input_data['WOW_PERCENT_WORKS_3'] = input_data['num_words_title_good'] / input_data['num_words_text']
        # ser = input_data['category'].map(input_data.groupby('category').agg(
        #     {'num_words_text': 'mean'}).to_dict()['num_words_text'])
        # input_data['num_words_text_wow'] = input_data['ctr'] / ser
        # input_data['num_words_text_wow2'] = input_data['ctr'] * ser
        # input_data['num_words_text_wow3'] = ser / input_data['ctr']
        return input_data


class NatashaTextTransformer(Transformer):
    def transform(self, input_data: pd.DataFrame, mode='train') -> pd.DataFrame:
        # for i in [
        #     'PER_TITLE',
        #     'ORG_TITLE',
        #     'LOC_TITLE'
        # ]:
        #     input_data[f'{i}_NUM_WORDS'] = input_data[i].apply(
        #         lambda x: len(x) if isinstance(x, list) else 0
        #     )
        # input_data['mean_text'] = (
        #         input_data['num_words_text']
        #         / input_data['news_amount_of_paragraphs_parsed']
        # )
        return input_data
