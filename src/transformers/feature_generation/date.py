import pandas as pd

from src.transformers.base import Transformer


def get_part_day(x):
    if (x > 4) and (x <= 8):
        return 'Early Morning'
    elif (x > 8) and (x <= 12):
        return 'Morning'
    elif (x > 12) and (x <= 16):
        return 'Noon'
    elif (x > 16) and (x <= 20):
        return 'Evening'
    elif (x > 20) and (x <= 24):
        return 'Night'
    elif x <= 4:
        return 'Late Night'


class DatetimeTransformer(Transformer):
    def transform(self, input_data: pd.DataFrame, mode='train') -> pd.DataFrame:
        input_data['publish_date'] = pd.to_datetime(input_data['publish_date'])
        input_data['date'] = input_data['publish_date'].dt.date
        input_data['date'] = input_data['date'].astype(str).astype('category')
        input_data['hour'] = input_data['publish_date'].dt.hour
        if self.kwargs.get('target_variable') == 'views':
            input_data['minute'] = input_data['publish_date'].dt.minute

        input_data['dayofweek'] = input_data['publish_date'].dt.dayofweek
        if self.kwargs.get('target_variable') != 'full_reads_percent':
            input_data['month'] = input_data['publish_date'].dt.month
            input_data['day'] = input_data['publish_date'].dt.day
        # df = (input_data['publish_date'] - datetime.datetime(2022, 2, 24)).dt.components
        # input_data = pd.concat((input_data, df[['days']]), axis=1)
        # input_data['is_after_24_february'] = (input_data['publish_date'] > datetime.datetime(2022, 2, 24)).astype(int)
        # input_data['is_weekend'] = np.where(
        #     input_data['publish_date']
        #     .dt.day_name()
        #     .isin(['Sunday', 'Saturday']),
        #     1,
        #     0,
        # )
        # input_data['diff_date'] = (
        #     input_data.index.map(input_data.sort_values(by='publish_date')['publish_date'].diff().to_dict()).values
        # )
        # input_data['diff_date_hour'] = input_data['diff_date'].dt.components['hours']
        # input_data['diff_date_minute'] = input_data['diff_date'].dt.components['minutes']
        # input_data['diff_date_days'] = input_data['diff_date'].dt.components['days']

        # input_data['minute_30'] = input_data['publish_date'].dt.minute % 30
        # input_data['minute_10'] = input_data['publish_date'].dt.minute % 10
        # input_data['minute_20'] = input_data['publish_date'].dt.minute % 20
        # input_data['minute_5'] = input_data['publish_date'].dt.minute % 5
        # input_data['median_ctr_by_median'] = input_data['date'].map((
        #     input_data[~(input_data['ctr'].isna())]
        #     .groupby(by='date')
        #     .agg({'ctr': 'median'})
        #     .to_dict()['ctr']
        # ))
        # input_data['median_ctr_by_category'] = input_data['category'].map((
        #     input_data[~(input_data['ctr'].isna())]
        #     .groupby(by='category')
        #     .agg({'ctr': 'median'})
        #     .to_dict()['ctr']
        # ))
        # input_data['median_parag_news_amount_of_paragraphs_parsed'] = input_data['date'].map((
        #     input_data[~(input_data['ctr'].isna())]
        #     .groupby(by='date')
        #     .agg({'news_amount_of_paragraphs_parsed': 'median'})
        #     .to_dict()['news_amount_of_paragraphs_parsed']
        # ))

        # input_data['news']
        #
        # input_data['hour'] = input_data['hour'].astype('category')
        # input_data['part_of_the_day'] = (
        #     input_data['hour'].apply(get_part_day).astype('category')
        # )
        #
        # #

        # FEATURE
        # input_data['minute'] = input_data['publish_date'].dt.minute
        # input_data['year'] = input_data['publish_date'].dt.year
        # input_data['day'] = input_data['publish_date'].dt.day
        #
        #
        # input_data['is_after_24_february'] = input_data[
        #     'publish_date'
        # ] > datetime.datetime(2022, 2, 24)
        # #
        # input_data['quarter'] = input_data['publish_date'].dt.quarter
        # input_data['time_sec'] = (
        #     input_data['publish_date'].dt.hour * 3600
        #     + input_data['publish_date'].dt.minute * 60
        #     + input_data['publish_date'].dt.second
        # )
        return input_data
