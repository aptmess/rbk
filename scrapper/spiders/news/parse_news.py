import datetime
import time
from typing import Any, Dict, Iterator

import pandas as pd
from scrapy import Request, Spider
from scrapy.http import Response

from definitions import INPUT_DATA_DIR
from scrapper.spiders.news.parse_one_new import parse_one_new

dd, dt = datetime.datetime, datetime.timedelta


class NewsSpider(Spider):
    name: str = 'news_spider'
    base_url: str = 'https://www.rbc.ru/rbcfreenews'
    start_urls = ['https://www.rbc.ru/rbcfreenews']
    handle_httpstatus_list = [301, 302]

    def __init__(
        self,
        path: str = 'train_data_train.csv',
        **kwargs,
    ):
        self.data = pd.read_csv(INPUT_DATA_DIR / path)
        super().__init__(**kwargs)

    def parse_news(self, response: Response) -> Iterator[Any]:
        yield from parse_one_new(response, self.base_url)

    def parse(
        self, response: Response, **kwargs: Dict[Any, Any]
    ) -> Iterator[Any]:
        news = [x[:24] for x in self.data['document_id']]
        for original, ids in zip(self.data['document_id'], news):
            time.sleep(0.01)
            yield Request(
                f'{self.base_url}/{ids}',
                callback=self.parse_news,
                meta={
                    'document_id': original,
                },
            )
