import json
import logging
import os
import typing as t

from itemadapter import ItemAdapter
from scrapy.spiders import Spider

from definitions import PARSED_DATA_DIR
from scrapper.items import AuthorItem, NewsItem, PostItem, TopicItem

log = logging.getLogger(__name__)


class ScrapyMFDPipeline:
    total = 0
    count = 0

    filename = PARSED_DATA_DIR / os.environ.get('OUTPUT_PATH', 'parsed.json')

    with open(filename, mode='w', encoding='utf-8') as f:
        pass

    def process_item(
        self,
        item: t.Union[AuthorItem, PostItem, TopicItem, NewsItem],
        spider: Spider,
    ) -> t.Union[AuthorItem, PostItem, TopicItem, NewsItem]:
        adapter = ItemAdapter(item).asdict()
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(adapter, ensure_ascii=False, default=str))
            f.write('\n')
        self.total += 1
        print(self.total)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
