import logging
import typing as t

from scrapy import Item
from w3lib.html import remove_tags

from scrapper.services.transformers.base_transformer import Transformer

log = logging.getLogger(__name__)


class NewsText(Transformer):
    def transform(self, input_data: Item) -> t.Tuple[Item, bool]:
        result = [
            remove_tags(x).replace('\xa0', ' ') for x in self._xpath_value
        ]
        input_data[self._column_name] = '\n'.join(result)
        return input_data, True


class NewsAmountOfParagraphs(Transformer):
    def transform(self, input_data: Item) -> t.Tuple[Item, bool]:
        input_data[self._column_name] = len(self._xpath_value)
        return input_data, True


class NewsTitle(Transformer):
    def transform(self, input_data: Item) -> t.Tuple[Item, bool]:
        if self._xpath_value is None:
            return input_data, False
        input_data[self._column_name] = ' '.join(self._xpath_value.split())
        return input_data, True


class NewsTitleYandexHeader(Transformer):
    def transform(self, input_data: Item) -> t.Tuple[Item, bool]:
        if self._xpath_value is None:
            input_data[self._column_name] = ''
            return input_data, True
        input_data[self._column_name] = ' '.join(self._xpath_value.split())
        return input_data, True


class NewsAuthorsTags(Transformer):
    def transform(self, input_data: Item) -> t.Tuple[Item, bool]:
        if self._xpath_value is None:
            input_data[self._column_name] = []
            return input_data, True
        input_data[self._column_name] = [
            x.replace(',', '') for x in self._xpath_value
        ]
        return input_data, True


class NewsInlineTitles(Transformer):
    def transform(self, input_data: Item) -> t.Tuple[Item, bool]:
        if self._xpath_value is None:
            input_data[self._column_name] = []
            return input_data, True
        input_data[self._column_name] = [
            ' '.join(remove_tags(x).replace('\xa0', ' ').split())
            for x in self._xpath_value
        ]
        return input_data, True


class NewsHasImage(Transformer):
    def transform(self, input_data: Item) -> t.Tuple[Item, bool]:
        input_data[self._column_name] = 1 if len(self._xpath_value) >= 1 else 0
        return input_data, True


class NewsImageTitle(Transformer):
    def transform(self, input_data: Item) -> t.Tuple[Item, bool]:
        if self._xpath_value is None:
            input_data[self._column_name] = ''
            return input_data, True
        input_data[self._column_name] = ' '.join(self._xpath_value.split())
        return input_data, True


class NewsAmountOfInlineItems(Transformer):
    def transform(self, input_data: Item) -> t.Tuple[Item, bool]:
        input_data[self._column_name] = len(self._xpath_value)
        return input_data, True
