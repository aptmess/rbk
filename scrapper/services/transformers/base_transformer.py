import typing as t
from abc import ABC, abstractmethod

from scrapy import Item
from scrapy.selector.unified import Selector


class Transformer(ABC):
    def __init__(
        self,
        xpath: str,
        column_name: str,
        post: Selector,
        get_all: bool = False,
    ):
        self.post = post
        self._column_name = column_name
        self._xpath = xpath
        self._xpath_value = (
            post.xpath(self._xpath).get()
            if not get_all
            else post.xpath(self._xpath).getall()
        )

    @abstractmethod
    def transform(self, input_data: Item) -> t.Tuple[Item, bool]:
        raise NotImplementedError
