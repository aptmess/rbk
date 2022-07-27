import re
import sys
import datetime
import pandas as pd
import copy
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from abc import ABC, abstractmethod
from definitions import DATA_PATH
from functools import lru_cache
import pymorphy2

ru_lemmatizer = pymorphy2.MorphAnalyzer()


@lru_cache(maxsize=int(10e5))
def lru_lemmatize(word):
    return ru_lemmatizer.parse(word)[0].normal_form


class Transformer(ABC):

    def __init__(self, name: str = 'transformer', columns: Optional[List[str]] = None) -> None:
        self.name = name
        if columns is not None:
            self.columns = columns

    @abstractmethod
    def transform(self,
                  input_data: pd.DataFrame,
                  mode: str = 'train') -> pd.DataFrame:
        raise NotImplementedError


class Compose(object):
    def __init__(self, transforms: List[Transformer]) -> None:
        self.transforms = transforms

    def __call__(self,
                 data: pd.DataFrame,
                 mode: str) -> pd.DataFrame:
        input_data = copy.deepcopy(data)
        for t in self.transforms:
            input_data = t.transform(input_data, mode)
        return input_data


class CategoryFromText(Transformer):
    def transform(self, input_data: pd.DataFrame, mode='train') -> pd.DataFrame:
        data = copy.deepcopy(input_data)
        preprocessed_text = []
        topics = []
        for i in data['title']:
            d = re.findall(r'\s[\w ]+,\xa0[\d\w, ]*\d\d:\d\d', i)
            if len(d) > 0:
                topic = (
                    d[0]
                    .split('\xa0')[0]
                    .split('\n')[-1]
                    .lower()
                    .replace(',', '')
                    .lstrip()
                )
                i = i.replace(d[0], '\n')
            else:
                topic = np.nan
            topics.append(topic)
            prepro = ' '.join(re.sub(r'[\n\xa0]', '', i).split())
            preprocessed_text.append(prepro)
        data['topics_parsed_from_text'] = topics
        data['prepro_text'] = preprocessed_text
        data['lemmatized'] = data['prepro_text'].apply(
            lambda x: ' '.join(
                [
                    lru_lemmatize(p) if p.islower() else p for p in x.split()
                ]
            )
        )
        return data
