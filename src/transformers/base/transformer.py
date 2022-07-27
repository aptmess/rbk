from abc import ABC, abstractmethod

import pandas as pd


class Transformer(ABC):
    def __init__(self, name: str = 'transformer', **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs

    @abstractmethod
    def transform(
        self, input_data: pd.DataFrame, mode: str = 'train'
    ) -> pd.DataFrame:
        raise NotImplementedError
