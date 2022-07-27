import copy
from typing import List

import pandas as pd

from .transformer import Transformer


class Compose(object):
    def __init__(self, transforms: List[Transformer]) -> None:
        self.transforms = transforms

    def __call__(self, data: pd.DataFrame, mode: str) -> pd.DataFrame:
        input_data = copy.deepcopy(data)
        for t in self.transforms:
            input_data = t.transform(input_data, mode)
        return input_data
