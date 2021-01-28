from abc import abstractmethod
from typing import Iterator, List, Union, Any

import pandas as pd
from torch.utils.data import Dataset as TorchDataset

from lib.data.resources import Data


class AbstractLoader(TorchDataset):

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, id_: Union[int, str]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def list_ids(self) -> List[str]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Data]:
        for id_ in self.list_ids():
            yield self[id_]


class CsvLoader(AbstractLoader):

    def __init__(self, input_path: str, input_column_names: List[str], target_column_names: List[str]):
        self._input_columns = input_column_names
        self._target_columns = target_column_names

        self._dataset = pd.read_csv(input_path)

    def __len__(self) -> int:
        return self._dataset.shape[0]

    def __getitem__(self, id_: str) -> Data:
        entry = self._dataset.loc[id_]
        return Data(
            id_=id_,
            labels=self._target_columns,
            inputs={**entry[self._input_columns]},
            outputs=entry[self._target_columns].to_list()
        )

    def list_ids(self) -> List[str]:
        return list(range(len(self)))


class ExcelLoader(CsvLoader):

    def __init__(self, input_path: str, sheet_index: str, input_column_names: List[str], target_column_names: List[str]):
        self._input_columns = input_column_names
        self._target_columns = target_column_names

        self._dataset = pd.read_excel(input_path, sheet_name=sheet_index)


class ListLoader(AbstractLoader):

    def __init__(self, data: List[Data]):
        self._dataset = data

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, id_: int) -> Data:
        return self._dataset[id_]

    def list_ids(self) -> List[str]:
        return list(range(len(self)))
