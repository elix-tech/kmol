from abc import abstractmethod
from typing import Iterator, List, Union, Any
import ast

import pandas as pd
from torch.utils.data import Dataset as TorchDataset

from .resources import DataPoint


class AbstractLoader(TorchDataset):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, id_: Union[int, str]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def list_ids(self) -> List[Union[int, str]]:
        raise NotImplementedError

    @abstractmethod
    def get_labels(self) -> List[str]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[DataPoint]:
        for id_ in self.list_ids():
            yield self[id_]


class CsvLoader(AbstractLoader):
    def __init__(self, input_path: str, input_column_names: List[str], target_column_names: List[str]):
        self._input_columns = input_column_names
        self._target_columns = target_column_names

        self._dataset = pd.read_csv(input_path)

    def __len__(self) -> int:
        return self._dataset.shape[0]

    def __getitem__(self, id_: str) -> DataPoint:
        entry = self._dataset.loc[id_]
        return DataPoint(
            id_=id_,
            labels=self._target_columns,
            inputs={**entry[self._input_columns]},
            outputs=entry[self._target_columns].to_list(),
        )

    def list_ids(self) -> List[Union[int, str]]:
        return list(range(len(self)))

    def get_labels(self) -> List[str]:
        return self._target_columns


class MultitaskLoader(CsvLoader):
    """
    Loader for prediction of multiple target interaction given one ligand as an input.
    """

    def __init__(
        self,
        input_path: str,
        task_column_name: str,
        max_num_tasks: int,
        input_column_names: List[str],
        target_column_names: List[str],
    ):
        """
        @param input_path: path to csv dataset
        @param task_column_name: Name of the column containing the positive protein list.
        @param max_num_tasks: Number of unique protein in the dataset
        @input_column_names: Name of the smiles column
        @target_column_names: Name of the target column use as prediction.
        """

        self._input_columns = input_column_names
        self._target_columns = target_column_names
        self._task_column_name = task_column_name
        self._max_num_tasks = max_num_tasks

        converter_dict = {target: ast.literal_eval for target in target_column_names}
        converter_dict.update({task: ast.literal_eval for task in [task_column_name]})
        self._dataset = pd.read_csv(
            filepath_or_buffer=input_path,
            converters=converter_dict,
        )

    def __getitem__(self, id_: str) -> DataPoint:
        entry = self._dataset.loc[id_]

        tasks = entry[self._task_column_name]
        labels = entry[self._target_columns].to_list()[0]

        task_outputs = [float("nan")] * self._max_num_tasks

        for idx in range(len(tasks)):
            task = tasks[idx]
            task_outputs[task] = labels[idx]

        return DataPoint(
            id_=id_,
            labels=self._target_columns,
            inputs={**entry[self._input_columns]},
            outputs=task_outputs,
        )


class ExcelLoader(CsvLoader):
    def __init__(self, input_path: str, sheet_index: str, input_column_names: List[str], target_column_names: List[str]):
        self._input_columns = input_column_names
        self._target_columns = target_column_names

        self._dataset = pd.read_excel(input_path, sheet_name=sheet_index)


class SdfLoader(CsvLoader):
    def __init__(self, input_path: str, input_column_names: List[str], target_column_names: List[str]):
        self._input_columns = input_column_names
        self._target_columns = target_column_names

        self._dataset = self._load_dataset(input_path)

    def _load_dataset(self, input_path: str) -> pd.DataFrame:
        from rdkit import Chem
        from rdkit.Chem import PandasTools

        dataset = PandasTools.LoadSDF(input_path)
        dataset["smiles"] = [Chem.MolToSmiles(smiles) for smiles in dataset["ROMol"]]

        return dataset


class ListLoader(AbstractLoader):
    def __init__(self, data: List[DataPoint], indices: List[str]):
        self._dataset = data
        self._indices = indices

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, id_: str) -> DataPoint:
        return self._dataset[self._indices.index(id_)]

    def list_ids(self) -> List[Union[int, str]]:
        return self._indices

    def get_labels(self) -> List[str]:
        return self._dataset[0].labels
