from abc import abstractmethod
from typing import Iterator, List, Union, Any
import ast

import pandas as pd
import dask.dataframe as dd
from torch.utils.data import Dataset as TorchDataset
import multiprocessing
import pickle

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

    def get_all(self) -> List[DataPoint]:
        """
        Retrieve the dataset as List of Datapoint, faster than iterating though the
        list of data in case of very large dataset.
        """
        df = dd.from_pandas(self._dataset, npartitions=4*multiprocessing.cpu_count())
        result = df.map_partitions(lambda pdf: pdf.apply(
            lambda row: DataPoint(
            id_=row,
            labels=self._target_columns,
            inputs={**row[self._input_columns]},
            outputs=row[self._target_columns]
        ), axis=1))
        result = result.compute(scheduler='processes')
        return result.to_list()

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
        assert type(self._dataset.loc[0, self._task_column_name]) == list, \
            "Type of target must be a list in the MultitaskLoader"
        max_target = max(self._dataset.loc[:, self._task_column_name].max())
        assert max_target < self._max_num_tasks, "The values of the target in the " \
            "MultitaskLoader should range from 0 to max_num_tasks - 1 here "  \
            f"maximum in target is {max_target} and max_num_tasks is {max_num_tasks} " \
            f"should be {max_target + 1}"

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


class PickleLoader(AbstractLoader):
    def __init__(self, input_path: str):
        self._dataset = self._load_dataset(input_path)

    def _load_dataset(self, input_path: str) -> List[Any]:
        with open(input_path, "rb") as file:
            pickle.load(file)

    def __getitem__(self, id_: str) -> DataPoint:
        return DataPoint(
            id_=id_,
            **self._dataset[id_]
        ) 