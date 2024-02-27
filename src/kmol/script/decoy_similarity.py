from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from datetime import datetime

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from itertools import product

from mila.factories import AbstractScript
from kmol.core.helpers import SuperFactory
from kmol.core.logger import LOGGER as logging
from kmol.data.loaders import AbstractLoader


class DecoyTanimotoSimilarityScript(AbstractScript):
    def __init__(
        self,
        target_loader,
        decoy_loader,
        data_loader,
        fingerprint_radius: int = 2,
    ):
        """
        Compute Tanimoto similiraty between a group of molecule and a decoy set.
        We will filter the data in `data_loader` containing set of protein / molecule dataset
        based on the protein contains in `target_loader`.
        The output is a csv file with the following format:
        smiles, decoy_smiles, tanimoto_score

        The various column input / outputs of the loaders should match in terms of content
        since the decoy similarity will compare and regroup the various input.
        For example the target sequence in the target loader should have the same format
        as the target in `data_loader`.

        @param target_loader: A loader config for the target dataset. This dataset is used to
        filter out the `data_loader` dataset and keep only the target inside target_loader.
        @param decoy_loader: A loader config for the decoy dataset. This dataset contains all the
        decoy molecules we want to use in order to augment our dataset.
        @param data_loader: A loader config for the overall molecule / target dataset.
        @param fingerprint_radius: Radius of the featurization Morgan fingerprint is used.
        """
        assert (
            len(target_loader["target_column_names"]) == 1
            and len(decoy_loader["input_column_names"]) == 1
            and len(data_loader["input_column_names"]) == 1
            and len(data_loader["target_column_names"]) == 1
        ), "DecoyTanimotoSimilarity except the following inputs: \n /\
                - target_loader: `target_column_names` should contain one element /\
                    being the unique targets (sequence or ids) columns \n /\
                - decoy_loader: `input_column_names` should contain one element /\
                    being the smiles columns \n /\
                - data_loader: Should contain one element in both `target_column_names` /\
                    and `input_column_names` corresponding to the matching field for /\
                    the previous target_loader and decoy_loader. The `input_column_names` /\
                    the unique being the smiles columns and the `target_column_names` being /\
                    targets (sequence or ids) columns"

        self.target = SuperFactory.create(AbstractLoader, target_loader)
        self.decoy = SuperFactory.create(AbstractLoader, decoy_loader)
        self.data_loader = SuperFactory.create(AbstractLoader, data_loader)
        self.fingerprint_radius = fingerprint_radius
        self.featurization_jobs = 10

    def compute_fingerprint_pandas(self, list_smiles):
        df = pd.DataFrame(list_smiles, columns=["smiles"])
        df["decoy_smiles"] = self.decoy._dataset[self.decoy._input_columns[0]]
        smiles_df = pd.DataFrame(list_smiles, columns=["smiles"])
        smiles_df["decoy_smiles"] = self.decoy._dataset[self.decoy._input_columns[0]]
        # Compute fingerprint
        df = df.applymap(Chem.MolFromSmiles, na_action="ignore")
        df = df.applymap(partial(AllChem.GetMorganFingerprintAsBitVect, radius=self.fingerprint_radius), na_action="ignore")
        # Create all possible tuple
        uniques = [df[i].dropna().unique().tolist() for i in df.columns]
        mol_tuple = pd.DataFrame(product(*uniques), columns=df.columns)
        # Compute tanimoto similarity
        smiles_uniques = [smiles_df[i].dropna().unique().tolist() for i in smiles_df.columns]
        smiles_df = pd.DataFrame(product(*smiles_uniques), columns=smiles_df.columns)
        assert len(smiles_df) == len(mol_tuple)
        smiles_df["tanimoto_score"] = mol_tuple.progress_apply(self.compute_tanimoto, axis=1)
        return smiles_df

    def retrieve_smile(self, row, col: str):
        return Chem.MolToSmiles(row[col])

    def compute_tanimoto(self, row):
        return DataStructs.FingerprintSimilarity(row["smiles"], row["decoy_smiles"])

    def get_target_sequence(self):
        """
        Preprocess the target loader and retrieve all targets (should be a unique
        identifier identical to the one in the data_loader).
        If some sequence are nan there are skip.
        """
        targets = []
        for i, target in enumerate(self.target):
            target = target.outputs[0]
            if target != target:  # is nan for various type
                logging.warning(
                    f"In the Decoy selection process one target did \
                    not have a sequence skipping this target, target number {i}"
                )
            else:
                targets.append(target)
        return targets

    def get_filter_smile(self, filter_target):
        filter_data = self.data_loader._dataset[
            np.isin(self.data_loader._dataset.loc[:, self.data_loader._target_columns[0]].values, filter_target)
        ]
        return filter_data.loc[:, self.data_loader._input_columns[0]].unique()

    def _get_chunks(self, _list):
        n_jobs = self.featurization_jobs
        chunk_size = len(_list) // n_jobs
        chunks = [_list[i : i + chunk_size] for i in range(0, len(_list), chunk_size)]
        return chunks

    def run(self):
        filter_target = self.get_target_sequence()
        smiles = self.get_filter_smile(filter_target)

        chunks = self._get_chunks(smiles)
        start_time = datetime.now()
        out = pd.DataFrame([], columns=["smiles", "decoy_smiles", "tanimoto_score"])
        with ProcessPoolExecutor(self.featurization_jobs) as executor:
            for result in executor.map(self.compute_fingerprint_pandas, chunks):
                out = pd.concat([out, result])
        print("Final Time: ", datetime.now() - start_time)
        out.to_csv("result_tanimoto_score.csv", index=False)
        return out
