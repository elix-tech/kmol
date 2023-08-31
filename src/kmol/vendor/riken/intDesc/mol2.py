"""mol2 class"""
import re
import sys
import itertools

import yaml
import networkx as nx
import numpy as np
import pandas as pd
import logging

from biopandas.mol2 import PandasMol2

logger = logging.getLogger(__name__)


class Mol2:
    """Mol2クラス"""

    def __init__(self, interaction_type):
        """Mol2 クラス

        Args:
            interaction_type (str): 相互作用計算対象種別
                                    Lig: リガンド-タンパク質相互作用
                                    Mut: 抗体-抗原相互作用
                                    Met: 中分子相互作用
        """
        # 「@<TRIPOS>ATOM」情報を保持するメンバ
        self.df_atom = pd.DataFrame(
            columns=[
                "atom_id",
                "atom_name",
                "x",
                "y",
                "z",
                "atom_type",
                "resi",
                "resn",
                "chain",
                "charge",
                "molcular_type",
                "bone",
            ]
        )

        # 「@<TRIPOS>BOND」情報を保持するメンバ
        self.df_bond = pd.DataFrame(columns=["atom_id", "bond_atom_id", "bond_type"])

        # 「@<TRIPOS>SUBSTRUCTURE」情報を保持するメンバ
        self.df_subst = pd.DataFrame(columns=["subst_id", "subst_name", "root_atom", "subst_type", "dict_type", "chain"])

        if interaction_type not in ["Lig", "Mut", "Med"]:
            msg = f"'interaction_type' must be 'Lig' or 'Mut' or 'Med' :{interaction_type}"
            raise ValueError(msg)

        self.interaction_type = interaction_type

    def read_mol2(self, mol2_file):
        """Mol2ファイル読み込みメソッド

        Args:
            mol2_file (str): 相互作用記述子を計算する立体構造情報ファイル（.mol2）パス
        """
        df_atom = None
        try:
            df_atom = PandasMol2().read_mol2(mol2_file).df
        except ValueError:
            column = [
                ("atom_id", int),
                ("atom_name", str),
                ("x", float),
                ("y", float),
                ("z", float),
                ("atom_type", str),
                ("subst_id", int),
                ("subst_name", str),
                ("charge", float),
                ("status_bit", str),
            ]
            df_atom = PandasMol2().read_mol2(mol2_file, columns=column).df

        # 「@<TRIPOS>BOND」,「@<TRIPOS>SUBSTRUCTURE」セクションの読み込み処理
        with open(mol2_file, "r", encoding="utf8") as file:
            f_text = file.read()

        bond_section_pattern = r"@<TRIPOS>BOND\s*([a-zA-Z0-9\s\|]*)(@|\s+|)"
        bonds_pattern = r"(\s|)([a-zA-Z0-9\s\|]*)"
        if re.search(bond_section_pattern, f_text) is None:
            raise ValueError(f'"@<TRIPOS>BOND" Section not Found in {mol2_file}')
        else:
            bond_section = re.search(pattern=bond_section_pattern, string=f_text).group(1)
            bonds = np.array(
                [record.strip().split()[0:4] for record in bond_section.splitlines() if re.match(bonds_pattern, record)]
            )
            self.df_bond = pd.concat(
                [
                    pd.DataFrame(bonds[:, [1, 2, 3]], columns=self.df_bond.columns),
                    pd.DataFrame(bonds[:, [2, 1, 3]], columns=self.df_bond.columns),
                ]
            )
            self.df_bond.drop_duplicates(subset=["atom_id", "bond_atom_id"], keep="first", inplace=True)
            self.df_bond = self.df_bond.astype({"atom_id": int, "bond_atom_id": int, "bond_type": str})
            self.df_bond = self.df_bond.replace("nan", np.nan)
            self.df_bond.reset_index(inplace=True, drop=True)

        subst_section_pattern = r"@<TRIPOS>SUBSTRUCTURE\s([a-zA-Z0-9\s\-\*]*)(@|\s+|)"
        substs_pattern = r"^\s*$"
        if re.search(subst_section_pattern, f_text) is None:
            logger.warning(f'"@<TRIPOS>SUBSTRUCTURE" Section not Found in {mol2_file}')

        else:
            subst_section = re.search(pattern=subst_section_pattern, string=f_text).group(1)
            substs = np.array(
                [record.strip().split()[:6] for record in subst_section.splitlines() if not re.match(substs_pattern, record)]
            )
            self.df_subst = pd.DataFrame(substs[0:, :6], columns=self.df_subst.columns)

        # 原子の骨格情報（主鎖 or 側鎖）を付与
        df_atom["bone"] = df_atom["atom_name"].apply(lambda x: "main" if x in ["C", "N", "CA", "O", "H", "HA"] else "side")

        df_atom["resi"] = df_atom["subst_name"].apply(lambda x: re.search(r"([\-0-9]+)", x).group(0))
        df_atom["resn"] = df_atom["subst_name"].apply(lambda x: re.search(r"([A-Z]+)", x).group(0))

        # Chain 情報の付与
        self.df_subst["root_atom"] = self.df_subst["root_atom"].astype(int)
        df_atom = pd.merge(
            df_atom,
            self.df_subst[["root_atom", "chain"]],
            how="left",
            left_on="atom_id",
            right_on="root_atom",
        )

        df_atom_list = df_atom[["subst_name", "chain", "subst_id"]].to_numpy()
        chain_list = [chain[1] for chain in df_atom_list if isinstance(chain[1], str)]

        if len(chain_list) != 0:
            chain_idx = 1
            obj_chain = chain_list[0]
            obj_subst_name = df_atom_list[0][0]
            obj_subst_id = int(df_atom_list[0][2])
            for i, row in enumerate(df_atom_list):
                subst_name = row[0]
                subst_id = int(row[2])
                if subst_id == obj_subst_id:
                    assert subst_name == obj_subst_name, "Mol2 format error"
                    df_atom_list[i][1] = obj_chain
                else:
                    obj_chain = chain_list[chain_idx]
                    df_atom_list[i][1] = obj_chain
                    obj_subst_name = df_atom_list[i][0]
                    obj_subst_id = df_atom_list[i][2]
                    chain_idx += 1

        df_chain = pd.DataFrame({"chain": df_atom_list[:, 1]})
        df_atom["chain"] = df_chain

        df_atom.loc[:, "molcular_type"] = np.nan

        self.df_atom = df_atom[self.df_atom.columns]
        self.df_atom = self.df_atom.astype(
            {
                "atom_id": int,
                "atom_name": str,
                "x": float,
                "y": float,
                "z": float,
                "atom_type": str,
                "resi": int,
                "resn": str,
                "chain": str,
                "charge": float,
                "molcular_type": str,
                "bone": str,
            }
        )
        self.df_atom = self.df_atom.replace("nan", np.nan)

    def add_molcular_type(self, molcular_select_file):
        """分子タイプを付与するメソッド

        Args:
            molcular_select_file (str): 相互作用対象分子指定ファイル
        """

        with open(molcular_select_file, "r", encoding="utf8") as file:
            subst = yaml.safe_load(file)

        if self.interaction_type == "Lig":
            for mol_name in subst.keys():
                if "protein" not in mol_name and "ligand" not in mol_name and "solvent" not in mol_name:
                    raise ValueError(f'Only "protein", "ligand", and "solvnet" in {molcular_select_file}')

        elif self.interaction_type == "Mut":
            for mol_name in subst.keys():
                if (
                    "mutant" not in mol_name
                    and "antigen" not in mol_name
                    and "antibody" not in mol_name
                    and "solvent" not in mol_name
                ):
                    raise ValueError(f'Only "mutant", "antibody", "antigen" "solvent" in {molcular_select_file}')

        elif self.interaction_type == "Med":
            for mol_name in subst.keys():
                if all(
                    [
                        "protein" not in mol_name,
                        "ligand" not in mol_name,
                        "peptide" not in mol_name,
                        "membrane" not in mol_name,
                        "solvent" not in mol_name,
                    ]
                ):
                    raise ValueError(
                        'Only "protein", "ligand", "peptide", "membrane", "solvent" ' f"in {molcular_select_file}"
                    )

        assigned_molcular_type = {}
        for mol_name in subst.keys():
            cond = True
            if "type" in subst[mol_name].keys():
                cond = cond & (self.df_atom["bone"] == subst[mol_name]["type"])

            chains = None
            if "chain" in subst[mol_name].keys():
                chains = subst[mol_name]["chain"]
                if isinstance(chains, list):
                    chains = list(chains)

            if chains is not None and len(chains) > 1:
                cond = cond & (self.df_atom["chain"].isin(subst[mol_name]["chain"]))

            else:
                resn_list = None
                if "name" in subst[mol_name].keys():
                    if isinstance(subst[mol_name]["name"], list):
                        resn_list = subst[mol_name]["name"]
                    else:
                        resn_list = [subst[mol_name]["name"]]

                resi_list = None
                if "num" in subst[mol_name].keys():
                    resi_list = []
                    if isinstance(subst[mol_name]["num"], list):
                        for num_str in subst[mol_name]["num"]:
                            if isinstance(num_str, str):
                                for num in range(int(num_str.split(":")[0]), int(num_str.split(":")[1]) + 1):
                                    resi_list.append(num)
                            else:
                                resi_list.append(num_str)
                    elif isinstance(subst[mol_name]["num"], str):
                        num_str = subst[mol_name]["num"]
                        for num in range(int(num_str.split(":")[0]), int(num_str.split(":")[1]) + 1):
                            resi_list.append(num)

                    else:
                        resi_list = [subst[mol_name]["num"]]
                if resn_list is not None and resi_list is not None:
                    raise ValueError('"num" and "name" cannot be specified at the same time.')

                if chains is not None:
                    if isinstance(chains, list):
                        cond = cond & (self.df_atom["chain"] == chains[0])
                    else:
                        cond = cond & (self.df_atom["chain"] == chains)

                if resn_list is not None:
                    cond = cond & (self.df_atom["resn"].isin(resn_list))
                elif resi_list is not None:
                    cond = cond & (self.df_atom["resi"].isin(resi_list))
                elif chains is None:
                    raise ValueError('Specify "num" Or "name" Or "Chain".')

            # NOTE: 指定範囲重複時、先に更新されている値を上書きしない
            cond = cond & (self.df_atom["molcular_type"].isnull())

            if "protein" in mol_name:
                self.df_atom.loc[cond, "molcular_type"] = "Pro"
                assigned_molcular_type[mol_name.split("_")[0]] = "Pro"
            elif "ligand" in mol_name:
                self.df_atom.loc[cond, "molcular_type"] = "L"
                assigned_molcular_type[mol_name.split("_")[0]] = "L"
            elif "mutant" in mol_name:
                self.df_atom.loc[cond, "molcular_type"] = "Mut"
                assigned_molcular_type[mol_name.split("_")[0]] = "Mut"
            elif "antibody" in mol_name:
                self.df_atom.loc[cond, "molcular_type"] = "Ab"
                assigned_molcular_type[mol_name.split("_")[0]] = "Ab"
            elif "antigen" in mol_name:
                self.df_atom.loc[cond, "molcular_type"] = "Ag"
                assigned_molcular_type[mol_name.split("_")[0]] = "Ag"
            elif "peptide" in mol_name:
                self.df_atom.loc[cond, "molcular_type"] = "Pep"
                assigned_molcular_type[mol_name.split("_")[0]] = "Pep"
            elif "membrane" in mol_name:
                self.df_atom.loc[cond, "molcular_type"] = "Mem"
                assigned_molcular_type[mol_name.split("_")[0]] = "Mem"
            elif "solvent" in mol_name:
                solvent_no = ""
                if "_" in mol_name:
                    solvent_no = mol_name.split("_")[1]
                self.df_atom.loc[cond, "molcular_type"] = f"S{solvent_no}"
                assigned_molcular_type[mol_name] = f"S{solvent_no}"

        for mol_name, molcular_type in assigned_molcular_type.items():
            df = self.df_atom.loc[self.df_atom["molcular_type"] == molcular_type]
            if df.empty:
                logger.warning(f"No atoms were assigned to '{mol_name}' " f"specified by '{molcular_select_file}'.")
