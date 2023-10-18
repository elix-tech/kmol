"""interaction class"""
import itertools
import math
import re
from decimal import Decimal
import operator

import yaml
import numpy as np
import pandas as pd
import networkx as nx

from . import my_math


class Interaction:
    """Interactionクラス"""

    def __init__(
        self,
        df_atom,
        df_bond,
        interaction_parameter_file,
        vdw_difine_file,
        priority_file,
        exec_type,
    ):
        """コンストラクタ

        Args:
            df_atom (DataFrame): 相互作用記述子を計算する立体構造情報のATOMセクションのDataFrame
            df_bond (DataFrame): 相互作用記述子を計算する立体構造情報のBONDセクションのDataFrame
            interaction_parameter_file (str): 相互作用閾値設定ファイルパス
            vdw_difine_file (str): van del waals 半径定義ファイルパス
            exec_type (str): 実行の種類
        """
        columns = [
            "label1",
            "label2",
            "pair_type",
            "item1_1",
            "item1_2",
            "item1_3",
            "item1_4",
            "item1_5",
            "item1_6",
            "item1_7",
            "item1_8",
            "item1_9",
            "item1_10",
            "item2_1",
            "item2_2",
            "item2_3",
            "item2_4",
            "item2_5",
            "item2_6",
            "item2_7",
            "item2_8",
            "item2_9",
            "item2_10",
            "atom1_id",
            "atom2_id",
            "atom3_id",
            "atom4_id",
        ]

        self.df_atom = df_atom
        self.df_bond = df_bond
        self.read_setting(interaction_parameter_file)
        self.read_vdw_radius(vdw_difine_file)
        if priority_file is not None:
            self.read_priority_file(priority_file)
        self.exec_type = exec_type
        self.interaction_table = pd.DataFrame(index=[], columns=columns)

    def read_setting(self, interaction_parameter_file):
        """相互作用閾値情報取得

        Args:
            interaction_parameter_file (str): 相互作用対象分子指定ファイルパス
        """
        with open(interaction_parameter_file, "r", encoding="utf-8") as file:
            self.interaction_parameter = yaml.safe_load(file)
        for key, value in self.interaction_parameter.items():
            if isinstance(value, str):
                threshold_list = []
                for val in value.split(" "):
                    try:
                        threshold_list.append(float(val))
                    except ValueError:
                        threshold_list.append(val)
                self.interaction_parameter[key] = threshold_list
            else:
                self.interaction_parameter[key] = [value]

    def read_vdw_radius(self, vdw_difine_file):
        """vdW半径情報取得

        Args:
            vdw_difine_file (str): van del waals 半径定義ファイルパス
        """
        with open(vdw_difine_file, "r", encoding="utf-8") as file:
            self.vdw_difine = yaml.safe_load(file)

    def read_priority_file(self, priority_file):
        """相互作用優先順位情報取得

        Args:
            priority_file (str): 相互作用優先順位定義ファイルパス
        """
        with open(priority_file, "r", encoding="utf-8") as file:
            self.priority = yaml.safe_load(file)
        for key in self.priority.keys():
            if not isinstance(self.priority[key], int):
                raise ValueError(f"Specify an integer for '{key}' in '{priority_file}'.")

    def calculate(self, no_mediate, switch_ch_pi):
        """相互作用検出フロー実行メソッド

        Args:
            no_mediate (bool): True: 溶媒を介した相互作用を検出しない
            switch_ch_pi (bool): True: CH_PI, NH_PI, OH_PI を古い定義で判定する。
        """
        if self.exec_type == "Lig":
            target_atoms = self.df_atom.loc[
                self.df_atom["molcular_type"] == "L",
                ["atom_id", "atom_type", "x", "y", "z", "resi", "molcular_type", "charge"],
            ]
            dist = 8.0
        elif self.exec_type == "Mut":
            target_atoms = self.df_atom.loc[
                self.df_atom["molcular_type"] == "Mut",
                ["atom_id", "atom_type", "x", "y", "z", "resi", "molcular_type", "charge"],
            ]
            dist = 8.0
        elif self.exec_type == "Med":
            target_atoms = self.df_atom.loc[
                self.df_atom["molcular_type"].isin(["Pep", "L", "Pro"]),
                ["atom_id", "atom_type", "x", "y", "z", "resi", "molcular_type", "charge"],
            ]
            dist = 6.5

        if len(target_atoms) == 0:
            return
        target_atom1_atoms_list = target_atoms.to_numpy().tolist()

        for atom1 in target_atom1_atoms_list:
            atom1_atoms_coord = [atom1[2], atom1[3], atom1[4]]
            atom1_molcular_type = atom1[6]

            bond_atoms = self.df_bond.loc[
                self.df_bond["atom_id"] == atom1[0], "bond_atom_id"
            ].tolist()

            # NOTE: atom_1 と相互作用を判定する原子条件
            # # 条件1: atom_1 と指定距離以内に位置する
            # # 条件2: atom_1 と共有結合している原子は対象外
            # # 条件3: atom_1 と同じ残基は対象外
            atom2_cond = (
                (
                    np.sqrt(
                        np.sum(
                            self.df_atom[["x", "y", "z"]].subtract(atom1_atoms_coord, axis=1) ** 2,
                            axis=1,
                        )
                    )
                    < dist
                )
                & ~(self.df_atom["atom_id"].isin(bond_atoms))
                & ~(self.df_atom["resi"] == atom1[5])
            )
            if self.exec_type == "Med":
                # # 中分子相互作用の場合
                # # 条件4: atom_1 の molcular_type に対応する残基を対象とする
                atom2_mol_type_cond = None
                if atom1_molcular_type == "Pep":
                    atom2_mol_type_cond = (self.df_atom["molcular_type"].isin(["Pep", "Mem"])) | (
                        (self.df_atom["molcular_type"].str.contains("S"))
                    )
                elif atom1_molcular_type == "L":
                    atom2_mol_type_cond = (self.df_atom["molcular_type"] == "Pro") | (
                        (self.df_atom["molcular_type"].str.contains("S"))
                    )
                elif atom1_molcular_type == "Pro":
                    atom2_mol_type_cond = self.df_atom["molcular_type"].str.contains("S")

                atom2_cond = atom2_cond & atom2_mol_type_cond

            elif self.exec_type == "Lig":
                # # リガンド-タンパク質相互作用の場合
                # # 条件4: atom_1 の molcular_type に対応する残基（タンパク質、溶媒）を対象とする
                atom2_mol_type_cond = (self.df_atom["molcular_type"] == "Pro") | (
                    (self.df_atom["molcular_type"].str.contains("S"))
                )
                atom2_cond = atom2_cond & atom2_mol_type_cond

            else:
                # # 条件4: molcular_type が指定されているものを対象とする。
                atom2_cond = atom2_cond & ~(self.df_atom["molcular_type"].isna())

            target_atom2_atoms_list = (
                self.df_atom.loc[
                    atom2_cond,
                    ["atom_id", "atom_type", "x", "y", "z", "resi", "molcular_type", "charge"],
                ]
                .to_numpy()
                .tolist()
            )

            for atom2 in target_atom2_atoms_list:
                atom2_atoms_coord = [atom2[2], atom2[3], atom2[4]]
                self.calc_interaction(
                    atom1_id=atom1[0],
                    atom1_atom_type=atom1[1],
                    atom1_coord=atom1_atoms_coord,
                    atom1_resi=atom1[5],
                    atom1_molcular_type=atom1_molcular_type,
                    atom1_charge=atom1[7],
                    atom2_id=atom2[0],
                    atom2_atom_type=atom2[1],
                    atom2_coord=atom2_atoms_coord,
                    atom2_resi=atom2[5],
                    atom2_molcular_type=atom2[6],
                    atom2_charge=atom2[7],
                    switch_ch_pi=switch_ch_pi,
                )
                self.calc_interaction(
                    atom1_id=atom2[0],
                    atom1_atom_type=atom2[1],
                    atom1_coord=atom2_atoms_coord,
                    atom1_resi=atom2[5],
                    atom1_molcular_type=atom2[6],
                    atom1_charge=atom2[7],
                    atom2_id=atom1[0],
                    atom2_atom_type=atom1[1],
                    atom2_coord=atom1_atoms_coord,
                    atom2_resi=atom1[5],
                    atom2_molcular_type=atom1_molcular_type,
                    atom2_charge=atom1[7],
                    switch_ch_pi=switch_ch_pi,
                )

        # 溶媒を介した相互作用の検出処理
        if no_mediate:
            return

        if self.exec_type == "Lig":
            # [Lig-S]相互作用の溶媒に対して相互作用を検出する
            target_atoms_list = self.interaction_table.loc[
                (self.interaction_table["pair_type"].str.match("L-S[0-9]*$")), "atom2_id"
            ].tolist()
            dist = 8.0

        if self.exec_type == "Mut":
            # [Mut-S]相互作用の溶媒に対して相互作用を検出する
            target_atoms_list = self.interaction_table.loc[
                (self.interaction_table["pair_type"].str.match("Mut-S[0-9]*$")), "atom2_id"
            ].tolist()
            dist = 8.0

        elif self.exec_type == "Med":
            # [Pep-S*, L-S*, Pro-S*]相互作用の溶媒に対して相互作用を検出する
            target_atoms_list = self.interaction_table.loc[
                (self.interaction_table["pair_type"].str.match(".*-S[0-9]*$"))
                & (self.interaction_table["atom3_id"].isna()),
                "atom2_id",
            ].tolist()
            dist = 6.5

        if len(target_atoms_list) == 0:
            return

        target_atoms = (
            self.df_atom.loc[
                self.df_atom["atom_id"].isin(target_atoms_list),
                ["atom_id", "atom_type", "x", "y", "z", "resi", "molcular_type", "charge"],
            ]
            .to_numpy()
            .tolist()
        )

        for atom1 in target_atoms:
            atom1_id = atom1[0]
            atom1_type = atom1[1]
            atom1_atoms_coord = atom1[2:5]
            atom1_resi = atom1[5]
            atom1_molcular_type = atom1[6]

            bond_atoms = self.df_bond.loc[
                self.df_bond["atom_id"] == atom1_id, "bond_atom_id"
            ].tolist()

            # NOTE: atom_1(溶媒) と相互作用を判定する原子条件
            # # 条件1: atom_1 と指定距離以内に位置する
            # # 条件2: atom_1 と共有結合している原子は対象外
            # # 条件3: atom_1 と同じ残基は対象外
            atom2_cond = (
                (
                    np.sqrt(
                        np.sum(
                            self.df_atom[["x", "y", "z"]].subtract(atom1_atoms_coord, axis=1) ** 2,
                            axis=1,
                        )
                    )
                    < dist
                )
                & ~(self.df_atom["atom_id"].isin(bond_atoms))
                & ~(self.df_atom["resi"] == atom1_resi)
            )
            if self.exec_type == "Lig":
                atom2_cond = atom2_cond & (self.df_atom["molcular_type"] == "Pro")

            elif self.exec_type == "Mut":
                atom2_cond = atom2_cond & (self.df_atom["molcular_type"].isin(["Ab", "Ag"]))

            elif self.exec_type == "Med":
                atom2_cond = atom2_cond & (self.df_atom["molcular_type"].isin(["Pep", "Pro"]))

            target_atom2_atoms_list = (
                self.df_atom.loc[
                    atom2_cond,
                    ["atom_id", "atom_type", "x", "y", "z", "resi", "molcular_type", "charge"],
                ]
                .to_numpy()
                .tolist()
            )
            for atom2 in target_atom2_atoms_list:
                atom2_atoms_coord = [atom2[2], atom2[3], atom2[4]]
                self.calc_interaction(
                    atom1_id=atom1_id,
                    atom1_atom_type=atom1_type,
                    atom1_coord=atom1_atoms_coord,
                    atom1_resi=atom1_resi,
                    atom1_molcular_type=atom1_molcular_type,
                    atom1_charge=atom1[7],
                    atom2_id=atom2[0],
                    atom2_atom_type=atom2[1],
                    atom2_coord=atom2_atoms_coord,
                    atom2_resi=atom2[5],
                    atom2_molcular_type=atom2[6],
                    atom2_charge=atom2[7],
                    switch_ch_pi=switch_ch_pi,
                )
                self.calc_interaction(
                    atom1_id=atom2[0],
                    atom1_atom_type=atom2[1],
                    atom1_coord=atom2_atoms_coord,
                    atom1_resi=atom2[5],
                    atom1_molcular_type=atom2[6],
                    atom1_charge=atom2[7],
                    atom2_id=atom1[0],
                    atom2_atom_type=atom1[1],
                    atom2_coord=atom1_atoms_coord,
                    atom2_resi=atom1[5],
                    atom2_molcular_type=atom1[6],
                    atom2_charge=atom1[7],
                    switch_ch_pi=switch_ch_pi,
                )

    def calc_interaction(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_resi,
        atom1_molcular_type,
        atom1_charge,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_resi,
        atom2_molcular_type,
        atom2_charge,
        switch_ch_pi,
    ):
        """相互作用判定処理実行メソッド

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_resi (int): atom1 の残基番号
            atom1_molcular_type (string): atom1の分子タイプ
            atom1_charge (float): atom1の電荷
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_resi (int): atom2 の残基番号
            atom2_molcular_type (string): atom2の分子タイプ
            atom2_charge (float): atom2の電荷
            switch_ch_pi (bool): True: CH_PI, NH_PI, OH_PI を古い定義で判定する。
        """
        self.calc_hydrogen_bond1(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )
        self.calc_hydrogen_bond1_oh(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )
        self.calc_hydrogen_bond2(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )
        self.calc_hydrogen_bond2_oh(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )
        self.calc_ch_x(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )
        self.calc_ch_o(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )
        self.calc_sh_n_sh_o(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )

        self.calc_electrostatic(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )
        self.calc_electrostatic_oh(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )

        self.calc_van_der_waals(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )

        self.calc_pi_pi_stacking(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_resi,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_resi,
            atom2_molcular_type,
        )

        self.calc_dipo_dipo(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom1_charge,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
            atom2_charge,
        )

        self.calc_omulpol(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom1_charge,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
            atom2_charge,
        )

        if switch_ch_pi:
            self.calc_ch_oh_nh_pi_two(
                atom1_id,
                atom1_atom_type,
                atom1_coord,
                atom1_resi,
                atom1_molcular_type,
                atom2_id,
                atom2_atom_type,
                atom2_coord,
                atom2_molcular_type,
            )
        else:
            self.calc_xh_pi(
                atom1_id,
                atom1_atom_type,
                atom1_coord,
                atom1_molcular_type,
                atom2_id,
                atom2_atom_type,
                atom2_coord,
                atom2_resi,
                atom2_molcular_type,
            )

        self.calc_halogen_one(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )

        self.calc_xh_f(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )

        self.calc_xh_halogen(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )
        self.calc_halogen_pi(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_resi,
            atom2_molcular_type,
        )

        self.calc_nh_s(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )

        self.calc_oh_s_sh_s(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )

        self.calc_s_o(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )

        self.calc_s_n(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )

        self.calc_s_s(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )

        self.calc_s_f(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )

        self.calc_s_pi_1(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_resi,
            atom2_molcular_type,
        )

        self.calc_s_pi_2(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_resi,
            atom2_molcular_type,
        )

        self.calc_metal(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )

        self.calc_ion(
            atom1_id,
            atom1_atom_type,
            atom1_coord,
            atom1_molcular_type,
            atom2_id,
            atom2_atom_type,
            atom2_coord,
            atom2_molcular_type,
        )

    def update_interaction_table(
        self,
        label,
        atom1_id,
        atom1_molcular_type,
        atom2_id,
        atom2_molcular_type,
        item_1=np.nan,
        item_2=np.nan,
        item_3=np.nan,
        item_4=np.nan,
        item_5=np.nan,
        item_6=np.nan,
        item_7=np.nan,
        item_8=np.nan,
        item_9=np.nan,
        item_10=np.nan,
    ):
        """相互作用情報更新メソッド

        Args:
            label (str): 相互作用ラベル
            atom1_id (int): 原子1の通し番号
            atom1_molcular_type (str): 原子1のmolcular_type
            atom2_id (int): 原子2の通し番号
            atom2_molcular_type (str): 原子2のmolcular_type
            item_1 (float, optional): 相互作用判定閾値. Defaults to np.nan.
            item_2 (float, optional): 相互作用判定閾値. Defaults to np.nan.
            item_3 (float, optional): 相互作用判定閾値. Defaults to np.nan.
            item_4 (float, optional): 相互作用判定閾値. Defaults to np.nan.
            item_5 (float, optional): 相互作用判定閾値. Defaults to np.nan.
            item_6 (float, optional): 相互作用判定閾値. Defaults to np.nan.
            item_7 (float, optional): 相互作用判定閾値. Defaults to np.nan.
            item_8 (float, optional): 相互作用判定閾値. Defaults to np.nan.
            item_9 (float, optional): 相互作用判定閾値. Defaults to np.nan.
            item_10 (float, optional): 相互作用判定閾値. Defaults to np.nan.

        """
        # NOTE: 水及び溶媒を介さない相互作用について、既に登録されている場合は追加しない
        cond_label = label if "Dipo" not in label else label.split("_")[0]
        cond = (
            (
                (self.interaction_table["atom1_id"] == atom1_id)
                & (self.interaction_table["atom2_id"] == atom2_id)
            )
            | (
                (self.interaction_table["atom1_id"] == atom2_id)
                & (self.interaction_table["atom2_id"] == atom1_id)
            )
        ) & (self.interaction_table["label1"].str.contains(cond_label))
        if not self.interaction_table[cond].empty:
            return

        if any(
            [
                "S" in atom1_molcular_type
                and atom2_molcular_type in ["Pro", "Mut", "Ab", "Ag", "Pep"],
                "S" in atom2_molcular_type
                and atom1_molcular_type in ["Pro", "Mut", "Ab", "Ag", "Pep"],
            ]
        ):
            # atom3: 溶媒原子
            # atom4: 溶媒原子と相互作用している原子
            if "S" in atom1_molcular_type:
                atom3_id = atom1_id
                atom3_molcular_type = atom1_molcular_type
                atom4_id = atom2_id
                atom4_molcular_type = atom2_molcular_type
            else:
                atom3_id = atom2_id
                atom3_molcular_type = atom2_molcular_type
                atom4_id = atom1_id
                atom4_molcular_type = atom1_molcular_type

                if "Dipo" in label:
                    tmp_charge = item_6
                    item_6 = item_7
                    item_7 = tmp_charge

            # NOTE: 溶媒を介する相互作用について、既に登録されている場合は追加しない
            cond = (
                (self.interaction_table["atom3_id"] == atom3_id)
                & (self.interaction_table["atom4_id"] == atom4_id)
                & (self.interaction_table["label2"] == label)
            )
            if not self.interaction_table[cond].empty:
                return

            # 溶媒原子の残基番号を取得
            atom3_resi = int(self.df_atom.loc[self.df_atom["atom_id"] == atom3_id, "resi"])
            target_atom_list = self.df_atom.loc[
                self.df_atom["resi"] == atom3_resi, "atom_id"
            ].tolist()
            # 更新対象DataFrame条件1：同一の水分子及び溶媒分子と相互作用している & atom4 が atom1と同一原子ではない。
            condition = (
                (self.interaction_table["atom2_id"].isin(target_atom_list))
                & ~(self.interaction_table["atom1_id"] == atom4_id)
                & (self.interaction_table["label2"].isna())
            )
            if atom4_molcular_type == "Pep":
                # 更新対象DataFrame条件1-2: 「Pep-S」の相互作用に対して更新する。
                condition = condition & (
                    self.interaction_table["pair_type"] == f"Pep-{atom3_molcular_type}"
                )
            elif atom4_molcular_type == "Pro":
                if self.exec_type == "Med":
                    # 更新対象DataFrame条件1-2: 「L-S*, Pro-S*」の相互作用に対して更新する。
                    condition = condition & (
                        self.interaction_table["pair_type"].isin(
                            [f"L-{atom3_molcular_type}", f"Pro-{atom3_molcular_type}"]
                        )
                    )
                elif self.exec_type == "Lig":
                    # 更新対象DataFrame条件1-2: 「L-S*」の相互作用に対して更新する。
                    condition = condition & (
                        self.interaction_table["pair_type"] == f"L-{atom3_molcular_type}"
                    )

            elif atom4_molcular_type in ["Mut", "Ab", "Ag"]:
                # 更新対象DataFrame条件1-2: 「Mut-S*」の相互作用に対して更新する。
                condition = condition & (
                    self.interaction_table["pair_type"] == f"Mut-{atom3_molcular_type}"
                )

            medium_interaction_table = self.interaction_table.loc[condition].copy()
            if not medium_interaction_table.empty:
                # 相互作用情報として更新
                medium_interaction_table.loc[:, "label2"] = label
                medium_interaction_table.loc[:, "pair_type"] += f"-{atom4_molcular_type}"
                for idx, item in enumerate(
                    [
                        item_1,
                        item_2,
                        item_3,
                        item_4,
                        item_5,
                        item_6,
                        item_7,
                        item_8,
                        item_9,
                        item_10,
                    ],
                    start=1,
                ):
                    medium_interaction_table.loc[:, f"item2_{idx}"] = item
                medium_interaction_table.loc[:, "atom3_id"] = atom3_id
                medium_interaction_table.loc[:, "atom4_id"] = atom4_id
                self.interaction_table = pd.concat(
                    [self.interaction_table, medium_interaction_table]
                )
            if self.exec_type == "Lig":
                # NOTE: 「S-Pro」については、溶媒を介した相互作用のみで更新し個別に登録しない
                if atom4_molcular_type == "Pro":
                    return
            elif self.exec_type == "Mut":
                # NOTE: 「S-Ab, S-Ag」については、溶媒を介した相互作用のみで更新し個別に登録しない
                if atom4_molcular_type in ["Ab", "Ag"]:
                    return

        # NOTE: 相互作用ペア種類の値を以下に統一するための処理
        # # L-Pro, L-S*, Mut-Ab, Mut-Ag, Mut-Mut, Mut-S*, Pep-Pep, Pep-S, Pep-Mem, Pro-S*
        if any(
            [
                atom1_molcular_type == "Pep" and atom2_molcular_type == "Pep",
                atom1_molcular_type == "Mut" and atom2_molcular_type == "Mut",
            ]
        ):
            dup_table2 = self.interaction_table.loc[
                (self.interaction_table["atom1_id"] == atom2_id)
                & (self.interaction_table["atom2_id"] == atom1_id)
            ]
            if len(dup_table2) != 0:
                tmp_id = atom2_id
                atom2_id = atom1_id
                atom1_id = tmp_id
                if "Dipo" in label:
                    tmp_charge = item_6
                    item_6 = item_7
                    item_7 = tmp_charge

        elif any(
            [
                atom2_molcular_type in ["L", "Mut", "Pep"],
                "S" in atom1_molcular_type and atom2_molcular_type == "Pro",
            ]
        ):
            tmp_id = atom2_id
            tmp_mol_type = atom2_molcular_type
            atom2_id = atom1_id
            atom2_molcular_type = atom1_molcular_type
            atom1_id = tmp_id
            atom1_molcular_type = tmp_mol_type
            if "Dipo" in label:
                tmp_charge = item_6
                item_6 = item_7
                item_7 = tmp_charge

        # 相互作用情報の追加
        table = pd.DataFrame(
            {
                "label1": label,
                "pair_type": f"{atom1_molcular_type}-{atom2_molcular_type}",
                "item1_1": item_1,
                "item1_2": item_2,
                "item1_3": item_3,
                "item1_4": item_4,
                "item1_5": item_5,
                "item1_6": item_6,
                "item1_7": item_7,
                "item1_8": item_8,
                "item1_9": item_9,
                "item1_10": item_10,
                "atom1_id": atom1_id,
                "atom2_id": atom2_id,
            },
            columns=self.interaction_table.columns,
            index=[0],
        )
        self.interaction_table = pd.concat([self.interaction_table, table])

    def drop_13_14(self):
        """1-3, 1-4 相互作用削除"""
        if self.interaction_table.empty:
            return

        atom_1_list = list(set(self.interaction_table["atom1_id"].tolist()))

        # atom1 の BOND情報抽出
        df_bond = self.df_bond.loc[
            (self.df_bond["atom_id"].isin(atom_1_list)), ["atom_id", "bond_atom_id"]
        ].copy()

        # atom1 の BOND情報に 1-3関係にある原子番号情報を付与
        df_bond = pd.merge(
            df_bond,
            self.df_bond[["atom_id", "bond_atom_id"]],
            how="left",
            left_on="bond_atom_id",
            right_on="atom_id",
            suffixes=("_1", "_2"),
        )[["atom_id_1", "bond_atom_id_1", "bond_atom_id_2"]]

        # atom1 の BOND情報に 1-4関係にある原子番号情報を付与
        df_bond = pd.merge(
            df_bond,
            self.df_bond[["atom_id", "bond_atom_id"]],
            how="left",
            left_on="bond_atom_id_2",
            right_on="atom_id",
        )[["atom_id_1", "bond_atom_id_1", "bond_atom_id_2", "bond_atom_id"]]

        droped_interaction_table = pd.DataFrame(columns=self.interaction_table.columns)
        for atom1 in atom_1_list:
            # atom1 と 1-3, 1-4 関係にない相互作用を登録
            atom_13_list = df_bond.loc[df_bond["atom_id_1"] == atom1, "bond_atom_id_2"].tolist()
            atom_14_list = df_bond.loc[df_bond["atom_id_1"] == atom1, "bond_atom_id"].tolist()
            list_13_14 = atom_13_list + atom_14_list
            cond = (self.interaction_table["atom1_id"] == atom1) & ~(
                self.interaction_table["atom2_id"].isin(list_13_14)
            )
            droped_interaction_table = pd.concat(
                [droped_interaction_table, self.interaction_table[cond]]
            )
        # NOTE 双極子-双極子相互作用は、2組の相互作用情報で1組であるため、labelが重複しないDipoは削除する。
        self.interaction_table = droped_interaction_table[
            ~(droped_interaction_table["label1"].str.contains("Dipo"))
            | (
                (droped_interaction_table["label1"].str.contains("Dipo"))
                & (droped_interaction_table.duplicated(subset=["label1"], keep=False))
            )
        ].copy()

    def drop_duplicate(self):
        """重複する相互作用の削除"""
        if self.interaction_table.empty:
            return

        atom_names = {}
        for atom in self.df_atom[["atom_id", "atom_type"]].to_numpy().tolist():
            symbol = atom[1].split(".")[0]
            if symbol in atom_names.keys():
                atom_names[symbol].append(atom[0])
            else:
                atom_names[symbol] = [atom[0]]

        init_table = self.interaction_table.copy()

        # 相互作用テーブルにスコア値を付与する
        for key, val in self.priority.items():
            if "vdW" in key:
                symbol1 = key.split("_")[0]
                symbol2 = key.split("_")[1]
                if symbol1 not in atom_names.keys() or symbol2 not in atom_names.keys():
                    continue

                init_table.loc[
                    (init_table["label1"] == "vdW")
                    & (init_table["atom1_id"].isin(atom_names[symbol1]))
                    & (init_table["atom2_id"].isin(atom_names[symbol2])),
                    "score1",
                ] = val

                init_table.loc[
                    (init_table["label1"] == "vdW")
                    & (init_table["atom1_id"].isin(atom_names[symbol2]))
                    & (init_table["atom2_id"].isin(atom_names[symbol1])),
                    "score1",
                ] = val

                init_table.loc[
                    (init_table["label2"] == "vdW")
                    & (init_table["atom3_id"].isin(atom_names[symbol1]))
                    & (init_table["atom4_id"].isin(atom_names[symbol2])),
                    "score2",
                ] = val

                init_table.loc[
                    (init_table["label2"] == "vdW")
                    & (init_table["atom3_id"].isin(atom_names[symbol2]))
                    & (init_table["atom4_id"].isin(atom_names[symbol1])),
                    "score2",
                ] = val

            else:
                init_table.loc[(init_table["label1"] == key), "score1"] = val

                init_table.loc[(init_table["label2"] == key), "score2"] = val

        # 重複していない原子間と”Dipo”相互作用の原子間について、keepフラグを立てる。
        cond = ~(init_table.duplicated(subset=["atom1_id", "atom2_id"], keep=False)) | (
            init_table["label1"].str.contains("Dipo")
        )
        init_table.loc[cond, "keep"] = True
        init_table.loc[~cond, "keep"] = False

        # 重複している原子間について、Score値が最大のものに keep フラグを立てる。
        dup_pairs = (
            init_table.loc[
                (init_table.duplicated(subset=["atom1_id", "atom2_id"], keep="first"))
                & ~(init_table["keep"]),
                ["atom1_id", "atom2_id"],
            ]
            .to_numpy()
            .tolist()
        )
        for pair in dup_pairs:
            max_score = init_table.loc[
                (init_table["atom1_id"] == pair[0]) & (init_table["atom2_id"] == pair[1]), "score1"
            ].max(skipna=True)
            if math.isnan(max_score):
                labels = list(
                    set(
                        init_table.loc[
                            (init_table["atom1_id"] == pair[0])
                            & (init_table["atom2_id"] == pair[1])
                            & ~(init_table["label1"].str.contains("Dipo")),
                            "label1",
                        ].tolist()
                    )
                )
                if len(labels) > 1:
                    raise ValueError(
                        f"Overlapping interaction({labels}) "
                        f"between atom_id={pair[0]} and atom_id={pair[1]} "
                        "that does not have a defined priority."
                    )

                init_table.loc[
                    (init_table["atom1_id"] == pair[0]) & (init_table["atom2_id"] == pair[1]),
                    "keep",
                ] = True
            else:
                init_table.loc[
                    (init_table["atom1_id"] == pair[0])
                    & (init_table["atom2_id"] == pair[1])
                    & (init_table["score1"] == max_score)
                    & ~(init_table["score1"].isna()),
                    "keep",
                ] = True

        # atom3-atom4 原子間に対して重複する相互作用を削除する。
        dup_pairs = (
            init_table.loc[
                (
                    init_table.duplicated(
                        subset=["atom1_id", "atom2_id", "atom3_id", "atom4_id"], keep="first"
                    )
                )
                & ~(init_table["atom3_id"].isna())
                & (init_table["keep"]),
                ["atom1_id", "atom2_id", "atom3_id", "atom4_id"],
            ]
            .to_numpy()
            .tolist()
        )
        for pair in dup_pairs:
            atom1_id = pair[0]
            atom2_id = pair[1]
            atom3_id = pair[2]
            atom4_id = pair[3]
            # NOTE 重複組み合わせの中で最大スコア値を取得
            max_score = init_table.loc[
                (init_table["atom1_id"] == atom1_id)
                & (init_table["atom2_id"] == atom2_id)
                & (init_table["atom3_id"] == atom3_id)
                & (init_table["atom4_id"] == atom4_id),
                "score2",
            ].max()
            if math.isnan(max_score):
                labels = list(
                    set(
                        init_table.loc[
                            (init_table["atom1_id"] == atom1_id)
                            & (init_table["atom2_id"] == atom2_id)
                            & (init_table["atom3_id"] == atom3_id)
                            & (init_table["atom4_id"] == atom4_id),
                            "label2",
                        ].tolist()
                    )
                )
                labels = [label for label in labels if "Dipo" not in label]
                if len(labels) > 1:
                    raise ValueError(
                        f"Overlapping interaction({labels}) "
                        f"between atom_id={atom3_id} and atom_id={atom4_id} "
                        "that does not have a defined priority."
                    )

            else:
                init_table.loc[
                    (init_table["atom1_id"] == atom1_id)
                    & (init_table["atom2_id"] == atom2_id)
                    & (init_table["atom3_id"] == atom3_id)
                    & (init_table["atom4_id"] == atom4_id)
                    & ~(init_table["score2"] == max_score)
                    & ~(init_table["label2"].str.contains("Dipo", na=False)),
                    "keep",
                ] = False

        # keep フラグが True の行を再登録
        self.interaction_table = init_table.loc[
            init_table["keep"], self.interaction_table.columns
        ].copy()

    def drop_mediate_interaction(self, allow_mediate_position):
        """溶媒を介する相互作用情報の削除

        溶媒を介する相互作用のうち、atom_2, atom_3 の位置関係が
        指定を超えているものを削除するメソッド

        Args:
            allow_mediate_position (int): 溶媒原子間の位置関係を表す数値
        """
        if self.interaction_table.empty:
            return

        assert allow_mediate_position >= 1, "1-[N] (N: 1 or more)"

        # 溶媒を介する相互作用のうち溶媒原子（atom2）の情報を取得
        atom_2_list = list(
            set(
                self.interaction_table.loc[
                    ~(self.interaction_table["label2"].isna()), "atom2_id"
                ].tolist()
            )
        )

        # atom2 の BOND情報抽出
        df_bond = self.df_bond.loc[
            (self.df_bond["atom_id"].isin(atom_2_list)), ["atom_id", "bond_atom_id"]
        ].copy()

        columns = ["atom_id", "bond_atom_id"]
        if allow_mediate_position > 2:
            # NOTE: 許容する位置関係まで、BOND情報を紐づける
            for i in range(3, allow_mediate_position + 1):
                columns.append(f"bond_atom_id_{i}")
                if i == 3:
                    df_bond = pd.merge(
                        df_bond,
                        self.df_bond[["atom_id", "bond_atom_id"]],
                        how="left",
                        left_on="bond_atom_id",
                        right_on="atom_id",
                        suffixes=("", f"_{i}"),
                    )[columns]
                else:
                    df_bond = pd.merge(
                        df_bond,
                        self.df_bond[["atom_id", "bond_atom_id"]],
                        how="left",
                        left_on=f"bond_atom_id_{i - 1}",
                        right_on="atom_id",
                        suffixes=("", f"_{i}"),
                    )[columns]

        droped_interaction_table = self.interaction_table.copy()
        droped_interaction_table["keep"] = True

        for atom2 in atom_2_list:
            if allow_mediate_position == 1:
                cond = (
                    (self.interaction_table["atom2_id"] == atom2)
                    & ~(self.interaction_table["atom3_id"] == atom2)
                    & ~(self.interaction_table["atom3_id"].isna())
                )
            else:
                atom2_allow_bond_list = list(
                    set(df_bond.loc[df_bond["atom_id"] == atom2].to_numpy().flatten())
                )
                cond = (
                    (self.interaction_table["atom2_id"] == atom2)
                    & ~(self.interaction_table["atom3_id"].isin(atom2_allow_bond_list))
                    & ~(self.interaction_table["atom3_id"].isna())
                )
            droped_interaction_table.loc[cond, "keep"] = False

        self.interaction_table = droped_interaction_table.loc[
            droped_interaction_table["keep"], self.interaction_table.columns
        ].copy()

    def calc_hydrogen_bond1(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """水素結合1

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """

        # AtomType条件
        atom1_atom_type_cond = ["N.3", "N.2", "N.1", "N.am", "N.pl3", "N.4", "N.ar"]
        atom2_atom_type_cond = [
            "O.3",
            "O.2",
            "O.co2",
            "O.spc",
            "O.t3p",
            "O.ar",
            "N.1",
            "N.2",
            "N.ar",
        ]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 閾値情報の取得
        dist_param = self.interaction_parameter["HB_NH_(N,O)"][0]
        angle_param = None
        if atom1_atom_type == "N.4":
            angle_param = self.interaction_parameter["HB_NH_(N,O)"][2]
        else:
            angle_param = self.interaction_parameter["HB_NH_(N,O)"][1]
        angle2_min_param = self.interaction_parameter["HB_NH_(N,O)"][3]
        angle2_max_param = self.interaction_parameter["HB_NH_(N,O)"][4]

        # 原子1と原子2の距離を取得
        distance = my_math.distance_two_points(atom1_coord, atom2_coord)
        if distance > dist_param:
            return

        # atom1  BOND条件
        # AtomType が「H」の原子（水素原子）と結合している
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        if len(atom1_bonds) == 0:
            return
        else:
            atom1_bonds_atom_type = self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom1_bonds), "atom_type"
            ].tolist()
            if "H" not in atom1_bonds_atom_type:
                return

        atom1_bonds = (
            self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom1_bonds), ["atom_type", "x", "y", "z"]
            ]
            .to_numpy()
            .tolist()
        )

        # atom2  BOND条件：無し
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        atom2_bonds = (
            self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom2_bonds), ["atom_type", "x", "y", "z"]
            ]
            .to_numpy()
            .tolist()
        )

        # atom1の結合が複数の場合があるためfor文
        for atom1_bond in atom1_bonds:
            atom1_bond_atom_type = atom1_bond[0]
            if atom1_bond_atom_type[0] != "H":
                continue

            # 原子1の各座標を取得
            atom1_bond_coord = atom1_bond[1:]

            # 原子2,原子1と原子1と共有結合した原子の角度を取得（取得するのは原子1付近の角度）
            angle = my_math.angle_three_points(atom2_coord, atom1_coord, atom1_bond_coord)

            atom2_symbol = atom2_atom_type.split(".")[0]

            # 閾値判定
            if angle > angle_param:
                continue

            for atom2_bond in atom2_bonds:
                atom2_bond_atom_type = atom2_bond[0]
                if atom2_bond_atom_type not in [
                    "C.1",
                    "C.2",
                    "C.3",
                    "C.ar",
                    "C.cat",
                    "N.1",
                    "N.2",
                    "N.3",
                    "N.4",
                    "N.ar",
                    "N.am",
                    "N.pl3",
                    "S.2",
                    "S.3",
                    "S.o2",
                    "S.o",
                    "P.3",
                    "H",
                ]:
                    continue
                atom2_bond_coord = atom2_bond[1:]
                angle2 = my_math.angle_three_points(atom1_bond_coord, atom2_coord, atom2_bond_coord)
                if angle2_min_param <= angle2 <= angle2_max_param:
                    self.update_interaction_table(
                        label=f"HB_NH_{atom2_symbol}",
                        atom1_id=atom1_id,
                        atom1_molcular_type=atom1_molcular_type,
                        atom2_id=atom2_id,
                        atom2_molcular_type=atom2_molcular_type,
                        item_1=distance,
                        item_2=angle,
                        item_3=angle2,
                    )
                    return None

    def calc_hydrogen_bond1_oh(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """水素結合1_oh

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        # AtomType条件
        atom1_atom_type_cond = ["N.3", "N.2", "N.1", "N.am", "N.pl3", "N.4", "N.ar"]
        if atom1_atom_type not in atom1_atom_type_cond or "O." not in atom2_atom_type:
            return

        # 閾値情報の取得
        dist_param = self.interaction_parameter["HB_NH_OH"][0]
        angle_param = None
        if atom1_atom_type == "N.4":
            angle_param = self.interaction_parameter["HB_NH_OH"][2]
        else:
            angle_param = self.interaction_parameter["HB_NH_OH"][1]
        angle2_param = self.interaction_parameter["HB_NH_OH"][3]
        angle3_param = self.interaction_parameter["HB_NH_OH"][4]

        distance = my_math.distance_two_points(atom1_coord, atom2_coord)

        if distance > dist_param:
            return

        # atom1  BOND条件
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        atom1_bonds = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] == "H"),
            ["x", "y", "z"],
        ].to_numpy()

        # atom2 BOND条件
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) != 2:
            return

        atom2_bonds = (
            self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom2_bonds), ["atom_id", "atom_type", "x", "y", "z"]
            ]
            .to_numpy()
            .tolist()
        )

        # atom1の結合が複数の場合があるためfor文
        for atom1_bond_coord in atom1_bonds:
            angle1 = my_math.angle_three_points(atom2_coord, atom1_coord, atom1_bond_coord)

            # 閾値判定
            if angle1 > angle_param:
                continue

            angle2 = None
            angle3 = None

            for atom2_x, atom2_h in itertools.product(atom2_bonds, atom2_bonds):
                if atom2_x[0] == atom2_h[0] or atom2_h[1] != "H":
                    continue
                elif atom2_x[1] not in [
                    "C.1",
                    "C.2",
                    "C.3",
                    "C.ar",
                    "C.cat",
                    "N.1",
                    "N.2",
                    "N.3",
                    "N.4",
                    "N.ar",
                    "N.am",
                    "N.pl3",
                    "S.2",
                    "S.3",
                    "S.o2",
                    "S.o",
                    "P.3",
                    "H",
                ]:
                    continue

                angle2 = my_math.angle_three_points(atom1_bond_coord, atom2_coord, atom2_h[2:])
                angle3 = my_math.angle_three_points(atom1_bond_coord, atom2_coord, atom2_x[2:])

                if angle2 >= angle2_param and angle3 >= angle3_param:
                    self.update_interaction_table(
                        label="HB_NH_O",
                        atom1_id=atom1_id,
                        atom1_molcular_type=atom1_molcular_type,
                        atom2_id=atom2_id,
                        atom2_molcular_type=atom2_molcular_type,
                        item_1=distance,
                        item_2=angle1,
                        item_3=angle2,
                        item_4=angle3,
                    )
                    return

    def calc_hydrogen_bond2(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """水素結合2

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = ["O.3", "O.2", "O.co2", "O.spc", "O.t3p", "O.ar"]
        atom2_atom_type_cond = [
            "O.3",
            "O.2",
            "O.co2",
            "O.spc",
            "O.t3p",
            "O.ar",
            "N.1",
            "N.2",
            "N.ar",
        ]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 原子1と原子2の距離を取得
        distance = my_math.distance_two_points(atom1_coord, atom2_coord)
        if distance > self.interaction_parameter["HB_OH_(N,O)"][0]:
            return

        # atom1  BOND条件
        # AtomType が「H」の原子（水素原子）と結合している
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        if len(atom1_bonds) == 0:
            return
        else:
            atom1_bonds = self.df_atom.loc[
                (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] == "H"),
                ["atom_type", "x", "y", "z"],
            ].to_numpy()

            if len(atom1_bonds) == 0:
                return

        # atom2 BOND条件：無し
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) == 0:
            return
        atom2_bonds = (
            self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom2_bonds), ["atom_type", "x", "y", "z"]
            ]
            .to_numpy()
            .tolist()
        )

        atom1_bonds = atom1_bonds.tolist()

        # atom1の結合が複数の場合があるためfor文
        for atom1_bond in atom1_bonds:
            # 原子1の各座標を取得
            atom1_bond_coord = atom1_bond[1:]

            # 原子2,原子1と原子1と共有結合した原子の角度を取得（取得するのは原子1付近の角度）
            angle = my_math.angle_three_points(atom2_coord, atom1_coord, atom1_bond_coord)

            # 閾値判定
            if angle > self.interaction_parameter["HB_OH_(N,O)"][1]:
                continue

            for atom2_bond in atom2_bonds:
                atom2_bond_atom_type = atom2_bond[0]
                if atom2_bond_atom_type not in [
                    "C.1",
                    "C.2",
                    "C.3",
                    "C.ar",
                    "C.cat",
                    "N.1",
                    "N.2",
                    "N.3",
                    "N.4",
                    "N.ar",
                    "N.am",
                    "N.pl3",
                    "S.2",
                    "S.3",
                    "S.o2",
                    "S.o",
                    "P.3",
                    "H",
                ]:
                    continue
                atom2_bond_coord = atom2_bond[1:]
                angle2 = my_math.angle_three_points(atom1_bond_coord, atom2_coord, atom2_bond_coord)
                if (
                    self.interaction_parameter["HB_OH_(N,O)"][2]
                    <= angle2
                    <= self.interaction_parameter["HB_OH_(N,O)"][3]
                ):
                    atom2_symbol = atom2_atom_type.split(".")[0]
                    self.update_interaction_table(
                        label=f"HB_OH_{atom2_symbol}",
                        atom1_id=atom1_id,
                        atom1_molcular_type=atom1_molcular_type,
                        atom2_id=atom2_id,
                        atom2_molcular_type=atom2_molcular_type,
                        item_1=distance,
                        item_2=angle,
                        item_3=angle2,
                    )
                    return None

    def calc_hydrogen_bond2_oh(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """水素結合2_oh

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        # AtomType条件
        atom1_atom_type_cond = ["O.3", "O.2", "O.co2", "O.spc", "O.t3p", "O.ar"]
        if atom1_atom_type not in atom1_atom_type_cond or "O." not in atom2_atom_type:
            return

        # 閾値情報の取得
        dist_param = self.interaction_parameter["HB_OH_OH"][0]
        angle_param = self.interaction_parameter["HB_OH_OH"][1]
        angle2_param = self.interaction_parameter["HB_OH_OH"][2]
        angle3_param = self.interaction_parameter["HB_OH_OH"][3]

        distance = my_math.distance_two_points(atom1_coord, atom2_coord)

        if distance > dist_param:
            return

        # atom1  BOND条件
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        atom1_bonds = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] == "H"),
            ["x", "y", "z"],
        ].to_numpy()

        # atom2 BOND条件
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) != 2:
            return

        atom2_bonds = (
            self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom2_bonds), ["atom_id", "atom_type", "x", "y", "z"]
            ]
            .to_numpy()
            .tolist()
        )

        # atom1の結合が複数の場合があるためfor文
        for atom1_bond_coord in atom1_bonds:
            angle1 = my_math.angle_three_points(atom2_coord, atom1_coord, atom1_bond_coord)

            # 閾値判定
            if angle1 > angle_param:
                continue

            angle2 = None
            angle3 = None

            for atom2_x, atom2_h in itertools.product(atom2_bonds, atom2_bonds):
                if atom2_x[0] == atom2_h[0] or atom2_h[1] != "H":
                    continue
                elif atom2_x[1] not in [
                    "C.1",
                    "C.2",
                    "C.3",
                    "C.ar",
                    "C.cat",
                    "N.1",
                    "N.2",
                    "N.3",
                    "N.4",
                    "N.ar",
                    "N.am",
                    "N.pl3",
                    "S.2",
                    "S.3",
                    "S.o2",
                    "S.o",
                    "P.3",
                    "H",
                ]:
                    continue

                angle2 = my_math.angle_three_points(atom1_bond_coord, atom2_coord, atom2_h[2:])
                angle3 = my_math.angle_three_points(atom1_bond_coord, atom2_coord, atom2_x[2:])

                if angle2 >= angle2_param and angle3 >= angle3_param:
                    self.update_interaction_table(
                        label="HB_OH_O",
                        atom1_id=atom1_id,
                        atom1_molcular_type=atom1_molcular_type,
                        atom2_id=atom2_id,
                        atom2_molcular_type=atom2_molcular_type,
                        item_1=distance,
                        item_2=angle1,
                        item_3=angle2,
                        item_4=angle3,
                    )
                    return

    def calc_ch_x(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """CH_N, CH_S 相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        # AtomType条件
        atom1_atom_type_cond = ["C.1", "C.2", "C.3", "C.cat", "C.ar"]
        atom2_atom_type_cond = [
            "N.1",
            "N.2",
            "N.ar",
            "N.am",
            "N.pl3",
            "S.2",
            "S.3",
            "S.ar",
        ]
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 閾値取得
        atom2_symbol = atom2_atom_type.split(".")[0]
        buffer_param = self.interaction_parameter[f"CH_{atom2_symbol}"][0]
        dist1_param = self.interaction_parameter[f"CH_{atom2_symbol}"][1]
        dist2_param = self.interaction_parameter[f"CH_{atom2_symbol}"][2]
        angle1_param = self.interaction_parameter[f"CH_{atom2_symbol}"][3]
        angle2_param = self.interaction_parameter[f"CH_{atom2_symbol}"][4]

        vdw_radius = Decimal(str(self.vdw_difine["C"])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        if distance1 > buffer_param + float(vdw_radius):
            return

        # atom2 BOND条件
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) == 0:
            return

        # atom1に共有結合する原子情報を取得
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        atom1_bonds = self.df_atom.loc[
            self.df_atom["atom_id"].isin(atom1_bonds),
            ["atom_type", "x", "y", "z"],
        ].to_numpy()

        distance2 = np.nan
        distance3 = np.nan
        angle1 = None
        angle2 = None
        is_true = False

        heavy_atoms = atom1_bonds[np.where(atom1_bonds[:, 0] != "H"), 1:][0]
        hydro_atoms = atom1_bonds[np.where(atom1_bonds[:, 0] == "H"), 1:][0]
        for heavy_atom, hydro_atom in itertools.product(heavy_atoms, hydro_atoms):
            distance2 = my_math.distance_two_points(heavy_atom, atom2_coord)
            distance3 = my_math.distance_two_points(hydro_atom, atom2_coord)
            angle1 = my_math.angle_three_points(heavy_atom, atom1_coord, atom2_coord)
            angle2 = my_math.angle_three_points(atom1_coord, hydro_atom, atom2_coord)
            if distance1 < distance2 and distance3 < distance1 and angle1 <= angle1_param:
                if any(
                    [
                        distance3 <= dist1_param,
                        dist1_param < distance3 <= dist2_param and angle2 >= angle2_param,
                    ]
                ):
                    is_true = True
                    break

        if not is_true:
            return

        atom2_bonds = self.df_atom.loc[
            self.df_atom["atom_id"].isin(atom2_bonds), ["atom_id", "atom_type", "x", "y", "z"]
        ].to_numpy()

        is_true = False
        distance4 = np.nan
        distance5 = np.nan
        if len(atom2_bonds) == 1 and atom2_bonds[0][1] != "H":
            distance4 = my_math.distance_two_points(atom2_bonds[0][2:], atom1_coord)
            if distance1 < distance4:
                is_true = True
        else:
            for atom2_heaby, atom2_bond in itertools.product(atom2_bonds, atom2_bonds):
                if atom2_heaby[0] == atom2_bond[0] or atom2_heaby[1] == "H":
                    continue
                distance4 = my_math.distance_two_points(atom2_heaby[2:], atom1_coord)
                distance5 = my_math.distance_two_points(atom2_bond[2:], atom1_coord)
                if distance1 < distance4 and distance1 < distance5:
                    is_true = True
                    break

        if not is_true:
            return

        self.update_interaction_table(
            label=f"CH_{atom2_symbol}",
            atom1_id=atom1_id,
            atom1_molcular_type=atom1_molcular_type,
            atom2_id=atom2_id,
            atom2_molcular_type=atom2_molcular_type,
            item_1=distance1,
            item_2=distance2,
            item_3=distance3,
            item_4=distance4,
            item_5=distance5,
            item_6=angle1,
            item_7=angle2,
        )

    def calc_ch_o(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """CH_O 相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """

        # AtomType条件
        atom1_atom_type_cond = ["C.1", "C.2", "C.3", "C.cat", "C.ar"]
        atom2_atom_type_cond = ["O.2", "O.3", "O.co2", "O.ar"]
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 閾値取得
        buffer_param = self.interaction_parameter["CH_O"][0]
        dist1_param = self.interaction_parameter["CH_O"][1]
        angle1_param = self.interaction_parameter["CH_O"][2]
        angle2_param = self.interaction_parameter["CH_O"][3]

        vdw_radius = Decimal(str(self.vdw_difine["C"])) + Decimal(str(self.vdw_difine["O"]))
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        if distance1 > buffer_param + float(vdw_radius):
            return

        # atom2 BOND条件
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) == 0:
            return

        # atom1  BOND条件
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        atom1_bonds = self.df_atom.loc[
            self.df_atom["atom_id"].isin(atom1_bonds),
            ["atom_type", "x", "y", "z"],
        ].to_numpy()

        distance2 = np.nan
        distance3 = np.nan
        angle1 = None
        is_true = False

        heavy_atoms = atom1_bonds[np.where(atom1_bonds[:, 0] != "H"), 1:][0]
        hydro_atoms = atom1_bonds[np.where(atom1_bonds[:, 0] == "H"), 1:][0]
        for heavy_atom, hydro_atom in itertools.product(heavy_atoms, hydro_atoms):
            distance2 = my_math.distance_two_points(heavy_atom, atom2_coord)
            distance3 = my_math.distance_two_points(hydro_atom, atom2_coord)
            angle1 = my_math.angle_three_points(heavy_atom, atom1_coord, atom2_coord)
            angle2 = my_math.angle_three_points(atom1_coord, hydro_atom, atom2_coord)
            if (
                distance1 < distance2
                and distance3 < distance1
                and distance3 < dist1_param
                and angle1 <= angle1_param
                and angle2_param <= angle2
            ):
                is_true = True
                break

        if not is_true:
            return

        atom2_bonds = self.df_atom.loc[
            self.df_atom["atom_id"].isin(atom2_bonds), ["atom_id", "atom_type", "x", "y", "z"]
        ]
        has_heaby_atom2_bond = len(atom2_bonds.loc[atom2_bonds["atom_type"] != "H"]) != 0
        atom2_bonds = atom2_bonds.to_numpy().tolist()

        is_true = False
        distance4 = np.nan
        distance5 = np.nan
        if len(atom2_bonds) == 1:
            distance4 = my_math.distance_two_points(atom2_bonds[0][2:], atom1_coord)
            if distance1 < distance4:
                is_true = True

        else:
            # atom2 に共有結合する原子群の直積で組み合わせを算出
            # # 組み合わせ条件1: (atom_x, atom_h) = (重原子, 水素原子)
            # # 組み合わせ条件2: (atom_x, atom_h) = (水素原子, 水素原子) *共有結合する原子群に重原子が存在しない場合
            for atom2_x, atom2_h in itertools.product(atom2_bonds, atom2_bonds):
                if atom2_x[0] == atom2_h[0]:
                    continue
                elif has_heaby_atom2_bond and atom2_x[1] == "H":
                    continue

                distance4 = my_math.distance_two_points(atom2_x[2:], atom1_coord)
                distance5 = my_math.distance_two_points(atom2_h[2:], atom1_coord)
                if distance1 < distance4 and distance1 < distance5:
                    is_true = True
                    break
        if not is_true:
            return

        self.update_interaction_table(
            label="CH_O",
            atom1_id=atom1_id,
            atom1_molcular_type=atom1_molcular_type,
            atom2_id=atom2_id,
            atom2_molcular_type=atom2_molcular_type,
            item_1=distance1,
            item_2=distance2,
            item_3=distance3,
            item_4=distance4,
            item_5=distance5,
            item_6=angle1,
            item_7=angle2,
        )

    def calc_sh_n_sh_o(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """SH-N, SH-O 相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = ["S.3", "S.2", "S.o", "S.o2", "S.ar"]
        atom2_atom_type_cond = [
            "O.3",
            "O.2",
            "O.co2",
            "O.spc",
            "O.t3p",
            "O.ar",
            "N.1",
            "N.2",
            "N.ar",
        ]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        atom2_symbol = atom2_atom_type.split(".")[0]
        label = "SH_" + atom2_symbol

        vdw_radius = Decimal(str(self.vdw_difine["S"])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )

        # 原子1と原子2の距離を取得
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        if distance1 > self.interaction_parameter[label][0] + float(vdw_radius):
            return

        # atom1  BOND条件
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        if len(atom1_bonds) == 0:
            return
        atom1_bonds_x = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] != "H"),
            ["atom_type", "x", "y", "z"],
        ].to_numpy()
        if len(atom1_bonds_x) == 0:
            return
        atom1_bonds_h = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] == "H"),
            ["atom_type", "x", "y", "z"],
        ].to_numpy()
        if len(atom1_bonds_h) == 0:
            return

        # atom2 BOND条件
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) == 0:
            return
        atom2_bonds = self.df_atom.loc[
            self.df_atom["atom_id"].isin(atom2_bonds), ["atom_type", "x", "y", "z"]
        ].to_numpy()
        if atom2_atom_type == "N.ar":
            # AtomType が「N.ar」の時、「H」の原子（水素原子）と結合していない
            if "H" in atom2_bonds[:, 0]:
                return

        # atom1 と共有結合する全ての重原子に対して判定処理を実施
        x_flag = False
        for atom1_bond_x in atom1_bonds_x:
            atom1_bond_x_coord = atom1_bond_x[1:]

            angle = my_math.angle_three_points(atom1_bond_x_coord, atom1_coord, atom2_coord)
            # 閾値判定
            if angle > self.interaction_parameter[label][1]:
                continue

            # 原子1のBondXと原子2の距離を取得
            distance2 = my_math.distance_two_points(atom1_bond_x_coord, atom2_coord)
            if distance1 >= distance2:
                continue
            x_flag = True
            break

        if not x_flag:
            return

        # atom1 と共有結合する全ての水素原子に対して判定処理を実施
        h_flag = False
        for atom1_bond_h in atom1_bonds_h:
            atom1_bond_h_coord = atom1_bond_h[1:]
            # 原子1のBondXと原子2の距離を取得
            distance3 = my_math.distance_two_points(atom1_bond_h_coord, atom2_coord)
            if distance3 >= distance1:
                continue
            h_flag = True
            break

        if not h_flag:
            return

        # atom2 と共有結合する全ての原子に対して判定処理を実施
        if len(atom2_bonds) == 1:
            distance4 = my_math.distance_two_points(atom1_coord, atom2_bonds[0][1:])
            if distance1 >= distance4:
                return
            distance5 = np.nan
        else:
            for atom2_bond_pair in itertools.combinations(atom2_bonds, 2):
                atom2_z = atom2_bond_pair[0][1:]
                atom2_y = atom2_bond_pair[1][1:]
                if atom2_bond_pair[0][0] == "H" and atom2_bond_pair[1][0] != "H":
                    atom2_z = atom2_bond_pair[1][1:]
                    atom2_y = atom2_bond_pair[0][1:]
                distance4 = my_math.distance_two_points(atom1_coord, atom2_z)
                if distance1 >= distance4:
                    continue
                distance5 = my_math.distance_two_points(atom1_coord, atom2_y)
                if distance1 >= distance5:
                    continue
                break

        self.update_interaction_table(
            label=label,
            atom1_id=atom1_id,
            atom1_molcular_type=atom1_molcular_type,
            atom2_id=atom2_id,
            atom2_molcular_type=atom2_molcular_type,
            item_1=distance1,
            item_2=distance2,
            item_3=distance3,
            item_4=distance4,
            item_5=distance5,
            item_6=angle,
        )

    def calc_electrostatic(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """静電相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = [
            "O.3",
            "O.2",
            "O.co2",
            "O.spc",
            "O.t3p",
            "O.ar",
            "N.3",
            "N.2",
            "N.1",
            "N.am",
            "N.pl3",
            "N.4",
            "N.ar",
        ]
        atom2_atom_type_cond = [
            "O.3",
            "O.2",
            "O.co2",
            "O.spc",
            "O.t3p",
            "O.ar",
            "N.1",
            "N.2",
            "N.ar",
        ]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 閾値情報の取得
        dist_param = self.interaction_parameter["Elec_(NH,OH)_(N,O)"][0]
        buffer_param = self.interaction_parameter["Elec_(NH,OH)_(N,O)"][1]
        angle_param = None
        if atom1_atom_type == "N.4":
            angle_param = self.interaction_parameter["Elec_(NH,OH)_(N,O)"][3]
        else:
            angle_param = self.interaction_parameter["Elec_(NH,OH)_(N,O)"][2]
        angle2_min_param = self.interaction_parameter["Elec_(NH,OH)_(N,O)"][4]
        angle2_max_param = self.interaction_parameter["Elec_(NH,OH)_(N,O)"][5]

        atom1_symbol = atom1_atom_type.split(".")[0]
        atom2_symbol = atom2_atom_type.split(".")[0]

        vdw_radius = Decimal(str(self.vdw_difine[atom1_symbol])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )

        # 原子1と原子2の距離を取得
        distance = my_math.distance_two_points(atom1_coord, atom2_coord)
        if not (dist_param <= distance <= buffer_param + float(vdw_radius)):
            return

        # atom1  BOND条件
        # AtomType が「H」の原子（水素原子）と結合している
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        if len(atom1_bonds) == 0:
            return
        else:
            atom1_bonds = self.df_atom.loc[
                (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] == "H"),
                ["x", "y", "z"],
            ].to_numpy()

            if len(atom1_bonds) == 0:
                return

        # atom2 BOND条件
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) == 0:
            return
        atom2_bonds = (
            self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom2_bonds), ["atom_type", "x", "y", "z"]
            ]
            .to_numpy()
            .tolist()
        )

        atom1_bonds = atom1_bonds.tolist()

        # atom1 と共有結合する全ての水素原子に対して判定処理を実施
        for atom1_bond in atom1_bonds:
            # 原子1の各座標を取得
            atom1_bond_coord = atom1_bond[0:]

            angle = my_math.angle_three_points(atom2_coord, atom1_coord, atom1_bond_coord)
            # 閾値判定
            if angle > angle_param:
                continue
            for atom2_bond in atom2_bonds:
                atom2_bond_atom_type = atom2_bond[0]
                if atom2_bond_atom_type not in [
                    "C.1",
                    "C.2",
                    "C.3",
                    "C.ar",
                    "C.cat",
                    "N.1",
                    "N.2",
                    "N.3",
                    "N.4",
                    "N.ar",
                    "N.am",
                    "N.pl3",
                    "S.2",
                    "S.3",
                    "S.o2",
                    "S.o",
                    "P.3",
                    "H",
                ]:
                    continue
                atom2_bond_coord = atom2_bond[1:]
                angle2 = my_math.angle_three_points(atom1_bond_coord, atom2_coord, atom2_bond_coord)
                if angle2_min_param <= angle2 <= angle2_max_param:
                    self.update_interaction_table(
                        label=f"Elec_{atom1_symbol}H_{atom2_symbol}",
                        atom1_id=atom1_id,
                        atom1_molcular_type=atom1_molcular_type,
                        atom2_id=atom2_id,
                        atom2_molcular_type=atom2_molcular_type,
                        item_1=distance,
                        item_2=angle,
                        item_3=angle2,
                    )
                    return None

    def calc_electrostatic_oh(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """静電相互作用_oh

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        # AtomType条件
        atom1_atom_type_cond = [
            "O.3",
            "O.2",
            "O.co2",
            "O.spc",
            "O.t3p",
            "O.ar",
            "N.3",
            "N.2",
            "N.1",
            "N.am",
            "N.pl3",
            "N.4",
            "N.ar",
        ]
        if atom1_atom_type not in atom1_atom_type_cond or "O." not in atom2_atom_type:
            return

        # 閾値情報の取得
        dist_param = self.interaction_parameter["Elec_(N,O)H_OH"][0]
        buffer_param = self.interaction_parameter["Elec_(N,O)H_OH"][1]
        angle_param = None
        if atom1_atom_type == "N.4":
            angle_param = self.interaction_parameter["Elec_(N,O)H_OH"][3]
        else:
            angle_param = self.interaction_parameter["Elec_(N,O)H_OH"][2]
        angle2_param = self.interaction_parameter["Elec_(N,O)H_OH"][4]
        angle3_param = self.interaction_parameter["Elec_(N,O)H_OH"][5]

        atom1_symbol = atom1_atom_type.split(".")[0]
        atom2_symbol = atom2_atom_type.split(".")[0]

        vdw_radius = Decimal(str(self.vdw_difine[atom1_symbol])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )

        distance = my_math.distance_two_points(atom1_coord, atom2_coord)

        if not (dist_param <= distance <= buffer_param + float(vdw_radius)):
            return

        # atom1  BOND条件
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        atom1_bonds = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] == "H"),
            ["x", "y", "z"],
        ].to_numpy()

        # atom2 BOND条件
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) != 2:
            return

        atom2_bonds = (
            self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom2_bonds), ["atom_id", "atom_type", "x", "y", "z"]
            ]
            .to_numpy()
            .tolist()
        )

        # atom1の結合が複数の場合があるためfor文
        for atom1_bond_coord in atom1_bonds:
            angle1 = my_math.angle_three_points(atom2_coord, atom1_coord, atom1_bond_coord)

            # 閾値判定
            if angle1 > angle_param:
                continue

            angle2 = None
            angle3 = None

            for atom2_x, atom2_h in itertools.product(atom2_bonds, atom2_bonds):
                if atom2_x[0] == atom2_h[0] or atom2_h[1] != "H":
                    continue
                elif atom2_x[1] not in [
                    "C.1",
                    "C.2",
                    "C.3",
                    "C.ar",
                    "C.cat",
                    "N.1",
                    "N.2",
                    "N.3",
                    "N.4",
                    "N.ar",
                    "N.am",
                    "N.pl3",
                    "S.2",
                    "S.3",
                    "S.o2",
                    "S.o",
                    "P.3",
                    "H",
                ]:
                    continue

                angle2 = my_math.angle_three_points(atom1_bond_coord, atom2_coord, atom2_h[2:])
                angle3 = my_math.angle_three_points(atom1_bond_coord, atom2_coord, atom2_x[2:])

                if angle2 >= angle2_param and angle3 >= angle3_param:
                    self.update_interaction_table(
                        label=f"Elec_{atom1_symbol}H_O",
                        atom1_id=atom1_id,
                        atom1_molcular_type=atom1_molcular_type,
                        atom2_id=atom2_id,
                        atom2_molcular_type=atom2_molcular_type,
                        item_1=distance,
                        item_2=angle1,
                        item_3=angle2,
                        item_4=angle3,
                    )
                    return

    def calc_van_der_waals(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """van der waals 相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom_type_cond = ["Fe", "Zn", "Ca", "Mg", "Ni", "K", "Na", "H"]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if atom1_atom_type in atom_type_cond or atom2_atom_type in atom_type_cond:
            return

        # 閾値取得
        buffer_param = self.interaction_parameter["vdW"][0]
        diff_param = self.interaction_parameter["vdW"][1]
        dist1_param = self.interaction_parameter["vdW"][2]
        dist2_param = self.interaction_parameter["vdW"][3]
        dist3_param = self.interaction_parameter["vdW"][4]

        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        vdw_radius = Decimal(str(self.vdw_difine[atom1_atom_type.split(".")[0]])) + Decimal(
            str(self.vdw_difine[atom2_atom_type.split(".")[0]])
        )
        if distance1 > buffer_param + float(vdw_radius):
            return

        dnearest = distance1
        diff_vdw = distance1 - float(vdw_radius)

        # atom1に共有結合する水素原子の取得
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        atom1_targets = (
            self.df_atom.loc[
                (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] == "H"),
                ["x", "y", "z"],
            ]
            .to_numpy()
            .tolist()
        )
        atom1_targets.append(atom1_coord)

        # atom2に共有結合する水素原子の取得
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        atom2_targets = (
            self.df_atom.loc[
                (self.df_atom["atom_id"].isin(atom2_bonds)) & (self.df_atom["atom_type"] == "H"),
                ["x", "y", "z"],
            ]
            .to_numpy()
            .tolist()
        )
        atom2_targets.append(atom2_coord)

        for atom1_h, atom2_h in itertools.product(atom1_targets, atom2_targets):
            dist = my_math.distance_two_points(atom1_h, atom2_h)
            dnearest = min(dnearest, dist)

        if any(
            [
                dnearest <= dist1_param,
                dist2_param < dnearest <= dist3_param,
                dist3_param < dnearest and diff_vdw <= diff_param,
            ]
        ):
            self.update_interaction_table(
                label="vdW",
                atom1_id=atom1_id,
                atom1_molcular_type=atom1_molcular_type,
                atom2_id=atom2_id,
                atom2_molcular_type=atom2_molcular_type,
                item_1=distance1,
                item_2=dnearest,
                item_3=diff_vdw,
            )
            return None

    def calc_pi_pi_stacking(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_resi,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_resi,
        atom2_molcular_type,
    ):
        """π-π stacking 相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_resi (int): atom1 の残基番号
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_resi (int): atom2 の残基番号
            atom2_molcular_type (string): atom2の分子タイプ
        """
        # AtomType条件
        atom1_atom_type_cond = ["C.ar", "N.ar", "C.2", "N.2", "N.pl3", "O.3", "S.3", "N.am"]
        atom2_atom_type_cond = ["C.ar", "N.ar", "C.2", "N.2", "N.pl3", "O.3", "S.3", "N.am"]
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 閾値取得
        buffer_param = self.interaction_parameter["PI_PI"][0]
        angle1_param = self.interaction_parameter["PI_PI"][1]
        angle2_param = self.interaction_parameter["PI_PI"][2]

        # 距離
        distance = my_math.distance_two_points(atom1_coord, atom2_coord)
        atom1_symbol = atom1_atom_type.split(".")[0]
        atom2_symbol = atom2_atom_type.split(".")[0]
        vdw_radius = Decimal(str(self.vdw_difine[atom1_symbol])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )

        # 距離閾値判定
        if distance > buffer_param + float(vdw_radius):
            return None

        # 原子 atom1 が属する芳香環原子を抽出
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        aroma_atoms1 = self.df_atom.loc[
            (self.df_atom["resi"] == atom1_resi)
            & (self.df_atom["atom_type"].isin(atom1_atom_type_cond)),
            ["atom_id", "x", "y", "z"],
        ].to_numpy()

        # 原子 atom2 が属する芳香環原子を抽出
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        aroma_atoms2 = self.df_atom.loc[
            (self.df_atom["resi"] == atom2_resi)
            & (self.df_atom["atom_type"].isin(atom2_atom_type_cond)),
            ["atom_id", "x", "y", "z"],
        ].to_numpy()

        def get_cycle_atom(aroma_atoms, atom_id):
            # networkx形成
            cycle_list = (
                self.df_bond.loc[
                    (self.df_bond["atom_id"].isin(aroma_atoms[:, 0]))
                    & (self.df_bond["bond_atom_id"].isin(aroma_atoms[:, 0])),
                    ["atom_id", "bond_atom_id"],
                ]
                .to_numpy()
                .tolist()
            )
            graph = nx.Graph()
            graph.add_edges_from(cycle_list)
            cycle_atom_id = [c for c in nx.cycle_basis(graph) if atom_id in c]
            return cycle_atom_id

        # 原子 Atom1, Atom2 が属する芳香環を構成する原子を取得
        atom1_cycles = get_cycle_atom(aroma_atoms1, atom1_id)
        atom2_cycles = get_cycle_atom(aroma_atoms2, atom2_id)

        # atom1の芳香環を形成するatom_idリスト全てに対して処理
        for atom1_cycle in atom1_cycles:
            atom1_bond_aroma_atom = [
                c for c in aroma_atoms1 if c[0] in atom1_bonds and c[0] in atom1_cycle
            ]

            atom1_ar1 = atom1_bond_aroma_atom[0][1:]
            atom1_ar2 = atom1_bond_aroma_atom[1][1:]

            # 原子 Atom2 から原子 Atom1 が属する芳香環面に垂線を下した交点
            n1 = my_math.intersection_point_vertical_line_and_plane(
                atom1_ar1,
                atom1_coord,
                atom1_ar2,
                atom2_coord,
            )
            angle2 = my_math.angle_three_points(atom2_coord, atom1_coord, n1)
            if angle2 < angle2_param:
                continue

            # 原子 Atom1 が属する芳香環面がなす法線ベクトル
            nv1 = my_math.normal_vector_three_points(atom1_ar1, atom1_coord, atom1_ar2)

            # atom2の芳香環を形成するatom_idリスト全てに対して処理
            for atom2_cycle in atom2_cycles:
                atom2_bond_aroma_atom = [
                    c for c in aroma_atoms2 if c[0] in atom2_bonds and c[0] in atom2_cycle
                ]

                atom2_ar1 = atom2_bond_aroma_atom[0][1:]
                atom2_ar2 = atom2_bond_aroma_atom[1][1:]

                # 原子 Atom2 が属する芳香環面がなす法線ベクトル
                nv2 = my_math.normal_vector_three_points(atom2_ar1, atom2_coord, atom2_ar2)

                # 芳香環面に対する法線ベクトルがなす角
                angle1 = my_math.angle_two_vectors(nv1, nv2)
                angle = angle1 if angle1 <= 90 else 180 - angle1

                if angle <= angle1_param:
                    self.update_interaction_table(
                        label="PI_PI",
                        atom1_id=atom1_id,
                        atom1_molcular_type=atom1_molcular_type,
                        atom2_id=atom2_id,
                        atom2_molcular_type=atom2_molcular_type,
                        item_1=distance,
                        item_2=angle1,
                        item_3=angle2,
                    )
                    return None

    def calc_dipo_dipo(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom1_charge,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
        atom2_charge,
    ):
        """双極子-双極子相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom1_charge (float): atom1の電荷
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
            atom2_charge (float): atom2の電荷
        """
        # 閾値取得
        buffer_param = self.interaction_parameter["Dipo"][0]
        angle1_param = self.interaction_parameter["Dipo"][1]
        angle2_param = self.interaction_parameter["Dipo"][2]
        angle3_param = self.interaction_parameter["Dipo"][3]
        charge_param = self.interaction_parameter["Dipo"][4]
        add_hydro_param = self.interaction_parameter["Dipo"][5]

        # AtomType条件
        atom1_atom_type_cond = ["O.2", "O.co2", "N.1", "S.2", "F", "Cl", "Br", "I"]
        atom2_atom_type_cond = ["C.2", "C.1", "C.ar", "S.o2", "S.o", "P.3"]
        if add_hydro_param == "add":
            atom1_atom_type_cond.append("H")
            atom2_atom_type_cond.append("H")

        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 距離1
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)

        atom1_symbol = atom1_atom_type.split(".")[0]
        atom2_symbol = atom2_atom_type.split(".")[0]

        vdw_radius = Decimal(str(self.vdw_difine[atom1_symbol])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )

        # 距離1閾値判定
        if distance1 > buffer_param + float(vdw_radius):
            return None

        # Atom1のBond
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()

        if add_hydro_param == "except":
            atom1_bonds = self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom1_bonds) & ~(self.df_atom["atom_type"] == "H"),
                ["atom_id", "charge", "x", "y", "z"],
            ].to_numpy()
        else:
            atom1_bonds = self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom1_bonds),
                ["atom_id", "charge", "x", "y", "z"],
            ].to_numpy()

        # 結合先原子が空である場合
        if len(atom1_bonds) == 0:
            return None

        # Atom2のBond
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if add_hydro_param == "except":
            atom2_bonds = self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom2_bonds) & ~(self.df_atom["atom_type"] == "H"),
                ["atom_id", "charge", "x", "y", "z"],
            ].to_numpy()
        else:
            atom2_bonds = self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom2_bonds),
                ["atom_id", "charge", "x", "y", "z"],
            ].to_numpy()

        # 結合先原子が空である場合
        if len(atom2_bonds) == 0:
            return None

        # atom1 と共有結合する全ての原子に対して判定処理を実施
        for atom1_bond in atom1_bonds:
            atom3_id = int(atom1_bond[0])
            atom3_charge = atom1_bond[1]
            atom3_coord = atom1_bond[2:]

            if atom3_charge * atom1_charge > 0:
                continue
            # 電荷1
            e1 = abs(atom1_charge - atom3_charge)
            if e1 < charge_param:
                continue

            # 重心
            c1 = my_math.center_of_gravity([atom1_coord, atom3_coord])
            # atom1とatom3のなすベクトル
            if atom1_charge > 0:
                v1 = my_math.vector_two_points(atom1_coord, atom3_coord)
            else:
                v1 = my_math.vector_two_points(atom3_coord, atom1_coord)

            # atom2 と共有結合する全ての原子に対して判定処理を実施
            for atom2_bond in atom2_bonds:
                atom4_id = int(atom2_bond[0])
                atom4_charge = atom2_bond[1]
                atom4_coord = atom2_bond[2:]

                # 双極子ではない
                if atom3_id == atom4_id:
                    continue

                if atom4_charge * atom2_charge > 0:
                    continue
                # 電荷2
                e2 = abs(atom2_charge - atom4_charge)
                if e2 < charge_param:
                    continue

                # 重心
                c2 = my_math.center_of_gravity([atom2_coord, atom4_coord])
                # 距離
                distance2 = my_math.distance_two_points(c1, c2)
                # 距離2閾値判定
                if distance2 > buffer_param + float(vdw_radius):
                    continue

                # atom2とatom4のなすベクトル
                if atom2_charge > 0:
                    v2 = my_math.vector_two_points(atom2_coord, atom4_coord)
                else:
                    v2 = my_math.vector_two_points(atom4_coord, atom2_coord)

                # v1とv2のなす角
                angle1 = my_math.angle_two_vectors(v1, v2)
                angle2 = my_math.angle_three_points(atom1_coord, atom4_coord, atom2_coord)
                angle3 = my_math.angle_three_points(atom4_coord, atom2_coord, atom3_coord)

                if all([angle1 >= angle1_param, angle2 <= angle2_param, angle3 <= angle3_param]):

                    # NOTE 双極子-双極子 相互作用は、Bond間の相互作用であるため各原子間で相互作用情報を定義する。
                    self.update_interaction_table(
                        label=f"Dipo_{atom1_id}_{atom2_id}",
                        atom1_id=atom1_id,
                        atom1_molcular_type=atom1_molcular_type,
                        atom2_id=atom2_id,
                        atom2_molcular_type=atom2_molcular_type,
                        item_1=distance1,
                        item_2=distance2,
                        item_3=angle1,
                        item_4=angle2,
                        item_5=angle3,
                        item_6=atom1_charge,
                        item_7=atom2_charge,
                    )
                    self.update_interaction_table(
                        label=f"Dipo_{atom1_id}_{atom2_id}",
                        atom1_id=atom3_id,
                        atom1_molcular_type=atom1_molcular_type,
                        atom2_id=atom4_id,
                        atom2_molcular_type=atom2_molcular_type,
                        item_1=distance1,
                        item_2=distance2,
                        item_3=angle1,
                        item_4=angle2,
                        item_5=angle3,
                        item_6=atom3_charge,
                        item_7=atom4_charge,
                    )
                    return None

    def calc_omulpol(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom1_charge,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
        atom2_charge,
    ):
        """直交多極子相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom1_charge (float): atom1の電荷
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
            atom2_charge (float): atom2の電荷
        """
        # 閾値取得
        buffer_param = self.interaction_parameter["OMulPol"][0]
        angle1_min_param = self.interaction_parameter["OMulPol"][1]
        angle1_max_param = self.interaction_parameter["OMulPol"][2]
        angle2_param = self.interaction_parameter["OMulPol"][3]
        angle3_param = self.interaction_parameter["OMulPol"][4]
        charge_param = self.interaction_parameter["OMulPol"][5]

        # AtomType条件判定
        atom1_atom_type_cond = ["O.2", "O.co2", "F", "Cl", "Br", "I", "N.1", "S.2"]
        atom2_atom_type_cond = ["C.ar", "C.2", "C.1"]

        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 電荷判定
        if atom1_charge > 0 or atom2_charge < 0:
            return

        # 距離1 閾値判定
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        atom1_symbol = atom1_atom_type.split(".")[0]
        atom2_symbol = atom2_atom_type.split(".")[0]
        vdw_radius = Decimal(str(self.vdw_difine[atom1_symbol])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )
        if distance1 > buffer_param + float(vdw_radius):
            return None

        # Atom1 の Bond条件判定
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        atom1_bonds = self.df_atom.loc[
            self.df_atom["atom_id"].isin(atom1_bonds) & ~(self.df_atom["atom_type"] == "H"),
            ["charge", "x", "y", "z"],
        ].to_numpy()
        if len(atom1_bonds) != 1:
            return

        atom3_charge = atom1_bonds[0][0]
        atom3_coord = atom1_bonds[0][1:]
        e1 = abs(atom1_charge - atom3_charge)
        distance2 = my_math.distance_two_points(atom2_coord, atom3_coord)
        if atom1_charge * atom3_charge > 0 or e1 < charge_param or distance1 > distance2:
            return

        # Atom2 と共有結合している原子を取得
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        atom2_bonds = self.df_atom.loc[
            self.df_atom["atom_id"].isin(atom2_bonds) & ~(self.df_atom["atom_type"] == "H"),
            ["atom_id", "charge", "x", "y", "z"],
        ].to_numpy()

        is_true = False
        if len(atom2_bonds) == 1:
            atom4_id = int(atom2_bonds[0][0])
            atom4_charge = atom2_bonds[0][1]
            atom4_coord = atom2_bonds[0][2:]
            e2 = abs(atom2_charge - atom4_charge)
            if atom2_charge * atom4_charge > 0 or e2 < charge_param:
                return

            angle1 = my_math.angle_three_points(atom1_coord, atom2_coord, atom4_coord)
            if angle1 < angle1_min_param or angle1 > angle1_max_param:
                return

            atom4_bonds = self.df_bond.loc[
                (self.df_bond["atom_id"] == atom4_id) & ~(self.df_bond["bond_atom_id"] == atom2_id),
                "bond_atom_id",
            ].tolist()
            atom4_bond_coords = self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom4_bonds) & ~(self.df_atom["atom_type"] == "H"),
                ["x", "y", "z"],
            ].to_numpy()

            for atom4_bond_coord in atom4_bond_coords:
                np = my_math.intersection_point_vertical_line_and_plane(
                    atom4_bond_coord,
                    atom4_coord,
                    atom2_coord,
                    atom1_coord,
                )
                angle2 = my_math.angle_three_points(np, atom1_coord, atom2_coord)
                angle3 = my_math.angle_three_points(np, atom1_coord, atom3_coord)

                if angle2 <= angle2_param and angle3 >= angle3_param:
                    is_true = True

        else:
            # atom2 に共有結合する原子群の直積で組み合わせを算出
            # # atom4: 重原子。atom2 と電荷符号が逆
            # # atom5: 重原子。
            for atom4, atom5 in itertools.product(atom2_bonds, atom2_bonds):
                if atom4[0] == atom5[0]:
                    continue

                e2 = abs(atom2_charge - atom4[1])
                if atom2_charge * atom4[1] > 0 or e2 < charge_param:
                    continue

                angle1 = my_math.angle_three_points(atom1_coord, atom2_coord, atom4[2:])
                np = my_math.intersection_point_vertical_line_and_plane(
                    atom4[2:], atom2_coord, atom5[2:], atom1_coord
                )
                angle2 = my_math.angle_three_points(np, atom1_coord, atom2_coord)
                angle3 = my_math.angle_three_points(np, atom1_coord, atom3_coord)
                if (
                    angle1_min_param <= angle1 <= angle1_max_param
                    and angle2 <= angle2_param
                    and angle3 >= angle3_param
                ):
                    is_true = True
                    break

        if is_true:
            self.update_interaction_table(
                label="OMulPol",
                atom1_id=atom1_id,
                atom1_molcular_type=atom1_molcular_type,
                atom2_id=atom2_id,
                atom2_molcular_type=atom2_molcular_type,
                item_1=distance1,
                item_2=distance2,
                item_3=angle1,
                item_4=angle2,
                item_5=angle3,
                item_6=e1,
                item_7=e2,
            )

    def calc_xh_pi(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_resi,
        atom2_molcular_type,
    ):
        """CH_PI, OH_PI, NH_PI, SH_PI 相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_resi (int): atom1 の残基番号
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = [
            "O.3",
            "O.spc",
            "O.t3p",
            "C.3",
            "C.2",
            "C.1",
            "C.ar",
            "N.3",
            "N.2",
            "N.pl3",
            "N.am",
            "N.4",
            "N.ar",
            "N.1",
            "S.3",
            "S.o2",
            "S.o",
        ]
        atom2_atom_type_cond = ["C.ar", "N.ar", "C.2", "N.2", "N.pl3", "O.3", "S.3", "N.am"]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        atom1_symbol = atom1_atom_type.split(".")[0]
        atom2_symbol = atom2_atom_type.split(".")[0]

        # 閾値取得
        buffer_param = self.interaction_parameter[f"{atom1_symbol}H_PI"][0]
        coef = self.interaction_parameter[f"{atom1_symbol}H_PI"][1]
        dist1_param = self.interaction_parameter[f"{atom1_symbol}H_PI"][2]
        dist2_param = self.interaction_parameter[f"{atom1_symbol}H_PI"][3]
        angle1_param = self.interaction_parameter[f"{atom1_symbol}H_PI"][4]
        angle2_param = self.interaction_parameter[f"{atom1_symbol}H_PI"][5]

        # 2点間距離
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        # 距離1閾値判定
        vdw_radius = Decimal(str(self.vdw_difine[atom1_symbol])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )

        if distance1 > buffer_param + float(vdw_radius):
            return None

        # atom1  BOND条件
        # 共有結合で隣接する原子数が「2以上」である。
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        atom1_bonds = (
            self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom1_bonds),
                ["atom_id", "atom_type", "x", "y", "z"],
            ]
            .to_numpy()
            .tolist()
        )

        # atom2  BOND条件
        # 共有結合で隣接する原子数が「2以上」である。
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) < 2:
            return

        # 原子2 と原子1 に共有結合している原子間の距離条件判定
        is_true = False
        distance2 = np.nan
        distance3 = np.nan
        angle2 = np.nan
        atom1_h_coord = None
        # atom1 に共有結合する原子群の直積で組み合わせを算出
        # CH_PI, NH_PI, OH_PI の組み合わせ
        # # 組み合わせ条件1: (atom_x, atom_h) = (重原子, 水素原子)
        # # 組み合わせ条件2: (atom_x, atom_h) = (水素原子, 水素原子) *共有結合する原子群に重原子が存在しない場合
        # SH_PI の組み合わせ
        # # 組み合わせ条件1: (atom_x, atom_h) = (重原子, 水素原子)
        for atom1_x, atom1_h in itertools.product(atom1_bonds, atom1_bonds):
            if atom1_x[0] == atom1_h[0] or atom1_h[1] != "H":
                continue
            elif atom1_symbol == "S" and atom1_x[1] == "H":
                continue

            distance2 = my_math.distance_two_points(atom1_x[2:], atom2_coord)
            distance3 = my_math.distance_two_points(atom1_h[2:], atom2_coord)
            angle2 = my_math.angle_three_points(atom1_coord, atom1_h[2:], atom2_coord)
            if distance3 <= distance1 <= distance2:
                if any(
                    [
                        distance3 <= dist1_param,
                        dist1_param < distance3 <= dist2_param and angle2 >= angle2_param,
                    ]
                ):
                    is_true = True
                    atom1_h_coord = atom1_h[2:]
                    break

        if not is_true:
            return

        # atom2 が属する芳香環原子を抽出
        aroma_atoms = self.df_atom.loc[
            (self.df_atom["resi"] == atom2_resi)
            & (self.df_atom["atom_type"].isin(atom2_atom_type_cond)),
            ["atom_id", "x", "y", "z"],
        ].to_numpy()

        # networkx形成
        cycle_list = (
            self.df_bond.loc[
                (self.df_bond["atom_id"].isin(aroma_atoms[:, 0]))
                & (self.df_bond["bond_atom_id"].isin(aroma_atoms[:, 0])),
                ["atom_id", "bond_atom_id"],
            ]
            .to_numpy()
            .tolist()
        )

        graph = nx.Graph()
        graph.add_edges_from(cycle_list)
        cycle_atom_id = [c for c in nx.cycle_basis(graph) if atom2_id in c]

        # 芳香環でfor文
        for cycle_atom in cycle_atom_id:
            cycle_aroma_coords = [ar[1:] for ar in aroma_atoms if ar[0] in cycle_atom]
            if len(cycle_aroma_coords) < 5 or len(cycle_aroma_coords) > 6:
                continue
            atom2_bond_aroma_atom = [
                ar[1:] for ar in aroma_atoms if ar[0] in atom2_bonds and ar[0] in cycle_atom
            ]

            # 芳香環の重心
            cn = my_math.center_of_gravity(cycle_aroma_coords)
            # 重心からの距離
            distance4 = my_math.distance_two_points(cn, atom2_coord)

            # 芳香環面に対して、原子1から垂線の下した点 Nrm
            nrm = my_math.intersection_point_vertical_line_and_plane(
                atom2_bond_aroma_atom[0], atom2_coord, atom2_bond_aroma_atom[1], atom1_coord
            )
            # 点 Nrm と重心との距離
            dnrm = my_math.distance_two_points(nrm, cn)
            angle1 = my_math.angle_three_points(nrm, atom1_coord, atom1_h_coord)

            if dnrm <= distance4 * coef and angle1 <= angle1_param:

                self.update_interaction_table(
                    label=f"{atom1_symbol}H_PI",
                    atom1_id=atom1_id,
                    atom1_molcular_type=atom1_molcular_type,
                    atom2_id=atom2_id,
                    atom2_molcular_type=atom2_molcular_type,
                    item_1=distance1,
                    item_2=distance2,
                    item_3=distance3,
                    item_4=distance4,
                    item_5=distance4 * coef,
                    item_6=dnrm,
                    item_7=angle1,
                    item_8=angle2,
                )
                return None

    def calc_ch_oh_nh_pi_two(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_resi,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """CH_PI, OH_PI, NH_PI 相互作用（旧定義）

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_resi (int): atom1 の残基番号
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_symbol = atom1_atom_type.split(".")[0]
        atom2_symbol = atom2_atom_type.split(".")[0]
        atom2_type_cond = [
            "C.3",
            "C.2",
            "C.1",
            "C.cat",
            "C.ar",
            "N.1",
            "N.2",
            "N.3",
            "N.4",
            "N.am",
            "N.pl3",
            "N.ar",
            "O.3",
            "O.2",
            "O.co2",
            "O.spc",
            "O.t3p",
            "O.ar",
        ]

        # AtomType条件
        if ".ar" not in atom1_atom_type or atom2_atom_type not in atom2_type_cond:
            return

        # 閾値判定：2点間距離
        vdw_radius = Decimal(str(self.vdw_difine[atom1_symbol])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        if distance1 > float(vdw_radius) + self.interaction_parameter[f"{atom2_symbol}H_PI"][0]:
            return None

        # atom2 BOND条件
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) == 0:
            return

        atom2_bonds = (
            self.df_atom.loc[
                (self.df_atom["atom_id"].isin(atom2_bonds)) & (self.df_atom["atom_type"] == "H"),
                ["x", "y", "z"],
            ]
            .to_numpy()
            .tolist()
        )
        if len(atom2_bonds) == 0:
            return

        # 原子 atom1 が属する芳香環原子の抽出
        aroma_atoms = self.df_atom.loc[
            (self.df_atom["resi"] == atom1_resi) & (self.df_atom["atom_type"].str.contains(".ar")),
            ["atom_id", "x", "y", "z"],
        ].to_numpy()
        cycle_list = (
            self.df_bond.loc[
                (self.df_bond["atom_id"].isin(aroma_atoms[:, 0]))
                & (self.df_bond["bond_atom_id"].isin(aroma_atoms[:, 0])),
                ["atom_id", "bond_atom_id"],
            ]
            .to_numpy()
            .tolist()
        )
        graph = nx.Graph()
        graph.add_edges_from(cycle_list)
        cycle_atom_ids = [c for c in nx.cycle_basis(graph) if atom1_id in c]

        # atom1 と隣接する芳香環原子
        atom1_bonds = self.df_bond.loc[
            (self.df_bond["atom_id"] == atom1_id)
            & (self.df_bond["bond_atom_id"].isin(aroma_atoms[:, 0])),
            "bond_atom_id",
        ].tolist()
        atom1_bond_ar = [ar[1:] for ar in aroma_atoms if ar[0] in atom1_bonds]

        for cycle_atom in cycle_atom_ids:
            # 芳香環の重心座標
            cycle_coords = [ar[1:] for ar in aroma_atoms if ar[0] in cycle_atom]
            c1 = my_math.center_of_gravity(cycle_coords)

            # 芳香環の重心と芳香環の各原子のとの間の最大距離+閾値距離
            r = 0
            for coord in cycle_coords:
                r_tmp = my_math.distance_two_points(c1, coord)
                r = r_tmp if r_tmp > r else r
            r = r + self.interaction_parameter[f"{atom2_symbol}H_PI"][0]

            # atom2 から atom1 の属する芳香環がなす平面に垂線を下した時の交点を取得
            p1 = my_math.intersection_point_vertical_line_and_plane(
                atom1_bond_ar[0], atom1_coord, atom1_bond_ar[1], atom2_coord
            )
            distance2 = my_math.distance_two_points(c1, p1)
            if distance2 > 2 * r:
                continue

            # atom1 を通る、芳香環面に垂直な法線ベクトルと atom2 との交点
            nv1 = my_math.normal_vector_three_points(
                atom1_bond_ar[0], atom1_coord, atom1_bond_ar[1]
            )
            p2 = my_math.intersection_point_vertical_line_and_line(nv1, atom1_coord, atom2_coord)

            for atom2_bond_coord in atom2_bonds:
                angle1 = my_math.angle_three_points(p2, atom2_coord, atom2_bond_coord)
                if (
                    not self.interaction_parameter[f"{atom2_symbol}H_PI"][1]
                    <= angle1
                    <= self.interaction_parameter[f"{atom2_symbol}H_PI"][2]
                ):
                    continue

                dihedral1 = my_math.dihedral_angle_four_points(
                    atom1_coord, p2, atom2_coord, atom2_bond_coord
                )
                if not dihedral1 <= self.interaction_parameter[f"{atom2_symbol}H_PI"][3]:
                    continue

                angle2 = None
                dihedral2 = None
                if r < distance2:
                    angle2 = my_math.angle_three_points(atom1_coord, c1, atom2_coord)
                    dihedral2 = my_math.dihedral_angle_four_points(c1, atom1_coord, p2, atom2_coord)
                    if not (
                        angle2 > 45
                        and dihedral2 > self.interaction_parameter[f"{atom2_symbol}H_PI"][4]
                    ):
                        continue

                self.update_interaction_table(
                    label=f"{atom2_symbol}H_PI",
                    atom1_id=atom1_id,
                    atom1_molcular_type=atom1_molcular_type,
                    atom2_id=atom2_id,
                    atom2_molcular_type=atom2_molcular_type,
                    item_1=distance1,
                    item_2=angle1,
                    item_3=angle2,
                    item_4=dihedral1,
                    item_5=dihedral2,
                )

    def calc_halogen_one(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """ハロゲン相互作用Ⅰ

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = ["Cl", "Br", "I"]
        atom2_atom_type_cond_1 = ["O.2", "O.co2", "N.1", "S.2"]
        atom2_atom_type_cond_2 = ["O.3", "O.spc", "O.tp3", "N.2", "N.ar", "S.3"]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond_1 + atom2_atom_type_cond_2
        ):
            return

        atom1_symbol = atom1_atom_type.split(".")[0]
        atom2_symbol = atom2_atom_type.split(".")[0]

        # 閾値取得
        buffer_param = self.interaction_parameter[f"Hal_(X)_{atom2_symbol}"][0]
        angle_param = self.interaction_parameter[f"Hal_(X)_{atom2_symbol}"][1]

        vdw_radius = Decimal(str(self.vdw_difine[atom1_symbol])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )

        # 原子1と原子2の距離を取得
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        if distance1 > buffer_param + float(vdw_radius):
            return

        # atom1  BOND条件
        # 共有結合で隣接する原子数が「1」である
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        if len(atom1_bonds) != 1:
            return
        atom1_bonds = (
            self.df_atom.loc[
                (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] != "H"),
                ["x", "y", "z"],
            ]
            .to_numpy()
            .tolist()
        )
        if len(atom1_bonds) == 0:
            return
        atom1_bond_coord = atom1_bonds[0]

        distance3 = my_math.distance_two_points(atom1_bond_coord, atom2_coord)
        angle1 = my_math.angle_three_points(atom1_coord, atom1_bond_coord, atom2_coord)
        if distance1 > distance3 or angle1 > angle_param:
            return

        # atom2 BOND条件
        # 共有結合で隣接する原子数が「1 or 2」である
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if any(
            [
                atom2_atom_type in atom2_atom_type_cond_1 and len(atom2_bonds) != 1,
                atom2_atom_type in atom2_atom_type_cond_2 and len(atom2_bonds) != 2,
            ]
        ):
            return

        atom2_bonds = self.df_atom.loc[
            self.df_atom["atom_id"].isin(atom2_bonds), ["atom_id", "atom_type", "x", "y", "z"]
        ]
        has_heaby_atom2_bond = len(atom2_bonds.loc[atom2_bonds["atom_type"] != "H"]) != 0
        atom2_bonds = atom2_bonds.to_numpy().tolist()

        if len(atom2_bonds) == 1 and has_heaby_atom2_bond:
            # 原子2に共有結合する原子が1個の場合
            distance2 = my_math.distance_two_points(atom1_coord, atom2_bonds[0][2:])
            if distance1 <= distance2:
                self.update_interaction_table(
                    label=f"Hal_{atom1_symbol}_{atom2_symbol}",
                    atom1_id=atom1_id,
                    atom1_molcular_type=atom1_molcular_type,
                    atom2_id=atom2_id,
                    atom2_molcular_type=atom2_molcular_type,
                    item_1=distance1,
                    item_2=distance2,
                    item_3=distance3,
                    item_4=angle1,
                )
                return None

        elif len(atom2_bonds) == 2:
            # 原子2に共有結合する原子が2個の場合
            # 組み合わせ条件1: (atom_x, atom_h) = (重原子, 重原子)
            # 組み合わせ条件2: (atom_x, atom_h) = (重原子, 水素原子)
            # 組み合わせ条件3: (atom_x, atom_h) = (水素原子, 水素原子) *共有結合する原子群に重原子が存在しない場合
            for atom2_x, atom2_y in itertools.product(atom2_bonds, atom2_bonds):
                if any([atom2_x[0] == atom2_y[0], has_heaby_atom2_bond and atom2_x[1] == "H"]):
                    continue
                distance2 = my_math.distance_two_points(atom1_coord, atom2_x[2:])
                distance4 = my_math.distance_two_points(atom1_coord, atom2_y[2:])
                distance5 = my_math.distance_two_points(atom1_bond_coord, atom2_y[2:])

                if distance1 <= distance2 and distance4 <= distance5:
                    self.update_interaction_table(
                        label=f"Hal_{atom1_symbol}_{atom2_symbol}",
                        atom1_id=atom1_id,
                        atom1_molcular_type=atom1_molcular_type,
                        atom2_id=atom2_id,
                        atom2_molcular_type=atom2_molcular_type,
                        item_1=distance1,
                        item_2=distance2,
                        item_3=distance3,
                        item_4=distance4,
                        item_5=distance5,
                        item_6=angle1,
                    )
                    return None

    def calc_xh_f(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """CH_F, NH_F, OH_F, SH_F 相互作用検出メソッド

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = [
            "C.1",
            "C.2",
            "C.3",
            "C.cat",
            "C.ar",
            "N.1",
            "N.2",
            "N.3",
            "N.4",
            "N.pl3",
            "N.am",
            "N.ar",
            "O.3",
            "O.spc",
            "O.t3p",
            "S.3",
            "S.o",
            "S.o2",
        ]
        atom2_atom_type_cond = ["F"]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        atom1_symbol = atom1_atom_type.split(".")[0]

        # 閾値取得
        buffer_param = self.interaction_parameter[f"{atom1_symbol}H_F"][0]
        dist1_param = self.interaction_parameter[f"{atom1_symbol}H_F"][1]
        angle1_param = self.interaction_parameter[f"{atom1_symbol}H_F"][2]
        angle2_param = self.interaction_parameter[f"{atom1_symbol}H_F"][3]

        vdw_radius = Decimal(str(self.vdw_difine[f"{atom1_symbol}"])) + Decimal(
            str(self.vdw_difine["F"])
        )

        # 原子1と原子2の距離を取得
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        if distance1 > buffer_param + float(vdw_radius):
            return

        # atom1  BOND条件
        # 共有結合で隣接する原子数が「2以上」である。
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        if len(atom1_bonds) < 2:
            return

        atom1_bonds = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom1_bonds)),
            ["atom_id", "atom_type", "x", "y", "z"],
        ]
        is_water = (
            atom1_symbol == "O" and len(atom1_bonds.loc[atom1_bonds["atom_type"] == "H"]) == 2
        )
        dist1_param = (
            self.interaction_parameter[f"{atom1_symbol}H_F"][4] if is_water else dist1_param
        )

        # atom2  BOND条件
        # 重原子1つと共有結合している。
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        atom2_heavy_bonds = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom2_bonds)) & (self.df_atom["atom_type"] != "H"),
            ["x", "y", "z"],
        ].to_numpy()
        if len(atom2_bonds) != len(atom2_heavy_bonds) or len(atom2_heavy_bonds) != 1:
            return

        distance2 = np.nan
        distance3 = np.nan
        angle1 = None

        atom1_bonds = atom1_bonds.to_numpy().tolist()
        # atom1 に共有結合する原子群の直積で組み合わせを算出
        # 組み合わせ条件1: (atom_x, atom_h) = (重原子, 水素原子)
        # 組み合わせ条件2: (atom_x, atom_h) = (水素原子, 水素原子) *原子1 が水分子の場合
        for atom_x, atom_h in itertools.product(atom1_bonds, atom1_bonds):
            if atom_x[0] == atom_h[0] or atom_h[1] != "H":
                continue
            elif not is_water and atom_x[1] == "H":
                continue

            distance2 = my_math.distance_two_points(atom_x[2:], atom2_coord)
            distance3 = my_math.distance_two_points(atom_h[2:], atom2_coord)
            angle1 = my_math.angle_three_points(atom_x[2:], atom1_coord, atom2_coord)
            angle2 = my_math.angle_three_points(atom1_coord, atom_h[2:], atom2_coord)
            if (
                distance1 <= distance2
                and distance3 <= distance1
                and angle1 <= angle1_param
                and (
                    distance3 <= dist1_param or (distance3 > dist1_param and angle2 > angle2_param)
                )
            ):
                self.update_interaction_table(
                    label=f"{atom1_symbol}H_F",
                    atom1_id=atom1_id,
                    atom1_molcular_type=atom1_molcular_type,
                    atom2_id=atom2_id,
                    atom2_molcular_type=atom2_molcular_type,
                    item_1=distance1,
                    item_2=distance2,
                    item_3=distance3,
                    item_4=angle1,
                    item_5=angle2,
                )
                return None

    def calc_xh_halogen(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """XH-ハロゲン相互作用(CH-Hal, OH-Hal, NH-Hal, SH-Hal)

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = [
            "C.3",
            "C.2",
            "C.1",
            "C.cat",
            "O.3",
            "O.spc",
            "O.t3p",
            "O.2",
            "S.3",
            "S.O",
            "S.O2",
            "S.2",
            "N.3",
            "N.2",
            "N.am",
            "N.pl3",
            "N.4",
            "N.ar",
            "N.1",
        ]
        atom2_atom_type_cond = ["Cl", "Br", "I"]

        atom1_symbol = atom1_atom_type.split(".")[0]
        atom2_symbol = atom2_atom_type.split(".")[0]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 閾値取得
        buffer_param = self.interaction_parameter[f"{atom1_symbol}H_Hal_(X)"][0]
        angle1_param = self.interaction_parameter[f"{atom1_symbol}H_Hal_(X)"][1]
        angle2_param = self.interaction_parameter[f"{atom1_symbol}H_Hal_(X)"][2]
        angle3_param = self.interaction_parameter[f"{atom1_symbol}H_Hal_(X)"][3]
        angle4_param = self.interaction_parameter[f"{atom1_symbol}H_Hal_(X)"][4]

        vdw_radius = Decimal(str(self.vdw_difine[atom1_symbol])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )

        # 原子1と原子2の距離を取得
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        if distance1 > buffer_param + float(vdw_radius):
            return

        # atom1  BOND条件
        # 共有結合で隣接する原子数が「2以上」
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        if len(atom1_bonds) < 2:
            return

        atom1_bonds = self.df_atom.loc[
            self.df_atom["atom_id"].isin(atom1_bonds),
            ["atom_id", "atom_type", "x", "y", "z"],
        ]
        has_heaby_atom1_bond = len(atom1_bonds.loc[atom1_bonds["atom_type"] != "H"]) != 0
        atom1_bonds = atom1_bonds.to_numpy().tolist()

        # atom2 BOND条件
        # 共有結合で隣接する原子数が「1つ」
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) != 1:
            return
        atom2_bonds = (
            self.df_atom.loc[
                (self.df_atom["atom_id"].isin(atom2_bonds)) & (self.df_atom["atom_type"] != "H"),
                ["x", "y", "z"],
            ]
            .to_numpy()
            .tolist()
        )
        if len(atom2_bonds) != 1:
            return
        atom2_bond_coord = atom2_bonds[0]

        angle2 = my_math.angle_three_points(atom2_bond_coord, atom2_coord, atom1_coord)
        if angle2 > angle2_param:
            return

        # atom1 に共有結合する2組の原子について、判定指標を算出
        distance2 = np.nan
        distance3 = np.nan
        angle1 = np.nan
        angle3 = np.nan
        angle4 = np.nan
        is_true = False
        # atom1 に共有結合する原子群の直積で組み合わせを算出
        # 組み合わせ条件1: (atom_x, atom_h) = (重原子, 水素原子)
        # 組み合わせ条件2: (atom_x, atom_h) = (水素原子, 水素原子) *共有結合する原子群に重原子が存在しない場合
        for atom1_x, atom1_h in itertools.product(atom1_bonds, atom1_bonds):
            if atom1_x[0] == atom1_h[0] or atom1_h[1] != "H":
                continue
            elif has_heaby_atom1_bond and atom1_x[1] == "H":
                continue

            tmp_distance2 = my_math.distance_two_points(atom1_x[2:], atom2_coord)
            if not np.isnan(distance2) and tmp_distance2 > tmp_distance2:
                continue

            # atom_x の候補が複数ある場合は、distance2 が最小となるものを採用する
            # より短い distance2 が算出されて場合は、すでに判定条件を満たしていても採用しない。
            is_true = False if tmp_distance2 < distance2 else is_true

            tmp_distance3 = my_math.distance_two_points(atom1_h[2:], atom2_coord)

            tmp_angle1 = my_math.angle_three_points(atom2_bond_coord, atom2_coord, atom1_h[2:])
            tmp_angle3 = my_math.angle_three_points(atom2_coord, atom1_coord, atom1_h[2:])
            tmp_angle4 = my_math.angle_three_points(atom2_bond_coord, atom1_coord, atom1_h[2:])

            if (
                distance1 <= tmp_distance2
                and tmp_distance3 <= distance1
                and tmp_angle1 <= angle1_param
                and tmp_angle3 <= angle3_param
                and tmp_angle4 <= angle4_param
            ):
                distance2 = tmp_distance2
                distance3 = tmp_distance3
                angle1 = tmp_angle1
                angle3 = tmp_angle3
                angle4 = tmp_angle4
                is_true = True

        if is_true:
            self.update_interaction_table(
                label=f"{atom1_symbol}H_Hal_{atom2_symbol}",
                atom1_id=atom1_id,
                atom1_molcular_type=atom1_molcular_type,
                atom2_id=atom2_id,
                atom2_molcular_type=atom2_molcular_type,
                item_1=distance1,
                item_2=distance2,
                item_3=distance3,
                item_4=angle1,
                item_5=angle2,
                item_6=angle3,
                item_7=angle4,
            )

    def calc_halogen_pi(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_resi,
        atom2_molcular_type,
    ):
        """ハロゲン_PI 相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_resi (int): atom1 の残基番号
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = ["Cl", "Br", "I"]
        atom2_atom_type_cond = ["C.ar", "N.ar", "C.2", "N.2", "N.pl3", "O.3", "S.3", "N.am"]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 閾値取得
        buffer_param = self.interaction_parameter["Hal_PI_(X)"][0]
        coef = self.interaction_parameter["Hal_PI_(X)"][1]
        angle1_param = self.interaction_parameter["Hal_PI_(X)"][2]
        dihedral_param = self.interaction_parameter["Hal_PI_(X)"][3]

        # 2点間距離
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        # 距離1閾値判定
        atom1_symbol = atom1_atom_type.split(".")[0]
        atom2_symbol = atom2_atom_type.split(".")[0]
        vdw_radius = Decimal(str(self.vdw_difine[atom1_symbol])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )

        if distance1 > buffer_param + float(vdw_radius):
            return None

        # atom1  BOND条件
        # 共有結合で隣接する原子数が「1」である。
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        if len(atom1_bonds) != 1:
            return
        atom1_bond_coord = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] != "H"),
            ["x", "y", "z"],
        ].to_numpy()
        if len(atom1_bond_coord) != 1:
            return
        atom1_bond_coord = atom1_bond_coord[0]

        distance2 = my_math.distance_two_points(atom1_bond_coord, atom2_coord)
        if distance1 > distance2:
            return

        # atom2  BOND条件
        # 共有結合で隣接する原子数が「2以上」である。
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) < 2:
            return

        # atom2 が属する芳香環原子を抽出
        aroma_atoms = self.df_atom.loc[
            (self.df_atom["resi"] == atom2_resi)
            & (self.df_atom["atom_type"].isin(atom2_atom_type_cond)),
            ["atom_id", "x", "y", "z"],
        ].to_numpy()

        # networkx形成
        cycle_list = (
            self.df_bond.loc[
                (self.df_bond["atom_id"].isin(aroma_atoms[:, 0]))
                & (self.df_bond["bond_atom_id"].isin(aroma_atoms[:, 0])),
                ["atom_id", "bond_atom_id"],
            ]
            .to_numpy()
            .tolist()
        )

        graph = nx.Graph()
        graph.add_edges_from(cycle_list)
        cycle_atom_id = [c for c in nx.cycle_basis(graph) if atom2_id in c]

        # 芳香環でfor文
        for cycle_atom in cycle_atom_id:
            cycle_aroma_coords = [ar[1:] for ar in aroma_atoms if ar[0] in cycle_atom]
            if len(cycle_aroma_coords) < 5 or len(cycle_aroma_coords) > 6:
                continue
            atom2_bond_aroma_atom = [
                ar[1:] for ar in aroma_atoms if ar[0] in atom2_bonds and ar[0] in cycle_atom
            ]

            # 芳香環の重心
            cn = my_math.center_of_gravity(cycle_aroma_coords)
            # 重心との各原子との距離
            distance3 = my_math.distance_two_points(cn, atom2_coord)
            distance5 = my_math.distance_two_points(cn, atom1_coord)
            distance6 = my_math.distance_two_points(cn, atom1_bond_coord)

            # 芳香環面に対して、原子1から垂線の下した点 Nrm
            nrm = my_math.intersection_point_vertical_line_and_plane(
                atom2_bond_aroma_atom[0], atom2_coord, atom2_bond_aroma_atom[1], atom1_coord
            )
            # 点 Nrm と重心との距離
            dnrm = my_math.distance_two_points(nrm, cn)

            angle1 = my_math.angle_three_points(nrm, atom1_coord, atom1_bond_coord)

            dihedral = my_math.dihedral_angle_four_points(cn, nrm, atom1_coord, atom1_bond_coord)

            if (
                dnrm <= distance3 * coef
                and distance5 <= distance6
                and angle1_param <= angle1
                and dihedral_param <= dihedral
            ):
                self.update_interaction_table(
                    label=f"Hal_PI_{atom1_symbol}",
                    atom1_id=atom1_id,
                    atom1_molcular_type=atom1_molcular_type,
                    atom2_id=atom2_id,
                    atom2_molcular_type=atom2_molcular_type,
                    item_1=distance1,
                    item_2=distance2,
                    item_3=distance3,
                    item_4=distance3 * coef,
                    item_5=dnrm,
                    item_6=distance5,
                    item_7=distance6,
                    item_8=angle1,
                    item_9=dihedral,
                )
                return None

    def calc_nh_s(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """NH_S 相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = ["N.1", "N.2", "N.3", "N.4", "N.am", "N.pl3", "N.ar"]
        atom2_atom_type_cond = ["S.2", "S.3", "S.ar"]
        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 閾値取得
        buffer_param = self.interaction_parameter["NH_S"][0]
        dist1_param = self.interaction_parameter["NH_S"][1]
        dist2_param = self.interaction_parameter["NH_S"][2]
        angle1_min_param = self.interaction_parameter["NH_S"][3]
        angle1_max_param = self.interaction_parameter["NH_S"][4]
        angle2_min_param = self.interaction_parameter["NH_S"][5]
        angle2_max_param = self.interaction_parameter["NH_S"][6]

        # 距離1 判定
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        vdw_radius = Decimal(str(self.vdw_difine["N"])) + Decimal(str(self.vdw_difine["S"]))
        if distance1 > buffer_param + float(vdw_radius):
            return

        # atom1 と共有結合している原子を取得
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        atom1_bonds = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom1_bonds)),
            ["atom_id", "atom_type", "x", "y", "z"],
        ].to_numpy()
        if len(atom1_bonds) != 2:
            return

        is_true = False
        distance2 = None
        distance3 = None
        angle1 = None
        for atom1_x, atom1_h in itertools.product(atom1_bonds, atom1_bonds):
            if atom1_x[0] == atom1_h[0] or atom1_x[1] == "H" or atom1_h[1] != "H":
                continue

            distance2 = my_math.distance_two_points(atom1_x[2:], atom2_coord)
            distance3 = my_math.distance_two_points(atom1_h[2:], atom2_coord)
            angle1 = my_math.angle_three_points(atom1_coord, atom1_h[2:], atom2_coord)
            if distance3 <= distance1 <= distance2 and any(
                [
                    distance3 <= dist1_param and angle1_min_param <= angle1 <= angle1_max_param,
                    dist1_param < distance3 <= dist2_param
                    and angle2_min_param <= angle1 <= angle2_max_param,
                ]
            ):
                is_true = True
                break
        if not is_true:
            return

        # atom2 と共有結合している原子を取得
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        atom2_bonds = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom2_bonds)),
            ["atom_id", "atom_type", "x", "y", "z"],
        ].to_numpy()

        is_true = False
        distance4 = None
        distance5 = None
        if atom2_atom_type == "S.2" and len(atom2_bonds) == 1 and atom2_bonds[0][1] != "H":
            distance4 = my_math.distance_two_points(atom2_bonds[0][2:], atom1_coord)
            is_true = distance1 <= distance4

        elif atom2_atom_type in ["S.3", "S.ar"]:
            for atom2_y, atom2_z in itertools.product(atom2_bonds, atom2_bonds):
                if atom2_y[0] == atom2_z[0] or atom2_y[1] == "H":
                    continue

                distance4 = my_math.distance_two_points(atom2_y[2:], atom1_coord)
                distance5 = my_math.distance_two_points(atom2_z[2:], atom1_coord)
                if distance1 <= distance4 and distance1 <= distance5:
                    is_true = True
                    break

        if is_true:
            self.update_interaction_table(
                label="NH_S",
                atom1_id=atom1_id,
                atom1_molcular_type=atom1_molcular_type,
                atom2_id=atom2_id,
                atom2_molcular_type=atom2_molcular_type,
                item_1=distance1,
                item_2=distance2,
                item_3=distance3,
                item_4=distance4,
                item_5=distance5,
                item_6=angle1,
            )

    def calc_oh_s_sh_s(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """OH_S, SH_S 相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = ["O.2", "O.3", "O.co2", "O.ar", "S.2", "S.3", "S.o", "S.o2"]
        atom2_atom_type_cond = [
            "S.2",
            "S.3",
            "S.ar",
        ]
        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        atom1_symbol = atom1_atom_type.split(".")[0]
        label = f"{atom1_symbol}H_S"

        vdw_radius = Decimal(str(self.vdw_difine[atom1_symbol])) + Decimal(
            str(self.vdw_difine["S"])
        )

        # 原子1と原子2の距離を取得
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        if distance1 > self.interaction_parameter[label][0] + float(vdw_radius):
            return None

        # atom1  BOND条件
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        if len(atom1_bonds) == 0:
            return

        # atom2  BOND条件
        # 共有結合で隣接する原子数が「1以上」である。
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) == 0:
            return

        atom1_bonds_x = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] != "H"),
            ["x", "y", "z"],
        ].to_numpy()
        if len(atom1_bonds_x) == 0:
            return
        atom1_bonds_h = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] == "H"),
            ["x", "y", "z"],
        ].to_numpy()
        if len(atom1_bonds_h) == 0:
            return

        atom2_bonds = self.df_atom.loc[
            self.df_atom["atom_id"].isin(atom2_bonds), ["atom_type", "x", "y", "z"]
        ].to_numpy()

        # atom1 と共有結合する全ての重原子に対して判定処理を実施
        x_flag = False
        for atom1_bond_x in atom1_bonds_x:
            atom1_bond_x_coord = atom1_bond_x[0:]
            # 原子1のBondXと原子2の距離を取得
            distance2 = my_math.distance_two_points(atom1_bond_x_coord, atom2_coord)
            if distance1 >= distance2:
                continue

            angle = my_math.angle_three_points(atom1_bond_x_coord, atom1_coord, atom2_coord)
            # 閾値判定
            if angle > self.interaction_parameter[label][1]:
                continue
            x_flag = True
            break

        if not x_flag:
            return

        # atom1 と共有結合する全ての水素原子に対して判定処理を実施
        h_flag = False
        for atom1_bond_h in atom1_bonds_h:
            atom1_bond_h_coord = atom1_bond_h[0:]
            # 原子1のBondXと原子2の距離を取得
            distance3 = my_math.distance_two_points(atom1_bond_h_coord, atom2_coord)
            if distance3 >= distance1:
                continue
            h_flag = True
            break

        if not h_flag:
            return

        # atom2 と共有結合する全ての原子に対して判定処理を実施
        atom2_flag = False
        if len(atom2_bonds) == 1:
            if atom2_bonds[0][0] == "H":
                return
            distance4 = my_math.distance_two_points(atom1_coord, atom2_bonds[0][1:])
            if distance1 >= distance4:
                return
            distance5 = np.nan
            atom2_flag = True
        else:
            for atom2_bond_pair in itertools.combinations(atom2_bonds, 2):
                if atom2_bond_pair[0][0] == "H" and atom2_bond_pair[1][0] == "H":
                    continue
                atom2_z = atom2_bond_pair[0][1:]
                atom2_y = atom2_bond_pair[1][1:]
                if atom2_bond_pair[0][0] == "H" and atom2_bond_pair[1][0] != "H":
                    atom2_z = atom2_bond_pair[1][1:]
                    atom2_y = atom2_bond_pair[0][1:]
                distance4 = my_math.distance_two_points(atom1_coord, atom2_z)
                if distance1 >= distance4:
                    continue
                distance5 = my_math.distance_two_points(atom1_coord, atom2_y)
                if distance1 >= distance5:
                    continue
                atom2_flag = True
                break

        if atom2_flag:
            self.update_interaction_table(
                label=label,
                atom1_id=atom1_id,
                atom1_molcular_type=atom1_molcular_type,
                atom2_id=atom2_id,
                atom2_molcular_type=atom2_molcular_type,
                item_1=distance1,
                item_2=distance2,
                item_3=distance3,
                item_4=distance4,
                item_5=distance5,
                item_6=angle,
            )

    def calc_s_o(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """S_O 相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = ["S.2", "S.3"]
        atom2_atom_type_cond = ["O.2", "O.3", "O.co2"]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 閾値取得
        buffer_param = self.interaction_parameter["S_O"][0]

        # 距離
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        # 距離1閾値判定
        vdw_radius = Decimal(str(self.vdw_difine["S"])) + Decimal(str(self.vdw_difine["O"]))

        if distance1 > buffer_param + float(vdw_radius):
            return None

        # atom1  BOND条件
        # 共有結合で隣接する原子数が「2以上」である。
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        if len(atom1_bonds) < 2:
            return

        # atom2  BOND条件
        # 共有結合で隣接する原子数が「1 or 2」である。
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        atom2_bond_atoms = None
        if atom2_atom_type == "O.3" and len(atom2_bonds) == 2:
            atom2_bond_atoms = self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom2_bonds), ["atom_id", "x", "y", "z"]
            ].to_numpy()

        elif atom2_atom_type != "O.3" and len(atom2_bonds) == 1:
            atom2_bond_atoms = self.df_atom.loc[
                (self.df_atom["atom_id"].isin(atom2_bonds)) & (self.df_atom["atom_type"] != "H"),
                ["atom_type", "x", "y", "z"],
            ].to_numpy()

        else:
            return

        atom1_bonds_x = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] != "H"),
            ["atom_id", "x", "y", "z"],
        ].to_numpy()
        atom1_bonds_w = self.df_atom.loc[
            self.df_atom["atom_id"].isin(atom1_bonds), ["atom_id", "x", "y", "z"]
        ].to_numpy()

        is_true = False

        # 原子2 と原子1 に共有結合している原子間の距離条件判定
        distance2 = np.nan
        distance3 = np.nan
        for atom1_x, atom1_w in itertools.product(atom1_bonds_x, atom1_bonds_w):
            if atom1_x[0] == atom1_w[0]:
                continue
            distance2 = my_math.distance_two_points(atom1_x[1:], atom2_coord)
            distance3 = my_math.distance_two_points(atom1_w[1:], atom2_coord)
            if distance1 <= distance2 and distance1 <= distance3:
                is_true = True
                break
        if not is_true:
            return

        # 原子1 と原子2 に共有結合している原子間の距離条件判定
        distance4 = np.nan
        distance5 = np.nan
        if len(atom2_bond_atoms) == 1:
            distance4 = my_math.distance_two_points(atom2_bond_atoms[0][1:], atom1_coord)
            if distance1 > distance4:
                return
        elif len(atom2_bond_atoms) == 2:
            is_true = False
            for atom2_bond_pair in itertools.combinations(atom2_bond_atoms, 2):
                atom2_x = atom2_bond_pair[0][1:]
                atom2_y = atom2_bond_pair[1][1:]
                # atom2_x を優先的に重原子にする
                if atom2_bond_pair[0][0] == "H" and atom2_bond_pair[1][0] != "H":
                    atom2_x = atom2_bond_pair[1][1:]
                    atom2_y = atom2_bond_pair[0][1:]
                distance4 = my_math.distance_two_points(atom2_x, atom1_coord)
                distance5 = my_math.distance_two_points(atom2_y, atom1_coord)
                if distance1 <= distance4 and distance1 <= distance5:
                    is_true = True
                    break

        if is_true:
            self.update_interaction_table(
                label="S_O",
                atom1_id=atom1_id,
                atom1_molcular_type=atom1_molcular_type,
                atom2_id=atom2_id,
                atom2_molcular_type=atom2_molcular_type,
                item_1=distance1,
                item_2=distance2,
                item_3=distance3,
                item_4=distance4,
                item_5=distance5,
            )

    def calc_s_n(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """S_N 相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = ["S.3"]
        atom2_atom_type_cond = ["N.ar", "N.2"]
        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 閾値取得
        buffer_param = self.interaction_parameter["S_N"][0]

        # 距離1 判定
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        vdw_radius = Decimal(str(self.vdw_difine["N"])) + Decimal(str(self.vdw_difine["S"]))
        if distance1 > buffer_param + float(vdw_radius):
            return

        # atom1 と共有結合している原子を取得
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        atom1_bonds = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom1_bonds)),
            ["atom_id", "atom_type", "x", "y", "z"],
        ].to_numpy()
        if len(atom1_bonds) != 2:
            return

        is_true = False
        distance2 = None
        distance3 = None
        for atom1_x, atom1_y in itertools.product(atom1_bonds, atom1_bonds):
            if atom1_x[0] == atom1_y[0] or atom1_x[1] == "H":
                continue

            distance2 = my_math.distance_two_points(atom1_x[2:], atom2_coord)
            distance3 = my_math.distance_two_points(atom1_y[2:], atom2_coord)
            if distance1 <= distance2 and distance1 <= distance3:
                is_true = True
                break
        if not is_true:
            return

        # atom2 と共有結合している原子を取得
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        atom2_bonds = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom2_bonds)),
            ["atom_id", "atom_type", "x", "y", "z"],
        ].to_numpy()
        if len(atom2_bonds) != 2:
            return

        is_true = False
        distance4 = None
        distance5 = None
        for atom2_x, atom2_y in itertools.product(atom2_bonds, atom2_bonds):
            if atom2_x[0] == atom2_y[0] or atom2_x[1] == "H":
                continue

            distance4 = my_math.distance_two_points(atom2_x[2:], atom1_coord)
            distance5 = my_math.distance_two_points(atom2_y[2:], atom1_coord)
            if distance1 <= distance4 and distance1 <= distance5:
                is_true = True
                break

        if is_true:
            self.update_interaction_table(
                label="NH_S",
                atom1_id=atom1_id,
                atom1_molcular_type=atom1_molcular_type,
                atom2_id=atom2_id,
                atom2_molcular_type=atom2_molcular_type,
                item_1=distance1,
                item_2=distance2,
                item_3=distance3,
                item_4=distance4,
                item_5=distance5,
            )

    def calc_s_s(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """S_S 相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = ["S.2", "S.3", "S.ar"]
        atom2_atom_type_cond = ["S.2", "S.3", "S.ar"]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 閾値取得
        buffer_param = self.interaction_parameter["S_S"][0]
        angle1_param = self.interaction_parameter["S_S"][1]

        # 原子1と原子2の距離を取得
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        vdw_radius = Decimal(str(self.vdw_difine["S"])) + Decimal(str(self.vdw_difine["S"]))
        if distance1 > buffer_param + float(vdw_radius):
            return None

        # atom1  BOND条件
        # 共有結合で隣接する原子数が「1以上」である。
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        atom1_bonds = self.df_atom.loc[
            self.df_atom["atom_id"].isin(atom1_bonds),
            ["atom_id", "atom_type", "x", "y", "z"],
        ].to_numpy()

        # atom1 と共有結合する全ての原子に対して判定処理を実施
        atom1_flag = False
        distance2 = None
        distance3 = None
        if len(atom1_bonds) == 1 and atom1_bonds[0][1] != "H":
            distance2 = my_math.distance_two_points(atom2_coord, atom1_bonds[0][2:])
            angle = my_math.angle_three_points(atom1_bonds[0][2:], atom1_coord, atom2_coord)
            atom1_flag = distance1 < distance2 and angle <= angle1_param

        else:
            for atom1_x, atom1_w in itertools.product(atom1_bonds, atom1_bonds):
                if atom1_x[0] == atom1_w[0] or atom1_x[1] == "H":
                    continue

                distance2 = my_math.distance_two_points(atom2_coord, atom1_x[2:])
                distance3 = my_math.distance_two_points(atom2_coord, atom1_w[2:])
                angle = my_math.angle_three_points(atom1_x[2:], atom1_coord, atom2_coord)
                if distance1 < distance2 and distance1 < distance3 and angle <= angle1_param:
                    atom1_flag = True
                    break

        if not atom1_flag:
            return

        # atom2  BOND条件
        # 共有結合で隣接する原子数が「1以上」である。
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        atom2_bonds = self.df_atom.loc[
            self.df_atom["atom_id"].isin(atom2_bonds),
            ["atom_id", "atom_type", "x", "y", "z"],
        ].to_numpy()

        # atom2 と共有結合する全ての原子に対して判定処理を実施
        atom2_flag = False
        distance4 = None
        distance5 = None
        if len(atom2_bonds) == 1 and atom2_bonds[0][1] != "H":
            distance4 = my_math.distance_two_points(atom1_coord, atom2_bonds[0][2:])
            atom2_flag = distance1 < distance4

        else:
            for atom2_y, atom2_z in itertools.product(atom2_bonds, atom2_bonds):
                if atom2_y[0] == atom2_z[0] or atom2_y[1] == "H":
                    continue

                distance4 = my_math.distance_two_points(atom1_coord, atom2_z[2:])
                distance5 = my_math.distance_two_points(atom1_coord, atom2_y[2:])
                if distance1 < distance4 and distance1 < distance5:
                    atom2_flag = True
                    break

        if atom2_flag:
            self.update_interaction_table(
                label="S_S",
                atom1_id=atom1_id,
                atom1_molcular_type=atom1_molcular_type,
                atom2_id=atom2_id,
                atom2_molcular_type=atom2_molcular_type,
                item_1=distance1,
                item_2=distance2,
                item_3=distance3,
                item_4=distance4,
                item_5=distance5,
                item_6=angle,
            )

    def calc_s_f(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """S_F 相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = ["S.2", "S.3", "S.ar"]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if atom1_atom_type not in atom1_atom_type_cond or atom2_atom_type != "F":
            return

        # 距離1
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)

        vdw_radius = Decimal(str(self.vdw_difine["S"])) + Decimal(str(self.vdw_difine["F"]))

        # 距離1閾値判定
        if distance1 > self.interaction_parameter["S_F"][0] + float(vdw_radius):
            return None

        # atom1  BOND条件
        # 共有結合で隣接する原子数が「1以上」である。
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        if len(atom1_bonds) == 0:
            return

        # atom2  BOND条件
        # 共有結合で隣接する原子数が「1」である。
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) != 1:
            return
        atom2_bond_coord = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom2_bonds)) & (self.df_atom["atom_type"] != "H"),
            ["x", "y", "z"],
        ].to_numpy()[0]

        distance4 = my_math.distance_two_points(atom1_coord, atom2_bond_coord)

        # 距離1,4判定
        if distance1 >= distance4:
            return None

        atom1_bonds_x = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] != "H"),
            ["atom_id", "x", "y", "z"],
        ].to_numpy()
        atom1_bonds_w = self.df_atom.loc[
            self.df_atom["atom_id"].isin(atom1_bonds), ["atom_id", "x", "y", "z"]
        ].to_numpy()

        # atom1 と共有結合する全ての水素原子に対して判定処理を実施
        for atom1_bond in atom1_bonds_x:
            atom1_bond_coord = atom1_bond[1:]

            distance2 = my_math.distance_two_points(atom1_bond_coord, atom2_coord)
            # 距離1,2判定
            if distance1 >= distance2:
                return None

            angle1 = my_math.angle_three_points(atom1_bond_coord, atom1_coord, atom2_coord)
            # 角度1判定
            if angle1 > self.interaction_parameter["S_F"][1]:
                continue

            for atom1_bond_w in atom1_bonds_w:
                if atom1_bond_w[0] == atom1_bond[0]:
                    continue
                atom1_bond_w_coord = atom1_bond_w[1:]

                distance3 = my_math.distance_two_points(atom1_bond_w_coord, atom2_coord)
                # 距離1,3判定
                if distance1 < distance3:
                    self.update_interaction_table(
                        label="S_F",
                        atom1_id=atom1_id,
                        atom1_molcular_type=atom1_molcular_type,
                        atom2_id=atom2_id,
                        atom2_molcular_type=atom2_molcular_type,
                        item_1=distance1,
                        item_2=distance2,
                        item_3=distance3,
                        item_4=distance4,
                        item_5=angle1,
                    )
                    return None

            if len(atom1_bonds_w) != 0:
                continue

            self.update_interaction_table(
                label="S_F",
                atom1_id=atom1_id,
                atom1_molcular_type=atom1_molcular_type,
                atom2_id=atom2_id,
                atom2_molcular_type=atom2_molcular_type,
                item_1=distance1,
                item_2=distance2,
                item_3=np.nan,
                item_4=distance4,
                item_5=angle1,
            )
            return None

    def calc_s_pi_1(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_resi,
        atom2_molcular_type,
    ):
        """S_PI 相互作用1

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_resi (int): atom1 の残基番号
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = ["S.2", "S.3", "S.ar"]
        atom2_atom_type_cond = ["C.ar", "N.ar", "C.2", "N.2", "N.pl3", "O.3", "S.3", "N.am"]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 閾値取得
        buffer_param = self.interaction_parameter["S_PI"][0]
        coef = self.interaction_parameter["S_PI"][1]
        angle1_param = self.interaction_parameter["S_PI"][2]

        # 2点間距離
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        # 距離1閾値判定
        atom2_symbol = atom2_atom_type.split(".")[0]
        vdw_radius = Decimal(str(self.vdw_difine["S"])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )

        if distance1 > buffer_param + float(vdw_radius):
            return None

        # atom1  BOND条件
        # 共有結合で隣接する原子数が「2以上」である。
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        if len(atom1_bonds) < 2:
            return
        atom1_bonds = (
            self.df_atom.loc[
                self.df_atom["atom_id"].isin(atom1_bonds),
                ["atom_type", "x", "y", "z"],
            ]
            .to_numpy()
            .tolist()
        )

        # atom2  BOND条件
        # 共有結合で隣接する原子数が「2以上」である。
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) < 2:
            return

        # 原子2 と原子1 に共有結合している原子間の距離条件判定
        is_true = False
        distance2 = np.nan
        distance3 = np.nan
        nvs = None
        for atom1_bond_pair in itertools.combinations(atom1_bonds, 2):
            atom1_x = atom1_bond_pair[0][1:]
            atom1_w = atom1_bond_pair[1][1:]
            # atom1_x を優先的に重原子にする
            if atom1_bond_pair[0][0] == "H" and atom1_bond_pair[1][0] == "H":
                continue

            elif atom1_bond_pair[0][0] == "H" and atom1_bond_pair[1][0] != "H":
                atom1_x = atom1_bond_pair[1][1:]
                atom1_w = atom1_bond_pair[0][1:]
            # 原子1 と共有結合している原子がなす法線ベクトル
            nvs = my_math.normal_vector_three_points(atom1_x, atom1_coord, atom1_w)

            distance2 = my_math.distance_two_points(atom1_x, atom2_coord)
            distance3 = my_math.distance_two_points(atom1_w, atom2_coord)
            if distance1 <= distance2 and distance1 <= distance3:
                is_true = True
                break

        if not is_true:
            return

        # atom2 が属する芳香環原子を抽出
        aroma_atoms = self.df_atom.loc[
            (self.df_atom["resi"] == atom2_resi)
            & (self.df_atom["atom_type"].isin(atom2_atom_type_cond)),
            ["atom_id", "x", "y", "z"],
        ].to_numpy()

        # networkx形成
        cycle_list = (
            self.df_bond.loc[
                (self.df_bond["atom_id"].isin(aroma_atoms[:, 0]))
                & (self.df_bond["bond_atom_id"].isin(aroma_atoms[:, 0])),
                ["atom_id", "bond_atom_id"],
            ]
            .to_numpy()
            .tolist()
        )

        graph = nx.Graph()
        graph.add_edges_from(cycle_list)
        cycle_atom_id = [c for c in nx.cycle_basis(graph) if atom2_id in c]

        # 芳香環でfor文
        for cycle_atom in cycle_atom_id:
            cycle_aroma_coords = [ar[1:] for ar in aroma_atoms if ar[0] in cycle_atom]
            if len(cycle_aroma_coords) < 5 or len(cycle_aroma_coords) > 6:
                continue
            atom2_bond_aroma_atom = [
                ar[1:] for ar in aroma_atoms if ar[0] in atom2_bonds and ar[0] in cycle_atom
            ]

            # 芳香環の重心
            cn = my_math.center_of_gravity(cycle_aroma_coords)
            # 重心との原子2との距離
            distance4 = my_math.distance_two_points(cn, atom2_coord)

            # 芳香環面に対して、原子1から垂線の下した点 Nrm
            nrm = my_math.intersection_point_vertical_line_and_plane(
                atom2_bond_aroma_atom[0], atom2_coord, atom2_bond_aroma_atom[1], atom1_coord
            )
            # 点 Nrm と重心との距離
            dnrm = my_math.distance_two_points(nrm, cn)

            # 原子2 と共有結合している芳香環原子がなす法線ベクトル
            nva = my_math.normal_vector_three_points(
                atom2_bond_aroma_atom[0], atom2_coord, atom2_bond_aroma_atom[1]
            )
            angle1 = my_math.angle_two_vectors(nvs, nva)
            angle1 = 180 - angle1 if angle1 > 90 else angle1

            if dnrm <= distance4 * coef and angle1_param <= angle1:

                self.update_interaction_table(
                    label="S_PI",
                    atom1_id=atom1_id,
                    atom1_molcular_type=atom1_molcular_type,
                    atom2_id=atom2_id,
                    atom2_molcular_type=atom2_molcular_type,
                    item_1=distance1,
                    item_2=distance2,
                    item_3=distance3,
                    item_4=distance4,
                    item_5=distance4 * coef,
                    item_6=dnrm,
                    item_7=angle1,
                )
                return None

    def calc_s_pi_2(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_resi,
        atom2_molcular_type,
    ):
        """S_PI 相互作用2

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_resi (int): atom1 の残基番号
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = ["S.2"]
        atom2_atom_type_cond = ["C.ar", "N.ar", "C.2", "N.2", "N.pl3", "O.3", "S.3", "N.am"]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 閾値取得
        buffer_param = self.interaction_parameter["S_PI"][0]
        coef = self.interaction_parameter["S_PI"][1]
        angle2_param = self.interaction_parameter["S_PI"][3]
        dihedral_param = self.interaction_parameter["S_PI"][4]

        # 2点間距離
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        # 距離1閾値判定
        atom2_symbol = atom2_atom_type.split(".")[0]
        vdw_radius = Decimal(str(self.vdw_difine["S"])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )

        if distance1 > buffer_param + float(vdw_radius):
            return None

        # atom1  BOND条件
        # 共有結合で隣接する原子数が「1」である。
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        if len(atom1_bonds) != 1:
            return
        atom1_bond_coord = self.df_atom.loc[
            (self.df_atom["atom_id"].isin(atom1_bonds)) & (self.df_atom["atom_type"] != "H"),
            ["x", "y", "z"],
        ].to_numpy()
        if len(atom1_bond_coord) != 1:
            return
        atom1_bond_coord = atom1_bond_coord[0]

        distance2 = my_math.distance_two_points(atom1_bond_coord, atom2_coord)
        if distance1 > distance2:
            return

        # atom2  BOND条件
        # 共有結合で隣接する原子数が「2以上」である。
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        if len(atom2_bonds) < 2:
            return

        # atom2 が属する芳香環原子を抽出
        aroma_atoms = self.df_atom.loc[
            (self.df_atom["resi"] == atom2_resi)
            & (self.df_atom["atom_type"].isin(atom2_atom_type_cond)),
            ["atom_id", "x", "y", "z"],
        ].to_numpy()

        # networkx形成
        cycle_list = (
            self.df_bond.loc[
                (self.df_bond["atom_id"].isin(aroma_atoms[:, 0]))
                & (self.df_bond["bond_atom_id"].isin(aroma_atoms[:, 0])),
                ["atom_id", "bond_atom_id"],
            ]
            .to_numpy()
            .tolist()
        )

        graph = nx.Graph()
        graph.add_edges_from(cycle_list)
        cycle_atom_id = [c for c in nx.cycle_basis(graph) if atom2_id in c]

        # 芳香環でfor文
        for cycle_atom in cycle_atom_id:
            cycle_aroma_coords = [ar[1:] for ar in aroma_atoms if ar[0] in cycle_atom]
            if len(cycle_aroma_coords) < 5 or len(cycle_aroma_coords) > 6:
                continue
            atom2_bond_aroma_atom = [
                ar[1:] for ar in aroma_atoms if ar[0] in atom2_bonds and ar[0] in cycle_atom
            ]

            # 芳香環の重心
            cn = my_math.center_of_gravity(cycle_aroma_coords)
            # 重心との各原子との距離
            distance4 = my_math.distance_two_points(cn, atom2_coord)
            distance6 = my_math.distance_two_points(cn, atom1_coord)
            distance7 = my_math.distance_two_points(cn, atom1_bond_coord)

            # 芳香環面に対して、原子1から垂線の下した点 Nrm
            nrm = my_math.intersection_point_vertical_line_and_plane(
                atom2_bond_aroma_atom[0], atom2_coord, atom2_bond_aroma_atom[1], atom1_coord
            )
            # 点 Nrm と重心との距離
            dnrm = my_math.distance_two_points(nrm, cn)

            angle2 = my_math.angle_three_points(nrm, atom1_coord, atom1_bond_coord)

            dihedral = my_math.dihedral_angle_four_points(cn, nrm, atom1_coord, atom1_bond_coord)

            if (
                dnrm <= distance4 * coef
                and distance6 <= distance7
                and angle2_param <= angle2
                and dihedral_param <= dihedral
            ):
                self.update_interaction_table(
                    label="S_PI",
                    atom1_id=atom1_id,
                    atom1_molcular_type=atom1_molcular_type,
                    atom2_id=atom2_id,
                    atom2_molcular_type=atom2_molcular_type,
                    item_1=distance1,
                    item_2=distance2,
                    item_3=distance4,
                    item_4=distance4 * coef,
                    item_5=dnrm,
                    item_6=distance6,
                    item_7=distance7,
                    item_8=angle2,
                    item_9=dihedral,
                )
                return None

    def calc_metal(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """金属相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = ["Fe", "Zn", "Ca", "Mg", "Ni"]
        atom2_atom_type_cond = [
            "O.2",
            "O.co2",
            "O.3",
            "N.2",
            "N.ar",
            "N.1",
            "N.3",
            "N.am",
            "N.pl3",
            "S.2",
            "S.3",
        ]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if (
            atom1_atom_type not in atom1_atom_type_cond
            or atom2_atom_type not in atom2_atom_type_cond
        ):
            return

        # 閾値の算出
        buffer_param = self.interaction_parameter["(Met)_(X)"][0]

        atom1_symbol = atom1_atom_type.split(".")[0]
        atom2_symbol = atom2_atom_type.split(".")[0]
        vdw_radius = Decimal(str(self.vdw_difine[atom1_symbol])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )

        # 距離算出
        distance1 = my_math.distance_two_points(atom1_coord, atom2_coord)
        if distance1 > float(vdw_radius) + buffer_param:
            return None

        # atom2  BOND条件：無し
        atom2_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom2_id, "bond_atom_id"].tolist()
        atom2_bonds = (
            self.df_atom.loc[
                (self.df_atom["atom_id"].isin(atom2_bonds)) & (self.df_atom["atom_type"] != "H"),
                ["atom_id", "x", "y", "z"],
            ]
            .to_numpy()
            .tolist()
        )

        is_true = False
        distance2 = np.nan
        distance3 = np.nan
        if len(atom2_bonds) == 1:
            distance2 = my_math.distance_two_points(atom1_coord, atom2_bonds[0][1:])
            if distance1 <= distance2:
                is_true = True
        else:
            for atom_x, atom_y in itertools.product(atom2_bonds, atom2_bonds):
                if atom_x[0] == atom_y[0]:
                    continue
                distance2 = my_math.distance_two_points(atom1_coord, atom_x[1:])
                distance3 = my_math.distance_two_points(atom1_coord, atom_y[1:])
                if distance1 <= distance2 and distance1 <= distance3:
                    is_true = True

        if is_true:
            self.update_interaction_table(
                label=f"{atom1_symbol}_{atom2_symbol}",
                atom1_id=atom1_id,
                atom1_molcular_type=atom1_molcular_type,
                atom2_id=atom2_id,
                atom2_molcular_type=atom2_molcular_type,
                item_1=distance1,
                item_2=distance2,
                item_3=distance3,
            )
            return None

    def calc_ion(
        self,
        atom1_id,
        atom1_atom_type,
        atom1_coord,
        atom1_molcular_type,
        atom2_id,
        atom2_atom_type,
        atom2_coord,
        atom2_molcular_type,
    ):
        """イオン相互作用

        Args:
            atom1_id (int): atom1のatom_id
            atom1_atom_type (string): atom1の原子タイプ
            atom1_coord (float): atom1の座標(x,y,z)
            atom1_molcular_type (string): atom1の分子タイプ
            atom2_id (int): atom2のatom_id
            atom2_atom_type (string): atom2の原子タイプ
            atom2_coord (float): atom2の座標(x,y,z)
            atom2_molcular_type (string): atom2の分子タイプ
        """
        atom1_atom_type_cond = ["Na", "K", "Cl"]
        atom2_atom_type_cond = ["H"]

        # atom1/atom2がAtomType条件に合致していない場合は終了
        if atom1_atom_type not in atom1_atom_type_cond or atom2_atom_type in atom2_atom_type_cond:
            return

        # 距離算出
        distance = my_math.distance_two_points(atom1_coord, atom2_coord)

        atom1_symbol = atom1_atom_type.split(".")[0]
        atom2_symbol = atom2_atom_type.split(".")[0]

        # van del waals半径の和
        vdw_radius = Decimal(str(self.vdw_difine[atom1_symbol])) + Decimal(
            str(self.vdw_difine[atom2_symbol])
        )

        # 閾値判定
        if distance > float(vdw_radius) + self.interaction_parameter["(Ion)_(X)"][0]:
            return

        # Bond条件(atom1)
        atom1_bonds = self.df_bond.loc[self.df_bond["atom_id"] == atom1_id, "bond_atom_id"].tolist()
        if len(atom1_bonds) == 0:
            label = atom1_symbol + "_" + atom2_symbol

            self.update_interaction_table(
                label=label,
                atom1_id=atom1_id,
                atom1_molcular_type=atom1_molcular_type,
                atom2_id=atom2_id,
                atom2_molcular_type=atom2_molcular_type,
                item_1=distance,
            )
            return None

    def write_one_hot_list(self, output, group_difine_file="group.yaml"):
        """One-hot listファイルを出力するメソッド

        Args:
            output (str): 出力ファイルのprefix
            group_difine_file (str): 相互作用のグループ定義ファイル
        """
        # 相互作用ラベルの一覧を取得
        labels = None
        with open(group_difine_file, "r", encoding="utf-8") as file:
            groups = yaml.safe_load(file)
            labels = groups.keys()

        # 分子タイプペアごとにone-hot-vector用のラベルを作成
        label_columns = [f"LP_{label}" for label in labels]
        molcular_types = list(
            set(self.df_atom.loc[~(self.df_atom["molcular_type"].isna()), "molcular_type"].tolist())
        )
        solvents = [mol_type for mol_type in molcular_types if "S" in mol_type]
        if len(solvents) == 0:
            label_columns.extend([f"LS_{label}" for label in labels])
            label_columns.extend([f"SP_{label}" for label in labels])

        for solvent in solvents:
            label_columns.extend([f"L{solvent}_{label}" for label in labels])
            label_columns.extend([f"{solvent}P_{label}" for label in labels])

        #
        info_columns = [
            "dist",
            "interaction_label",
            "molcular_type",
            "chain",
            "residue",
            "residue_number",
            "atom_name",
            "atom_number",
            "atom_type",
            "partner_molcular_type",
            "partner_chain",
            "partner_residue",
            "partner_residue_number",
            "partner_atom_name",
            "partner_atom_number",
            "partner_atom_type",
        ]
        one_hot_list = dict()
        for col in label_columns + info_columns:
            one_hot_list[col] = []

        # 「L-S-Pro」のみを抽出し、重複を削除したリストを取得
        df_sol_interaction = self.interaction_table.loc[~(self.interaction_table["label2"].isna())]
        sol_pro_lists = (
            df_sol_interaction.loc[
                ~(
                    df_sol_interaction.duplicated(
                        subset=["label2", "atom3_id", "atom4_id"], keep="first"
                    )
                ),
                ["label2", "pair_type", "item2_1", "atom3_id", "atom4_id"],
            ]
            .to_numpy()
            .tolist()
        )

        # 溶媒を介さない相互作用のリストを取得
        lig_lists = (
            self.interaction_table.loc[
                self.interaction_table["label2"].isna(),
                ["label1", "pair_type", "item1_1", "atom1_id", "atom2_id"],
            ]
            .to_numpy()
            .tolist()
        )
        atoms_dict = dict(
            zip(
                self.df_atom["atom_id"],
                self.df_atom[["atom_name", "atom_type", "resi", "resn", "chain"]]
                .to_numpy()
                .tolist(),
            )
        )

        for row in lig_lists + sol_pro_lists:
            label = "Dipo" if "Dipo" in row[0] else row[0]

            if label.split("_")[0] in ["Fe", "Zn", "Ca", "Mg", "Ni", "Na", "K", "Cl"]:
                one_hot_label = f"{label.split('_')[0]}_X"
            else:
                one_hot_label = label

            pair_type = row[1].split("-")
            dist = row[2]
            atom1_id = row[3]
            atom2_id = row[4]

            mol_type = None
            partner_mol_type = None

            if len(pair_type) == 3:
                mol_type = "solvent"
                partner_mol_type = "protein"
                one_hot_label = f"{pair_type[1]}P_{one_hot_label}"

            elif pair_type[1] == "Pro":
                mol_type = "ligand"
                partner_mol_type = "protein"
                one_hot_label = f"LP_{one_hot_label}"

            else:
                mol_type = "ligand"
                partner_mol_type = "solvent"
                one_hot_label = f"L{pair_type[1]}_{one_hot_label}"

            for label_name in label_columns:
                if one_hot_label == label_name:
                    one_hot_list[label_name].append(1)
                else:
                    one_hot_list[label_name].append(0)

            one_hot_list["dist"].append(f"{dist:.4f}")
            one_hot_list["interaction_label"].append(label)
            one_hot_list["molcular_type"].append(mol_type)
            one_hot_list["chain"].append(str(atoms_dict[atom1_id][4]))
            one_hot_list["residue"].append(atoms_dict[atom1_id][3])
            one_hot_list["residue_number"].append(atoms_dict[atom1_id][2])
            one_hot_list["atom_name"].append(atoms_dict[atom1_id][0])
            one_hot_list["atom_number"].append(atom1_id)
            one_hot_list["atom_type"].append(atoms_dict[atom1_id][1])

            one_hot_list["partner_molcular_type"].append(partner_mol_type)
            one_hot_list["partner_chain"].append(str(atoms_dict[atom2_id][4]))
            one_hot_list["partner_residue"].append(atoms_dict[atom2_id][3])
            one_hot_list["partner_residue_number"].append(atoms_dict[atom2_id][2])
            one_hot_list["partner_atom_name"].append(atoms_dict[atom2_id][0])
            one_hot_list["partner_atom_number"].append(atom2_id)
            one_hot_list["partner_atom_type"].append(atoms_dict[atom2_id][1])

        df_one_hot = pd.DataFrame(one_hot_list)
        df_one_hot = df_one_hot.loc[:, ~(df_one_hot.sum(axis=0) == 0)]
        df_one_hot.to_csv(f"{output}_one_hot_list.csv", index=False)

    def write_interaction_sum_list(self, output, group_difine_file="group.yaml"):
        """Interaction Sum listファイル

        Args:
            output (str): 出力ファイルのprefix
            group_difine_file (str): 相互作用のグループ定義ファイル
        """
        totals = {}
        groups = {}
        with open(group_difine_file, "r", encoding="utf-8") as file:
            groups = yaml.safe_load(file)

        # 「L-S-Pro」のみを抽出し、重複を削除して、相互作用グループごとにカウント
        df_sol_interaction = self.interaction_table.loc[~(self.interaction_table["label2"].isna())]
        sol_pro_lists = (
            df_sol_interaction.loc[
                ~(
                    df_sol_interaction.duplicated(
                        subset=["label2", "atom3_id", "atom4_id"], keep="first"
                    )
                ),
                ["pair_type", "label2"],
            ]
            .to_numpy()
            .tolist()
        )

        for pair_type, label in sol_pro_lists:
            solvent = pair_type.split("-")[1]
            if label.split("_")[0] in ["Fe", "Zn", "Ca", "Mg", "Ni", "Na", "K", "Cl"]:
                label = f"{label.split('_')[0]}_X"
            elif "Dipo" in label:
                label = "Dipo"

            totals.setdefault(f"{solvent}P_{groups[label]}", 0)
            totals[f"{solvent}P_{groups[label]}"] += 1

        # 溶媒を介さない相互作用情報を取得し、相互作用グループごとにカウント
        label1_table = (
            self.interaction_table.loc[
                self.interaction_table["label2"].isna(),
                ["pair_type", "label1"],
            ]
            .to_numpy()
            .tolist()
        )
        for pair_type, label in label1_table:
            prefix = None
            if "S" in pair_type:
                prefix = f"L{pair_type.split('-')[1]}"
            else:
                prefix = "LP"

            if label.split("_")[0] in ["Fe", "Zn", "Ca", "Mg", "Ni", "Na", "K", "Cl"]:
                label = f"{label.split('_')[0]}_X"
            elif "Dipo" in label:
                label = "Dipo"

            totals.setdefault(f"{prefix}_{groups[label]}", 0)
            totals[f"{prefix}_{groups[label]}"] += 1
        pd.DataFrame(totals, index=[0]).to_csv(f"{output}_interaction_sum_list.csv", index=False)

    def write_interaction(self, mol2_name, output):
        """相互作用検出結果(Raw list ファイル)出力機能

        Args:
            output (str): 出力ファイルのprefix
        """

        molcular_types = list(
            map(
                list,
                set(
                    map(
                        tuple,
                        self.df_atom.loc[
                            ~(self.df_atom["molcular_type"].isna()), ["molcular_type", "resn"]
                        ]
                        .to_numpy()
                        .tolist(),
                    )
                ),
            )
        )
        solvents = [m for m in molcular_types if "S" in m[0]]
        solvents = sorted(
            solvents, key=lambda x: [x[0][0], int(x[0][1:])] if len(x[0]) != 1 else [x[0]]
        )

        atoms_dict = dict(
            zip(
                self.df_atom["atom_id"],
                self.df_atom[["atom_name", "atom_type", "resi", "resn", "chain"]]
                .to_numpy()
                .tolist(),
            )
        )
        bond_list = self.df_bond.to_numpy()

        with open(f"{output}_raw_list.txt", "w", encoding="utf8") as file:
            file.write(f"R {mol2_name}\n")
            if self.exec_type == "Lig":
                res_list = [m[1] for m in molcular_types if m[0] == "L"]
                file.write(f"L {','.join(sorted(res_list))}\n")
            elif self.exec_type == "Mut":
                res_list = [m[1] for m in molcular_types if m[0] == "Mut"]
                file.write(f"A {','.join(sorted(res_list))}\n")

            if len(solvents) == 0:
                file.write("S None\n")

            for solvent in solvents:
                file.write(f"{solvent[0]} {solvent[1]}\n")
            file.write("\n")

            table = self.interaction_table.to_numpy()
            for row in sorted(table, key=operator.itemgetter(0)):
                label1 = row[0] if "Dipo" not in row[0] else row[0].split("_")[0]
                if self.exec_type == "Lig":
                    file.write(f"K {row[2]}\n")
                    items = " ".join([f"{v:.4f}" for v in row[3:13] if not math.isnan(v)])
                    file.write(f"I1-2 {label1} {items}\n")
                    if isinstance(row[1], str):
                        label2 = row[1] if "Dipo" not in row[1] else row[1].split("_")[0]
                        items = " ".join([f"{v:.4f}" for v in row[13:23] if not math.isnan(v)])
                        file.write(f"I3-4 {label2} {items}\n")

                    for i, atom_id in enumerate(row[23:]):
                        if math.isnan(atom_id):
                            continue

                        bond_atoms = bond_list[np.where(bond_list[:, 0] == atom_id), 1:][0]

                        infos = atoms_dict[atom_id]
                        if i == 0:
                            mol = "L"

                        elif i == 1:
                            mol = "P" if row[2] == "L-Pro" else "S"

                        elif i == 2:
                            mol = "S"

                        elif i == 3:
                            mol = "P"

                        file.write(
                            f"{mol}C{i + 1} {infos[3]} {infos[2]} "
                            f"{infos[0]} {atom_id} {infos[1]}\n"
                        )
                        for bond_atom in bond_atoms:
                            infos = atoms_dict[bond_atom[0]]
                            file.write(
                                f"{mol}N{i + 1} {infos[3]} {infos[2]} {infos[0]} "
                                f"{bond_atom[0]} {infos[1]} {bond_atom[1]}\n"
                            )

                elif self.exec_type == "Mut":
                    pair_label = row[2].replace("Mut", "M")
                    file.write(f"MID {pair_label} {label1} {row[3]:.4f} ")
                    for i, atom_id in enumerate(row[23:]):
                        if math.isnan(atom_id):
                            continue

                        if i != 0:
                            file.write("- ")
                        infos = atoms_dict[atom_id]
                        file.write(f"{infos[4]} {infos[2]} {infos[3]} {infos[0]} {atom_id} ")

                    if isinstance(row[1], str):
                        label2 = row[1] if "Dipo" not in row[1] else row[1].split("_")[0]
                        file.write(f"{label2} {row[13]:.4f}")

                elif self.exec_type == "Med":
                    pair_label = row[2].replace("Pep", "P")
                    file.write(f"K {pair_label}\n")
                    items = " ".join([f"{v:.4f}" for v in row[3:13] if not math.isnan(v)])
                    file.write(f"I1-2 {label1} {items}\n")
                    if isinstance(row[1], str):
                        label2 = row[1] if "Dipo" not in row[1] else row[1].split("_")[0]
                        items = " ".join([f"{v:.4f}" for v in row[13:23] if not math.isnan(v)])
                        file.write(f"I3-4 {label2} {items}\n")

                    for i, atom_id in enumerate(row[23:]):
                        if math.isnan(atom_id):
                            continue

                        bond_atoms = []
                        for idx in np.where(bond_list[:, 0] == atom_id):
                            bond_atoms.append(bond_list[idx][1:])

                        bond_atoms = bond_list[np.where(bond_list[:, 0] == atom_id), 1:][0]

                        infos = atoms_dict[atom_id]
                        mol = pair_label.split("-")
                        if i == 0:
                            mol = mol[0]
                        elif i == 1 or i == 2:
                            mol = mol[1]
                        else:
                            mol = mol[2]

                        file.write(
                            f"{mol}C{i + 1} {infos[3]} {infos[2]}, "
                            f"{infos[0]}, {atom_id} {infos[1]}\n"
                        )
                        for bond_atom in bond_atoms:
                            infos = atoms_dict[bond_atom[0]]
                            file.write(
                                f"{mol}N{i + 1} {infos[3]} {infos[2]}, {infos[0]}, "
                                f"{bond_atom[0]} {infos[1]} {bond_atom[1]}\n"
                            )
                file.write("\n")

    def write_total_interaction(self, output, group_difine_file="group.yaml", is_output=True):
        """相互作用検出数集計結果(Interaction Count list ファイル)出力機能

        Args:
            output (str): 出力ファイルのprefix
            group_difine_file (str): 相互作用のグループ定義ファイル
            is_output(bool): ファイル出力のスイッチ
        """

        # 相互作用ラベルの一覧を取得
        labels = None
        with open(group_difine_file, "r", encoding="utf-8") as file:
            groups = yaml.safe_load(file)
            labels = groups.keys()

        molcular_types = list(
            set(self.df_atom.loc[~(self.df_atom["molcular_type"].isna()), "molcular_type"].tolist())
        )
        solvents = [mol_type for mol_type in molcular_types if "S" in mol_type]
        if len(solvents) == 0:
            solvents = ["S"]

        table = self.interaction_table[["label1", "label2", "pair_type"]].to_numpy().tolist()

        total_list = dict()
        if self.exec_type == "Lig":
            for label in labels:
                total_list[f"L#{label}#Pro"] = 0

            for solvent in solvents:
                for label in labels:
                    total_list[f"L#{label}#{solvent}"] = 0
                    for label_1, label_2 in itertools.product(labels, labels):
                        total_list[f"L#{label_1}#{solvent}#{label_2}#Pro"] = 0

        elif self.exec_type == "Mut":
            for mol_type in ["Ab", "Ag", "M"]:
                for label in labels:
                    total_list[f"M#{label}#{mol_type}"] = 0

            for solvent in solvents:
                for label in labels:
                    total_list[f"M#{label}#{solvent}"] = 0

                for mol_type in ["Ab", "Ag", "M"]:
                    for label_1, label_2 in itertools.product(labels, labels):
                        total_list[f"M#{label_1}#{solvent}#{label_2}#{mol_type}"] = 0

        elif self.exec_type == "Med":
            for mol_type in ["P-P", "P-Mem", "L-Pro"]:
                mol1 = mol_type.split("-")[0]
                mol2 = mol_type.split("-")[1]
                for label in labels:
                    total_list[f"{mol1}#{label}#{mol2}"] = 0

            for solvent in solvents:
                for mol_type in ["P", "L", "Pro"]:
                    for label in labels:
                        total_list[f"{mol_type}#{label}#{solvent}"] = 0

                for label_1, label_2 in itertools.product(labels, labels):
                    total_list[f"P#{label_1}#{solvent}#{label_2}#P"] = 0

                for label_1, label_2 in itertools.product(labels, labels):
                    total_list[f"L#{label_1}#{solvent}#{label_2}#Pro"] = 0

                for label_1, label_2 in itertools.product(labels, labels):
                    total_list[f"Pro#{label_1}#{solvent}#{label_2}#Pro"] = 0

        # 検出された相互作用の集計処理
        for row in table:
            label1 = row[0]
            if "Dipo" in label1:
                label1 = "Dipo"
            elif label1.split("_")[0] in ["Fe", "Zn", "Ca", "Mg", "Ni", "Na", "K", "Cl"]:
                label1 = f"{label1.split('_')[0]}_X"

            pair_type = row[2]
            pair_type = pair_type.replace("Mut", "M")
            pair_type = pair_type.replace("Pep", "P")

            total_list_label = pair_type.replace("-", f"#{label1}#", 1)
            label2 = row[1]
            if isinstance(label2, str):
                if "Dipo" in label2:
                    label2 = "Dipo"
                elif label2.split("_")[0] in ["Fe", "Zn", "Ca", "Mg", "Ni", "Na", "K", "Cl"]:
                    label2 = f"{label2.split('_')[0]}_X"

                total_list_label = total_list_label.replace("-", f"#{label2}#", 1)
            total_list[total_list_label] += 1

        # NOTE: 双極子-双極子相互作用はBOND間の相互作用の為、総数は相互作用情報の半分となる。
        for key, val in total_list.items():
            if "Dipo" in key:
                total_list[key] = int(val / 2)

        # 各項目をラベルで並び替えて出力
        total_list = sorted([[key, val] for key, val in total_list.items()], key=lambda x: x[0])

        if is_output:
            with open(f"{output}_interaction_count_list.csv", "w", encoding="utf8") as file:
                for row in total_list:
                    file.write(f"{row[0]},{row[1]}\n")
        return total_list

    def write_pml(self, output, suffix, model_prefix, water_def_file):
        """可視化ファイル出力機能

        Args:
            output (str): 出力ファイルのprefix
            suffix (str): グループ名、距離オブジェクトのサフィックス
                          NOTE: 複数構造を同時に可視化した際に重複しないようにするため
            model_prefix (str): 可視化対象構造名
                                NOTE: 距離オブジェクト作成時に指定する。
            water_def_file (str): 水分子の名称定義ファイル
        """

        def convert_pair_label(pair_type):
            convert_dict = {
                "Lig": {},
                "Mut": {"Ab": "Ab", "Ag": "Ag"},
                "Med": {"Pro": "Pro", "Mem": "Mem"},
            }
            mol_types = pair_type.split("-")
            labels = []
            atom1_mol_type = convert_dict[self.exec_type].get(mol_types[0], mol_types[0][0])
            atom2_mol_type = None
            if mol_types[1][0] == "S":
                atom2_mol_type = mol_types[1]
            else:
                atom2_mol_type = convert_dict[self.exec_type].get(mol_types[1], mol_types[1][0])
            labels.append("".join([atom1_mol_type, atom2_mol_type]))

            if len(mol_types) == 3:
                solvent = mol_types[1]
                other = convert_dict[self.exec_type].get(mol_types[2], mol_types[2][0])
                if mol_types[0][0] == mol_types[2][0]:
                    # NOTE: M-S-M, P-S-P の場合は、 MS, PS に揃える
                    labels.append("".join([other, solvent]))
                elif other == "Pro":
                    # NOTE: 中分子かつ L-S-Pro の場合は、ProSに揃える
                    labels.append("".join([other, solvent]))
                else:
                    labels.append("".join([solvent, other]))
            return labels

        water_names = set()
        with open(water_def_file, "r", encoding="utf8") as file:
            for line in file.readlines():
                if line.startswith("#"):
                    continue
                water_names.add(line.strip())

        # stick表示するatom_id取得
        target_type = None
        if self.exec_type == "Lig":
            target_type = "L"
        elif self.exec_type == "Mut":
            target_type = "Mut"
        elif self.exec_type == "Med":
            target_type = "Pep"
        stick_atom = (
            self.df_atom.loc[self.df_atom["molcular_type"] == target_type, "atom_id"]
            .to_numpy()
            .tolist()
        )
        stick_atom = set(stick_atom)

        with open(f"{output}.pml", "w", encoding="utf8") as file:
            # 表示設定
            # 全ての構造を一度非表示にする。
            file.write(f"hide everything, {model_prefix}\n")

            # 相互作用
            atoms_list = set()
            table = (
                self.interaction_table[
                    [
                        "pair_type",
                        "label1",
                        "label2",
                        "atom1_id",
                        "atom2_id",
                        "atom3_id",
                        "atom4_id",
                    ]
                ]
                .sort_values(by="pair_type")
                .to_numpy()
                .tolist()
            )
            group_list = {}
            obj_name_list = {}
            for row in table:
                pair_types = convert_pair_label(row[0])

                interaction_sets = [(row[1], row[3], row[4], pair_types[0])]
                if len(pair_types) == 2:
                    interaction_sets.append((row[2], row[6], row[5], pair_types[1]))

                for label, atom1_id, atom2_id, pair_label in interaction_sets:
                    obj_name_list.setdefault(pair_label, set())
                    group_list.setdefault(pair_label, set())

                    # 双極子-双極子相互作用について電荷の符号が反対である原子間の相互作用のみ表示
                    if "Dipo" in label:
                        charges = self.df_atom.loc[
                            self.df_atom["atom_id"].isin([atom1_id, atom2_id]), "charge"
                        ].tolist()
                        if charges[0] * charges[1] > 0:
                            continue
                        label = "Dipo"

                    elif label.split("_")[0] in ["Fe", "Zn", "Ca", "Mg", "Ni", "Na", "K", "Cl"]:
                        label = f"{label.split('_')[0]}_X"

                    id_name = None
                    if pair_label.startswith("S"):
                        id_name = f"{atom2_id}_{atom1_id}"
                    else:
                        id_name = f"{atom1_id}_{atom2_id}"
                    obj_name_list[pair_label].add(
                        tuple([f"{pair_label}_{label}_{id_name}", atom1_id, atom2_id])
                    )
                    atoms_list |= set([atom1_id, atom2_id])
                    group_list[pair_label].add(f"{pair_label}_{label}")

            for mol_type, obj_sets in obj_name_list.items():
                # 分子タイプグループの作成
                file.write(f"group {mol_type}_{suffix}\n")

                obj_sets = sorted(obj_sets, key=operator.itemgetter(0, 1, 2))
                for row in obj_sets:
                    target_atom1 = f"id {row[1]} & {model_prefix}"
                    target_atom2 = f"id {row[2]} & {model_prefix}"
                    file.write(f"distance {row[0]}_{suffix}, {target_atom1}, {target_atom2}\n")

                # group設定
                for group in sorted(group_list[mol_type]):
                    file.write(f"group {group}_{suffix}, {group}_*_{suffix}, add\n")
                    file.write(f"group {mol_type}_{suffix}, {group}_{suffix}, add\n")

            # color 設定
            file.write("color marine, *_HB_*\n")
            file.write("color marine, *_SH_N*\n")
            file.write("color marine, *_SH_O*\n")
            file.write("color cyan, *_Elec_*\n")
            file.write("color pink, *_CH_O*\n")
            file.write("color pink, *_CH_N*\n")
            file.write("color warmpink, *_PI_PI*\n")
            file.write("color brown, *_vdW*\n")
            file.write("color violet, *_Dipo*\n")
            file.write("color yelloworange, *_OMulPol*\n")
            file.write("color orange, *_CH_PI*\n")
            file.write("color brightorange, *_OH_PI*\n")
            file.write("color tv_orange, *_NH_PI*\n")
            file.write("color tv_orange, *_SH_PI*\n")
            file.write("color lightorange, *_S_PI*\n")
            file.write("color palegreen, *H_F*\n")
            file.write("color Splitpea, *H_Hal_*\n")
            file.write("color Splitpea, *_S_O*\n")
            file.write("color chocolate, *H_S*\n")
            file.write("color sand, *_NH_S*\n")
            file.write("color sand, *_CH_S*\n")
            file.write("color sand, *_S_F*\n")
            file.write("color sand, *_S_S*\n")
            file.write("color deeppurple, *_Hal_PI_*\n")
            file.write("color violetpurple, *_Hal_Cl_*\n")
            file.write("color violetpurple, *_Hal_Br_*\n")
            file.write("color violetpurple, *_Hal_I_*\n")
            file.write("color purple, *_Fe_X*\n")
            file.write("color purple, *_Zn_X*\n")
            file.write("color purple, *_Ca_X*\n")
            file.write("color purple, *_Mg_X*\n")
            file.write("color purple, *_Ni_X*\n")
            file.write("color lightteal, *_Na_X*\n")
            file.write("color lightteal, *_K_X*\n")
            file.write("color lightteal, *_Cl_X*\n")

            # 表示設定
            # line表示
            atoms_list = atoms_list - stick_atom
            target_atoms = "+".join([str(i) for i in sorted(list(atoms_list))])
            file.write(f"show lines, byres (id {target_atoms} & {model_prefix})\n")
            # stick表示
            target_atoms = "+".join([str(i) for i in sorted(list(stick_atom))])
            file.write(f"show sticks, id {target_atoms} & {model_prefix}\n")
            # リガンド表示色設定
            if self.exec_type == "Lig":
                file.write(f"util.cbas id {target_atoms} & {model_prefix}\n")
            # ラベルの非表示設定
            file.write(f"hide labels, *_{suffix}\n")
