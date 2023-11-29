import argparse
import glob
import os

from .mol2 import Mol2
from .interaction import Interaction


def calculate(
    exec_type,
    mol2,
    molcular_select_file,
    parametar_file,
    vdw_file,
    priority_file,
    water_definition_file,
    interaction_group_file,
    output,
    allow_mediate_position,
    on_14,
    dup,
    no_mediate,
    no_out_total,
    no_out_pml,
    switch_ch_pi,
):
    """メイン関数

    Args:
        exec_type (str): 実行機能（Lig, Mut, Med）
        mol2 (str): 結晶構造情報ファイル（.mol）
        molcular_select_file (str): 分子構造指定ファイル
        parametar_file (str): 相互作用閾値設定ファイル
        vdw_file (str): Van Del Waals 半径設定ファイル
        priority_file (str): 相互作用優先順位設定ファイル
        water_definition_file (str): 水分子名指定ファイル
        output (str): 出力ファイルのprefix
        allow_mediate_position (int): 溶媒原子間の位置関係を表す数値
        on_14 (bool): True: 1-3, 1-4 相互作用を検出する
        dup (bool): True: 同一重原子間で重複した相互作用の検出を認める
        no_mediate (bool): True: 溶媒を介した相互作用を検出しない
        no_out_total (bool): True: 集計結果ファイルを出力しない
        no_out_pml (bool): True: 可視化ファイルを出力しない
        switch_ch_pi (bool): True: CH_PI, NH_PI, OH_PI を古い定義で判定する。
    """
    mol2_files = []
    if os.path.isdir(mol2):
        mol2_files = glob.glob(os.path.join(mol2, "flame*.mol2"))

    else:
        mol2_files = [mol2]

    is_flame = True if len(mol2_files) != 1 else False
    trajectory_total = {}
    for file in mol2_files:
        output_prefix = output
        prefix = os.path.basename(file).split(".")[0]
        if is_flame:
            if "/" in output:
                split_path = os.path.split(output)
                output_prefix = f"{prefix}_{split_path[1]}"
                output_prefix = os.path.join(split_path[0], output_prefix)
            else:
                output_prefix = f"{prefix}_{output}"
        mol = Mol2(interaction_type=exec_type)
        mol.read_mol2(mol2_file=file)
        mol.add_molcular_type(molcular_select_file=molcular_select_file)

        init = Interaction(
            df_atom=mol.df_atom,
            df_bond=mol.df_bond,
            interaction_parameter_file=parametar_file,
            vdw_difine_file=vdw_file,
            priority_file=priority_file,
            exec_type=exec_type,
        )
        init.calculate(no_mediate, switch_ch_pi)

        if not on_14:
            init.drop_13_14()

        if not dup:
            # 重複削除
            init.drop_duplicate()

        # 溶媒を介する相互作用の削除
        if allow_mediate_position is not None and no_mediate is False:
            init.drop_mediate_interaction(allow_mediate_position)

        # 相互作用検出結果出力
        init.write_interaction(file, output_prefix)

        if not no_out_total:
            # 相互作用集計結果出力
            init.write_total_interaction(output_prefix, interaction_group_file)
            with open(f"{output_prefix}_interaction_count_list.csv", "r", encoding="utf8") as file:
                for row in file.readlines():
                    label = row.split(",")[0]
                    num = int(row.split(",")[1])
                    if label in trajectory_total:
                        trajectory_total[label] += num
                    else:
                        trajectory_total[label] = num

        if not no_out_pml:
            # 相互作用可視化ファイル出力
            init.write_pml(
                output=output_prefix,
                suffix=os.path.basename(output_prefix),
                model_prefix=prefix,
                water_def_file=water_definition_file,
            )

        if exec_type == "Lig":
            # One-hot listファイル出力
            init.write_one_hot_list(output_prefix, interaction_group_file)

            # Interaction Sum listファイル出力
            init.write_interaction_sum_list(output_prefix, interaction_group_file)

    if len(mol2_files) != 1:
        # 相互作用集計結果（トラジェクトリ）出力
        with open(f"{output}_trajectory.csv", "w", encoding="utf8") as file:
            file.write(f"flames,{len(mol2_files)}\n")
            for key, val in trajectory_total.items():
                file.write(f"{key},{val}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="exec_type")
    subparsers.required = True

    # リガンド-タンパク質相互作用記述子計算機能
    ligand_parser = subparsers.add_parser("ligand")
    ligand_parser.add_argument(
        "mol2_file", help="Tripos Mol2 file (.mol2) or directory containing Mol2"
    )
    ligand_parser.add_argument("molcular_select_file", help="Molecule difinication file (.yaml)")
    ligand_parser.add_argument("vdw_file", help="Van Der Waals Radius difinication file (.yaml)")
    ligand_parser.add_argument("parameter_file", help="parameter setting file (.yaml)")
    ligand_parser.add_argument("priority_file", help="priority difinication file (.yaml)")
    ligand_parser.add_argument("output", help="Output file prefix")
    ligand_parser.add_argument("--on_14", help="detect 1-3, 1-4 interaction", action="store_true")
    ligand_parser.add_argument("--dup", help="detect duplicate interactions", action="store_true")
    ligand_parser.add_argument(
        "--allow_mediate_pos",
        default=None,
        type=int,
        help="Position between solvent atoms that "
        "allow detection of solvent-mediated interactions (≧ 1)",
    )
    ligand_parser.add_argument(
        "--no_mediate", help="Not detect solvent-mediated interactions.", action="store_true"
    )
    ligand_parser.add_argument(
        "--no_out_total", help=".csv will not be output", action="store_true"
    )
    ligand_parser.add_argument("--no_out_pml", help=".pml will not be output", action="store_true")
    ligand_parser.add_argument(
        "--switch_ch_pi",
        help="CH_PI, NH_PI, OH_PI Determined by the old definition.",
        action="store_true",
    )

    # 抗体-抗原相互作用記述子計算成機能
    mutant_parser = subparsers.add_parser("mutant")
    mutant_parser.add_argument(
        "mol2_file", help="Tripos Mol2 file (.mol2) or directory containing Mol2"
    )
    mutant_parser.add_argument("molcular_select_file", help="Molecule difinication file (.yaml)")
    mutant_parser.add_argument("vdw_file", help="Van Der Waals Radius difinication file (.yaml)")
    mutant_parser.add_argument("parameter_file", help="parameter setting file (.yaml)")
    mutant_parser.add_argument("priority_file", help="priority difinication file (.yaml)")
    mutant_parser.add_argument("output", help="Output file prefix")
    mutant_parser.add_argument("--on_14", help="detect 1-3, 1-4 interaction", action="store_true")
    mutant_parser.add_argument("--dup", help="detect duplicate interactions", action="store_true")
    mutant_parser.add_argument(
        "--allow_mediate_pos",
        default=None,
        type=int,
        help="Position between solvent atoms that "
        "allow detection of solvent-mediated interactions (≧ 1)",
    )
    mutant_parser.add_argument(
        "--no_mediate", help="Not detect solvent-mediated interactions.", action="store_true"
    )
    mutant_parser.add_argument(
        "--no_out_total", help=".csv will not be output", action="store_true"
    )
    mutant_parser.add_argument("--no_out_pml", help=".pml will not be output", action="store_true")
    mutant_parser.add_argument(
        "--switch_ch_pi",
        help="CH_PI, NH_PI, OH_PI Determined by the old definition.",
        action="store_true",
    )

    # 中分子相互作用記述子
    medium_parser = subparsers.add_parser("medium")
    medium_parser.add_argument(
        "mol2_file", help="Tripos Mol2 file (.mol2) or directory containing Mol2"
    )
    medium_parser.add_argument("molcular_select_file", help="Molecule difinication file (.yaml)")
    medium_parser.add_argument("vdw_file", help="Van Der Waals Radius difinication file (.yaml)")
    medium_parser.add_argument("parameter_file", help="parameter setting file (.yaml)")
    medium_parser.add_argument("priority_file", help="priority difinication file (.yaml)")
    medium_parser.add_argument("output", help="Output file prefix")
    medium_parser.add_argument("--on_14", help="detect 1-3, 1-4 interaction", action="store_true")
    medium_parser.add_argument("--dup", help="detect duplicate interactions", action="store_true")
    medium_parser.add_argument(
        "--allow_mediate_pos",
        default=None,
        type=int,
        help="Position between solvent atoms that "
        "allow detection of solvent-mediated interactions (≧ 1)",
    )
    medium_parser.add_argument(
        "--no_mediate", help="Not detect solvent-mediated interactions.", action="store_true"
    )
    medium_parser.add_argument(
        "--no_out_total", help=".csv will not be output", action="store_true"
    )
    medium_parser.add_argument("--no_out_pml", help=".pml will not be output", action="store_true")
    medium_parser.add_argument(
        "--switch_ch_pi",
        help="CH_PI, NH_PI, OH_PI Determined by the old definition.",
        action="store_true",
    )

    args = parser.parse_args()

    if args.allow_mediate_pos is not None and args.allow_mediate_pos < 1:
        raise ValueError("'--allow_mediate_pos' is 1 or more")

    water_definition_file = os.path.join(os.path.dirname(__file__), "water_definition.txt")
    interaction_group_file = os.path.join(os.path.dirname(__file__), "group.yaml")

    calculate(
        exec_type=str(args.exec_type[0:3]).capitalize(),
        mol2=args.mol2_file,
        molcular_select_file=args.molcular_select_file,
        parametar_file=args.parameter_file,
        vdw_file=args.vdw_file,
        priority_file=args.priority_file,
        water_definition_file=water_definition_file,
        interaction_group_file=interaction_group_file,
        output=args.output,
        allow_mediate_position=args.allow_mediate_pos,
        on_14=args.on_14,
        dup=args.dup,
        no_mediate=args.no_mediate,
        no_out_total=args.no_out_total,
        no_out_pml=args.no_out_pml,
        switch_ch_pi=args.switch_ch_pi,
    )
