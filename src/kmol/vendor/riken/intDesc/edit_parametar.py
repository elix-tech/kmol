"""edit_parametar"""
import argparse
import sys
from argparse import RawTextHelpFormatter

import yaml


def write_param(param_dict: dict, output: str):
    """パラメータファイルを出力する関数

    Args:
        param_dict (dict): パラメータ辞書（key=相互作用名、value=パラメータのリスト）
        output (str): 出力ファイル
    """
    # 出力対象の相互作用のリスト
    # 出力順序は本リストの順序に準拠する。
    interaction_type_list = [
        "HB_OH_(N,O)",
        "HB_OH_OH",
        "HB_NH_(N,O)",
        "HB_NH_OH",
        "vdW",
        "(Met)_(X)",
        "(Ion)_(X)",
        "SH_N",
        "SH_O",
        "OH_S",
        "SH_S",
        "PI_PI",
        "Hal_(X)_(N,O)",
        "(NH,OH,SH,CH)_F",
        "S_O",
        "CH_O",
        "CH_N",
        "CH_S",
        "S_PI",
        "Elec_(NH,OH)_(N,O)",
        "Elec_(N,O)H_OH",
        "Dipo",
        "CH_PI",
        "NH_PI",
        "OH_PI",
        "S_NH",
        "S_N",
        "S_F",
    ]

    try:
        with open(output, "w") as wf:
            for interaction_type in interaction_type_list:
                if interaction_type == "HB_OH_(N,O)":
                    wf.write(
                        "# interaction_type dist(Ang) angle1(deg) angle2_min(deg) angle2_max(deg)\n"
                    )

                elif interaction_type == "HB_OH_OH":
                    wf.write("\n# interaction_type dist(Ang) angle1(deg) angle2(deg) angle3(deg)\n")

                elif interaction_type == "HB_NH_(N,O)":
                    wf.write(
                        "\n# interaction_type dist(Ang) angle1(deg) angle1_N4(deg) angle2_min(deg) angle2_max(deg)\n"
                    )

                elif interaction_type == "HB_NH_OH":
                    wf.write(
                        "\n# interaction_type dist(Ang) angle1(deg) angle1_N4(deg) angle2(deg) angle3(deg)\n"
                    )

                elif interaction_type == "vdW":
                    wf.write("\n# interaction_type buffer(Ang)\n")

                elif interaction_type == "SH_N":
                    wf.write("\n# interaction_type buffer(Ang) angle(deg)\n")

                elif interaction_type == "S_PI":
                    wf.write("\n# interaction_type buffer(Ang) angle2(deg)\n")

                elif interaction_type == "Elec_(NH,OH)_(N,O)":
                    wf.write(
                        "\n# interaction_type dist(Ang) buffer(Ang) angle1(deg) angle1_N4(deg) angle2_min(deg) angle2_max(deg)\n"
                    )

                elif interaction_type == "Elec_(N,O)H_OH":
                    wf.write(
                        "\n# interaction_type dist(Ang) buffer(Ang) angle1(deg) angle1_N4(deg)  angle2(deg) angle3(deg)\n"
                    )

                elif interaction_type == "Dipo":
                    wf.write(
                        "\n# interaction_type buffer(Ang) angle1(Ang) angle2(Ang) angle3(Ang) charge hydro\n"
                    )

                elif interaction_type == "CH_PI":
                    wf.write(
                        "\n# interaction_type buffer1(Ang) buffer2(Ang) dist_ratio1 dist_ratio2 angle1_min angle1_max\n"
                    )

                elif interaction_type == "S_NH":
                    wf.write(
                        "\n# interaction_type buffer(Ang) angle1_min(deg) angle1_max(deg) angle2 dihedral_min(deg) dihedral_max(deg)\n"
                    )

                elif interaction_type == "S_N":
                    wf.write(
                        "\n# interaction_type buffer(Ang) angle1_min(deg) angle1_max(deg) angle2(deg) dihedral1_min(deg) dihedral1_max(deg) dihedral2_min(deg) dihedral2_max(deg)\n"
                    )

                elif interaction_type == "S_F":
                    wf.write("\n# interaction_type buffer(Ang) angle1(deg)\n")

                wf.write(
                    "{key}: {val}\n".format(
                        key=interaction_type, val=" ".join(param_dict[interaction_type])
                    )
                )

    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


def edit_param(parameter_file: str, buffer_dist: float) -> dict:
    """パラメータファイルを読み込み、バッファ距離の値を変更したパラメータ辞書を出力する関数

    Args:
        parameter_file (str): パラメータファイル
        buffer_dist (float): バッファ距離

    Returns:
        dict: パラメータ辞書（key=相互作用名、value=パラメータのリスト）
    """

    # バッファ距離をパラメータに持つ相互作用リスト
    interaction_type_list = [
        "vdW",
        "(Met)_(X)",
        "(Ion)_(X)",
        "SH_N",
        "SH_O",
        "OH_S",
        "SH_S",
        "PI_PI",
        "Hal_(X)_(N,O)",
        "(NH,OH,SH,CH)_F",
        "S_O",
        "CH_O",
        "CH_N",
        "CH_S",
        "S_PI",
        "Elec_(NH,OH)_(N,O)",
        "Elec_(N,O)H_OH",
        "Dipo",
        "S_NH",
        "S_N",
        "S_F",
    ]

    try:
        param_dict = dict()
        with open(parameter_file, "r") as rf:
            yml = yaml.safe_load(rf)
            for key in yml.keys():
                values = str(yml[key]).split()

                # バッファ距離を指定値に変更
                if key in interaction_type_list:
                    if key in ["Elec_(NH,OH)_(N,O)", "Elec_(N,O)H_OH"]:
                        values[1] = str(buffer_dist)
                    else:
                        values[0] = str(buffer_dist)

                param_dict[key] = values
        return param_dict

    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='edit "parameter.txt".', formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("parameter_file", help="File to edit.", type=str)
    parser.add_argument("output", help="File to edit.", type=str)
    parser.add_argument("buffer", help="buffer distance parameter", type=float)

    args = parser.parse_args()
    param = edit_param(parameter_file=args.parameter_file, buffer_dist=args.buffer)
    write_param(param_dict=param, output=args.output)
