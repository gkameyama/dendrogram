from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXCEL_PATH = SCRIPT_DIR / "SBT2501_data.xlsx"

LINE_WIDTH = 1.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a dendrogram from a co-occurrence matrix using the SPSS-equivalent "
            "chi-square distance (chisqd) and Ward's method."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Excel file path (data sheet = square co-occurrence matrix, name sheet = labels).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path.",
    )
    parser.add_argument(
        "--output-matrix",
        type=Path,
        default=None,
        help="Output Excel path for the distance matrix (default: auto-generated alongside the image).",
    )
    return parser.parse_args()


def resolve_excel_path(candidate: Path | None) -> Path:
    if candidate is not None:
        path = candidate.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")
        return path

    if DEFAULT_EXCEL_PATH.exists():
        return DEFAULT_EXCEL_PATH

    xlsx_files = sorted(SCRIPT_DIR.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError("No xlsx file was found in the script directory.")
    if len(xlsx_files) == 1:
        return xlsx_files[0]

    names = ", ".join(p.name for p in xlsx_files)
    raise FileNotFoundError(
        "Multiple xlsx files were found. Use --input to specify one: " f"{names}"
    )


def build_output_path(excel_path: Path, output_path: Path | None) -> tuple[Path, str]:
    stem_prefix = excel_path.stem.split("_", 1)[0].strip() or excel_path.stem
    timestamp = datetime.now().strftime("%m%d%H%M")
    if output_path is not None:
        return output_path.expanduser().resolve(), timestamp
    return (SCRIPT_DIR / f"{stem_prefix}_dendrogram_{timestamp}.png").resolve(), timestamp


def build_matrix_output_path(
    excel_path: Path, matrix_path: Path | None, timestamp: str
) -> Path:
    if matrix_path is not None:
        return matrix_path.expanduser().resolve()
    stem_prefix = excel_path.stem.split("_", 1)[0].strip() or excel_path.stem
    return (SCRIPT_DIR / f"{stem_prefix}_chisqd_{timestamp}.xlsx").resolve()


def set_japanese_font() -> None:
    candidates = [
        "Yu Gothic",
        "Meiryo",
        "BIZ UDGothic",
        "MS Gothic",
        "Noto Sans CJK JP",
        "IPAexGothic",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["lines.linewidth"] = LINE_WIDTH


def normalize_code(value: object) -> str:
    return str(value).strip()


def find_name_sheet(workbook: pd.ExcelFile) -> str:
    preferred = [s for s in workbook.sheet_names if s.lower() == "name"]
    if preferred:
        return preferred[0]

    for sheet in workbook.sheet_names:
        df = pd.read_excel(workbook, sheet_name=sheet, header=None)
        if df.shape[1] >= 2 and df.iloc[:, :2].notna().all().all():
            return sheet

    raise ValueError("Could not find a label sheet like the expected 'name' sheet.")


def find_data_sheet(workbook: pd.ExcelFile) -> str:
    preferred = [s for s in workbook.sheet_names if s.lower() == "data"]
    if preferred:
        return preferred[0]

    for sheet in workbook.sheet_names:
        df = pd.read_excel(workbook, sheet_name=sheet)
        if df.empty:
            continue
        numeric = df.apply(pd.to_numeric, errors="coerce")
        if numeric.shape[0] == numeric.shape[1] and not numeric.isna().any().any():
            return sheet

    raise ValueError("Could not find a square numeric sheet like the expected 'data' sheet.")


def load_labels(workbook: pd.ExcelFile, sheet_name: str) -> pd.Series:
    name_df = pd.read_excel(workbook, sheet_name=sheet_name, header=None).iloc[:, :2]
    if name_df.empty:
        raise ValueError("The label sheet is empty.")

    name_df.columns = ["code", "label"]
    name_df["code"] = name_df["code"].map(normalize_code)
    name_df["label"] = name_df["label"].astype(str).str.strip()
    name_df = name_df[(name_df["code"] != "") & (name_df["label"] != "")]

    if name_df.empty:
        raise ValueError("No valid code-label rows were found in the label sheet.")
    if name_df["code"].duplicated().any():
        duplicates = name_df.loc[name_df["code"].duplicated(), "code"].tolist()
        raise ValueError(f"Duplicate codes were found in the label sheet: {duplicates}")

    return name_df.set_index("code")["label"]


def load_matrix(workbook: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name=sheet_name)
    if df.empty:
        raise ValueError("The data sheet is empty.")

    df.columns = [normalize_code(col) for col in df.columns]
    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("The data sheet contains non-numeric values.")
    if numeric.shape[0] != numeric.shape[1]:
        raise ValueError("The data sheet must be a square matrix.")

    numeric.index = numeric.columns
    return numeric


def chisqd(X: np.ndarray) -> np.ndarray:
    """SPSS PROXIMITIES MEASURE=CHISQ VIEW=VARIABLE と同等の距離行列を計算する。

    各列ペア (i, j) について共起行列の2列を取り出し、カイ2乗統計量の平方根を距離とする。
    R の chisqd() 関数（chisqd.r）と完全に同じ結果を返す。

    Parameters
    ----------
    X : ndarray, shape (V, V)
        正方な共起行列（対称・非負）。

    Returns
    -------
    D : ndarray, shape (V, V)
        ペア間のカイ2乗距離行列（対角 = 0）。
    """
    V = X.shape[1]
    D = np.zeros((V, V))

    for i in range(V):
        for j in range(i + 1, V):
            mat = np.column_stack([X[:, i], X[:, j]])   # V×2
            row_sums = mat.sum(axis=1, keepdims=True)
            col_sums = mat.sum(axis=0)
            grand = mat.sum()

            if grand == 0:
                continue

            ex = (row_sums * col_sums) / grand          # V×2 期待値

            with np.errstate(invalid="ignore", divide="ignore"):
                y = (mat - ex) ** 2 / ex
                y[np.isnan(y)] = 0                       # 期待値=0 のセルを除外

            d = np.sqrt(y.sum())
            D[i, j] = D[j, i] = d

    return D


def format_label(code: str, label: str) -> str:
    match = re.search(r"\.(\d+)$", code)
    suffix = match.group(1).zfill(2) if match else code
    return f"{label} {suffix}"


def save_distance_matrix(
    dist_matrix: np.ndarray, codes: list[str], matrix_path: Path
) -> None:
    df = pd.DataFrame(dist_matrix, index=codes, columns=codes)
    df.index.name = None
    with pd.ExcelWriter(matrix_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="chisqd")


def main() -> None:
    args = parse_args()
    excel_path = resolve_excel_path(args.input)
    output_path, timestamp = build_output_path(excel_path, args.output)
    matrix_path = build_matrix_output_path(excel_path, args.output_matrix, timestamp)

    set_japanese_font()

    workbook = pd.ExcelFile(excel_path)
    data_sheet = find_data_sheet(workbook)
    name_sheet = find_name_sheet(workbook)

    label_map = load_labels(workbook, name_sheet)
    matrix = load_matrix(workbook, data_sheet)

    missing_codes = [code for code in matrix.columns if code not in label_map.index]
    if missing_codes:
        raise ValueError(f"Codes missing in the label sheet: {missing_codes}")

    labels = [format_label(code, label_map[code]) for code in matrix.columns]
    codes = list(matrix.columns)

    dist_matrix = chisqd(matrix.to_numpy(dtype=float))

    save_distance_matrix(dist_matrix, codes, matrix_path)

    condensed = squareform(dist_matrix)
    linkage_matrix = linkage(condensed, method="ward")

    plt.figure(figsize=(12, 18))
    dendrogram(
        linkage_matrix,
        labels=labels,
        orientation="right",
        leaf_rotation=0,
        leaf_font_size=10,
        color_threshold=0,
        link_color_func=lambda _: "black",
    )
    plt.title("Dendrogram (SPSS chi-square distance, Ward)")
    plt.xlabel("Distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"input  : {excel_path}")
    print(f"data   : {data_sheet}")
    print(f"label  : {name_sheet}")
    print(f"matrix : {matrix_path}")
    print(f"saved  : {output_path}")


if __name__ == "__main__":
    main()
