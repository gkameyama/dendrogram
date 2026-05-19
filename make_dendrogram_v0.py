from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXCEL_PATH = SCRIPT_DIR / "SBT2501_data.xlsx"
VALID_CLUSTER_MODES = ("chi2_average", "chi2_ward", "sqeuclidean_average")

# Priority:
# 1. chi2_ward: keep Ward's method to stay close to the conventional output.
# 2. chi2_average: fallback that is more natural for pure chi-square distance.
CLUSTER_MODE = "chi2_ward"
LINE_WIDTH = 1.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dendrogram image from an Excel workbook."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Excel file path. If omitted, the script auto-detects an xlsx file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path.",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_CLUSTER_MODES,
        default=CLUSTER_MODE,
        help="Clustering mode.",
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
    if len(xlsx_files) == 1:
        return xlsx_files[0]
    if not xlsx_files:
        raise FileNotFoundError("No xlsx file was found in the script directory.")

    names = ", ".join(path.name for path in xlsx_files)
    raise FileNotFoundError(
        "Multiple xlsx files were found. Use --input to specify one: "
        f"{names}"
    )


def build_output_path(excel_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()

    stem_prefix = excel_path.stem.split("_", 1)[0].strip() or excel_path.stem
    timestamp = datetime.now().strftime("%m%d%H%M")
    file_name = f"{stem_prefix}_dendrogram_{timestamp}.png"
    return (SCRIPT_DIR / file_name).resolve()


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
    preferred = [sheet for sheet in workbook.sheet_names if sheet.lower() == "name"]
    if preferred:
        return preferred[0]

    for sheet in workbook.sheet_names:
        df = pd.read_excel(workbook, sheet_name=sheet, header=None)
        if df.shape[1] >= 2 and df.iloc[:, :2].notna().all().all():
            return sheet

    raise ValueError("Could not find a label sheet like the expected 'name' sheet.")


def find_data_sheet(workbook: pd.ExcelFile) -> str:
    preferred = [sheet for sheet in workbook.sheet_names if sheet.lower() == "data"]
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


def format_label(code: str, label: str) -> str:
    match = re.search(r"\.(\d+)$", code)
    suffix = match.group(1).zfill(2) if match else code
    return f"{label} {suffix}"


def chi_square_distance_matrix(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    grand_total = values.sum()
    if grand_total <= 0:
        raise ValueError("data sheet total must be greater than 0.")

    row_sums = values.sum(axis=1, keepdims=True)
    col_masses = values.sum(axis=0) / grand_total

    if np.any(row_sums == 0):
        raise ValueError("Chi-square distance cannot be computed because some row sums are 0.")
    if np.any(col_masses == 0):
        raise ValueError(
            "Chi-square distance cannot be computed because some column masses are 0."
        )

    row_profiles = values / row_sums
    weighted_profiles = row_profiles / np.sqrt(col_masses)
    condensed = pdist(weighted_profiles, metric="euclidean")
    return squareform(condensed)


def chi_square_ward_features(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    grand_total = values.sum()
    row_sums = values.sum(axis=1, keepdims=True)
    col_masses = values.sum(axis=0) / grand_total

    if grand_total <= 0:
        raise ValueError("data sheet total must be greater than 0.")
    if np.any(row_sums == 0):
        raise ValueError("Ward features cannot be built because some row sums are 0.")
    if np.any(col_masses == 0):
        raise ValueError("Ward features cannot be built because some column masses are 0.")

    row_profiles = values / row_sums
    return row_profiles / np.sqrt(col_masses)


def squared_euclidean_distance_matrix(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("The data sheet is empty.")
    condensed = pdist(values, metric="sqeuclidean")
    return squareform(condensed)


def build_linkage(matrix: pd.DataFrame, cluster_mode: str) -> np.ndarray:
    if cluster_mode == "chi2_average":
        chi2_dist = chi_square_distance_matrix(matrix)
        return linkage(squareform(chi2_dist), method="average")

    if cluster_mode == "chi2_ward":
        features = chi_square_ward_features(matrix)
        return linkage(features, method="ward", metric="euclidean")

    if cluster_mode == "sqeuclidean_average":
        sqeuclidean_dist = squared_euclidean_distance_matrix(matrix)
        return linkage(squareform(sqeuclidean_dist), method="average")

    raise ValueError(
        "CLUSTER_MODE must be one of "
        f"{', '.join(repr(mode) for mode in VALID_CLUSTER_MODES)}."
    )


def main() -> None:
    args = parse_args()
    excel_path = resolve_excel_path(args.input)
    output_path = build_output_path(excel_path, args.output)
    cluster_mode = args.mode

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
    linkage_matrix = build_linkage(matrix, cluster_mode)

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
    plt.title(f"Dendrogram ({cluster_mode})")
    plt.xlabel("Distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"input : {excel_path}")
    print(f"data  : {data_sheet}")
    print(f"label : {name_sheet}")
    print(f"saved : {output_path}")


if __name__ == "__main__":
    main()
