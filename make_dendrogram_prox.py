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
DEFAULT_EXCEL_PATH = SCRIPT_DIR / "SBT2501_prox.xlsx"

LINE_WIDTH = 1.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dendrogram from a pre-computed proximity matrix (e.g. SPSS CHISQ output)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Excel file containing a proximity matrix in the 'data' sheet and labels in the 'name' sheet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path.",
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


def build_output_path(excel_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()

    stem_prefix = excel_path.stem.split("_", 1)[0].strip() or excel_path.stem
    timestamp = datetime.now().strftime("%m%d%H%M")
    return (SCRIPT_DIR / f"{stem_prefix}_dendrogram_{timestamp}.png").resolve()


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


def load_proximity_matrix(workbook: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name=sheet_name, index_col=0)
    if df.empty:
        raise ValueError("The data sheet is empty.")

    df.columns = [normalize_code(col) for col in df.columns]
    df.index = [normalize_code(idx) for idx in df.index]

    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("The data sheet contains non-numeric values.")
    if numeric.shape[0] != numeric.shape[1]:
        raise ValueError("The data sheet must be a square matrix.")
    if not np.allclose(numeric.values, numeric.values.T, atol=1e-6):
        raise ValueError("The proximity matrix must be symmetric.")
    if not np.allclose(np.diag(numeric.values), 0, atol=1e-6):
        raise ValueError("The diagonal of the proximity matrix must be 0.")

    numeric.index = numeric.columns
    return numeric


def format_label(code: str, label: str) -> str:
    match = re.search(r"\.(\d+)$", code)
    suffix = match.group(1).zfill(2) if match else code
    return f"{label} {suffix}"


def main() -> None:
    args = parse_args()
    excel_path = resolve_excel_path(args.input)
    output_path = build_output_path(excel_path, args.output)

    set_japanese_font()

    workbook = pd.ExcelFile(excel_path)
    data_sheet = find_data_sheet(workbook)
    name_sheet = find_name_sheet(workbook)

    label_map = load_labels(workbook, name_sheet)
    prox = load_proximity_matrix(workbook, data_sheet)

    missing_codes = [code for code in prox.columns if code not in label_map.index]
    if missing_codes:
        raise ValueError(f"Codes missing in the label sheet: {missing_codes}")

    labels = [format_label(code, label_map[code]) for code in prox.columns]

    condensed = squareform(prox.values)
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
    plt.title("Dendrogram (SPSS chi-square measure, Ward)")
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
