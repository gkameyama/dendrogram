# Dendrogram Maker

共起行列（Excel）からSPSS互換のカイ2乗距離とWard法を用いてデンドログラムを生成するツールです。
GUIとコマンドライン（CLI）の両方で使用できます。

---

## ファイル構成

```
(保存先)/
├── make_dendrogram.py        # コア処理・CLI
├── make_dendrogram_gui.py    # GUIアプリ（tkinter）
└── start_dendrogram_gui.bat  # GUIをダブルクリックで起動するバッチ
```

---

## 必要環境

- Python 3.x
- 以下のパッケージ（`pip install` で導入）:

```
matplotlib
numpy
pandas
scipy
openpyxl
```

---

## 入力ファイル形式（Excel）

Excelファイルには以下の2つのシートが必要です。

| シート名 | 内容 |
|----------|------|
| `data`   | 正方な共起行列（行・列ともに変数コード） |
| `name`   | コードとラベルの対応表（A列：コード、B列：ラベル） |

シート名は大文字・小文字を問いません。`data` / `name` 以外の名前でも、内容から自動判別します。

---

## 使い方

### GUI（推奨）

`start_dendrogram_gui.bat` をダブルクリックして起動します。

1. **Input Excel** — 入力Excelファイルを選択（省略時はスクリプトと同じフォルダの `.xlsx` を自動検出）
2. **Output PNG** — デンドログラムの保存先を指定（省略時は自動命名）
3. **Distance Matrix** — カイ2乗距離行列の保存先を指定（省略時は自動命名）
4. **Create Dendrogram** ボタンをクリック

### コマンドライン（CLI）

```bash
python make_dendrogram.py [--input <Excelファイル>] [--output <出力PNG>] [--output-matrix <距離行列Excel>]
```

#### オプション

| オプション | 説明 |
|------------|------|
| `--input`  | 入力Excelファイルのパス（省略時はスクリプトと同じフォルダの `.xlsx` を自動検出） |
| `--output` | 出力PNGのパス（省略時は `<入力ファイル名>_dendrogram_<日時>.png`） |
| `--output-matrix` | 距離行列Excelのパス（省略時は `<入力ファイル名>_chisqd_<日時>.xlsx`） |

#### 実行例

```bash
# スクリプトフォルダの .xlsx を自動検出して実行
python make_dendrogram.py

# 入力ファイルを指定して実行
python make_dendrogram.py --input data/beer.xlsx

# すべてのパスを指定して実行
python make_dendrogram.py --input data/beer.xlsx --output out/dendrogram.png --output-matrix out/chisqd.xlsx
```

---

## 出力ファイル

| ファイル | 内容 |
|----------|------|
| `<名前>_dendrogram_<日時>.png` | デンドログラム画像（300 dpi） |
| `<名前>_chisqd_<日時>.xlsx`   | カイ2乗距離行列（`chisqd` シート） |

---

## アルゴリズム

- **距離尺度**: SPSS `PROXIMITIES MEASURE=CHISQ` と同等のカイ2乗距離
  - 列ペアごとに共起行列の2列を取り出し、カイ2乗統計量の平方根を距離とする
- **クラスタリング**: Ward 法（`scipy.cluster.hierarchy.linkage` の `method="ward"`）
- **日本語フォント**: Yu Gothic → Meiryo → BIZ UDGothic → MS Gothic → Noto Sans CJK JP → IPAexGothic の順に自動選択
