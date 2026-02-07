# Intel LiDAR 断面図化

護岸前面の LiDAR 点群から「1本の垂直断面」の形状を抽出し、時間経過による変形（初期との差分）を mm 単位で可視化する。

---

## 1. 操作方法

### 初回セットアップ

```bash
cp -r params_sample params
```

その後 `params/<case>.yaml` を編集し、`points`（point1, point2, revetment）と必要なら `ply_directory` を設定。PLY ファイルは `data/<case>/` に配置。

### 実行

1. `notebooks/LiDAR_viz.ipynb` を開く
2. 先頭セルの `params_path` を編集（例: `No7-friction-8mm.yaml` に変更）
3. セルをすべて実行

出力は `output/<case>/figures/` に保存される。

### Google Colab

プロジェクトを Drive に置き、先頭セルを実行。Colab は自動で Drive をマウントし、`pip install -e .` は不要（既存の numpy/scipy/matplotlib で動作）。

---

## 2. パラメータ（YAML）

### ケース YAML（`params/<case>.yaml`）で必須

| 名前 | 意味 |
|------|------|
| **point1** | 断面直線の端（例: 水槽側）[x, y, z] m |
| **point2** | 断面直線のもう一方の端（例: 土層側）[x, y, z] m |
| **revetment** | 護岸内側の点。Zoom の 0 mm 基準 [x, y, z] m |
| **ply_directory** | （省略可）PLY フォルダ。省略時は `data/<case>/` |

### 共通パラメータ（`params/base.yaml`）

断面・前処理・描画の詳細は `params_sample/base.yaml` を参照。主な項目:

- `cross_section`: 断面の閾値・Z 範囲・インデックス
- `common_grid`: 共通グリッドの範囲・点数
- `filter`: 等間隔化・メディアンフィルタ・Savitzky–Golay
- `ground`: 地面推定 RANSAC の範囲・閾値
- `plot`: 凡例・Zoom 幅・Y 軸範囲など

---

## 3. 処理の流れ（概要）

```
PLY 読み込み → 地面で鉛直補正 → 断面抽出 → プロファイル作成
  → 前処理（等間隔化・メディアン・平滑化）→ 共通グリッドへ再サンプリング
  → 初期との差分計算 → 図化（断面・差分・Zoom）
```

- **断面**: point1–point2 直線を含む鉛直平面から、一定距離以内の点を抽出
- **プロファイル**: 距離 [mm] vs 高さ [mm]
- **差分**: 最初の PLY との高さ差 [mm]

---

## 4. 環境

- Python 3.12
- 依存: numpy, scipy, matplotlib, PyYAML

```bash
pip install numpy scipy matplotlib PyYAML
# Jupyter 用（ローカル）
pip install notebook ipykernel
```

Colab では既存パッケージで動作（不要に応じて `!pip install PyYAML` のみ）。
