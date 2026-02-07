# context.md — LiDAR 断面図化（Notebook主導・パラメータ可変）

## このプロジェクトの目的（Cursor向け）
Intel LiDAR 点群（PLY）から、護岸前面の「ある垂直断面（指定直線スライス）」の 1D 形状プロファイル（距離–Z）を抽出し、前処理（等間隔化・スパイク除去・平滑化）を適用したうえで、共通グリッドに揃えて「初期断面との差分（mm）」の時間変化を可視化する。  
実行・パラメータ変更は **Jupyter Notebook が唯一の入口** とし、CLI は作らない。  
点群の読み込み・地面平面推定・幾何計算は **Open3D に依存せず**、NumPy・SciPy・標準ライブラリで実装している。

---

## LiDAR 断面処理の全体フロー

このプロジェクトでは、Intel LiDAR から得られた点群（PLY）を用いて、護岸前面の断面変位プロファイルを作成し、その時間変化（初期との差分）を評価しています。処理の主な流れは次のとおりです。

---

### 1. データ構成

- `data/LiDAR/<case>/`  
  - LiDAR 点群（`.ply`）を格納するディレクトリ。`<case>` は例: `No4-friction-8mm`, `No7-friction-8mm`。  
  - 例: `4s80.ply`, `4s82.ply`, `4s810.ply`, ...
- `params/<case>.yaml`  
  - 断面直線の2点（`point1`, `point2`）と護岸点（`revetment`）を YAML で定義。データフォルダ名と同名（例: `No4-friction-8mm.yaml`）で配置。
- `src/lidar/preprocess_lidar.py`  
  - 断面プロファイルの平滑化・前処理関数群。
- `notebooks/LiDAR_visualize.ipynb`  
  - 点群から断面を抽出し、Z軸断面グラフと 3D / X-Y / Y-Z 図を作成する解析ノートブック。

---

### 2. 点群から断面への変換（`LiDAR_visualize.ipynb`）

1. **パラメータと PLY 読み込み**
   - プロジェクトルートは `(Path.cwd() / "src").is_dir()` で判定。`ply_directory` はプロジェクト直下の `data/LiDAR/<case>` を優先し、無ければ親ディレクトリの同パスを使用。ノートブックが `notebooks/` から実行されても、プロジェクトルートから実行されても正しく `data/LiDAR/<case>` を指すようにパス解決している。
   - 断面直線の2点（`point1`, `point2`）と護岸点（`revetment`）は、`params/<case>.yaml`（`<case>` は `Path(ply_directory).name`）から YAML で読み込み、`np.array` に変換。
   - `src.lidar.io_ply.load_point_cloud(ply_file)` で点群を読み込む。PLY は **標準ライブラリ（`re`, `struct`, `pathlib`）＋ NumPy** でパース（Open3D 非依存）。ASCII の「1頂点複数行」形式（Realsense Viewer 形式）およびバイナリ形式（`float32` プロパティ）に対応。型サイズと struct 書式はモジュール定数で整理されている。

2. **鉛直（傾き）補正（地面平面の自動推定を基本とする）**
   - 点群には地面が含まれているため、手動の tilt 点指定を必須とせず、地面（または卓越平面）を **NumPy ベースの RANSAC 平面推定**（`estimate_ground_plane`）で抽出する。
   - 推定した平面の法線ベクトルは **Z 成分が正になる向きに統一**し、回転行列で +Z を「上」に揃える（天地の向きが Open3D と互換になる）。全点を回転して Z 軸を鉛直方向に揃える。
   - 地面抽出の ROI（高さ範囲・XY 範囲）は `_apply_roi_filter` で適用した後に RANSAC を実行する。（必要なら）Notebook 上で ROI を調整可能。

   **Implementation note:**
   - 地面抽出のROIパラメータ（Notebook先頭セルで定義）：
     - `ground_z_range`: Z座標の範囲フィルタ `(min_z, max_z)` または `None`。`None` の場合はフィルタなし。
     - `ground_xy_range`: XY座標の範囲フィルタ `((min_x, max_x), (min_y, max_y))` または `None`。`None` の場合はフィルタなし。
   - RANSAC平面推定のパラメータ（Notebook先頭セルで定義）：
     - `ground_distance_threshold`: RANSACの距離閾値（単位: m、デフォルト: 0.01）
     - `ground_ransac_n`: RANSACのサンプル数（デフォルト: 3）
     - `ground_num_iterations`: RANSACの反復回数（デフォルト: 1000）
   - これらのパラメータは `estimate_ground_plane` 関数に渡され、ROIフィルタ適用後にRANSAC平面推定が実行される。

3. **断面の抽出（回転前の元データ座標で指定した直線に沿ったスライス）**
   - X-Y 平面で 2点 `point1`, `point2` を結ぶ直線を定義する（`point1`, `point2` は `params/<case>.yaml` の `points` から読み込む。回転前の元データ座標系）。
   - 断面抽出は、**断面直線を含み、新しいz軸（地面平面の法線）に平行な平面**から一定距離（例: 0.01 m）以内の点を抽出する。
   - 平面の法線ベクトルは、断面直線の方向ベクトルと新しいz軸の外積で計算される。
   - 抽出条件：
     - 平面からの距離が `cross_section_threshold` 以内
     - `point1` と `point2` の間にある点（射影距離が0以上、直線長以下）
     - （オプション）新しいz軸でのZ座標が `cross_section_z_range` の範囲内（床などの点を除外するため）
   - 抽出した「断面点」に対して 2. の鉛直補正（回転）を適用し、断面プロファイル作成に用いる。

   **Implementation note:**
   - 平面の法線ベクトルは、断面直線の方向ベクトル `line_dir = (point2 - point1) / ||point2 - point1||` と新しいz軸 `plane_normal` の外積 `plane_normal_vec = line_dir × plane_normal` で計算される。外積が0（断面直線が新しいz軸と平行）の場合は特別処理を行う。
   - 平面からの距離は、各点 `p` から平面上の点（`point1`）へのベクトル `vec_to_plane = p - point1` を平面の法線ベクトル `plane_normal_vec` に射影した値：`distance = dot(vec_to_plane, plane_normal_vec)`（単位: m、符号付き）
   - 射影距離は、各点 `p` から `point1` へのベクトル `vec_to_point1 = p - point1` を直線方向ベクトル `line_dir` に射影した値：`proj_length = dot(vec_to_point1, line_dir)`（単位: m、符号付き）。この値が0以上かつ `||point2 - point1||` 以下である点のみを抽出する。

4. **断面 Z データの切り出し**
   - 鉛直補正後の断面点の Z 座標を取り出し、平均値オフセットは取らずに mm 換算：
     \[
       \text{displacements} = z\_\text{coords} \times 1000 \;[\text{mm}]
     \]
   - 回転後のZ座標は、地面平面の法線方向（新しいZ軸方向）への変位を表しており、  
     これは地面平面からの垂直距離を表している。
   - 断面方向の距離インデックスとして、`start_index`〜`end_index_max` を指定し、  
     その範囲をクリップして利用する。

---

### 3. 断面プロファイルの前処理（`preprocess_lidar_profile`）

`src/lidar/preprocess_lidar.py` の `preprocess_lidar_profile(x, z, ...)` を用いて、  
距離 `x` と変位 `z` の 1D プロファイルに対して次の処理を行います。

1. **前処理共通**
   - NaN / 無限大を除去。
   - `x` の昇順ソート。
   - 重複する `x` がある場合は、対応する `z` の平均を取り 1点にまとめる。

2. **(必要に応じて) 等間隔化：`resample_to_uniform_grid`**
   - `diff(x)` のばらつき（CV および range_ratio）から不等間隔かどうかを判定。
     - CV = `std(diff) / median(diff)`
     - range_ratio = `(max(diff) - min(diff)) / median(diff)`
   - いずれかが `uneven_tol`（デフォルト 0.05）を超える場合、「不等間隔」とみなして補間を実施。
   - `dx` が None の場合、`median(diff(x))` をグリッド間隔として使用。
   - `scipy.interpolate.interp1d` により、指定の補間方法（線形など）で等間隔グリッド上に再サンプリング。

3. **メディアンフィルタによるスパイク除去：`despike_median`**
   - `scipy.ndimage.median_filter` を用いてウィンドウ長 `k`（例: 5点）でローカルメディアンを計算。
   - LiDAR 特有のスパイク（孤立した突出値）を抑制する目的。

4. **Savitzky–Golay フィルタによる平滑化：`smooth_savgol`**
   - `scipy.signal.savgol_filter` を用いて，窓長 `window`（奇数）と多項式次数 `polyorder`（例: 3）を指定。
   - ローカルな多項式フィットにより、**ピークや曲率を保ちつつギザギザを平滑化**。
   - `window * dx` が物理空間での平滑化スケール（長さ）に相当する。

5. **まとめ関数：`preprocess_lidar_profile`**
   - 上記 2〜4 をまとめて実行し、以下を返す：
     - `x`: 等間隔化後の距離
     - `z_raw`: 等間隔化後（メディアン前）の z
     - `z_median`: メディアンフィルタ後
     - `z_smooth`: Savitzky–Golay フィルタ後
     - `meta`: 使用したパラメータや不等間隔判定の統計量

---

### 4. 共通グリッドへの再サンプリング（全断面で長さを統一）

`notebooks/LiDAR_visualize.ipynb` では、各断面の前処理後プロファイル（`pp.x`, `pp.z_smooth`）を、  
**共通の距離グリッド上に再サンプリング**してから差分を取っています。

1. **共通グリッドの定義**
   - 例: 0〜165 mm を 400 点に分割
   - `common_x = np.linspace(common_x_min, common_x_max, common_x_points)`
   - `common_x_max = None` の場合、断面直線（point1からpoint2）の長さから自動設定される

   **Implementation note:**
   - `common_x_max = None` の場合、自動設定は以下の式で行われる：
     - `line_length = ||point2 - point1||`（単位: m）
     - `common_x_max = line_length * 1000.0`（単位: mmに変換）
   - 自動設定後、`common_x_max > common_x_min` が成立することを安全チェックで確認する。

2. **各断面に対して**
   - `preprocess_lidar_profile` で前処理した後、以下のデータを共通グリッドに再サンプリングする：
     - **等間隔処理前のデータ**（`x_clipped, z_clipped`）→ `all_before_resample`：処理検証用
     - **フィルター適用前のデータ**（`x_processed, z_raw`）→ `all_before_filter`：処理検証用
     - **最終処理済みデータ**（`x_processed, z_smooth`）→ `all_displacements`：計測値として使用

     ```python
     # 等間隔処理前のデータを共通グリッドに再サンプリング
     z_before_resample = resample_to_common_grid(x_clipped, z_clipped, common_x, method=filter_interp_method)
     
     # フィルター適用前のデータを共通グリッドに再サンプリング
     z_before_filter = resample_to_common_grid(x_processed, z_raw, common_x, method=filter_interp_method)
     
     # 最終処理済みデータを共通グリッドに再サンプリング
     z_on_common = resample_to_common_grid(x_processed, z_smooth, common_x, method=filter_interp_method)
     ```

   - これにより、どの断面も `z_on_common` が共通の `common_x`（長さ `common_x_points`）の上に定義される。
   - 処理検証用のデータ（`all_before_resample`, `all_before_filter`）も共通グリッド上に再サンプリングされ、可視化で使用される。

   **Implementation note:**
   - `all_before_resample`, `all_before_filter`, `all_displacements`, `all_differences` は、各要素が長さ `common_x_points` の `np.ndarray` である `List[np.ndarray]` 型のデータ構造である。
   - 各リストの `idx` 番目の要素は、`idx` 番目のPLYファイル（自然順ソート後）に対応する。
   - `all_differences[idx]` は、`idx == 0` の場合は `np.zeros_like(all_displacements[0])`（NaN除く）、それ以外の場合は `all_displacements[idx] - all_displacements[0]` で計算される。

3. **初期断面との差分**
   - 最初の断面（idx == 0）をベースラインとし、その後の断面とは同じ `common_x` 上で差分を取る：

     ```python
     if idx == 0:
         baseline_displacement = shifted_displacements
         difference_displacement = np.zeros_like(shifted_displacements)
     else:
         difference_displacement = shifted_displacements - baseline_displacement
     ```

   - これにより、**全断面で距離軸・サンプル数が揃った状態での時間差分プロファイル**が得られる。

---

### 5. 可視化（断面差分＋3D / X-Y / Y-Z）

1. **初期断面の可視化（`plot_initial_section_with_lines`）**
   - 初期断面（idx==0）の点群を可視化し、断面直線と新しいZ軸方向（地面平面の法線）を重ねて表示する。
   - **元座標系**と**新しいxyz軸（回転後）**の両方を表示する（2行3列のサブプロット）。
   - 1行目：元座標系でのXY平面、YZ平面、XZ平面
   - 2行目：新しいxyz軸でのXY平面、YZ平面、XZ平面
   - 地面と判定された点は青色で表示され、断面点は赤色で表示される。
   - 断面直線（point1とpoint2を結ぶ直線）と新しいZ軸方向（地面平面の法線）が表示される。
   - **抽出面の範囲**も表示される（紫色の破線）：
     - 元座標系：平面の法線方向に `cross_section_threshold` の範囲で線を描画
     - 新しいxyz軸：Z軸方向に `cross_section_z_range` の範囲で線を描画（指定されている場合）

2. **断面計測値と差分の可視化（`plot_section_differences`）**
   - 各グラフを別々のFigureとして出力する。
   - **Full Range（全域）のグラフの表示順序**：
     1. 等間隔処理前（Before Resampling）：前処理前の元のプロファイル（`all_before_resample`）を表示。等間隔化処理の効果を確認できる。
     2. フィルター適用前（Before Filtering）：等間隔化後、フィルター適用前のデータ（`all_before_filter`）を表示。フィルター処理の効果を確認できる。
     3. 処理後（計測値）（Section Measurement）：各断面の新しいxyz軸での変換値（`all_displacements`）を距離方向に沿ってプロット。
        - `all_displacements`は、各断面の共通グリッド上での計測値（地面平面からの垂直距離、mm単位）を格納している。
        - これは回転後のZ座標（新しいZ軸方向への変位）を表しており、地面平面からの垂直距離を表している。
     4. 処理後の差分（Section Difference）：初期断面からの変位差（`all_differences`）を距離方向に沿ってプロット。
        - `all_differences`は、各断面の計測値から初期断面（idx==0）の計測値を引いた差分（mm単位）を格納している。
        - 初期断面（idx==0）の差分は0（NaN除く）になる。
   - **特定区間（Zoom）のグラフ**：
     - `show_zoom=True`の場合、特定区間（`xlim_zoom`で指定）のグラフを表示する。
     - 特定区間では、**差分グラフのみ**を表示する（等間隔処理前、フィルター適用前、計測値のグラフは表示しない）。
     - 各グラフは別々のFigureとして出力される。
   - **グラフの設定**：
     - 各グラフのサイズの縦横比は4:6（横:縦）に設定される（`graph_size_ratio`パラメータで指定可能）。
     - データ座標での縦横比は1:1に設定される（`set_aspect('equal')`）。
     - グラフの文字は英文のみで、英文フォント（DejaVu Sans）で表示される。
     - X軸の範囲（`xlim`）、Y軸の範囲（`ylim_displacement`, `ylim_difference`）を指定可能。
     - X軸を0からスタートさせるオプション（`x_start_from_zero`）がある。Trueの場合、全域とZoomグラフの両方でX軸が0からスタートする。
     - グラフの凡例は、`plot_legend_names`パラメータで指定可能。Noneの場合はPLYファイル名が使用される。

---

## Notebook主導の実装仕様（Cursor向けに追加：必須）

### A. Notebookがパラメータの唯一の操作面
- `notebooks/LiDAR_visualize.ipynb` が以下を **Notebook先頭セル**で定義し、処理関数へ渡す。
- **src側に固定値を書かない**（デフォルト値は持ってよいが、Notebookから常に上書きできる設計）。
- 断面直線の2点（`point1`, `point2`）と護岸点（`revetment`）は **`params/<case>.yaml`** から読み込む。`<case>` は `Path(ply_directory).name`（例: `No4-friction-8mm`）。YAML の `points` 下に `point1`, `point2`, `revetment` を `[x, y, z]` のリストで定義。

Notebook先頭セルで定義するパラメータ（例の変数名を維持）：
- データディレクトリ：
  - `ply_directory`（PLYが格納されているディレクトリのパス。プロジェクト直下の `data/LiDAR/<case>` を優先、無ければ親の同パス。文字列）
- 幾何（`params/<case>.yaml` から読み込み後、`point1`, `point2`, `revetment` が `np.ndarray` で設定される）：
  - `point1`, `point2`（断面直線の2点：回転前の元データ座標系、3要素 `[x, y, z]`）
  - `revetment`（護岸内側の点、Zoom グラフのゼロ点計算に使用）
  - `cross_section_threshold`（例：0.005 m、断面抽出の距離閾値）
  - `cross_section_z_range`（例：None または (min_z, max_z)、新しいz軸でのZ座標の範囲。Noneの場合はフィルタなし）
- 地面抽出ROI（オプション、Noneの場合は全点を使用）：
  - `ground_z_range`（例：(-1.6, -1.4) または None、Z座標の範囲フィルタ `(min_z, max_z)`）
  - `ground_xy_range`（例：None または ((min_x, max_x), (min_y, max_y))、XY座標の範囲フィルタ）
- 地面平面推定パラメータ：
  - `ground_distance_threshold`（例：0.01 m、RANSACの距離閾値）
  - `ground_ransac_n`（例：3、RANSACのサンプル数）
  - `ground_num_iterations`（例：1000、RANSACの反復回数）
- 前処理：
  - `filter_dx`（目標グリッド間隔、mm単位。Noneの場合はmedian(diff(x))を使用）
  - `filter_uneven_tol`（不等間隔判定のしきい値、デフォルト: 0.05）
  - `filter_median_k`（メディアンフィルタの窓長、サンプル数、奇数に丸められる）
  - `filter_sg_window`（Savitzky–Golayフィルタの窓長、サンプル数、奇数に調整される）
  - `filter_sg_poly`（Savitzky–Golayフィルタの多項式次数）
  - `filter_interp_method`（補間方法: "linear", "nearest", "quadratic", "cubic"）
- 共通グリッド：
  - `common_x_min`（共通グリッドの最小値、mm単位）
  - `common_x_max`（共通グリッドの最大値、mm単位。Noneの場合は断面直線の長さから自動設定）
  - `common_x_points`（共通グリッドの点数）
- サブセット：
  - `start_index`（サンプリング開始インデックス、0以上）
  - `end_index_max`（サンプリング終了インデックスの最大値、start_indexより大きい必要がある）
- 可視化：
  - `plot_xlim`（X軸の範囲 `(x_min, x_max)`、Noneの場合は自動設定）
  - `plot_ylim_displacement`（計測値グラフのY軸の範囲 `(y_min, y_max)`、Noneの場合は自動設定）
  - `plot_ylim_difference`（差分グラフのY軸の範囲 `(y_min, y_max)`、Noneの場合は自動設定）
  - `plot_x_start_from_zero`（Trueの場合、全域とZoomグラフの両方でX軸を0からスタートさせる）
  - `plot_show_zoom`（Trueの場合、全域と特定区間の両方のグラフを表示）
  - 特定区間のX軸は護岸交点（`revetment_intersection_distance_mm(revetment, point1, point2)`）を基準に `x_zero_zoom` でシフトし、`plot_zoom_width` の幅で表示
  - `plot_ylim_difference_zoom`（特定区間の差分グラフのY軸の範囲 `(y_min, y_max)`、Noneの場合は自動設定）
  - `plot_ylim_displacement_zoom`（特定区間の計測値グラフ（Section Measurement - Zoom）のY軸の範囲 `(y_min, y_max)`、Noneの場合は`plot_ylim_displacement`と同じ設定を使用）
  - `plot_graph_size_ratio`（各グラフのサイズの縦横比 `(width, height)`、例: (4, 6)で横:縦=4:6）
  - `plot_legend_names`（グラフの凡例に表示する名前のリスト。Noneの場合はPLYファイル名を使用。例: ["Initial", "5 minutes", "10 minutes", ...]。特殊な値として、"Initial" は自然順ソートで最も若い番号のファイル、"Last" は時間番号が最大のファイルに対応する。時間を表す数字（例: 20）が存在しない場合は、近傍で最も近い番号のファイルを自動選択し（差が同じ場合は大きい方を優先）、凡例中の数字も実際に使用した番号に書き換える）

### B. srcは「純粋関数」中心（Notebookから呼び出しやすく）
Notebookでの試行錯誤に耐えるよう、srcは次の方針：
- できるだけ **stateless**（状態を保持しない）
- 1関数=1責務（デバッグ容易）
- 型ヒントとdocstring必須
- I/O（PLY）と地面平面推定（RANSAC）は numpy/標準ライブラリで実装、処理は numpy 中心

推奨モジュール分割（CLIは作らない）：
- `src/lidar/io_ply.py`
  - `load_point_cloud(ply_path: str | Path) -> np.ndarray`（N×3、各点のx, y, z座標）
  - `natural_sort_paths(paths: List[Path]) -> List[Path]`（自然順ソート）
- `src/lidar/geometry.py`
  - 直線の単位方向ベクトルと長さは内部ヘルパー `_normalized_line_direction(point1, point2)` で共通化。`distance_to_line`, `project_to_line`, `distance_to_plane_containing_line`, `extract_cross_section`, `extract_cross_section_by_plane` で利用。
  - `distance_to_line(points, point1, point2) -> np.ndarray`（各点から直線への最短距離、単位: m）
  - `project_to_line(points, point1, point2) -> np.ndarray`（各点のpoint1からの射影距離、単位: m）
  - `revetment_intersection_distance_mm(revetment, point1, point2) -> float`（護岸点から断面直線への射影距離、mm単位）
  - `distance_to_plane_containing_line(points, point1, point2, plane_normal) -> np.ndarray`（各点から平面への距離、単位: m、符号付き）
  - `extract_cross_section(points, point1, point2, threshold) -> np.ndarray`（直線からの距離による抽出、M×3）
  - `extract_cross_section_by_plane(points, point1, point2, plane_normal, threshold, rotation_matrix, z_range) -> np.ndarray`（平面からの距離による抽出、新しいz軸での範囲フィルタ対応、M×3）
- `src/lidar/leveling.py`
  - 地面抽出用 ROI（z_range, xy_range）は `_apply_roi_filter` で適用してから RANSAC を実行。平面法線は Z 成分が正になる向きに統一。
  - `estimate_ground_plane(points, distance_threshold, ransac_n, num_iterations, z_range, xy_range) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`（平面の法線、平面上の点、地面マスク）
  - `create_rotation_matrix(plane_normal) -> np.ndarray`（3×3の回転行列、法線→+Z）
  - `apply_rotation(points, rotation_matrix) -> np.ndarray`（回転後の点群、N×3）
  - `distance_to_plane(points, plane_normal, plane_point) -> np.ndarray`（各点から平面への垂直距離、単位: m、符号付き）
- `src/lidar/profile.py`
  - `create_profile(cross_section_points_original, cross_section_points_rotated, point1, point2) -> Tuple[np.ndarray, np.ndarray]`（x_mm, z_mm、単位: mm）
  - `clip_profile_indices(x, z, start_index, end_index_max) -> Tuple[np.ndarray, np.ndarray, int, int]`（x_clipped, z_clipped, start_actual, end_actual）
- `src/lidar/preprocess_lidar.py`（既存）
  - `resample_to_uniform_grid(x, z, dx, method) -> Tuple[np.ndarray, np.ndarray]`（等間隔化）
  - `despike_median(z, k) -> np.ndarray`（メディアンフィルタ）
  - `smooth_savgol(z, window_length, polyorder, dx) -> np.ndarray`（Savitzky-Golayフィルタ）
  - `preprocess_lidar_profile(x, z, dx, uneven_tol, median_k, sg_window, sg_poly, interp_method) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]`（x, z_raw, z_median, z_smooth, meta）
- `src/lidar/resample.py`
  - `resample_to_common_grid(x, z, common_x, method) -> np.ndarray`（共通グリッド上のz値、長さlen(common_x)）
- `src/lidar/plotting.py`
  - `plot_initial_section_with_lines(points_original, points_rotated, point1, point2, cross_section_points_original, cross_section_points_rotated, plane_normal, ground_mask, cross_section_threshold, cross_section_z_range, title, figsize) -> plt.Figure`（初期断面の点群を可視化、2行3列のサブプロット、元座標系と新しいxyz軸の両方を表示、抽出面の範囲も表示）
  - `plot_section_differences(all_common_x, all_differences, all_displacements, all_before_resample, all_before_filter, ply_names, title, xlabel, ylabel_displacement, ylabel_difference, figsize, xlim, ylim_displacement, ylim_difference, x_start_from_zero, show_zoom, xlim_zoom, ylim_difference_zoom, graph_size_ratio) -> List[plt.Figure]`（断面計測値と差分の可視化、各グラフを別々のFigureとして出力、Full Range 4枚＋Zoom差分1枚の順）

### C. PLYファイルの並び順（Notebookで制御）
PLY名の文字列ソートで時系列が崩れる可能性があるため、srcに
- `natural_sort_paths(paths: list[Path]) -> list[Path]`
を用意し、Notebook側でそれを使って処理順を決める。

### D. 受け入れ条件（Acceptance Criteria）
- Notebookを上から実行すると、エラーなく図が出る
- Notebook先頭セルのパラメータ変更が結果に反映される（srcの改変不要）
- `idx==0` の差分は `0`（NaN除く）になる
- `common_x` 上の差分プロファイルが全断面で同一長さ（`common_x_points`）になる
- 主要関数に型ヒントとdocstringがあり、Notebookで利用しやすいAPIになっている

### E. Notebookセル構成と処理フロー
Notebookは以下のセル構成で実行される：

1. **パラメータ設定セル**（先頭セル）
   - プロジェクトルート（`_project_root`）を判定し、`ply_directory` を解決（プロジェクト直下の `data/LiDAR/<case>` を優先、無ければ親の同パス）
   - `params/<case>.yaml`（`<case>` = `Path(ply_directory).name`）を読み込み、`point1`, `point2`, `revetment` を設定
   - その他の処理パラメータを定義（変数名は context.md「Notebook先頭セル：フィルターパラメータ設定」に準拠）

2. **インポートセル**
   - `src.lidar` モジュールから必要な関数をインポート

3. **安全チェックセル**
   - 共通グリッドパラメータのチェック（`common_x_max`が指定されている場合のみ）
   - 前処理パラメータのチェック（`filter_sg_window > filter_sg_poly`, `filter_median_k >= 1`, `filter_interp_method`の有効性）
   - インデックスパラメータのチェック（`start_index >= 0`, `end_index_max > start_index`）

4. **PLYファイル取得セル**
   - `ply_directory`からPLYファイルを検索
   - `natural_sort_paths`で自然順ソート

5. **共通グリッド定義セル**
   - `common_x_max = None` の場合、断面直線の長さから自動設定
   - `common_x = np.linspace(common_x_min, common_x_max, common_x_points)` で共通グリッドを生成
   - 自動設定後の安全チェック（`common_x_max > common_x_min`）

6. **各PLYファイル処理セル**（ループ）
   - 各PLYファイルに対して以下を実行：
     - PLY読み込み（`load_point_cloud`）
     - 地面平面推定（`estimate_ground_plane`）と回転行列作成（`create_rotation_matrix`）
     - 断面抽出（`extract_cross_section_by_plane`）
     - 回転適用（`apply_rotation`）
     - プロファイル化（`create_profile`）→ `x_mm, z_mm`
     - インデックスクリップ（`clip_profile_indices`）→ `x_clipped, z_clipped`
     - クリップ後の安全チェック（`end_actual > start_actual >= 0`）
     - 前処理（`preprocess_lidar_profile`）→ `x_processed, z_raw, z_median, z_smooth, meta`
     - 共通グリッドへの再サンプリング（`resample_to_common_grid`）：
       - `z_before_resample`（等間隔処理前）→ `all_before_resample` に追加
       - `z_before_filter`（フィルター適用前）→ `all_before_filter` に追加
       - `z_on_common`（最終処理済み）→ `all_displacements` に追加
     - 差分計算：
       - `idx == 0` の場合：`baseline_displacement = z_on_common`, `difference_displacement = np.zeros_like(z_on_common)`
       - それ以外：`difference_displacement = z_on_common - baseline_displacement`
     - `all_differences` に `difference_displacement` を追加
     - 初期断面（`idx == 0`）のデータを保存（`initial_points_original`, `initial_points_rotated`, `initial_cross_section_original`, `initial_cross_section_rotated`, `initial_plane_normal`, `initial_ground_mask`）

7. **初期断面可視化セル**
   - `plot_initial_section_with_lines` を呼び出し（初期断面データが存在する場合のみ）

8. **断面差分可視化セル**
   - 凡例名の決定（`plot_legend_names` が指定されていればそれを使用、そうでなければ `all_ply_names` を使用）
   - `plot_section_differences` を呼び出し
   - 各Figureを個別に表示（`plt.show()`）
   - 不要な出力を抑制するため、最後に `None` を返す

---

## 実装制約（Cursor向け）
- **Open3D は使用しない**。点群の読み込み・地面平面推定・幾何計算は NumPy・SciPy・標準ライブラリで実装する。
- 補間は `scipy.interpolate.interp1d`（bounds_error=False, fill_value=np.nan）
- 図は `matplotlib`（スタイル固定はしない）
- グラフの文字は英文のみで、英文フォント（DejaVu Sans）で表示する
- グラフのデータ座標での縦横比は1:1に設定する（`set_aspect('equal')`）
- 各グラフのサイズの縦横比は4:6（横:縦）に設定する（`graph_size_ratio`パラメータで指定可能）
- 各グラフは別々のFigureとして出力する
- Notebook側で試行錯誤する前提なので、例外メッセージは原因が分かるように具体的にする

---

## Notebook先頭セル：フィルターパラメータ設定（そのまま貼り付け）

```python
# ============================================
# フィルターパラメータ設定
# ============================================

# 断面抽出パラメータ
cross_section_threshold = 0.01  # 断面抽出の距離閾値 (m)
cross_section_z_range = None  # 新しいz軸でのZ座標の範囲 (min_z, max_z)。Noneの場合はフィルタなし
start_index = 0  # サンプリング開始インデックス
end_index_max = 5000  # サンプリング終了インデックスの最大値

# 共通グリッドパラメータ
common_x_min = 0.0  # 共通グリッドの最小値 (mm)
common_x_max = None  # 共通グリッドの最大値 (mm)。Noneの場合は断面直線の長さから自動設定
common_x_points = 1000  # 共通グリッドの点数

# 前処理パラメータ (preprocess_lidar_profile)
filter_dx = None  # 目標グリッド間隔。None の場合は median(diff(x)) を使用
filter_uneven_tol = 0.05  # 不等間隔判定のしきい値
filter_median_k = 5  # メディアンフィルタの窓長（サンプル数、奇数に丸められる）
filter_sg_window = 21  # Savitzky–Golay フィルタの窓長（サンプル数、奇数に調整される）
filter_sg_poly = 3  # Savitzky–Golay フィルタの多項式次数
filter_interp_method = "linear"  # 補間方法: "linear", "nearest", "cubic", "quadratic"

# 地面抽出ROI（オプション、Noneの場合は全点を使用）
ground_z_range = None  # (min_z, max_z) または None
ground_xy_range = None  # ((min_x, max_x), (min_y, max_y)) または None

# 地面平面推定パラメータ
ground_distance_threshold = 0.01  # RANSACの距離閾値 (m)
ground_ransac_n = 3  # RANSACのサンプル数
ground_num_iterations = 1000  # RANSACの反復回数

# 可視化パラメータ
plot_xlim = None  # X軸の範囲 (x_min, x_max)。Noneの場合は自動設定
plot_ylim_displacement = None  # 計測値グラフのY軸の範囲 (y_min, y_max)。Noneの場合は自動設定
plot_ylim_difference = None  # 差分グラフのY軸の範囲 (y_min, y_max)。Noneの場合は自動設定
plot_x_start_from_zero = True  # Trueの場合、X軸を0からスタートさせる
plot_show_zoom = True  # Trueの場合、全域と特定区間の両方のグラフを表示
# 特定区間のX軸は護岸交点を基準に x_zero_zoom (= x_revetment_mm + revetment_shift_mm)、幅 plot_zoom_width で自動設定
plot_ylim_difference_zoom = None  # 特定区間の差分グラフのY軸の範囲 (y_min, y_max)。Noneの場合は自動設定
plot_graph_size_ratio = (4, 6)  # 各グラフのサイズの縦横比 (width, height)。例: (4, 6)で横:縦=4:6。データ座標の縦横比は常に1:1
plot_legend_names = None  # グラフの凡例に表示する名前のリスト。Noneの場合はPLYファイル名を使用。例: ["Initial", "Day 1", "Day 2", ...]
```

---

## 変更履歴

### 2026-01-31: Open3D 削除・NumPy/SciPy 中心への移行
- **Open3D 依存の削除**: 点群読み込み・地面平面推定・幾何計算を NumPy・SciPy・標準ライブラリで再実装。プロジェクトは Open3D に依存しない。
- **`src/lidar/io_ply.py`**: PLY を `re` / `struct` / `pathlib` と NumPy でパース。ASCII「1頂点複数行」（Realsense Viewer 形式）およびバイナリ `float32` に対応。型サイズ・struct 書式をモジュール定数で整理。
- **`src/lidar/leveling.py`**: RANSAC 平面推定を NumPy で実装。平面法線の Z 成分が正になる向きに統一（天地の向きを Open3D と互換に）。ROI フィルタを `_apply_roi_filter` に抽出し、`xy_range` 適用時のマスク長不一致バグを修正。
- **`src/lidar/geometry.py`**: 直線の単位方向ベクトル・長さを `_normalized_line_direction` で共通化し、`distance_to_line` / `project_to_line` / `distance_to_plane_containing_line` / `extract_cross_section` / `extract_cross_section_by_plane` で利用。
- **`pyproject.toml`**: プロジェクト名を `lidar-section` に変更。不要な依存（pandas, pynufft, h5py, seaborn 等）を削除し、`numpy`, `scipy`, `matplotlib`, `notebook`, `ipykernel`, `PyQt6`, `PyYAML` を明示。`uv` で推移依存を解決する最小構成に。
- **ノートブック**: `ply_directory` のパス解決を堅牢化（`notebooks/` 実行時・プロジェクトルート実行時の両方で `data/LiDAR/<case>` を正しく指す）。`idx=0` で断面が 0 点でも初期表示用変数（`initial_points_original` 等）が設定されるように修正。`src/lidar/plotting.py` の scatter に `len(...) > 0` ガードを追加し、断面点群が空でもエラーにならないようにした。`revetment_position` の不要な参照を削除。
- 実行確認: `uv sync` の実行と Jupyter カーネルの再起動後にノートブックを再実行して動作を確認することを推奨。

### 2026-01-31: 構成・パラメータの更新
- ノートブックを `LiDAR_visualize.ipynb` に統一。データは `data/LiDAR/<case>/`、パラメータは `params/<case>.yaml`。
- `point1`, `point2`, `revetment` を YAML から読み込む方式に変更。YAML は `points.point1`, `points.point2`, `points.revetment` を `[x, y, z]` のリストで定義。
- `ply_directory` はプロジェクト直下の `data/LiDAR/<case>` を優先し、無ければ親ディレクトリの同パスを使用。
- 可視化の Zoom 区間は護岸交点（`revetment_intersection_distance_mm`）を基準に `x_zero_zoom`・`plot_zoom_width` で指定。`revetment_position` は削除済み。
- geometry に `revetment_intersection_distance_mm` の説明を追記。

### 2026-01-24: 実装仕様の明文化

以下の内容を追記・整理しました（既存の処理仕様は変更していません）：

1. **「2. 点群から断面への変換」セクション**：
   - `extract_cross_section_by_plane`の平面距離計算式と射影距離の定義を明文化（Implementation noteとして追加）
   - 平面の法線ベクトルの計算方法（外積）と特別処理（断面直線が新しいz軸と平行な場合）を明文化

2. **「2. 鉛直（傾き）補正」セクション**：
   - 地面抽出のROIパラメータ（`ground_z_range`, `ground_xy_range`）の説明を追加
   - RANSAC平面推定のパラメータ（`ground_distance_threshold`, `ground_ransac_n`, `ground_num_iterations`）の説明を追加

3. **「4. 共通グリッドへの再サンプリング」セクション**：
   - `common_x_max = None` の場合の自動設定の定義を明文化（断面線長からmm換算）
   - `all_before_resample`, `all_before_filter`, `all_displacements`, `all_differences` のデータ構造を明文化（List[np.ndarray]型、各要素の長さは`common_x_points`）
   - 差分計算の定義を明文化（`idx == 0`の場合は`np.zeros_like`、それ以外は`all_displacements[idx] - all_displacements[0]`）

4. **「B. srcは「純粋関数」中心」セクション**：
   - 推奨モジュール分割に、実際の関数名・引数・戻り値を追記（`io_ply.py`の`natural_sort_paths`、`geometry.py`の各関数、`leveling.py`の各関数、`profile.py`の各関数、`resample.py`の関数、`plotting.py`の各関数）

5. **「A. Notebookがパラメータの唯一の操作面」セクション**：
   - Notebook先頭セルで定義するパラメータ一覧に、データディレクトリ（`ply_directory`）、地面抽出ROIパラメータ、地面平面推定パラメータを追記
   - 各パラメータの型と説明を明確化

6. **「Notebook先頭セル：フィルターパラメータ設定」セクション**：
   - コードブロックに、データディレクトリ、地面抽出ROIパラメータ、地面平面推定パラメータを追記

7. **新規セクション「E. Notebookセル構成と処理フロー」**：
   - Notebookのセル構成（8つのセル）と各セルの処理内容を明文化
   - 各セルで保持する中間データの名称を明文化（`all_before_resample`, `all_before_filter`, `all_displacements`, `all_differences`, `baseline_displacement`, 初期断面データなど）