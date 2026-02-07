"""可視化関数"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from typing import List, Optional, Tuple


def plot_section_differences(
    all_common_x: np.ndarray,
    all_differences: List[np.ndarray],
    all_displacements: Optional[List[np.ndarray]] = None,
    all_before_resample: Optional[List[np.ndarray]] = None,
    all_before_filter: Optional[List[np.ndarray]] = None,
    ply_names: Optional[List[str]] = None,
    title: str = "Section Displacements and Differences",
    xlabel: str = "Distance (mm)",
    ylabel_displacement: str = "Displacement (mm)",
    ylabel_difference: str = "Difference (mm)",
    figsize: tuple = (15, 6),
    xlim: Optional[Tuple[float, float]] = None,
    ylim_displacement: Optional[Tuple[float, float]] = None,
    ylim_difference: Optional[Tuple[float, float]] = None,
    x_start_from_zero: bool = False,
    show_zoom: bool = False,
    xlim_zoom: Optional[Tuple[float, float]] = None,
    ylim_difference_zoom: Optional[Tuple[float, float]] = None,
    ylim_displacement_zoom: Optional[Tuple[float, float]] = None,
    graph_size_ratio: Optional[Tuple[float, float]] = None,
    title_prefix: str = "",
    x_zero_zoom: Optional[float] = None
) -> List[plt.Figure]:
    """
    断面の計測値（新しいxyz軸での変換値）と差分を可視化する。
    各グラフを別々のFigureとして出力する。
    処理検証用に、等間隔処理前とフィルター適用前のグラフも表示可能。
    
    Args:
        all_common_x: 共通グリッドの距離配列（mm）
        all_differences: 各断面の差分配列のリスト（各要素はlen(all_common_x)）
        all_displacements: 各断面の計測値配列のリスト（各要素はlen(all_common_x)）。Noneの場合は計測値のグラフを表示しない
        all_before_resample: 等間隔処理前のデータ（共通グリッドに再サンプリング済み）。Noneの場合は表示しない
        all_before_filter: フィルター適用前のデータ（共通グリッドに再サンプリング済み）。Noneの場合は表示しない
        ply_names: PLYファイル名のリスト（凡例用、オプション）
        title: グラフタイトル（使用されない、互換性のため残している）
        xlabel: X軸ラベル
        ylabel_displacement: 計測値グラフのY軸ラベル
        ylabel_difference: 差分グラフのY軸ラベル
        figsize: 図のサイズ（使用されない、互換性のため残している）
        xlim: X軸の範囲 (x_min, x_max)。Noneの場合は自動設定
        ylim_displacement: 計測値グラフのY軸の範囲 (y_min, y_max)。Noneの場合は自動設定
        ylim_difference: 差分グラフのY軸の範囲 (y_min, y_max)。Noneの場合は自動設定
        x_start_from_zero: Trueの場合、X軸を0からスタートさせる（全域とZoomグラフの両方に適用。全域ではall_common_xから最小値を引き、Zoomではxlim_zoomの最小値から引く）
        show_zoom: Trueの場合、全域と特定区間の両方のグラフを表示する
        xlim_zoom: 特定区間のX軸の範囲 (x_min, x_max)。show_zoom=Trueの場合に使用
        ylim_difference_zoom: 特定区間の差分グラフのY軸の範囲 (y_min, y_max)。Noneの場合は自動設定
        ylim_displacement_zoom: 特定区間の計測値グラフ（Section Measurement - Zoom）のY軸の範囲
            (y_min, y_max)。Noneの場合は `ylim_displacement` と同じ設定を使用
        graph_size_ratio: 各グラフのサイズの縦横比 (width, height)。Noneの場合は自動設定。例: (4, 6)で横:縦=4:6
        title_prefix: グラフタイトルの先頭に付与するプレフィックス（例: フォルダ名）。空文字列の場合は付与しない
        x_zero_zoom: ZoomグラフのX軸ゼロ点（mm、point1からの距離）。指定時は護岸断面交点をゼロとして表示。Noneの場合は従来どおり
        
    Returns:
        matplotlib Figureオブジェクトのリスト（等間隔処理前、フィルター適用前、全域の計測値、
        全域の差分、特定区間の計測値、特定区間の差分の順。ただし、対応するデータが存在しない
        場合はそのFigureは生成されない）
    """
    # 英文フォント設定（DejaVu Sansを使用）
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化けを防ぐ
    
    # タイトル用のヘルパー関数
    def with_prefix(base_title: str) -> str:
        if title_prefix:
            return f"{title_prefix}: {base_title}"
        return base_title
    
    # X軸を0からスタートさせる場合
    if x_start_from_zero:
        x_offset = all_common_x.min()
        plot_x = all_common_x - x_offset
        # xlimも0から始まるように調整
        if xlim is not None:
            xlim = (xlim[0] - x_offset, xlim[1] - x_offset)
    else:
        x_offset = 0.0
        plot_x = all_common_x
    
    n_sections = len(all_differences)
    
    # カラーマップを生成
    colors = plt.cm.viridis(np.linspace(0, 1, n_sections))
    
    # 計測値のグラフを表示するかどうか
    show_displacements = all_displacements is not None and len(all_displacements) > 0
    
    # グラフのサイズの縦横比を計算（各グラフのサイズを4:6にする）
    if graph_size_ratio is not None:
        single_graph_width = 9.0
        single_graph_height = single_graph_width * graph_size_ratio[1] / graph_size_ratio[0]
    else:
        single_graph_width = 9.0
        single_graph_height = 13.5

    def figsize_for_aspect(x_range: float, y_range: float) -> Tuple[float, float]:
        """データの縦横比（1:1）に合わせてFigureサイズを計算し、上下の余白を抑える"""
        if y_range <= 0:
            return (single_graph_width, single_graph_height)
        axes_aspect = x_range / y_range
        if axes_aspect >= single_graph_width / single_graph_height:
            w, h = single_graph_width, single_graph_width / axes_aspect
        else:
            w, h = single_graph_height * axes_aspect, single_graph_height
        return (w, h)

    figures = []
    
    # ========== Full Range（全域）のグラフ ==========
    # 表示順序: 1. 等間隔処理前、2. フィルター適用前、3. 処理後（計測値）、4. 処理後の差分
    
    # 1. 等間隔処理前のグラフ（処理検証用）
    if all_before_resample is not None and len(all_before_resample) > 0:
        xr = (plot_x.min(), plot_x.max()) if xlim is None else xlim
        yr = (np.nanmin(np.concatenate(all_before_resample)), np.nanmax(np.concatenate(all_before_resample))) if ylim_displacement is None else ylim_displacement
        fw, fh = figsize_for_aspect(xr[1] - xr[0], yr[1] - yr[0])
        fig_before_resample = plt.figure(figsize=(fw, fh))
        ax_before_resample = fig_before_resample.add_subplot(111)
        
        for idx, data in enumerate(all_before_resample):
            label = ply_names[idx] if ply_names is not None else f"Section {idx}"
            ax_before_resample.plot(plot_x, data, label=label, color=colors[idx], alpha=0.7)
        
        ax_before_resample.set_xlabel(xlabel)
        ax_before_resample.set_ylabel(ylabel_displacement)
        ax_before_resample.set_title(with_prefix("Before Resampling (Original Profile) - Full Range"))
        if xlim is not None:
            ax_before_resample.set_xlim(xlim)
        if ylim_displacement is not None:
            ax_before_resample.set_ylim(ylim_displacement)
        # データ座標での縦横比を1:1に設定
        ax_before_resample.set_aspect('equal')
        ax_before_resample.grid(True, alpha=0.3)
        ax_before_resample.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        figures.append(fig_before_resample)
    
    # 2. フィルター適用前のグラフ（処理検証用）
    if all_before_filter is not None and len(all_before_filter) > 0:
        xr = (plot_x.min(), plot_x.max()) if xlim is None else xlim
        yr = (np.nanmin(np.concatenate(all_before_filter)), np.nanmax(np.concatenate(all_before_filter))) if ylim_displacement is None else ylim_displacement
        fw, fh = figsize_for_aspect(xr[1] - xr[0], yr[1] - yr[0])
        fig_before_filter = plt.figure(figsize=(fw, fh))
        ax_before_filter = fig_before_filter.add_subplot(111)
        
        for idx, data in enumerate(all_before_filter):
            label = ply_names[idx] if ply_names is not None else f"Section {idx}"
            ax_before_filter.plot(plot_x, data, label=label, color=colors[idx], alpha=0.7)
        
        ax_before_filter.set_xlabel(xlabel)
        ax_before_filter.set_ylabel(ylabel_displacement)
        ax_before_filter.set_title(with_prefix("Before Filtering (After Resampling) - Full Range"))
        if xlim is not None:
            ax_before_filter.set_xlim(xlim)
        if ylim_displacement is not None:
            ax_before_filter.set_ylim(ylim_displacement)
        # データ座標での縦横比を1:1に設定
        ax_before_filter.set_aspect('equal')
        ax_before_filter.grid(True, alpha=0.3)
        ax_before_filter.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        figures.append(fig_before_filter)
    
    # 3. 処理後（計測値）のグラフ
    if show_displacements:
        xr = (plot_x.min(), plot_x.max()) if xlim is None else xlim
        yr = (np.nanmin(np.concatenate(all_displacements)), np.nanmax(np.concatenate(all_displacements))) if ylim_displacement is None else ylim_displacement
        fw, fh = figsize_for_aspect(xr[1] - xr[0], yr[1] - yr[0])
        fig_displacement_full = plt.figure(figsize=(fw, fh))
        ax_displacement_full = fig_displacement_full.add_subplot(111)
        
        for idx, displacement in enumerate(all_displacements):
            label = ply_names[idx] if ply_names is not None else f"Section {idx}"
            ax_displacement_full.plot(plot_x, displacement, label=label, color=colors[idx], alpha=0.7)
        
        ax_displacement_full.set_xlabel(xlabel)
        ax_displacement_full.set_ylabel(ylabel_displacement)
        ax_displacement_full.set_title(with_prefix("Section Measurement (New XYZ Axis) - Full Range"))
        if xlim is not None:
            ax_displacement_full.set_xlim(xlim)
        if ylim_displacement is not None:
            ax_displacement_full.set_ylim(ylim_displacement)
        # データ座標での縦横比を1:1に設定
        ax_displacement_full.set_aspect('equal')
        ax_displacement_full.grid(True, alpha=0.3)
        ax_displacement_full.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        figures.append(fig_displacement_full)
    
    # 4. 処理後の差分グラフ
    xr = (plot_x.min(), plot_x.max()) if xlim is None else xlim
    yr = (np.nanmin(np.concatenate(all_differences)), np.nanmax(np.concatenate(all_differences))) if ylim_difference is None else ylim_difference
    fw, fh = figsize_for_aspect(xr[1] - xr[0], yr[1] - yr[0])
    fig_difference_full = plt.figure(figsize=(fw, fh))
    ax_difference_full = fig_difference_full.add_subplot(111)
    
    for idx, diff in enumerate(all_differences):
        label = ply_names[idx] if ply_names is not None else f"Section {idx}"
        ax_difference_full.plot(plot_x, diff, label=label, color=colors[idx], alpha=0.7)
    
    ax_difference_full.set_xlabel(xlabel)
    ax_difference_full.set_ylabel(ylabel_difference)
    ax_difference_full.set_title(with_prefix("Section Difference (from Initial Section) - Full Range"))
    if xlim is not None:
        ax_difference_full.set_xlim(xlim)
    if ylim_difference is not None:
        ax_difference_full.set_ylim(ylim_difference)
    # データ座標での縦横比を1:1に設定
    ax_difference_full.set_aspect('equal')
    ax_difference_full.grid(True, alpha=0.3)
    ax_difference_full.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    figures.append(fig_difference_full)
    
    # ========== 特定区間のグラフ ==========
    if show_zoom and xlim_zoom is not None:
        # 特定区間のX軸範囲を計算
        if x_zero_zoom is not None:
            # 護岸断面交点をゼロ点とする（x_zero_zoom = point1からの距離 mm）
            plot_x_zoom = all_common_x - x_zero_zoom
            xlim_zoom_final = (xlim_zoom[0] - x_zero_zoom, xlim_zoom[1] - x_zero_zoom)
        elif x_start_from_zero:
            xlim_zoom_adjusted = (xlim_zoom[0] - x_offset, xlim_zoom[1] - x_offset)
            x_offset_zoom = xlim_zoom_adjusted[0]
            plot_x_zoom = plot_x - x_offset_zoom
            xlim_zoom_final = (0, xlim_zoom_adjusted[1] - xlim_zoom_adjusted[0])
        else:
            xlim_zoom_adjusted = xlim_zoom
            plot_x_zoom = plot_x
            xlim_zoom_final = xlim_zoom_adjusted

        # 3'. 処理後（計測値）のグラフ（特定区間）
        if show_displacements:
            yz = ylim_displacement_zoom if ylim_displacement_zoom is not None else (ylim_displacement if ylim_displacement is not None else (np.nanmin(np.concatenate(all_displacements)), np.nanmax(np.concatenate(all_displacements))))
            fw, fh = figsize_for_aspect(xlim_zoom_final[1] - xlim_zoom_final[0], yz[1] - yz[0])
            fig_displacement_zoom = plt.figure(figsize=(fw, fh))
            ax_displacement_zoom = fig_displacement_zoom.add_subplot(111)

            for idx, displacement in enumerate(all_displacements):
                label = ply_names[idx] if ply_names is not None else f"Section {idx}"
                ax_displacement_zoom.plot(plot_x_zoom, displacement, label=label, color=colors[idx], alpha=0.7)

            ax_displacement_zoom.set_xlabel(xlabel)
            ax_displacement_zoom.set_ylabel(ylabel_displacement)
            ax_displacement_zoom.set_title(with_prefix("Section Measurement (New XYZ Axis) - Zoom"))
            ax_displacement_zoom.set_xlim(xlim_zoom_final)
            # Zoom用のY範囲が指定されていればそれを優先し、なければ全域用の設定を使う
            if ylim_displacement_zoom is not None:
                ax_displacement_zoom.set_ylim(ylim_displacement_zoom)
            elif ylim_displacement is not None:
                ax_displacement_zoom.set_ylim(ylim_displacement)
            # X軸は10 mm ピッチでグリッドを細かく表示
            ax_displacement_zoom.xaxis.set_major_locator(MultipleLocator(10.0))
            # データ座標での縦横比を1:1に設定
            ax_displacement_zoom.set_aspect('equal')
            ax_displacement_zoom.grid(True, alpha=0.3)
            ax_displacement_zoom.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.tight_layout()
            figures.append(fig_displacement_zoom)

        # 差分のグラフ（特定区間）
        yz = ylim_difference_zoom if ylim_difference_zoom is not None else (np.nanmin(np.concatenate(all_differences)), np.nanmax(np.concatenate(all_differences)))
        fw, fh = figsize_for_aspect(xlim_zoom_final[1] - xlim_zoom_final[0], yz[1] - yz[0])
        fig_difference_zoom = plt.figure(figsize=(fw, fh))
        ax_difference_zoom = fig_difference_zoom.add_subplot(111)
        
        for idx, diff in enumerate(all_differences):
            label = ply_names[idx] if ply_names is not None else f"Section {idx}"
            ax_difference_zoom.plot(plot_x_zoom, diff, label=label, color=colors[idx], alpha=0.7)
        
        ax_difference_zoom.set_xlabel(xlabel)
        ax_difference_zoom.set_ylabel(ylabel_difference)
        ax_difference_zoom.set_title(with_prefix("Section Difference (from Initial Section) - Zoom"))
        ax_difference_zoom.set_xlim(xlim_zoom_final)
        if ylim_difference_zoom is not None:
            ax_difference_zoom.set_ylim(ylim_difference_zoom)
        # データ座標での縦横比を1:1に設定
        ax_difference_zoom.set_aspect('equal')
        ax_difference_zoom.grid(True, alpha=0.3)
        ax_difference_zoom.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        figures.append(fig_difference_zoom)
    
    return figures


def plot_initial_section_with_lines(
    points_original: np.ndarray,
    points_rotated: np.ndarray,
    point1: np.ndarray,
    point2: np.ndarray,
    cross_section_points_original: np.ndarray,
    cross_section_points_rotated: np.ndarray,
    plane_normal: np.ndarray,
    ground_mask: np.ndarray,
    cross_section_threshold: float = 0.01,
    cross_section_z_range: Optional[Tuple[float, float]] = None,
    title: str = "Initial Section with Cross-Section Line",
    figsize: tuple = (15, 10),
    plot_range: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
) -> plt.Figure:
    """
    初期断面の点群を可視化し、断面直線と新しいZ軸方向（地面平面の法線）を重ねて表示する。
    元座標系と新しいxyz軸の両方を表示する。
    XY平面、YZ平面、XZ平面の3方向から表示する。
    抽出面の範囲も表示する。
    
    Args:
        points_original: 元データ座標系の点群（N×3）
        points_rotated: 鉛直補正後の点群（N×3）
        point1: 断面直線の第1点 [x, y, z]（元データ座標系）
        point2: 断面直線の第2点 [x, y, z]（元データ座標系）
        cross_section_points_original: 抽出された断面点群（元データ座標系、M×3）
        cross_section_points_rotated: 抽出された断面点群（鉛直補正後、M×3）
        plane_normal: 地面平面の法線ベクトル [nx, ny, nz]（新しいZ軸方向）
        ground_mask: 地面と判定された点のマスク（長さNのbool配列）
        cross_section_threshold: 断面抽出の距離閾値（m）
        cross_section_z_range: 新しいz軸でのZ座標の範囲 (min_z, max_z)。Noneの場合は表示しない
        title: グラフタイトル
        figsize: 図のサイズ
        
    Returns:
        matplotlib Figureオブジェクト
    """
    # 英文フォント設定
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 2行3列のサブプロットを作成（1行目：元座標系、2行目：新しいxyz軸）
    fig, axes = plt.subplots(2, 3, figsize=(figsize[0], figsize[1]))
    
    # 元座標系のサブプロット
    ax_xy_orig, ax_yz_orig, ax_xz_orig = axes[0, :]
    # 新しいxyz軸のサブプロット
    ax_xy_new, ax_yz_new, ax_xz_new = axes[1, :]
    
    # 地面と判定された点を別の色でプロット
    ground_points_orig = points_original[ground_mask]
    non_ground_points_orig = points_original[~ground_mask]
    ground_points_new = points_rotated[ground_mask]
    non_ground_points_new = points_rotated[~ground_mask]
    
    # 断面直線の中点を計算（元座標系）
    mid_point_orig = (point1 + point2) / 2
    # 断面直線の中点を計算（新しいxyz軸）
    point1_rotated = points_rotated[np.argmin(np.linalg.norm(points_original - point1, axis=1))]
    point2_rotated = points_rotated[np.argmin(np.linalg.norm(points_original - point2, axis=1))]
    mid_point_new = (point1_rotated + point2_rotated) / 2
    
    # 各軸の範囲を計算（元座標系）
    x_min_orig, x_max_orig = points_original[:, 0].min(), points_original[:, 0].max()
    y_min_orig, y_max_orig = points_original[:, 1].min(), points_original[:, 1].max()
    z_min_orig, z_max_orig = points_original[:, 2].min(), points_original[:, 2].max()
    x_range_orig = x_max_orig - x_min_orig
    y_range_orig = y_max_orig - y_min_orig
    z_range_orig = z_max_orig - z_min_orig
    axis_length_orig = max(x_range_orig, y_range_orig, z_range_orig) * 0.3
    
    # 各軸の範囲を計算（新しいxyz軸）
    x_min_new, x_max_new = points_rotated[:, 0].min(), points_rotated[:, 0].max()
    y_min_new, y_max_new = points_rotated[:, 1].min(), points_rotated[:, 1].max()
    z_min_new, z_max_new = points_rotated[:, 2].min(), points_rotated[:, 2].max()
    x_range_new = x_max_new - x_min_new
    y_range_new = y_max_new - y_min_new
    z_range_new = z_max_new - z_min_new
    axis_length_new = max(x_range_new, y_range_new, z_range_new) * 0.3

    # 可視化範囲は「元座標系と新しいxyz軸で共通」とする。
    # さらに、ユーザー指定のplot_rangeがあればそれを優先する。
    # plot_range: ((x_min, x_max), (y_min, y_max), (z_min, z_max)) [m]
    if plot_range is not None:
        (x_min_user, x_max_user), (y_min_user, y_max_user), (z_min_user, z_max_user) = plot_range
        x_min_xy = x_min_user
        x_max_xy = x_max_user
        y_min_xy = y_min_user
        y_max_xy = y_max_user
        y_min_yz = y_min_user
        y_max_yz = y_max_user
        z_min_yz = z_min_user
        z_max_yz = z_max_user
        x_min_xz = x_min_user
        x_max_xz = x_max_user
        z_min_xz = z_min_user
        z_max_xz = z_max_user
    else:
        # 自動決定（元座標系と新xyz軸の包絡）
        # XY平面用
        x_min_xy = min(x_min_orig, x_min_new)
        x_max_xy = max(x_max_orig, x_max_new)
        y_min_xy = min(y_min_orig, y_min_new)
        y_max_xy = max(y_max_orig, y_max_new)
        # YZ平面用
        y_min_yz = y_min_xy
        y_max_yz = y_max_xy
        z_min_yz = min(z_min_orig, z_min_new)
        z_max_yz = max(z_max_orig, z_max_new)
        # XZ平面用
        x_min_xz = x_min_xy
        x_max_xz = x_max_xy
        z_min_xz = z_min_yz
        z_max_xz = z_max_yz
    
    # 元座標系での断面直線の2点を回転後の座標に変換
    # 元座標系でのpoint1, point2に対応する回転後の点を探す
    point1_rotated_idx = np.argmin(np.linalg.norm(points_original - point1, axis=1))
    point2_rotated_idx = np.argmin(np.linalg.norm(points_original - point2, axis=1))
    point1_rotated = points_rotated[point1_rotated_idx]
    point2_rotated = points_rotated[point2_rotated_idx]
    
    # 新しいxyz軸でのZ軸方向（垂直方向）は[0, 0, 1]
    new_z_axis = np.array([0.0, 0.0, 1.0])
    
    # 抽出面の範囲を計算（断面直線を含み、新しいz軸に平行な平面）
    # 平面の法線ベクトル = 断面直線の方向ベクトル × 新しいz軸
    line_dir = point2 - point1
    line_length = np.linalg.norm(line_dir)
    if line_length > 1e-10:
        line_dir = line_dir / line_length
        plane_normal_vec = np.cross(line_dir, plane_normal)
        plane_normal_length = np.linalg.norm(plane_normal_vec)
        if plane_normal_length > 1e-10:
            plane_normal_vec = plane_normal_vec / plane_normal_length
        else:
            # 断面直線が新しいz軸と平行な場合
            plane_normal_vec = None
    else:
        plane_normal_vec = None
    
    # 新しいz軸でのZ座標範囲を取得
    if cross_section_z_range is not None:
        z_min, z_max = cross_section_z_range
    else:
        # 抽出された点の新しいz軸でのZ座標範囲を使用
        if len(cross_section_points_rotated) > 0:
            z_min = cross_section_points_rotated[:, 2].min()
            z_max = cross_section_points_rotated[:, 2].max()
        else:
            z_min = None
            z_max = None
    
    # ========== 元座標系のプロット ==========
    # XY平面
    if len(non_ground_points_orig) > 0:
        ax_xy_orig.scatter(non_ground_points_orig[:, 0], non_ground_points_orig[:, 1], 
                          c='lightgray', s=0.5, alpha=0.3, label='Non-ground points')
    if len(ground_points_orig) > 0:
        ax_xy_orig.scatter(ground_points_orig[:, 0], ground_points_orig[:, 1], 
                          c='blue', s=2, alpha=0.6, label='Ground points')
    if len(cross_section_points_original) > 0:
        ax_xy_orig.scatter(cross_section_points_original[:, 0], cross_section_points_original[:, 1], 
                          c='red', s=5, alpha=0.8, label='Cross-section points')
    ax_xy_orig.plot([point1[0], point2[0]], [point1[1], point2[1]], 'b-', linewidth=2, label='Cross-section line')
    # 抽出面の範囲を表示（平面の法線方向にthresholdの範囲で線を描画）
    if plane_normal_vec is not None:
        # 平面の法線ベクトルのXY投影
        plane_normal_xy = plane_normal_vec[:2]
        plane_normal_xy_length = np.linalg.norm(plane_normal_xy)
        if plane_normal_xy_length > 1e-10:
            plane_normal_xy_normalized = plane_normal_xy / plane_normal_xy_length
            # 断面直線の両端から平面の法線方向にthresholdの範囲で線を描画
            for p in [point1, point2]:
                offset_x = np.array([p[0] - plane_normal_xy_normalized[0] * cross_section_threshold,
                                    p[0] + plane_normal_xy_normalized[0] * cross_section_threshold])
                offset_y = np.array([p[1] - plane_normal_xy_normalized[1] * cross_section_threshold,
                                    p[1] + plane_normal_xy_normalized[1] * cross_section_threshold])
                ax_xy_orig.plot(offset_x, offset_y, 'm--', linewidth=1, alpha=0.5)
    z_axis_xy = plane_normal[:2]
    z_axis_xy_length = np.linalg.norm(z_axis_xy)
    if z_axis_xy_length > 1e-10:
        z_axis_xy_normalized = z_axis_xy / z_axis_xy_length
        z_axis_x = np.array([mid_point_orig[0] - z_axis_xy_normalized[0] * axis_length_orig,
                            mid_point_orig[0] + z_axis_xy_normalized[0] * axis_length_orig])
        z_axis_y = np.array([mid_point_orig[1] - z_axis_xy_normalized[1] * axis_length_orig,
                            mid_point_orig[1] + z_axis_xy_normalized[1] * axis_length_orig])
        ax_xy_orig.plot(z_axis_x, z_axis_y, 'g--', linewidth=2, label='New Z-axis (ground normal)')
    ax_xy_orig.plot(point1[0], point1[1], 'bo', markersize=10, label='Point 1')
    ax_xy_orig.plot(point2[0], point2[1], 'bs', markersize=10, label='Point 2')
    ax_xy_orig.set_xlabel('X (m)')
    ax_xy_orig.set_ylabel('Y (m)')
    ax_xy_orig.set_title('XY Plane (Original Coordinate)')
    ax_xy_orig.set_aspect('equal')
    # 元座標系と新しいxyz軸で同じ範囲を表示するための軸設定
    ax_xy_orig.set_xlim(x_min_xy, x_max_xy)
    ax_xy_orig.set_ylim(y_min_xy, y_max_xy)
    ax_xy_orig.grid(True, alpha=0.3)
    ax_xy_orig.legend(fontsize=8)
    
    # YZ平面
    if len(non_ground_points_orig) > 0:
        ax_yz_orig.scatter(non_ground_points_orig[:, 1], non_ground_points_orig[:, 2], 
                          c='lightgray', s=0.5, alpha=0.3, label='Non-ground points')
    if len(ground_points_orig) > 0:
        ax_yz_orig.scatter(ground_points_orig[:, 1], ground_points_orig[:, 2], 
                          c='blue', s=2, alpha=0.6, label='Ground points')
    if len(cross_section_points_original) > 0:
        ax_yz_orig.scatter(cross_section_points_original[:, 1], cross_section_points_original[:, 2], 
                          c='red', s=5, alpha=0.8, label='Cross-section points')
    ax_yz_orig.plot([point1[1], point2[1]], [point1[2], point2[2]], 'b-', linewidth=2, label='Cross-section line')
    # 抽出面の範囲を表示（平面の法線方向にthresholdの範囲で線を描画）
    if plane_normal_vec is not None:
        # 平面の法線ベクトルのYZ投影
        plane_normal_yz = plane_normal_vec[1:3]
        plane_normal_yz_length = np.linalg.norm(plane_normal_yz)
        if plane_normal_yz_length > 1e-10:
            plane_normal_yz_normalized = plane_normal_yz / plane_normal_yz_length
            # 断面直線の両端から平面の法線方向にthresholdの範囲で線を描画
            for p in [point1, point2]:
                offset_y = np.array([p[1] - plane_normal_yz_normalized[0] * cross_section_threshold,
                                    p[1] + plane_normal_yz_normalized[0] * cross_section_threshold])
                offset_z = np.array([p[2] - plane_normal_yz_normalized[1] * cross_section_threshold,
                                    p[2] + plane_normal_yz_normalized[1] * cross_section_threshold])
                ax_yz_orig.plot(offset_y, offset_z, 'm--', linewidth=1, alpha=0.5)
    z_axis_yz = plane_normal[1:3]
    z_axis_yz_length = np.linalg.norm(z_axis_yz)
    if z_axis_yz_length > 1e-10:
        z_axis_yz_normalized = z_axis_yz / z_axis_yz_length
        z_axis_y_coords = np.array([mid_point_orig[1] - z_axis_yz_normalized[0] * axis_length_orig,
                                   mid_point_orig[1] + z_axis_yz_normalized[0] * axis_length_orig])
        z_axis_z_coords = np.array([mid_point_orig[2] - z_axis_yz_normalized[1] * axis_length_orig,
                                   mid_point_orig[2] + z_axis_yz_normalized[1] * axis_length_orig])
        ax_yz_orig.plot(z_axis_y_coords, z_axis_z_coords, 'g--', linewidth=2, label='New Z-axis (ground normal)')
    ax_yz_orig.plot(point1[1], point1[2], 'bo', markersize=10, label='Point 1')
    ax_yz_orig.plot(point2[1], point2[2], 'bs', markersize=10, label='Point 2')
    ax_yz_orig.set_xlabel('Y (m)')
    ax_yz_orig.set_ylabel('Z (m)')
    ax_yz_orig.set_title('YZ Plane (Original Coordinate)')
    ax_yz_orig.set_aspect('equal')
    ax_yz_orig.set_xlim(y_min_yz, y_max_yz)
    ax_yz_orig.set_ylim(z_min_yz, z_max_yz)
    ax_yz_orig.grid(True, alpha=0.3)
    ax_yz_orig.legend(fontsize=8)
    
    # XZ平面
    if len(non_ground_points_orig) > 0:
        ax_xz_orig.scatter(non_ground_points_orig[:, 0], non_ground_points_orig[:, 2], 
                          c='lightgray', s=0.5, alpha=0.3, label='Non-ground points')
    if len(ground_points_orig) > 0:
        ax_xz_orig.scatter(ground_points_orig[:, 0], ground_points_orig[:, 2], 
                          c='blue', s=2, alpha=0.6, label='Ground points')
    if len(cross_section_points_original) > 0:
        ax_xz_orig.scatter(cross_section_points_original[:, 0], cross_section_points_original[:, 2], 
                          c='red', s=5, alpha=0.8, label='Cross-section points')
    ax_xz_orig.plot([point1[0], point2[0]], [point1[2], point2[2]], 'b-', linewidth=2, label='Cross-section line')
    # 抽出面の範囲を表示（平面の法線方向にthresholdの範囲で線を描画）
    if plane_normal_vec is not None:
        # 平面の法線ベクトルのXZ投影
        plane_normal_xz = np.array([plane_normal_vec[0], plane_normal_vec[2]])
        plane_normal_xz_length = np.linalg.norm(plane_normal_xz)
        if plane_normal_xz_length > 1e-10:
            plane_normal_xz_normalized = plane_normal_xz / plane_normal_xz_length
            # 断面直線の両端から平面の法線方向にthresholdの範囲で線を描画
            for p in [point1, point2]:
                offset_x = np.array([p[0] - plane_normal_xz_normalized[0] * cross_section_threshold,
                                    p[0] + plane_normal_xz_normalized[0] * cross_section_threshold])
                offset_z = np.array([p[2] - plane_normal_xz_normalized[1] * cross_section_threshold,
                                    p[2] + plane_normal_xz_normalized[1] * cross_section_threshold])
                ax_xz_orig.plot(offset_x, offset_z, 'm--', linewidth=1, alpha=0.5)
    z_axis_xz = np.array([plane_normal[0], plane_normal[2]])
    z_axis_xz_length = np.linalg.norm(z_axis_xz)
    if z_axis_xz_length > 1e-10:
        z_axis_xz_normalized = z_axis_xz / z_axis_xz_length
        z_axis_x_coords = np.array([mid_point_orig[0] - z_axis_xz_normalized[0] * axis_length_orig,
                                   mid_point_orig[0] + z_axis_xz_normalized[0] * axis_length_orig])
        z_axis_z_coords = np.array([mid_point_orig[2] - z_axis_xz_normalized[1] * axis_length_orig,
                                   mid_point_orig[2] + z_axis_xz_normalized[1] * axis_length_orig])
        ax_xz_orig.plot(z_axis_x_coords, z_axis_z_coords, 'g--', linewidth=2, label='New Z-axis (ground normal)')
    ax_xz_orig.plot(point1[0], point1[2], 'bo', markersize=10, label='Point 1')
    ax_xz_orig.plot(point2[0], point2[2], 'bs', markersize=10, label='Point 2')
    ax_xz_orig.set_xlabel('X (m)')
    ax_xz_orig.set_ylabel('Z (m)')
    ax_xz_orig.set_title('XZ Plane (Original Coordinate)')
    ax_xz_orig.set_aspect('equal')
    ax_xz_orig.set_xlim(x_min_xz, x_max_xz)
    ax_xz_orig.set_ylim(z_min_xz, z_max_xz)
    ax_xz_orig.grid(True, alpha=0.3)
    ax_xz_orig.legend(fontsize=8)
    
    # ========== 新しいxyz軸のプロット ==========
    # XY平面
    if len(non_ground_points_new) > 0:
        ax_xy_new.scatter(non_ground_points_new[:, 0], non_ground_points_new[:, 1], 
                         c='lightgray', s=0.5, alpha=0.3, label='Non-ground points')
    if len(ground_points_new) > 0:
        ax_xy_new.scatter(ground_points_new[:, 0], ground_points_new[:, 1], 
                         c='blue', s=2, alpha=0.6, label='Ground points')
    if len(cross_section_points_rotated) > 0:
        ax_xy_new.scatter(cross_section_points_rotated[:, 0], cross_section_points_rotated[:, 1], 
                         c='red', s=5, alpha=0.8, label='Cross-section points')
    ax_xy_new.plot([point1_rotated[0], point2_rotated[0]], [point1_rotated[1], point2_rotated[1]], 
                  'b-', linewidth=2, label='Cross-section line')
    # 抽出面の範囲を表示（平面の法線方向にthresholdの範囲で線を描画）
    # 新しいxyz軸では、平面の法線は断面直線に垂直な方向
    line_dir_rotated = point2_rotated - point1_rotated
    line_length_rotated = np.linalg.norm(line_dir_rotated)
    if line_length_rotated > 1e-10:
        line_dir_rotated = line_dir_rotated / line_length_rotated
        # 平面の法線ベクトル（XY平面に垂直）= [line_dir_rotated[1], -line_dir_rotated[0]]
        plane_normal_xy_new = np.array([line_dir_rotated[1], -line_dir_rotated[0]])
        plane_normal_xy_new_length = np.linalg.norm(plane_normal_xy_new)
        if plane_normal_xy_new_length > 1e-10:
            plane_normal_xy_new = plane_normal_xy_new / plane_normal_xy_new_length
            # 断面直線の両端から平面の法線方向にthresholdの範囲で線を描画
            for p in [point1_rotated, point2_rotated]:
                offset_x = np.array([p[0] - plane_normal_xy_new[0] * cross_section_threshold,
                                    p[0] + plane_normal_xy_new[0] * cross_section_threshold])
                offset_y = np.array([p[1] - plane_normal_xy_new[1] * cross_section_threshold,
                                    p[1] + plane_normal_xy_new[1] * cross_section_threshold])
                ax_xy_new.plot(offset_x, offset_y, 'm--', linewidth=1, alpha=0.5)
    # 新しいxyz軸ではZ軸は垂直方向[0, 0, 1]なので、XY平面への投影は点になる
    ax_xy_new.plot(mid_point_new[0], mid_point_new[1], 'go', markersize=10, label='Z-axis (vertical)')
    ax_xy_new.plot(point1_rotated[0], point1_rotated[1], 'bo', markersize=10, label='Point 1')
    ax_xy_new.plot(point2_rotated[0], point2_rotated[1], 'bs', markersize=10, label='Point 2')
    ax_xy_new.set_xlabel('X (m)')
    ax_xy_new.set_ylabel('Y (m)')
    ax_xy_new.set_title('XY Plane (New XYZ Axis)')
    ax_xy_new.set_aspect('equal')
    ax_xy_new.set_xlim(x_min_xy, x_max_xy)
    ax_xy_new.set_ylim(y_min_xy, y_max_xy)
    ax_xy_new.grid(True, alpha=0.3)
    ax_xy_new.legend(fontsize=8)
    
    # YZ平面
    if len(non_ground_points_new) > 0:
        ax_yz_new.scatter(non_ground_points_new[:, 1], non_ground_points_new[:, 2], 
                         c='lightgray', s=0.5, alpha=0.3, label='Non-ground points')
    if len(ground_points_new) > 0:
        ax_yz_new.scatter(ground_points_new[:, 1], ground_points_new[:, 2], 
                         c='blue', s=2, alpha=0.6, label='Ground points')
    if len(cross_section_points_rotated) > 0:
        ax_yz_new.scatter(cross_section_points_rotated[:, 1], cross_section_points_rotated[:, 2], 
                         c='red', s=5, alpha=0.8, label='Cross-section points')
    ax_yz_new.plot([point1_rotated[1], point2_rotated[1]], [point1_rotated[2], point2_rotated[2]], 
                  'b-', linewidth=2, label='Cross-section line')
    # 新しいxyz軸ではZ軸は垂直方向[0, 0, 1]なので、YZ平面では垂直線として表示
    z_axis_y_coords = np.array([mid_point_new[1], mid_point_new[1]])
    z_axis_z_coords = np.array([mid_point_new[2] - axis_length_new, mid_point_new[2] + axis_length_new])
    ax_yz_new.plot(z_axis_y_coords, z_axis_z_coords, 'g--', linewidth=2, label='Z-axis (vertical)')
    # 抽出面の範囲を表示（Z軸方向にz_rangeの範囲で線を描画）
    if z_min is not None and z_max is not None:
        # 断面直線に沿って、Z軸方向にz_rangeの範囲で線を描画
        for p in [point1_rotated, point2_rotated]:
            z_range_y = np.array([p[1], p[1]])
            z_range_z = np.array([z_min, z_max])
            ax_yz_new.plot(z_range_y, z_range_z, 'm--', linewidth=1, alpha=0.5)
    ax_yz_new.plot(point1_rotated[1], point1_rotated[2], 'bo', markersize=10, label='Point 1')
    ax_yz_new.plot(point2_rotated[1], point2_rotated[2], 'bs', markersize=10, label='Point 2')
    ax_yz_new.set_xlabel('Y (m)')
    ax_yz_new.set_ylabel('Z (m)')
    ax_yz_new.set_title('YZ Plane (New XYZ Axis)')
    ax_yz_new.set_aspect('equal')
    ax_yz_new.set_xlim(y_min_yz, y_max_yz)
    ax_yz_new.set_ylim(z_min_yz, z_max_yz)
    ax_yz_new.grid(True, alpha=0.3)
    ax_yz_new.legend(fontsize=8)
    
    # XZ平面
    if len(non_ground_points_new) > 0:
        ax_xz_new.scatter(non_ground_points_new[:, 0], non_ground_points_new[:, 2], 
                         c='lightgray', s=0.5, alpha=0.3, label='Non-ground points')
    if len(ground_points_new) > 0:
        ax_xz_new.scatter(ground_points_new[:, 0], ground_points_new[:, 2], 
                         c='blue', s=2, alpha=0.6, label='Ground points')
    if len(cross_section_points_rotated) > 0:
        ax_xz_new.scatter(cross_section_points_rotated[:, 0], cross_section_points_rotated[:, 2], 
                         c='red', s=5, alpha=0.8, label='Cross-section points')
    ax_xz_new.plot([point1_rotated[0], point2_rotated[0]], [point1_rotated[2], point2_rotated[2]], 
                  'b-', linewidth=2, label='Cross-section line')
    # 新しいxyz軸ではZ軸は垂直方向[0, 0, 1]なので、XZ平面では垂直線として表示
    z_axis_x_coords = np.array([mid_point_new[0], mid_point_new[0]])
    z_axis_z_coords = np.array([mid_point_new[2] - axis_length_new, mid_point_new[2] + axis_length_new])
    ax_xz_new.plot(z_axis_x_coords, z_axis_z_coords, 'g--', linewidth=2, label='Z-axis (vertical)')
    # 抽出面の範囲を表示（Z軸方向にz_rangeの範囲で線を描画）
    if z_min is not None and z_max is not None:
        # 断面直線に沿って、Z軸方向にz_rangeの範囲で線を描画
        for p in [point1_rotated, point2_rotated]:
            z_range_x = np.array([p[0], p[0]])
            z_range_z = np.array([z_min, z_max])
            ax_xz_new.plot(z_range_x, z_range_z, 'm--', linewidth=1, alpha=0.5)
    ax_xz_new.plot(point1_rotated[0], point1_rotated[2], 'bo', markersize=10, label='Point 1')
    ax_xz_new.plot(point2_rotated[0], point2_rotated[2], 'bs', markersize=10, label='Point 2')
    ax_xz_new.set_xlabel('X (m)')
    ax_xz_new.set_ylabel('Z (m)')
    ax_xz_new.set_title('XZ Plane (New XYZ Axis)')
    ax_xz_new.set_aspect('equal')
    ax_xz_new.set_xlim(x_min_xz, x_max_xz)
    ax_xz_new.set_ylim(z_min_xz, z_max_xz)
    ax_xz_new.grid(True, alpha=0.3)
    ax_xz_new.legend(fontsize=8)
    
    # 全体のタイトルを設定
    fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    return fig
