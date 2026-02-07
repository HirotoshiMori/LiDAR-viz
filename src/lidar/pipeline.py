"""LiDAR断面処理パイプライン"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import extract_cross_section_by_plane, revetment_intersection_distance_mm
from .io_ply import load_point_cloud, natural_sort_paths
from .leveling import apply_rotation, create_rotation_matrix, estimate_ground_plane
from .preprocess_lidar import preprocess_lidar_profile
from .profile import clip_profile_indices, create_profile
from .resample import resample_to_common_grid


@dataclass
class PipelineResult:
    """パイプライン実行結果"""

    common_x: np.ndarray
    all_differences: list[np.ndarray]
    all_displacements: list[np.ndarray]
    all_before_resample: list[np.ndarray]
    all_before_filter: list[np.ndarray]
    legend_names: list[str]
    point1: np.ndarray
    point2: np.ndarray
    revetment: np.ndarray
    config: dict[str, Any]

    # 3D可視化用（初期断面）
    initial_points_original: np.ndarray | None = None
    initial_points_rotated: np.ndarray | None = None
    initial_cross_section_original: np.ndarray | None = None
    initial_cross_section_rotated: np.ndarray | None = None
    initial_plane_normal: np.ndarray | None = None
    initial_ground_mask: np.ndarray | None = None

    @property
    def x_zero_zoom(self) -> float:
        """Zoomグラフのゼロ点（護岸交点 + シフト）[mm]"""
        shift = self.config.get("plot", {}).get("revetment_shift_mm", 0)
        return revetment_intersection_distance_mm(
            self.revetment, self.point1, self.point2
        ) + shift

    @property
    def plot_xlim_zoom(self) -> tuple[float, float]:
        """Zoom区間のX軸範囲"""
        w = self.config.get("plot", {}).get("zoom_width", 220)
        z0 = self.x_zero_zoom
        return (z0, z0 + w)


def _select_ply_files(
    ply_dir: Path,
    all_ply_files: list[Path],
    legend_names: list[str],
) -> tuple[list[Path], list[str]]:
    """plot_legend_names に基づいてPLYファイルを選択。"""
    if not legend_names:
        return all_ply_files, [p.name for p in all_ply_files]

    initial_ply = all_ply_files[0]

    file_time_candidates: list[tuple[Path, int]] = []
    for pf in all_ply_files:
        nums = re.findall(r"\d+", pf.stem)
        if not nums:
            continue
        raw_num_str = nums[-1]
        if len(raw_num_str) >= 2:
            time_val = int(raw_num_str[1:])
        else:
            time_val = int(raw_num_str)
        file_time_candidates.append((pf, time_val))

    available_times = sorted({t for _, t in file_time_candidates})

    def choose_last_kth(k: int) -> tuple[Path, int]:
        if not available_times:
            raise FileNotFoundError(
                f"時間情報を含むPLYファイルが {ply_dir} に見つかりません"
            )
        idx = max(0, len(available_times) - 1 - k)
        target_time = available_times[idx]
        for pf, t in file_time_candidates:
            if t == target_time:
                return pf, target_time
        return file_time_candidates[0]

    selected_files: list[Path] = []
    resolved_names: list[str] = []
    selected_times: list[int | None] = []

    for label in legend_names:
        label_stripped = label.strip()

        if label_stripped.lower() == "initial":
            selected_files.append(initial_ply)
            resolved_names.append(label_stripped)
            selected_times.append(None)
            continue

        if label_stripped.lower().startswith("last"):
            if not file_time_candidates:
                selected_files.append(all_ply_files[-1])
                resolved_names.append(label_stripped)
                selected_times.append(None)
                continue

            m_last = re.match(r"last-(\d+)", label_stripped, flags=re.IGNORECASE)
            k = int(m_last.group(1)) if m_last else 0
            chosen_pf, chosen_time = choose_last_kth(k)

            # Last より大きい時間指定を除外
            filtered = [
                (pf, lbl, t)
                for pf, lbl, t in zip(selected_files, resolved_names, selected_times)
                if t is None or t <= chosen_time
            ]
            selected_files = [x[0] for x in filtered]
            resolved_names = [x[1] for x in filtered]
            selected_times = [x[2] for x in filtered]

            selected_files.append(chosen_pf)
            resolved_names.append(f"{chosen_time} minutes")
            selected_times.append(chosen_time)
            continue

        m = re.search(r"\d+", label_stripped)
        if not m:
            raise ValueError(
                f"plot_legend_names の要素 '{label}' から時間を表す数字が抽出できません"
            )
        time_str = m.group(0)
        time_val = int(time_str)

        candidates = [
            pf for pf in all_ply_files
            if time_str in pf.stem or time_str in pf.name
        ]
        if candidates:
            selected_files.append(candidates[0])
            resolved_names.append(label_stripped)
            selected_times.append(time_val)
            continue

        if not file_time_candidates:
            raise FileNotFoundError(
                f"ラベル '{label}' に対応するPLYファイルが {ply_dir} に見つかりません"
            )

        def sort_key(item: tuple[Path, int]) -> tuple[int, int]:
            pf, t = item
            return (abs(t - time_val), -t)

        nearest_pf, nearest_time = sorted(file_time_candidates, key=sort_key)[0]
        selected_files.append(nearest_pf)
        selected_times.append(nearest_time)
        updated_label = re.sub(r"\d+", str(nearest_time), label, count=1)
        resolved_names.append(updated_label)

    return selected_files, resolved_names


def run_pipeline(config: dict[str, Any], verbose: bool = True) -> PipelineResult:
    """
    設定に基づいてLiDAR断面処理を一括実行する。

    Args:
        config: load_config() で取得した設定
        verbose: 進捗を標準出力に表示するか

    Returns:
        PipelineResult
    """
    ply_dir = Path(config["_ply_directory"])
    point1 = np.array(config["points"]["point1"], dtype=float)
    point2 = np.array(config["points"]["point2"], dtype=float)
    revetment = np.array(config["points"]["revetment"], dtype=float)

    cs = config.get("cross_section", {})
    cg = config.get("common_grid", {})
    flt = config.get("filter", {})
    grd = config.get("ground", {})
    plot_cfg = config.get("plot", {})

    cross_section_threshold = cs.get("threshold", 0.005)
    cross_section_z_range = cs.get("z_range")
    start_index = cs.get("start_index", 0)
    end_index_max = cs.get("end_index_max", 5000)

    common_x_min = cg.get("x_min", 0.0)
    common_x_max = cg.get("x_max")
    common_x_points = cg.get("x_points", 500)

    interp_method = flt.get("interp_method", "linear")

    # PLYファイル取得・選択
    ply_files = list(ply_dir.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"PLYファイルが見つかりません: {ply_dir}")

    ply_files_sorted = natural_sort_paths(ply_files)
    legend_names = plot_cfg.get("legend_names")
    if legend_names:
        ply_files_sorted, legend_names = _select_ply_files(
            ply_dir, ply_files_sorted, legend_names
        )
    else:
        legend_names = [p.name for p in ply_files_sorted]

    if verbose:
        print(f"✓ {len(ply_files_sorted)} 個のPLYファイルを使用（処理順）")
        for i, pf in enumerate(ply_files_sorted[:5]):
            print(f"  [{i}] {pf.name}")
        if len(ply_files_sorted) > 5:
            print(f"  ... (他 {len(ply_files_sorted) - 5} 個)")

    # 共通グリッド
    if common_x_max is None:
        line_length = float(np.linalg.norm(point2 - point1))
        common_x_max = line_length * 1000.0
        if verbose:
            print(f"✓ common_x_max 自動設定: {common_x_max:.2f} mm")

    common_x = np.linspace(common_x_min, common_x_max, common_x_points)

    # メインループ
    all_differences: list[np.ndarray] = []
    all_displacements: list[np.ndarray] = []
    all_before_resample: list[np.ndarray] = []
    all_before_filter: list[np.ndarray] = []
    baseline_displacement: np.ndarray | None = None

    initial_points_original: np.ndarray | None = None
    initial_points_rotated: np.ndarray | None = None
    initial_cross_section_original: np.ndarray | None = None
    initial_cross_section_rotated: np.ndarray | None = None
    initial_plane_normal: np.ndarray | None = None
    initial_ground_mask: np.ndarray | None = None

    for idx, ply_path in enumerate(ply_files_sorted):
        if verbose:
            print(f"\n[{idx}] 処理中: {ply_path.name}")

        points_original = load_point_cloud(ply_path)
        if verbose:
            print(f"  ✓ 点群読み込み: {len(points_original)} 点")

        if idx == 0:
            plane_normal, plane_point, ground_mask = estimate_ground_plane(
                points_original,
                distance_threshold=grd.get("distance_threshold", 0.001),
                ransac_n=grd.get("ransac_n", 3),
                num_iterations=grd.get("num_iterations", 1000),
                z_range=grd.get("z_range"),
                xy_range=grd.get("xy_range"),
            )
            if verbose:
                print(f"  ✓ 地面平面推定完了（法線: {plane_normal}）")
        else:
            plane_normal = initial_plane_normal
            plane_point = None
            ground_mask = np.zeros(len(points_original), dtype=bool)

        rotation_matrix = create_rotation_matrix(plane_normal)

        cross_section_original = extract_cross_section_by_plane(
            points_original,
            point1,
            point2,
            plane_normal,
            threshold=cross_section_threshold,
            rotation_matrix=rotation_matrix,
            z_range=cross_section_z_range,
        )
        if verbose:
            print(f"  ✓ 断面抽出: {len(cross_section_original)} 点")

        points_rotated = apply_rotation(points_original, rotation_matrix)

        if idx == 0:
            initial_points_original = points_original
            initial_points_rotated = points_rotated
            initial_cross_section_original = cross_section_original
            initial_cross_section_rotated = (
                apply_rotation(cross_section_original, rotation_matrix)
                if len(cross_section_original) > 0
                else np.empty((0, 3))
            )
            initial_plane_normal = plane_normal
            initial_ground_mask = ground_mask

        if len(cross_section_original) == 0:
            if verbose:
                print("  ⚠ 断面点が0点です。スキップします。")
            continue

        cross_section_rotated = apply_rotation(cross_section_original, rotation_matrix)

        x_mm, z_mm = create_profile(
            cross_section_original,
            cross_section_rotated,
            point1,
            point2,
        )

        x_clipped, z_clipped, start_actual, end_actual = clip_profile_indices(
            x_mm, z_mm, start_index, end_index_max
        )
        if verbose:
            print(f"  ✓ インデックスクリップ: [{start_actual}:{end_actual}] ({len(x_clipped)} 点)")

        x_processed, z_raw, z_median, z_smooth, meta = preprocess_lidar_profile(
            x_clipped,
            z_clipped,
            dx=flt.get("dx"),
            uneven_tol=flt.get("uneven_tol", 0.01),
            median_k=flt.get("median_k", 5),
            sg_window=flt.get("sg_window", 21),
            sg_poly=flt.get("sg_poly", 3),
            interp_method=interp_method,
        )
        if verbose:
            print(f"  ✓ 前処理完了: {len(x_processed)} 点")

        z_before_resample = resample_to_common_grid(
            x_clipped, z_clipped, common_x, method=interp_method
        )
        z_before_filter = resample_to_common_grid(
            x_processed, z_raw, common_x, method=interp_method
        )
        z_on_common = resample_to_common_grid(
            x_processed, z_smooth, common_x, method=interp_method
        )

        if idx == 0:
            baseline_displacement = z_on_common
            difference_displacement = np.zeros_like(z_on_common)
            if verbose:
                print("  ✓ ベースライン設定（idx=0）")
        else:
            difference_displacement = z_on_common - baseline_displacement
            if verbose:
                print("  ✓ 差分計算完了")

        all_differences.append(difference_displacement)
        all_displacements.append(z_on_common)
        all_before_resample.append(z_before_resample)
        all_before_filter.append(z_before_filter)

    if verbose:
        print(f"\n✓ 全処理完了: {len(all_differences)} 断面")

    return PipelineResult(
        common_x=common_x,
        all_differences=all_differences,
        all_displacements=all_displacements,
        all_before_resample=all_before_resample,
        all_before_filter=all_before_filter,
        legend_names=legend_names,
        point1=point1,
        point2=point2,
        revetment=revetment,
        config=config,
        initial_points_original=initial_points_original,
        initial_points_rotated=initial_points_rotated,
        initial_cross_section_original=initial_cross_section_original,
        initial_cross_section_rotated=initial_cross_section_rotated,
        initial_plane_normal=initial_plane_normal,
        initial_ground_mask=initial_ground_mask,
    )
