"""幾何計算：直線距離と断面抽出"""

import numpy as np
from typing import Tuple, Optional

# 直線の長さがこれ未満の場合は同一点とみなす
_LINE_LENGTH_EPS = 1e-10


def _normalized_line_direction(point1: np.ndarray, point2: np.ndarray) -> Tuple[np.ndarray, float]:
    """直線の単位方向ベクトルと長さを返す。同一点の場合は ValueError。"""
    line_dir = point2 - point1
    length = np.linalg.norm(line_dir)
    if length < _LINE_LENGTH_EPS:
        raise ValueError("point1とpoint2が同じ点です")
    return line_dir / length, length


def distance_to_line(
    points: np.ndarray,
    point1: np.ndarray,
    point2: np.ndarray
) -> np.ndarray:
    """
    点群の各点から、2点で定義される直線への最短距離を計算する。
    
    Args:
        points: N×3の点群配列（元データ座標系）
        point1: 直線の第1点 [x, y, z]（元データ座標系）
        point2: 直線の第2点 [x, y, z]（元データ座標系）
        
    Returns:
        各点から直線への最短距離の配列（長さN）
    """
    line_dir, _ = _normalized_line_direction(point1, point2)
    vec_to_point1 = points - point1
    proj_length = np.dot(vec_to_point1, line_dir)
    
    # 直線上で最も近い点
    closest_on_line = point1 + proj_length[:, np.newaxis] * line_dir
    
    # 各点から直線への距離
    distances = np.linalg.norm(points - closest_on_line, axis=1)
    
    return distances


def project_to_line(
    points: np.ndarray,
    point1: np.ndarray,
    point2: np.ndarray
) -> np.ndarray:
    """
    点群の各点を、2点で定義される直線に射影し、point1からの距離を返す。
    
    Args:
        points: N×3の点群配列（元データ座標系）
        point1: 直線の第1点 [x, y, z]（元データ座標系）
        point2: 直線の第2点 [x, y, z]（元データ座標系）
        
    Returns:
        各点のpoint1からの射影距離の配列（長さN）
    """
    line_dir, line_length = _normalized_line_direction(point1, point2)
    vec_to_point1 = points - point1
    proj_length = np.dot(vec_to_point1, line_dir)
    return proj_length


def profile_point_xyz(
    point1: np.ndarray,
    point2: np.ndarray,
    rotation_matrix: np.ndarray,
    along_mm: float,
    z_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """断面距離と回転後高さから、元座標と回転後座標の1点を返す。"""
    line_dir, _ = _normalized_line_direction(point1, point2)
    foot = np.asarray(point1, dtype=float) + line_dir * (along_mm / 1000.0)
    rotated = foot @ rotation_matrix.T
    rotated = rotated.copy()
    rotated[2] = z_mm / 1000.0
    original = rotated @ rotation_matrix
    return original, rotated


def extract_cross_section(
    points: np.ndarray,
    point1: np.ndarray,
    point2: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    元データ座標系で、指定直線から一定距離以内で、かつpoint1とpoint2の間にある点を抽出する。
    
    Args:
        points: N×3の点群配列（元データ座標系）
        point1: 直線の第1点 [x, y, z]（元データ座標系）
        point2: 直線の第2点 [x, y, z]（元データ座標系）
        threshold: 断面抽出の距離閾値（m）
        
    Returns:
        抽出された断面点群（M×3の配列、M <= N）
    """
    # 直線からの距離を計算
    distances = distance_to_line(points, point1, point2)
    
    # point1からpoint2への射影距離を計算
    proj_distances = project_to_line(points, point1, point2)
    
    _, line_length = _normalized_line_direction(point1, point2)
    mask = (distances <= threshold) & (proj_distances >= 0) & (proj_distances <= line_length)
    return points[mask]


def distance_to_plane_containing_line(
    points: np.ndarray,
    point1: np.ndarray,
    point2: np.ndarray,
    plane_normal: np.ndarray
) -> np.ndarray:
    """
    点群の各点から、指定直線を含み、指定された法線ベクトルに平行な平面への距離を計算する。
    
    この平面は、断面直線（point1, point2）を含み、新しいz軸（plane_normal）に平行な平面である。
    平面の法線ベクトルは、断面直線の方向ベクトルと新しいz軸の外積で計算される。
    
    Args:
        points: N×3の点群配列（元データ座標系）
        point1: 直線の第1点 [x, y, z]（元データ座標系）
        point2: 直線の第2点 [x, y, z]（元データ座標系）
        plane_normal: 新しいz軸方向のベクトル（地面平面の法線、3要素、正規化済み）
        
    Returns:
        各点から平面への距離の配列（長さN、符号付き）
    """
    line_dir, _ = _normalized_line_direction(point1, point2)
    # 平面の法線ベクトル = 断面直線の方向ベクトル × 新しいz軸
    # この平面は断面直線を含み、新しいz軸に平行
    plane_normal_vec = np.cross(line_dir, plane_normal)
    plane_normal_length = np.linalg.norm(plane_normal_vec)
    
    if plane_normal_length < 1e-10:
        # 断面直線が新しいz軸と平行な場合、外積が0になる
        # この場合は、断面直線を含み、新しいz軸に垂直な平面を定義
        # 平面の法線 = 新しいz軸に垂直な任意のベクトル（断面直線の方向ベクトルに垂直なベクトル）
        # 簡単のため、断面直線の方向ベクトルに垂直な単位ベクトルを計算
        # 断面直線がz軸と平行な場合、XY平面に垂直な平面を定義
        if abs(line_dir[2]) < 1e-10:
            # 断面直線がXY平面内にある場合
            plane_normal_vec = np.array([0.0, 0.0, 1.0])  # Z軸方向
        else:
            # 断面直線がZ軸方向の場合、X軸方向を法線とする
            plane_normal_vec = np.array([1.0, 0.0, 0.0])
    else:
        plane_normal_vec = plane_normal_vec / plane_normal_length
    
    # 平面上の点としてpoint1を使用
    plane_point = point1
    
    # 各点から平面へのベクトル
    vec_to_plane = points - plane_point
    
    # 法線方向への射影（これが平面からの距離）
    distances = np.dot(vec_to_plane, plane_normal_vec)
    
    return distances


def _local_surface_height(z: np.ndarray, gap_m: float) -> float:
    """ビン内の地表面高さ。上下に max の隙間があるときは、隙間より下の点の中央値。"""
    ordered = np.sort(np.asarray(z, dtype=float))
    if len(ordered) < 3:
        return float("nan")
    gaps = np.diff(ordered)
    split = int(np.argmax(gaps))
    if gaps[split] > gap_m and split >= 2:
        return float(np.median(ordered[: split + 1]))
    return float(np.median(ordered))


def drop_above_local_surface(
    points: np.ndarray,
    z: np.ndarray,
    along: np.ndarray,
    max_above_m: float,
    bin_m: float = 0.04,
) -> np.ndarray:
    """
    断面に沿った各地点の地表面より max_above_m 以上高い点を除く。

    計測器がビンの過半数でも、地表面との隙間が max_above_m より大きければ下側を地表面にする。
    面からの距離では落ちない。
    """
    if len(points) == 0:
        return points
    z = np.asarray(z, dtype=float)
    along = np.asarray(along, dtype=float)
    a_min = float(np.min(along))
    a_max = float(np.max(along))
    if not np.isfinite(a_min) or a_max - a_min < bin_m:
        med = _local_surface_height(z, max_above_m)
        if not np.isfinite(med):
            med = float(np.median(z))
        return points[z <= med + max_above_m]

    n_bins = int(np.floor((a_max - a_min) / bin_m + 1e-9)) + 1
    idx = np.clip(((along - a_min) / bin_m).astype(int), 0, n_bins - 1)
    medians = np.full(n_bins, np.nan)
    for i in range(n_bins):
        chosen = z[idx == i]
        if len(chosen) >= 3:
            medians[i] = _local_surface_height(chosen, max_above_m)
    valid = np.flatnonzero(np.isfinite(medians))
    if len(valid) == 0:
        med = float(np.median(z))
        return points[z <= med + max_above_m]
    for i in range(n_bins):
        if not np.isfinite(medians[i]):
            medians[i] = medians[valid[np.argmin(np.abs(valid - i))]]
    original = medians.copy()
    for i in range(n_bins):
        neighbors = []
        if i > 0:
            neighbors.append(original[i - 1])
        if i + 1 < n_bins:
            neighbors.append(original[i + 1])
        if neighbors and original[i] > max(neighbors) + max_above_m:
            medians[i] = min(neighbors)
    return points[z <= medians[idx] + max_above_m]


def extract_cross_section_by_plane(
    points: np.ndarray,
    point1: np.ndarray,
    point2: np.ndarray,
    plane_normal: np.ndarray,
    threshold: float,
    rotation_matrix: np.ndarray,
    z_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    断面直線を含み、新しいz軸（plane_normal）に平行な平面から一定距離以内の点を抽出する。
    かつ、point1とpoint2の間にある点のみを抽出する。
    新しいz軸での範囲指定も可能。
    
    Args:
        points: N×3の点群配列（元データ座標系）
        point1: 直線の第1点 [x, y, z]（元データ座標系）
        point2: 直線の第2点 [x, y, z]（元データ座標系）
        plane_normal: 新しいz軸方向のベクトル（地面平面の法線、3要素、正規化済み）
        threshold: 断面抽出の距離閾値（m）
        rotation_matrix: 回転行列（3×3）
        z_range: 新しいz軸でのZ座標の範囲フィルタ (min_z, max_z)。Noneの場合はフィルタなし
        
    Returns:
        抽出された断面点群（M×3の配列、M <= N、元データ座標系）
    """
    plane_distances = distance_to_plane_containing_line(points, point1, point2, plane_normal)
    proj_distances = project_to_line(points, point1, point2)
    _, line_length = _normalized_line_direction(point1, point2)
    mask = (np.abs(plane_distances) <= threshold) & (proj_distances >= 0) & (proj_distances <= line_length)
    
    # 新しいz軸での範囲フィルタ
    if z_range is not None:
        min_z, max_z = z_range
        # 点群を回転して新しいz軸でのZ座標を取得
        points_rotated = points @ rotation_matrix.T
        z_coords_new = points_rotated[:, 2]
        z_mask = (z_coords_new >= min_z) & (z_coords_new <= max_z)
        mask = mask & z_mask
    
    cross_section_points = points[mask]
    
    return cross_section_points
