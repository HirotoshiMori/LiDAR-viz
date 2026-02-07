"""鉛直補正：地面平面推定と回転"""

import numpy as np
from typing import Tuple, Optional

# RANSAC で 3 点が退化（共線）とみなす法線ノルムの閾値
_PLANE_NORMAL_EPS = 1e-10


def _apply_roi_filter(
    points: np.ndarray,
    z_range: Optional[Tuple[float, float]] = None,
    xy_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ROI（z_range, xy_range）を適用し、フィルタ後の点群と元の点群に対するマスクを返す。
    Returns: (filtered_points, roi_mask) ただし roi_mask は len(points) の bool 配列。
    """
    roi_mask = np.ones(len(points), dtype=bool)
    filtered = points.copy()
    if z_range is not None:
        min_z, max_z = z_range
        z_mask = (filtered[:, 2] >= min_z) & (filtered[:, 2] <= max_z)
        filtered = filtered[z_mask]
        roi_mask = roi_mask & z_mask
    if xy_range is not None:
        (min_x, max_x), (min_y, max_y) = xy_range
        xy_mask = (
            (points[:, 0] >= min_x) & (points[:, 0] <= max_x)
            & (points[:, 1] >= min_y) & (points[:, 1] <= max_y)
        )
        roi_mask = roi_mask & xy_mask
        filtered = points[roi_mask]
    return filtered, roi_mask


def _fit_plane_from_points(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    3点から平面 ax+by+cz+d=0 を fitting する。
    pts: (3, 3) の配列。法線は正規化済みで返す。
    Returns: (normal, d) where normal is (3,) unit vector, plane eq: dot(normal, p) + d = 0
    """
    p1, p2, p3 = pts[0], pts[1], pts[2]
    normal = np.cross(p2 - p1, p3 - p1)
    nnorm = np.linalg.norm(normal)
    if nnorm < _PLANE_NORMAL_EPS:
        return np.array([0.0, 0.0, 1.0]), 0.0  # 退化時は水平面
    normal = normal / nnorm
    d = -np.dot(normal, p1)
    return normal, d


def _count_inliers(
    points: np.ndarray,
    normal: np.ndarray,
    d: float,
    distance_threshold: float,
) -> np.ndarray:
    """各点から平面への符号付き距離 |dot(normal, p) + d| が threshold 以内のインデックスを返す。"""
    dists = np.abs(np.dot(points, normal) + d)
    return np.where(dists <= distance_threshold)[0]


def estimate_ground_plane(
    points: np.ndarray,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    z_range: Optional[Tuple[float, float]] = None,
    xy_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RANSAC平面推定で地面平面を抽出する。
    
    Args:
        points: N×3の点群配列
        distance_threshold: RANSACの距離閾値（デフォルト: 0.02 m）
        ransac_n: RANSACのサンプル数（デフォルト: 3）
        num_iterations: RANSACの反復回数（デフォルト: 1000）
        z_range: Z座標の範囲フィルタ (min_z, max_z)。Noneの場合はフィルタなし
        xy_range: XY座標の範囲フィルタ ((min_x, max_x), (min_y, max_y))。Noneの場合はフィルタなし
        
    Returns:
        (plane_normal, plane_point, ground_mask): 
        - 平面の法線ベクトル（3要素）
        - 平面上の点（3要素）
        - 地面と判定された点のマスク（元の点群に対する、長さNのbool配列）
        
    Raises:
        ValueError: 点群が空または平面推定に失敗した場合
    """
    if len(points) == 0:
        raise ValueError("点群が空です")

    filtered_points, roi_mask = _apply_roi_filter(points, z_range, xy_range)
    if len(filtered_points) < 3:
        raise ValueError(f"ROIフィルタ後の点が不足しています（{len(filtered_points)}点）")
    
    # RANSAC平面推定（numpy のみで実装）
    rng = np.random.default_rng()
    n_pts = len(filtered_points)
    best_inliers = np.array([], dtype=np.intp)
    best_normal = np.array([0.0, 0.0, 1.0])
    best_d = 0.0

    for _ in range(num_iterations):
        idx = rng.choice(n_pts, size=min(ransac_n, n_pts), replace=False)
        sample = filtered_points[idx]
        if len(sample) < 3:
            continue
        normal, d = _fit_plane_from_points(sample)
        inliers = _count_inliers(
            filtered_points, normal, d, distance_threshold
        )
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = normal
            best_d = d

    if len(best_inliers) == 0:
        raise ValueError("平面推定に失敗しました（インライアが0点）")

    plane_normal = best_normal.copy()
    plane_d = best_d
    # 法線を「上」向きに統一：回転後の +Z が空（上）、地面が下になるようにする
    if plane_normal[2] < 0:
        plane_normal = -plane_normal
        plane_d = -plane_d

    # 平面上の点を計算（原点から平面への射影）
    if abs(plane_d) > 1e-10:
        # 原点から平面への最短点
        plane_point = -plane_d * plane_normal / np.dot(plane_normal, plane_normal)
    else:
        # d=0の場合は原点が平面上
        plane_point = np.array([0.0, 0.0, 0.0])
    
    # 元の点群に対する地面マスクを作成
    # ROIフィルタ後の点群のインデックスを元の点群のインデックスにマッピング
    ground_mask = np.zeros(len(points), dtype=bool)
    roi_indices = np.where(roi_mask)[0]  # ROIフィルタ後の点群の元のインデックス
    ground_mask[roi_indices[best_inliers]] = True  # インライアの点をTrueに設定
    
    return plane_normal, plane_point, ground_mask


def create_rotation_matrix(plane_normal: np.ndarray) -> np.ndarray:
    """
    平面の法線ベクトルを +Z 方向に向ける回転行列を作成する。
    
    Args:
        plane_normal: 平面の法線ベクトル（3要素、正規化済み）
        
    Returns:
        3×3の回転行列
    """
    target_normal = np.array([0.0, 0.0, 1.0])  # +Z方向
    
    # 法線ベクトルが既に+Z方向を向いている場合
    if np.allclose(plane_normal, target_normal):
        return np.eye(3)
    
    # 法線ベクトルが-Z方向を向いている場合
    if np.allclose(plane_normal, -target_normal):
        # 180度回転（任意の軸で回転）
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
    
    # ロドリゲスの回転公式を使用
    # 回転軸 = plane_normal × target_normal
    rotation_axis = np.cross(plane_normal, target_normal)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # 回転角
    cos_angle = np.dot(plane_normal, target_normal)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    # ロドリゲスの回転公式
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    
    R = (np.eye(3) + 
         np.sin(angle) * K + 
         (1 - np.cos(angle)) * np.dot(K, K))
    
    return R


def apply_rotation(points: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    点群に回転行列を適用する。
    
    Args:
        points: N×3の点群配列
        rotation_matrix: 3×3の回転行列
        
    Returns:
        回転後の点群（N×3）
    """
    return points @ rotation_matrix.T


def distance_to_plane(
    points: np.ndarray,
    plane_normal: np.ndarray,
    plane_point: np.ndarray
) -> np.ndarray:
    """
    点群の各点から地面平面への垂直距離を計算する。
    
    Args:
        points: N×3の点群配列
        plane_normal: 平面の法線ベクトル（3要素、正規化済み）
        plane_point: 平面上の点（3要素）
        
    Returns:
        各点から平面への垂直距離の配列（長さN、符号付き）
    """
    # 各点から平面上の点へのベクトル
    vec_to_plane = points - plane_point
    
    # 法線方向への射影（これが垂直距離）
    distances = np.dot(vec_to_plane, plane_normal)
    
    return distances
