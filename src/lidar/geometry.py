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


def revetment_intersection_distance_mm(
    revetment: np.ndarray,
    point1: np.ndarray,
    point2: np.ndarray
) -> float:
    """
    護岸点を、断面直線（point1-point2）を含み新しいZ軸に平行な平面に垂直な平面と
    断面直線の交点に射影し、point1からの距離をmm単位で返す。
    
    幾何学的には、護岸点から断面直線への垂線の足（投影点）が交点であり、
    そのpoint1からの距離が返り値となる。
    
    Args:
        revetment: 護岸点 [x, y, z]（元データ座標系、m単位）
        point1: 断面直線の第1点 [x, y, z]（元データ座標系）
        point2: 断面直線の第2点 [x, y, z]（元データ座標系）
        
    Returns:
        point1からの射影距離（mm単位）
    """
    dist_m = project_to_line(np.atleast_2d(revetment), point1, point2)[0]
    return float(dist_m * 1000.0)


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
