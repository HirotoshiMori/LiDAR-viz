"""断面プロファイル化とインデックスクリップ"""

import numpy as np
from typing import Tuple

from scipy.signal import savgol_filter

from .geometry import project_to_line


def create_profile(
    cross_section_points_original: np.ndarray,
    cross_section_points_rotated: np.ndarray,
    point1: np.ndarray,
    point2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    断面点群から (x_mm, z_mm) のプロファイルを作成する。
    
    xは元座標系での直線への射影距離（point1からの距離）、
    zは地面平面からの垂直距離（回転後のZ座標、地面平面の法線方向への変位）。
    両方ともmm単位で返す。
    
    注意：回転後のZ座標は、地面平面の法線方向（新しいZ軸方向）への変位を表している。
    これは地面平面からの垂直距離を表している。
    
    Args:
        cross_section_points_original: 断面点群（M×3、元データ座標系、回転前）
        cross_section_points_rotated: 断面点群（M×3、鉛直補正後、回転後）
        point1: 直線の第1点 [x, y, z]（元データ座標系）
        point2: 直線の第2点 [x, y, z]（元データ座標系）
        
    Returns:
        (x_mm, z_mm): 距離（mm）と地面平面からの垂直距離（mm）の配列
    """
    # 元座標系での射影距離を計算
    x_proj = project_to_line(cross_section_points_original, point1, point2)
    
    # mm単位に変換
    x_mm = x_proj * 1000.0
    
    # 回転後のZ座標をmm単位で取得
    # 回転行列は地面平面の法線を+Z方向に向けるように作成されているため、
    # 回転後のZ座標は地面平面からの垂直距離を表している
    z_mm = cross_section_points_rotated[:, 2] * 1000.0
    
    return x_mm, z_mm


def prominent_inflection_x_mm(
    x_mm: np.ndarray,
    z_mm: np.ndarray,
    *,
    y_increases_with_x: bool,
    slope_window_mm: float = 20.0,
    min_angle_deg: float = 45.0,
) -> float | None:
    """断面の Y が負の側から見て、勾配角が最初に min_angle_deg 以上変わる位置を返す。

    x は point1 から point2 への断面距離（mm）。走査は Y の負側から正側へ進む。
    勾配角は arctan(dz/dx) で、前後 slope_window_mm の差を見る。
    """
    x = np.asarray(x_mm, dtype=float)
    z = np.asarray(z_mm, dtype=float)
    finite = np.isfinite(x) & np.isfinite(z)
    if int(finite.sum()) < 15:
        return None
    z_fill = z.copy()
    if not bool(finite.all()):
        z_fill[~finite] = np.interp(x[~finite], x[finite], z[finite])
    dx = float(np.median(np.diff(x)))
    if not np.isfinite(dx) or dx <= 0:
        return None
    n = len(z_fill)
    win = min(31, n if n % 2 == 1 else n - 1)
    if win < 7:
        return None
    slope = savgol_filter(z_fill, win, 3, deriv=1, delta=dx)
    theta = np.degrees(np.arctan(slope))
    half = max(1, int(round(slope_window_mm / dx)))
    if 2 * half >= n:
        return None
    order = range(half, n - half) if y_increases_with_x else range(n - half - 1, half - 1, -1)
    for i in order:
        if not finite[i]:
            continue
        behind = i - half if y_increases_with_x else i + half
        ahead = i + half if y_increases_with_x else i - half
        if abs(float(theta[ahead] - theta[behind])) >= min_angle_deg:
            return float(x[i])
    return None


def clip_profile_indices(
    x: np.ndarray,
    z: np.ndarray,
    start_index: int,
    end_index_max: int
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    プロファイルのインデックス範囲を安全にクリップする。
    
    Args:
        x: 距離配列（mm）
        z: Z座標配列（mm）
        start_index: 開始インデックス
        end_index_max: 終了インデックスの最大値
        
    Returns:
        (x_clipped, z_clipped, start_actual, end_actual):
        クリップ後のx, z配列と実際に使用された開始・終了インデックス
        
    Raises:
        ValueError: クリップ後の範囲が無効な場合
    """
    n = len(x)
    
    # インデックスを安全にクリップ
    start_actual = max(0, min(start_index, n - 1))
    end_actual = min(end_index_max, n)
    
    if end_actual <= start_actual:
        raise ValueError(
            f"クリップ後の範囲が無効です: start={start_actual}, end={end_actual}, "
            f"元の範囲: start_index={start_index}, end_index_max={end_index_max}, n={n}"
        )
    
    x_clipped = x[start_actual:end_actual]
    z_clipped = z[start_actual:end_actual]
    
    return x_clipped, z_clipped, start_actual, end_actual
