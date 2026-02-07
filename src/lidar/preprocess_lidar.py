"""断面プロファイルの前処理：等間隔化・スパイク除去・平滑化"""

import numpy as np
from scipy import interpolate, ndimage, signal
from typing import Tuple, Optional, Dict, Any


def resample_to_uniform_grid(
    x: np.ndarray,
    z: np.ndarray,
    dx: Optional[float] = None,
    method: str = "linear"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    不等間隔データを等間隔グリッドに再サンプリングする。
    
    Args:
        x: 距離配列（昇順ソート済み）
        z: 値配列
        dx: 目標グリッド間隔。Noneの場合はmedian(diff(x))を使用
        method: 補間方法（"linear", "nearest", "quadratic", "cubic"）
        
    Returns:
        (x_uniform, z_uniform): 等間隔化後のx, z配列
    """
    if len(x) < 2:
        return x.copy(), z.copy()
    
    if dx is None:
        dx = np.median(np.diff(x))
    
    if dx <= 0:
        raise ValueError(f"無効なグリッド間隔: dx={dx}")
    
    # 等間隔グリッドを作成
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    x_uniform = np.arange(x_min, x_max + dx, dx)
    
    # 補間
    f = interpolate.interp1d(
        x, z,
        kind=method,
        bounds_error=False,
        fill_value=np.nan
    )
    z_uniform = f(x_uniform)
    
    return x_uniform, z_uniform


def despike_median(z: np.ndarray, k: int = 5) -> np.ndarray:
    """
    メディアンフィルタによるスパイク除去。
    
    Args:
        z: 値配列
        k: メディアンフィルタの窓長（サンプル数、奇数に丸められる）
        
    Returns:
        フィルタ後のz配列
    """
    if k < 1:
        raise ValueError(f"メディアンフィルタの窓長は1以上である必要があります: k={k}")
    
    # 奇数に丸める
    k_odd = k if k % 2 == 1 else k + 1
    
    if len(z) < k_odd:
        # データが少ない場合はそのまま返す
        return z.copy()
    
    z_filtered = ndimage.median_filter(z, size=k_odd)
    return z_filtered


def smooth_savgol(
    z: np.ndarray,
    window_length: int,
    polyorder: int,
    dx: float = 1.0
) -> np.ndarray:
    """
    Savitzky-Golayフィルタによる平滑化。
    
    Args:
        z: 値配列
        window_length: 窓長（サンプル数、奇数に調整される）
        polyorder: 多項式次数
        dx: グリッド間隔（物理空間でのスケール計算用、デフォルト: 1.0）
        
    Returns:
        平滑化後のz配列
        
    Raises:
        ValueError: パラメータが無効な場合
    """
    if window_length <= polyorder:
        raise ValueError(
            f"window_length ({window_length}) は polyorder ({polyorder}) より大きい必要があります"
        )
    
    if len(z) < window_length:
        # データが少ない場合はそのまま返す
        return z.copy()
    
    # 奇数に調整
    window_odd = window_length if window_length % 2 == 1 else window_length + 1
    
    if len(z) < window_odd:
        window_odd = len(z) if len(z) % 2 == 1 else len(z) - 1
    
    if window_odd <= polyorder:
        return z.copy()
    
    z_smooth = signal.savgol_filter(z, window_odd, polyorder)
    return z_smooth


def preprocess_lidar_profile(
    x: np.ndarray,
    z: np.ndarray,
    dx: Optional[float] = None,
    uneven_tol: float = 0.05,
    median_k: int = 5,
    sg_window: int = 21,
    sg_poly: int = 3,
    interp_method: str = "linear"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    断面プロファイルの前処理をまとめて実行する。
    
    処理順序：
    1. NaN/無限大除去、昇順ソート、重複xの平均化
    2. 等間隔化（必要に応じて）
    3. メディアンフィルタ（スパイク除去）
    4. Savitzky-Golayフィルタ（平滑化）
    
    Args:
        x: 距離配列（mm）
        z: Z座標配列（mm）
        dx: 目標グリッド間隔（mm）。Noneの場合はmedian(diff(x))を使用
        uneven_tol: 不等間隔判定のしきい値
        median_k: メディアンフィルタの窓長
        sg_window: Savitzky-Golayフィルタの窓長
        sg_poly: Savitzky-Golayフィルタの多項式次数
        interp_method: 補間方法（"linear", "nearest", "quadratic", "cubic"）
        
    Returns:
        (x, z_raw, z_median, z_smooth, meta):
        - x: 等間隔化後の距離
        - z_raw: 等間隔化後（メディアン前）のz
        - z_median: メディアンフィルタ後
        - z_smooth: Savitzky-Golayフィルタ後
        - meta: 使用したパラメータや不等間隔判定の統計量
    """
    # 1. 前処理共通：NaN/無限大除去
    valid_mask = np.isfinite(x) & np.isfinite(z)
    x_clean = x[valid_mask].copy()
    z_clean = z[valid_mask].copy()
    
    if len(x_clean) == 0:
        raise ValueError("有効なデータ点がありません")
    
    # 昇順ソート
    sort_idx = np.argsort(x_clean)
    x_sorted = x_clean[sort_idx]
    z_sorted = z_clean[sort_idx]
    
    # 重複xの平均化
    unique_x, unique_indices = np.unique(x_sorted, return_inverse=True)
    z_unique = np.array([z_sorted[unique_indices == i].mean() 
                        for i in range(len(unique_x))])
    x_final = unique_x
    z_final = z_unique
    
    # 2. 等間隔化判定
    if len(x_final) < 2:
        # データが少ない場合はそのまま
        x_uniform = x_final
        z_uniform = z_final
        is_uniform = True
        cv_diff = 0.0
        range_ratio = 0.0
    else:
        diff_x = np.diff(x_final)
        median_diff = np.median(diff_x)
        
        if median_diff <= 0:
            # 重複や異常値がある場合
            x_uniform = x_final
            z_uniform = z_final
            is_uniform = True
            cv_diff = 0.0
            range_ratio = 0.0
        else:
            cv_diff = np.std(diff_x) / median_diff
            range_ratio = (np.max(diff_x) - np.min(diff_x)) / median_diff
            
            is_uniform = (cv_diff <= uneven_tol) and (range_ratio <= uneven_tol)
            
            if not is_uniform:
                # 等間隔化を実行
                x_uniform, z_uniform = resample_to_uniform_grid(
                    x_final, z_final, dx=dx, method=interp_method
                )
            else:
                x_uniform = x_final
                z_uniform = z_final
    
    # z_raw: 等間隔化後（メディアン前）
    z_raw = z_uniform.copy()
    
    # 3. メディアンフィルタ
    z_median = despike_median(z_raw, k=median_k)
    
    # 4. Savitzky-Golayフィルタ
    # dxを計算（等間隔化後のグリッド間隔）
    if len(x_uniform) >= 2:
        dx_actual = np.median(np.diff(x_uniform))
    else:
        dx_actual = 1.0
    
    z_smooth = smooth_savgol(z_median, sg_window, sg_poly, dx=dx_actual)
    
    # メタデータ
    meta = {
        "dx_used": dx if dx is not None else np.median(np.diff(x_final)) if len(x_final) >= 2 else None,
        "is_uniform": is_uniform,
        "cv_diff": cv_diff,
        "range_ratio": range_ratio,
        "median_k": median_k,
        "sg_window": sg_window,
        "sg_poly": sg_poly,
        "interp_method": interp_method,
        "n_points_original": len(x),
        "n_points_final": len(x_uniform)
    }
    
    return x_uniform, z_raw, z_median, z_smooth, meta
