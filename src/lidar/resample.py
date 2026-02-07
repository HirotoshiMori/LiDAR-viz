"""共通グリッドへの再サンプリング"""

import numpy as np
from scipy import interpolate
from typing import Union


def resample_to_common_grid(
    x: np.ndarray,
    z: np.ndarray,
    common_x: np.ndarray,
    method: str = "linear"
) -> np.ndarray:
    """
    プロファイルを共通グリッド上に再サンプリングする。
    
    Args:
        x: 元の距離配列（mm）
        z: 元の値配列（mm）
        common_x: 共通グリッドの距離配列（mm）
        method: 補間方法（"linear", "nearest", "quadratic", "cubic"）
        
    Returns:
        共通グリッド上のz値（長さlen(common_x)）
    """
    if len(x) == 0 or len(z) == 0:
        return np.full(len(common_x), np.nan)
    
    if len(x) != len(z):
        raise ValueError(f"xとzの長さが一致しません: len(x)={len(x)}, len(z)={len(z)}")
    
    # 有効なデータのみを使用
    valid_mask = np.isfinite(x) & np.isfinite(z)
    if np.sum(valid_mask) == 0:
        return np.full(len(common_x), np.nan)
    
    x_valid = x[valid_mask]
    z_valid = z[valid_mask]
    
    # 重複xを処理（平均化）
    unique_x, unique_indices = np.unique(x_valid, return_inverse=True)
    z_unique = np.array([z_valid[unique_indices == i].mean() 
                         for i in range(len(unique_x))])
    
    if len(unique_x) < 2:
        # データが少ない場合はNaNを返す
        return np.full(len(common_x), np.nan)
    
    # 補間関数を作成
    f = interpolate.interp1d(
        unique_x, z_unique,
        kind=method,
        bounds_error=False,
        fill_value=np.nan
    )
    
    # 共通グリッド上で補間
    z_on_common = f(common_x)
    
    return z_on_common
