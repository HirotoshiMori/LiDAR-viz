"""YAML設定の読み込み・マージ・検証"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """ネストされたdictを再帰的にマージ。override の値が優先。"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _to_tuple(value: Any) -> Any:
    """YAMLのリストをtupleに変換（範囲指定用）。"""
    if isinstance(value, list):
        return tuple(value)
    return value


def _resolve_config(raw: dict) -> dict:
    """生のYAMLを実行用の config に変換。"""
    cfg = copy.deepcopy(raw)

    # points -> numpy array
    if "points" in cfg:
        pts = cfg["points"]
        for k in ("point1", "point2", "revetment"):
            if k in pts:
                pts[k] = np.array(pts[k], dtype=float)

    # cross_section.z_range
    if "cross_section" in cfg and "z_range" in cfg["cross_section"]:
        zr = cfg["cross_section"]["z_range"]
        if zr is not None:
            cfg["cross_section"]["z_range"] = _to_tuple(zr)

    # ground.z_range, xy_range
    if "ground" in cfg:
        if cfg["ground"].get("z_range") is not None:
            cfg["ground"]["z_range"] = _to_tuple(cfg["ground"]["z_range"])
        if cfg["ground"].get("xy_range") is not None:
            cfg["ground"]["xy_range"] = tuple(
                _to_tuple(r) for r in cfg["ground"]["xy_range"]
            )

    # plot の範囲類
    if "plot" in cfg:
        for k in ("xlim", "ylim_displacement", "ylim_difference",
                  "ylim_difference_zoom", "ylim_displacement_zoom", "graph_size_ratio"):
            if cfg["plot"].get(k) is not None:
                cfg["plot"][k] = _to_tuple(cfg["plot"][k])

    # plot_3d.range
    if "plot_3d" in cfg and cfg["plot_3d"].get("range") is not None:
        cfg["plot_3d"]["range"] = tuple(
            _to_tuple(r) for r in cfg["plot_3d"]["range"]
        )

    return cfg


def validate_config(cfg: dict) -> None:
    """パラメータの正当性を検証。不正な場合は ValueError を送出。"""
    # points 必須
    pts = cfg.get("points", {})
    if not all(k in pts for k in ("point1", "point2", "revetment")):
        raise ValueError("points に point1, point2, revetment が必要です")

    # common_grid
    cg = cfg.get("common_grid", {})
    x_min = cg.get("x_min", 0)
    x_max = cg.get("x_max")
    x_points = cg.get("x_points", 500)
    if x_max is not None and x_max <= x_min:
        raise ValueError(
            f"common_grid.x_max ({x_max}) は x_min ({x_min}) より大きい必要があります"
        )
    if x_points <= 0:
        raise ValueError(f"common_grid.x_points ({x_points}) は正の値である必要があります")

    # filter
    flt = cfg.get("filter", {})
    sg_w = flt.get("sg_window", 21)
    sg_p = flt.get("sg_poly", 3)
    if sg_w <= sg_p:
        raise ValueError(
            f"filter.sg_window ({sg_w}) は filter.sg_poly ({sg_p}) より大きい必要があります"
        )
    if flt.get("median_k", 5) < 1:
        raise ValueError("filter.median_k は 1 以上である必要があります")
    valid_interp = {"linear", "nearest", "quadratic", "cubic"}
    if flt.get("interp_method", "linear") not in valid_interp:
        raise ValueError(f"filter.interp_method は {valid_interp} のいずれかである必要があります")

    # cross_section
    cs = cfg.get("cross_section", {})
    start_idx = cs.get("start_index", 0)
    end_idx = cs.get("end_index_max", 5000)
    if start_idx < 0:
        raise ValueError("cross_section.start_index は 0 以上である必要があります")
    if end_idx <= start_idx:
        raise ValueError(
            f"cross_section.end_index_max ({end_idx}) は start_index ({start_idx}) より大きい必要があります"
        )


def load_config(
    ply_directory: Path,
    project_root: Path | None = None,
) -> dict:
    """
    params/base.yaml と params/{folder_name}.yaml をマージして設定を読み込む。

    Args:
        ply_directory: データフォルダ（例: data/No7-friction-8mm）
        project_root: プロジェクトルート。None の場合は ply_directory の親を遡って判定

    Returns:
        マージ済み・解决済みの設定 dict
    """
    if project_root is None:
        project_root = ply_directory
        for _ in range(5):
            if (project_root / "src").is_dir():
                break
            parent = project_root.parent
            if parent == project_root:
                break
            project_root = parent

    ply_dir = Path(ply_directory)
    folder_name = ply_dir.name

    params_dir = project_root / "params"
    default_path = params_dir / "base.yaml"
    dataset_path = params_dir / f"{folder_name}.yaml"

    if not default_path.is_file():
        raise FileNotFoundError(f"ベース設定が見つかりません: {default_path}")

    with open(default_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if dataset_path.is_file():
        with open(dataset_path, "r", encoding="utf-8") as f:
            overlay = yaml.safe_load(f) or {}
        config = _deep_merge(config, overlay)

    # points が無い場合は データセット yaml 必須
    if "points" not in config or not config["points"]:
        raise FileNotFoundError(
            f"points が定義されていません。{dataset_path} に point1, point2, revetment を定義してください"
        )

    config = _resolve_config(config)
    validate_config(config)
    config["_ply_directory"] = str(ply_dir)
    config["_params_path"] = str(dataset_path.resolve())
    return config


def load_config_from_params_path(
    params_path: Path | str,
    project_root: Path | None = None,
) -> dict:
    """
    params_path を指定して設定を読み込む。
    params/base.yaml と 指定YAML をマージする。
    ply_directory は params の ply_directory 指定、または data/{case_id} を参照。
    """
    params_path = Path(params_path).resolve()
    if not params_path.is_file():
        raise FileNotFoundError(f"パラメータファイルが見つかりません: {params_path}")

    if project_root is None:
        project_root = params_path.parent.parent
        for _ in range(5):
            if (project_root / "src").is_dir():
                break
            parent = project_root.parent
            if parent == project_root:
                break
            project_root = parent

    project_root = Path(project_root)
    params_dir = project_root / "params"
    default_path = params_dir / "base.yaml"

    with open(default_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    with open(params_path, "r", encoding="utf-8") as f:
        overlay = yaml.safe_load(f) or {}
    raw_merged = _deep_merge(config, overlay)
    config = raw_merged

    if "points" not in config or not config["points"]:
        raise FileNotFoundError(
            f"points が定義されていません。{params_path} に point1, point2, revetment を定義してください"
        )

    case_id = config.get("case_id") or params_path.stem
    ply_dir = config.get("ply_directory")
    if ply_dir is None:
        ply_dir = project_root / "data" / case_id
    else:
        ply_dir = Path(ply_dir)
        if not ply_dir.is_absolute():
            ply_dir = project_root / ply_dir

    config = _resolve_config(config)
    validate_config(config)
    config["_ply_directory"] = str(ply_dir)
    config["_params_path"] = str(params_path)
    config["_raw_params"] = raw_merged  # base+case マージ済み（params.yml 出力用）
    return config
