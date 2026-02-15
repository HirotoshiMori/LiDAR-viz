"""ケース実行の一括エントリポイント"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .config import load_config_from_params_path
from .output_manager import OutputManager
from .pipeline import PipelineResult, run_pipeline
from .plotting import plot_initial_section_with_lines, plot_section_differences

# 出力する4図: initial_section, displacement_full, displacement_zoom, difference_zoom
# plot_section_differences の返り値順: [displacement_full, difference_full, displacement_zoom, difference_zoom]
_OUTPUT_FIGS = [
    ("displacement_full", "jpg"),
    None,  # difference_full はスキップ
    ("displacement_zoom", "png"),
    ("difference_zoom", "png"),
]


def _save_figures(out: OutputManager, result: PipelineResult) -> None:
    """図を4つに絞って保存する。"""
    plot_cfg = result.config.get("plot", {})
    figure_cfg = plot_cfg.get("figure") or {}
    font_cfg = plot_cfg.get("font") or {}
    line_cfg = plot_cfg.get("line") or {}
    dpi = figure_cfg.get("dpi", 150)

    # 1. initial_section.png
    if (
        result.initial_points_original is not None
        and result.initial_plane_normal is not None
        and result.initial_ground_mask is not None
    ):
        cs = result.config.get("cross_section", {})
        plot_3d = result.config.get("plot_3d", {})
        fig = plot_initial_section_with_lines(
            result.initial_points_original,
            result.initial_points_rotated,
            result.point1,
            result.point2,
            result.initial_cross_section_original,
            result.initial_cross_section_rotated,
            result.initial_plane_normal,
            result.initial_ground_mask,
            cross_section_threshold=cs.get("threshold", 0.005),
            cross_section_z_range=cs.get("z_range"),
            title="Initial Section with Cross-Section Line",
            plot_range=plot_3d.get("range"),
            font_title=font_cfg.get("title", 14),
            font_legend=font_cfg.get("legend", 8),
            line_width_main=line_cfg.get("main", 2),
            line_width_aux=line_cfg.get("aux", 1),
        )
        fig.savefig(out.fig_path("initial_section.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # 2-4. displacement_full.jpg, displacement_zoom.png, difference_zoom.png
    figures = plot_section_differences(
        result.common_x,
        result.all_differences,
        all_displacements=result.all_displacements,
        all_before_resample=None,
        all_before_filter=None,
        ply_names=result.legend_names,
        xlabel="Distance (mm)",
        ylabel_displacement="Displacement (mm)",
        ylabel_difference="Difference (mm)",
        xlim=plot_cfg.get("xlim"),
        ylim_displacement=plot_cfg.get("ylim_displacement"),
        ylim_difference=plot_cfg.get("ylim_difference"),
        x_start_from_zero=plot_cfg.get("x_start_from_zero", True),
        show_zoom=plot_cfg.get("show_zoom", True),
        xlim_zoom=result.plot_xlim_zoom,
        ylim_difference_zoom=plot_cfg.get("ylim_difference_zoom"),
        ylim_displacement_zoom=plot_cfg.get("ylim_displacement_zoom"),
        graph_size_ratio=plot_cfg.get("graph_size_ratio"),
        title_prefix=out.case_id,
        x_zero_zoom=result.x_zero_zoom,
        figure_base_width=figure_cfg.get("base_width", 9.0),
        font_legend=font_cfg.get("legend", 8),
        font_label=font_cfg.get("label", 10),
        line_width_data=line_cfg.get("data", 1.5),
    )
    for i, fig in enumerate(figures):
        spec = _OUTPUT_FIGS[i] if i < len(_OUTPUT_FIGS) else None
        if spec is not None:
            name, ext = spec
            fig.savefig(out.fig_path(f"{name}.{ext}"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _save_csv(out: OutputManager, result: PipelineResult) -> None:
    """CSVを保存する。"""
    with open(out.table_path(f"{out.case_id}_displacements.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["distance_mm"] + result.legend_names)
        for i in range(len(result.common_x)):
            w.writerow([result.common_x[i]] + [d[i] for d in result.all_displacements])

    with open(out.table_path(f"{out.case_id}_differences.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["distance_mm"] + result.legend_names)
        for i in range(len(result.common_x)):
            w.writerow([result.common_x[i]] + [d[i] for d in result.all_differences])


def run_case(
    params_path: Path | str,
    project_root: Path | None = None,
) -> PipelineResult:
    """
    ケースを一括実行する。config, pipeline, 図保存, CSV保存, metadata をすべて実行。

    Returns:
        PipelineResult
    """
    params_path = Path(params_path)
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

    config = load_config_from_params_path(params_path, project_root)
    out = OutputManager.from_params(params_path, project_root)

    run_started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    out.save_merged_params(config["_raw_params"])

    _run_start = time.time()
    with out.capture_stdout_stderr(tee_to_console=False):
        result = run_pipeline(config, verbose=True)

    run_finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    elapsed_sec = round(time.time() - _run_start, 2)

    out.save_metadata(
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
        elapsed_sec=elapsed_sec,
        params=config,
    )

    _save_figures(out, result)
    _save_csv(out, result)

    return result
