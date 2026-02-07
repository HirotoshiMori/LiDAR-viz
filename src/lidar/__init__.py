"""LiDAR点群処理モジュール"""

from .config import load_config, load_config_from_params_path
from .io_ply import load_point_cloud, natural_sort_paths
from .geometry import (
    distance_to_line,
    project_to_line,
    revetment_intersection_distance_mm,
    extract_cross_section,
    extract_cross_section_by_plane,
)
from .leveling import estimate_ground_plane, create_rotation_matrix, apply_rotation, distance_to_plane
from .profile import create_profile, clip_profile_indices
from .preprocess_lidar import preprocess_lidar_profile
from .resample import resample_to_common_grid
from .plotting import plot_section_differences, plot_initial_section_with_lines
from .pipeline import run_pipeline, PipelineResult
from .output_manager import OutputManager
from .run import run_case

__all__ = [
    "run_case",
    "load_config",
    "load_config_from_params_path",
    "OutputManager",
    "load_point_cloud",
    "natural_sort_paths",
    "distance_to_line",
    "project_to_line",
    "revetment_intersection_distance_mm",
    "extract_cross_section",
    "extract_cross_section_by_plane",
    "estimate_ground_plane",
    "create_rotation_matrix",
    "apply_rotation",
    "distance_to_plane",
    "create_profile",
    "clip_profile_indices",
    "preprocess_lidar_profile",
    "resample_to_common_grid",
    "plot_section_differences",
    "plot_initial_section_with_lines",
    "run_pipeline",
    "PipelineResult",
]
