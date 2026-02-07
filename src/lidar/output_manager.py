"""出力ディレクトリの集中管理（OutputManager）"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import yaml


class _Tee:
    """標準出力／標準エラーをファイルと元のストリームの両方に書き込む。"""

    def __init__(self, stream: Any, file_handle: Any):
        self.stream = stream
        self.file = file_handle

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
        return len(data)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()


class OutputManager:
    """
    ケース別の出力を output/{case_id}/ に整理する。
    figures/, tables/, logs/, artifacts/ を自動作成し、
    パス生成・メタデータ保存・ログ出力を一元管理する。
    """

    SUBDIRS = ("figures", "tables", "logs", "artifacts")

    def __init__(
        self,
        case_id: str,
        output_root: Path,
        params_path: Path,
    ):
        self.case_id = case_id
        self.output_root = Path(output_root)
        self.params_path = Path(params_path)
        self.case_dir = self.output_root / case_id

        for sub in self.SUBDIRS:
            (self.case_dir / sub).mkdir(parents=True, exist_ok=True)

        self._log_handler: logging.FileHandler | None = None
        self._log_initialized = False

    def fig_path(self, name: str) -> Path:
        """figures/ 配下のパスを返す。拡張子はそのまま。"""
        return self.case_dir / "figures" / name

    def table_path(self, name: str) -> Path:
        """tables/ 配下のパスを返す。"""
        return self.case_dir / "tables" / name

    def log_path(self, name: str = "run.log") -> Path:
        """logs/ 配下のパスを返す。"""
        return self.case_dir / "logs" / name

    def artifact_path(self, name: str) -> Path:
        """artifacts/ 配下のパスを返す。"""
        return self.case_dir / "artifacts" / name

    def setup_logging(self) -> None:
        """
        標準出力・標準エラーを run.log に記録するよう設定する。
        Notebook 内でもログファイルが残る。
        """
        if self._log_initialized:
            return

        log_path = self.log_path("run.log")
        self._log_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        self._log_handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        self._log_handler.setFormatter(fmt)

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(self._log_handler)

        self._log_initialized = True

    def teardown_logging(self) -> None:
        """ログハンドラを解除する。"""
        if self._log_handler is not None:
            root = logging.getLogger()
            root.removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None
        self._log_initialized = False

    @contextmanager
    def capture_stdout_stderr(self, tee_to_console: bool = False) -> Generator[None, None, None]:
        """
        標準出力・標準エラーを run.log に追記するコンテキストマネージャ。
        tee_to_console=False のときはファイルのみに記録（Notebook 出力を抑制）。
        """
        log_path = self.log_path("run.log")
        with open(log_path, "a", encoding="utf-8") as f:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            if tee_to_console:
                sys.stdout = _Tee(old_stdout, f)
                sys.stderr = _Tee(old_stderr, f)
            else:
                sys.stdout = f
                sys.stderr = f
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    def copy_params(self, params_path: Path | None = None) -> Path:
        """
        実行時点のパラメータを output/{case_id}/params.yml にコピーする。
        params_path のみ指定時はそのファイルをコピー。
        merged_params は save_merged_params() で別途保存する。
        """
        src = Path(params_path) if params_path is not None else self.params_path
        dst = self.case_dir / "params.yml"
        shutil.copy2(src, dst)
        return dst

    def save_merged_params(self, merged_params: dict) -> Path:
        """
        base + case マージ済みパラメータを output/{case_id}/params.yml に保存する。
        """
        dst = self.case_dir / "params.yml"

        def _to_serializable(obj: Any) -> Any:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(x) for x in obj]
            return obj

        data = _to_serializable(merged_params)
        with open(dst, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        return dst

    def save_metadata(
        self,
        *,
        run_started_at: str | None = None,
        run_finished_at: str | None = None,
        elapsed_sec: float | None = None,
        params: dict | None = None,
        **extra: Any,
    ) -> Path:
        """
        metadata.json を生成する。
        最低限: case_id, params_path, run_started_at, run_finished_at,
        elapsed_sec, python_version, platform, git_commit, seed
        """
        git_commit: str | None = None
        try:
            r = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.output_root.parent,
            )
            if r.returncode == 0 and r.stdout.strip():
                git_commit = r.stdout.strip()
        except Exception:
            pass

        seed = None
        if params:
            seed = params.get("seed")
            # nested
            if seed is None and "filter" in params:
                seed = params.get("filter", {}).get("seed")

        meta = {
            "case_id": self.case_id,
            "params_path": str(self.params_path),
            "run_started_at": run_started_at,
            "run_finished_at": run_finished_at,
            "elapsed_sec": elapsed_sec,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "git_commit": git_commit,
            "seed": seed,
            **extra,
        }

        dst = self.case_dir / "metadata.json"
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return dst

    @classmethod
    def from_params(
        cls,
        params_path: Path | str,
        project_root: Path | None = None,
        output_root: Path | None = None,
    ) -> "OutputManager":
        """
        params_path から OutputManager を初期化する。
        case_id は YAML の case_id があれば優先、なければファイル名の stem。
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
        output_root = output_root or (project_root / "output")

        case_id = params_path.stem
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
        if "case_id" in params and params["case_id"]:
            case_id = str(params["case_id"])

        return cls(
            case_id=case_id,
            output_root=output_root,
            params_path=params_path,
        )
