"""PLYファイルの読み込みと自然ソート"""

import re
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np

# バイナリ PLY の型サイズ（バイト）と struct 書式文字
_PLY_TYPE_SIZE = {
    "char": 1, "uchar": 1, "short": 2, "ushort": 2,
    "int": 4, "uint": 4, "float": 4, "float32": 4, "double": 8,
}
_PLY_STRUCT_CHAR = {
    "float": "f", "float32": "f", "double": "d",
    "int": "i", "uint": "I", "short": "h", "ushort": "H",
    "char": "b", "uchar": "B",
}


def _parse_ply_header(lines: List[str]) -> Tuple[int, List[Tuple[str, str]], str]:
    """
    PLYヘッダーをパースする。
    Returns: (vertex_count, list of (dtype, name), format_str)
    format_str is 'ascii' or 'binary_little_endian' or 'binary_big_endian'
    """
    vertex_count = 0
    properties: List[Tuple[str, str]] = []  # (dtype, name)
    format_str = "ascii"
    in_vertex_element = False

    for line in lines:
        line = line.strip()
        if not line or line.startswith("comment"):
            continue
        parts = line.split()
        if parts[0] == "format":
            format_str = parts[1]  # ascii 1.0 or binary_* 1.0
        elif parts[0] == "element":
            if parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex_element = True
                properties = []
            else:
                in_vertex_element = False
        elif parts[0] == "property" and in_vertex_element:
            # property float x / property list uchar int vertex_index など
            if len(parts) >= 3 and parts[1] != "list":
                properties.append((parts[1], parts[2]))
        elif parts[0] == "end_header":
            break

    return vertex_count, properties, format_str


def _read_ply_ascii(
    ply_path: Path, vertex_count: int, properties: List[Tuple[str, str]]
) -> np.ndarray:
    """
    ASCII形式のPLYから頂点のx,y,zを読み込む。
    - 1行に全プロパティが並ぶ形式（1行＝1頂点）と、
    - 1頂点が複数行の形式（Realsense形式: 1行目 x y z, 2行目 nx ny nz, 3行目 rgb）の両方に対応。
    """
    names = [p[1] for p in properties]
    ix = names.index("x")
    iy = names.index("y")
    iz = names.index("z")
    n_props = len(properties)

    with open(ply_path, "r", encoding="utf-8", errors="replace") as f:
        while f.readline().strip() != "end_header":
            pass

        def data_lines():
            for line in f:
                line = line.strip()
                if not line or line.startswith("comment"):
                    continue
                yield line

        it = iter(data_lines())
        points = []
        try:
            first_line = next(it)
        except StopIteration:
            return np.array([], dtype=np.float64)
        first_parts = first_line.split()

        if len(first_parts) >= n_props:
            # 1行＝1頂点（全プロパティが1行に並ぶ）
            if len(first_parts) >= max(ix, iy, iz) + 1:
                points.append([
                    float(first_parts[ix]), float(first_parts[iy]), float(first_parts[iz])
                ])
            for _ in range(vertex_count - 1):
                line = next(it)
                parts = line.split()
                if len(parts) >= max(ix, iy, iz) + 1:
                    points.append([float(parts[ix]), float(parts[iy]), float(parts[iz])])
        else:
            # 1頂点が複数行（Realsense形式: 1行目 x y z, 2行目 nx ny nz, 3行目 rgb）
            lines_per_vertex = max(1, (n_props + 2) // 3)
            if len(first_parts) >= 3:
                points.append([
                    float(first_parts[0]), float(first_parts[1]), float(first_parts[2])
                ])
            for _ in range(vertex_count - 1):
                for _ in range(lines_per_vertex - 1):
                    next(it)  # この頂点の残り行を読み飛ばす
                line = next(it)  # 次の頂点の1行目（x y z）
                parts = line.split()
                if len(parts) >= 3:
                    points.append([
                        float(parts[0]), float(parts[1]), float(parts[2])
                    ])
        return np.array(points, dtype=np.float64)


def _read_ply_binary_from_handle(
    f, vertex_count: int, properties: List[Tuple[str, str]], little_endian: bool
) -> np.ndarray:
    """バイナリ形式のPLYを、現在位置から頂点のx,y,zを読み込む。"""
    names = [p[1] for p in properties]
    dtypes = [p[0] for p in properties]
    try:
        ix = names.index("x")
        iy = names.index("y")
        iz = names.index("z")
    except ValueError:
        raise ValueError("PLYに property x, y, z が含まれていません")

    vertex_size = sum(_PLY_TYPE_SIZE.get(d, 4) for d in dtypes)
    order = "<" if little_endian else ">"
    fmt = order + "".join(_PLY_STRUCT_CHAR.get(d, "f") for d in dtypes)

    points = np.empty((vertex_count, 3), dtype=np.float64)
    for i in range(vertex_count):
        raw = f.read(vertex_size)
        if len(raw) < vertex_size:
            points = points[:i]
            break
        values = struct.unpack(fmt, raw)
        points[i, 0] = float(values[ix])
        points[i, 1] = float(values[iy])
        points[i, 2] = float(values[iz])
    return points


def natural_sort_paths(paths: List[Path]) -> List[Path]:
    """
    パスリストを自然順（natural sort）でソートする。
    
    文字列ソートでは "9ys810.ply" が "9ys82.ply" より前に来るが、
    自然順では "9ys82.ply" が先に来る。
    
    Args:
        paths: ソートするパスのリスト
        
    Returns:
        自然順でソートされたパスのリスト
    """
    def natural_key(path: Path) -> tuple:
        """自然順ソート用のキーを生成"""
        name = path.stem
        # 数字と非数字を分離
        parts = re.split(r'(\d+)', name)
        key = []
        for part in parts:
            if part.isdigit():
                key.append(int(part))
            else:
                key.append(part.lower())
        return tuple(key)
    
    return sorted(paths, key=natural_key)


def load_point_cloud(ply_path: str | Path) -> np.ndarray:
    """
    PLYファイルから点群を読み込み、N×3のnumpy配列として返す。
    ASCII形式および binary_little_endian / binary_big_endian に対応。

    Args:
        ply_path: PLYファイルのパス

    Returns:
        N×3のnumpy配列（各点のx, y, z座標）

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: 点群が空の場合、または x,y,z プロパティがない場合
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLYファイルが見つかりません: {ply_path}")

    with open(ply_path, "rb") as f:
        header_lines: List[str] = []
        while True:
            line = f.readline().decode("utf-8", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        data_start_offset = f.tell()

    vertex_count, properties, format_str = _parse_ply_header(header_lines)
    if vertex_count == 0:
        raise ValueError(f"点群が空です: {ply_path}")

    if format_str == "ascii":
        points = _read_ply_ascii(ply_path, vertex_count, properties)
    else:
        little = "little" in format_str
        with open(ply_path, "rb") as f:
            f.seek(data_start_offset)
            points = _read_ply_binary_from_handle(f, vertex_count, properties, little)
    if len(points) == 0:
        raise ValueError(f"点群が空です: {ply_path}")
    return points
