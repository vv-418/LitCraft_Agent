"""benchmark 结果统一输出目录。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = Path(__file__).resolve().parent / "benchmark_result"


def result_path(filename: str) -> Path:
    """返回 benchmark/benchmark_result 下的结果文件路径，并确保目录存在。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR / filename
