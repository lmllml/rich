"""
路径常量 - 统一管理项目根目录与输出目录
"""
from pathlib import Path

# 本文件位于 rich/src/paths.py
SRC_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SRC_DIR.parent
OUTPUT_DIR: Path = PROJECT_ROOT / 'output'
DB_DIR: Path = PROJECT_ROOT / 'cache'
DB_PATH: Path = DB_DIR / 'market_cache.sqlite'

__all__ = ['SRC_DIR', 'PROJECT_ROOT', 'OUTPUT_DIR', 'DB_DIR', 'DB_PATH']


