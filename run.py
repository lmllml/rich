#!/usr/bin/env python3
"""
Rich 项目启动脚本
从项目根目录运行主程序
"""
import sys
import os

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入并运行主程序
from main import main

if __name__ == "__main__":
    main()
