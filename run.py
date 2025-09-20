#!/usr/bin/env python3
"""
Rich 项目启动脚本
从项目根目录运行主程序
"""
import sys
import os

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def usage():
    print("用法: python run.py [backtest|factor]")
    print(" - backtest: 仅运行策略回测与绩效对比")
    print(" - factor  : 仅运行因子分析（含相关性/信号有效性/图表）")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else 'backtest'
    if mode == 'backtest':
        from backtest_main import main as backtest_main
        backtest_main()
    elif mode == 'factor':
        from factor_analysis_main import main as factor_main
        factor_main()
    else:
        usage()
