# Rich - 币安 ETHUSDT 因子分析系统

这是一个使用 Python、Backtrader 和 CCXT 实现的加密货币因子分析系统，专门用于分析币安交易所的 ETHUSDT 交易对。

## 功能特性

- 🔄 **实时数据获取**: 使用 CCXT 从币安获取 ETHUSDT 历史数据
- 📊 **多因子分析**: 实现 RSI、MACD、布林带、动量、波动率等技术指标
- 🤖 **自动化回测**: 基于 Backtrader 的策略回测框架
- 📈 **可视化分析**: 生成详细的因子分析图表
- 💾 **结果导出**: 自动保存分析结果为 CSV 文件

## 项目结构

```
rich/
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── main.py            # 主程序入口
│   ├── data_fetcher.py    # 数据获取模块
│   └── factor_strategy.py # 因子分析和策略实现
├── output/                # 程序运行结果（不提交到 git）
│   ├── factor_analysis_results.csv
│   ├── factor_correlation_matrix.csv
│   ├── backtest_results.csv
│   └── factor_analysis_charts.png
├── run.py                 # 启动脚本
├── requirements.txt       # 项目依赖
├── .gitignore            # Git 忽略文件
└── README.md             # 项目说明
```

## 技术指标

### 实现的因子

1. **RSI (相对强弱指标)**: 识别超买超卖信号
2. **MACD**: 趋势跟踪和动量分析
3. **布林带**: 价格波动区间分析
4. **动量因子**: 价格变化速度分析
5. **波动率因子**: 市场波动性分析

### 综合信号

- 多因子加权平均信号
- 动态仓位管理
- 风险控制机制

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

从项目根目录运行：

```bash
python run.py
```

或者进入 src 目录运行：

```bash
cd src
python -m main
```

## 输出文件

运行后会在 `output/` 目录中生成以下文件：

- `factor_analysis_results.csv`: 详细的因子数据和信号
- `factor_correlation_matrix.csv`: 因子相关性矩阵
- `backtest_results.csv`: 回测结果摘要
- `factor_analysis_charts.png`: 因子分析可视化图表

## 开发说明

- 所有源代码位于 `src/` 目录
- 程序运行结果自动保存到 `output/` 目录
- `output/` 目录已在 `.gitignore` 中忽略，不会提交到版本控制
- 使用相对导入来组织模块间的依赖关系

## 注意事项

- 本项目仅用于学习和研究目的
- 实际交易请谨慎评估风险
- 历史表现不代表未来收益
- 建议在实盘前进行充分的参数优化和风险评估