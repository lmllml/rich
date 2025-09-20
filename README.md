# Rich - 币安 ETHUSDT 因子分析系统

这是一个使用 Python、Backtrader 和 CCXT 实现的加密货币因子分析系统，专门用于分析币安交易所的 ETHUSDT 交易对。

## 功能特性

- 🔄 **实时数据获取**: 使用 CCXT 从币安获取 ETHUSDT 历史数据
- 📊 **多因子分析**: 实现 RSI、MACD、布林带、动量、波动率等技术指标
- 🤖 **自动化回测**: 基于 Backtrader 的策略回测框架
- 📈 **可视化分析**: 生成详细的因子分析图表
- 💾 **结果导出**: 自动保存分析结果为 CSV 文件

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

```bash
python main.py
```

## 输出文件

运行后会生成以下文件：
- `factor_analysis_results.csv`: 详细的因子数据和信号
- `factor_correlation_matrix.csv`: 因子相关性矩阵
- `backtest_results.csv`: 回测结果摘要
- `factor_analysis_charts.png`: 因子分析可视化图表

## 最近回测结果

基于最近30天的小时数据：
- 初始资金: $10,000
- 最终资金: $9,395
- 总收益: -$605 (-6.05%)
- 最大回撤: 9.20%
- 总交易次数: 23

## 因子相关性分析

```
              rsi   macd  momentum  volatility
rsi         1.000  0.488     0.755       0.059
macd        0.488  1.000     0.805      -0.095
momentum    0.755  0.805     1.000      -0.046
volatility  0.059 -0.095    -0.046       1.000
```

## 项目结构

```
rich/
├── main.py                 # 主程序入口
├── data_fetcher.py         # 数据获取模块
├── factor_strategy.py      # 因子分析策略
├── requirements.txt        # 依赖包列表
└── README.md              # 项目说明
```

## 注意事项

- 本项目仅用于学习和研究目的
- 实际交易请谨慎评估风险
- 历史表现不代表未来收益
- 建议在实盘前进行充分的参数优化和风险评估
