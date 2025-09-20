"""
主程序 - 币安 ETHUSDT 因子分析示例
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import BinanceDataFetcher
from factor_strategy import FactorAnalyzer
from chart_utils import create_enhanced_factor_chart


def analyze_factor_performance(factor_data: pd.DataFrame):
    """分析因子表现"""
    print("\n=== 因子分析报告 ===")
    
    # 计算各因子的信号统计
    signal_cols = ['rsi_signal', 'macd_signal', 'bb_signal', 'momentum_signal', 'volatility_signal']
    
    print("\n各因子信号分布:")
    for col in signal_cols:
        signal_counts = factor_data[col].value_counts().sort_index()
        print(f"{col}: {dict(signal_counts)}")
    
    # 计算因子相关性
    factor_cols = ['rsi', 'macd', 'momentum', 'volatility']
    correlation_matrix = factor_data[factor_cols].corr()
    
    print("\n因子相关性矩阵:")
    print(correlation_matrix.round(3))
    
    # 分析综合信号的分布
    print(f"\n综合信号统计:")
    print(f"平均值: {factor_data['combined_signal'].mean():.3f}")
    print(f"标准差: {factor_data['combined_signal'].std():.3f}")
    print(f"最大值: {factor_data['combined_signal'].max():.3f}")
    print(f"最小值: {factor_data['combined_signal'].min():.3f}")
    
    # 分析交易信号
    buy_signals = factor_data[factor_data['combined_signal'] > 0.3]
    sell_signals = factor_data[factor_data['combined_signal'] < -0.3]
    
    print(f"\n交易信号统计:")
    print(f"买入信号次数: {len(buy_signals)}")
    print(f"卖出信号次数: {len(sell_signals)}")
    
    return correlation_matrix


def plot_factor_analysis(factor_data: pd.DataFrame):
    """绘制因子分析图表"""
    # 配置中文字体
    try:
        # 尝试设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except:
        # 如果字体设置失败，使用英文标题
        print("警告：中文字体设置失败，将使用英文标题")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('ETHUSDT Factor Analysis', fontsize=16)
    
    # 价格和综合信号
    ax1 = axes[0, 0]
    ax1.plot(factor_data.index, factor_data['price'], label='ETH Price', color='black')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(factor_data.index, factor_data['combined_signal'], 
                  label='Combined Signal', color='red', alpha=0.7)
    ax1.set_title('Price vs Combined Signal')
    ax1.set_ylabel('Price (USDT)')
    ax1_twin.set_ylabel('Combined Signal')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # RSI
    ax2 = axes[0, 1]
    ax2.plot(factor_data.index, factor_data['rsi'], label='RSI', color='blue')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    ax2.set_title('RSI Indicator')
    ax2.set_ylabel('RSI')
    ax2.legend()
    
    # MACD
    ax3 = axes[1, 0]
    ax3.plot(factor_data.index, factor_data['macd'], label='MACD', color='purple')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('MACD Indicator')
    ax3.set_ylabel('MACD')
    ax3.legend()
    
    # 布林带
    ax4 = axes[1, 1]
    ax4.plot(factor_data.index, factor_data['price'], label='Price', color='black')
    ax4.plot(factor_data.index, factor_data['bb_upper'], label='BB Upper', color='red', alpha=0.7)
    ax4.plot(factor_data.index, factor_data['bb_lower'], label='BB Lower', color='green', alpha=0.7)
    ax4.fill_between(factor_data.index, factor_data['bb_lower'], factor_data['bb_upper'], 
                     alpha=0.1, color='gray')
    ax4.set_title('Bollinger Bands')
    ax4.set_ylabel('Price (USDT)')
    ax4.legend()
    
    # 动量指标
    ax5 = axes[2, 0]
    ax5.plot(factor_data.index, factor_data['momentum'], label='Momentum', color='orange')
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax5.set_title('Momentum Indicator')
    ax5.set_ylabel('Momentum')
    ax5.legend()
    
    # 仓位变化
    ax6 = axes[2, 1]
    ax6.plot(factor_data.index, factor_data['position'], label='Position', color='brown')
    ax6.set_title('Position Changes')
    ax6.set_ylabel('Position Size')
    ax6.legend()
    
    plt.tight_layout()
    import os
    os.makedirs('../output', exist_ok=True)
    plt.savefig('../output/factor_analysis_charts.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("图表已保存为 output/factor_analysis_charts.png")


def main():
    """主函数"""
    print("=== 币安 ETHUSDT 因子分析系统 ===")
    
    # 1. 获取数据
    print("\n1. 正在获取 ETHUSDT 数据...")
    fetcher = BinanceDataFetcher()
    
    # 获取最近30天的小时数据
    data = fetcher.fetch_recent_data(symbol='ETH/USDT', timeframe='1h', days=30)
    
    if data.empty:
        print("❌ 数据获取失败，请检查网络连接或API配置")
        return
    
    print(f"✅ 成功获取 {len(data)} 条数据")
    
    # 2. 运行因子分析
    print("\n2. 正在运行因子分析...")
    analyzer = FactorAnalyzer(data)
    
    # 运行回测
    results = analyzer.run_backtest(initial_cash=10000.0)
    
    # 3. 输出回测结果
    print("\n=== 回测结果 ===")
    print(f"初始资金: ${results['initial_cash']:,.2f}")
    print(f"最终资金: ${results['final_value']:,.2f}")
    print(f"总收益: ${results['total_return']:,.2f}")
    print(f"收益率: {results['return_pct']:.2f}%")
    sharpe_ratio = results['sharpe_ratio'] if results['sharpe_ratio'] is not None else 0.0
    max_drawdown = results['max_drawdown'] if results['max_drawdown'] is not None else 0.0
    print(f"夏普比率: {sharpe_ratio:.3f}")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"总交易次数: {results['total_trades']}")
    
    # 4. 获取因子数据并分析
    factor_data = analyzer.get_factor_data()
    
    if not factor_data.empty:
        # 设置日期索引
        factor_data['date'] = pd.to_datetime(factor_data['date'])
        factor_data.set_index('date', inplace=True)
        
        # 分析因子表现
        correlation_matrix = analyze_factor_performance(factor_data)
        
        # 5. 绘制分析图表
        print("\n5. 正在生成增强版分析图表...")
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            create_enhanced_factor_chart(factor_data, '../output/factor_analysis_charts.png')
            print("✅ 增强版图表已保存")
        except Exception as e:
            print(f"绘图时发生错误: {e}")
            print("尝试使用基础版本图表...")
            try:
                plot_factor_analysis(factor_data)
                print("✅ 基础版图表已保存")
            except Exception as e2:
                print(f"基础版图表也失败: {e2}")
                print("跳过绘图步骤，继续保存数据")
        
        # 6. 保存结果
        print("\n6. 正在保存分析结果...")
        
        # 确保输出目录存在
        import os
        os.makedirs('../output', exist_ok=True)
        
        # 保存因子数据
        factor_data.to_csv('../output/factor_analysis_results.csv')
        print("✅ 因子分析结果已保存到 output/factor_analysis_results.csv")
        
        # 保存相关性矩阵
        correlation_matrix.to_csv('../output/factor_correlation_matrix.csv')
        print("✅ 因子相关性矩阵已保存到 output/factor_correlation_matrix.csv")
        
        # 保存回测结果
        results_df = pd.DataFrame([results])
        results_df.to_csv('../output/backtest_results.csv', index=False)
        print("✅ 回测结果已保存到 output/backtest_results.csv")
    
    print("\n=== 分析完成 ===")
    print("📊 查看 output/ 目录中的CSV文件获取详细数据")
    print("📈 如果支持图形界面，应该已显示分析图表")
    print("\n输出文件位置:")
    print("  - output/factor_analysis_results.csv")
    print("  - output/factor_correlation_matrix.csv")
    print("  - output/backtest_results.csv")
    print("  - output/factor_analysis_charts.png")


if __name__ == "__main__":
    main()
