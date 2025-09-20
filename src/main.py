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
from strategies.factor_strategy import FactorAnalyzer
from strategies.adaptive_factor_strategy import AdaptiveFactorAnalyzer
from strategies.enhanced_factor_strategy import EnhancedFactorAnalyzer
from paths import OUTPUT_DIR


def analyze_factor_performance(factor_data: pd.DataFrame):
    """分析因子表现"""
    print("\n=== 因子分析报告 ===")
    
    # 检查数据结构并适配不同的策略输出
    if 'rsi_signal' in factor_data.columns:
        # 传统策略的数据结构
        signal_cols = ['rsi_signal', 'macd_signal', 'bb_signal', 'momentum_signal', 'volatility_signal']
        
        print("\n各因子信号分布:")
        for col in signal_cols:
            if col in factor_data.columns:
                signal_counts = factor_data[col].value_counts().sort_index()
                print(f"{col}: {dict(signal_counts)}")
    else:
        # 自适应策略的数据结构
        print("\n自适应策略因子数据:")
        if 'factors' in factor_data.columns and not factor_data.empty:
            # 如果factors是嵌套结构，需要展开
            first_factors = factor_data['factors'].iloc[0] if len(factor_data) > 0 else {}
            if isinstance(first_factors, dict):
                print(f"因子类型: {list(first_factors.keys())}")
    
    # 计算因子相关性
    factor_cols = ['rsi', 'macd', 'momentum', 'volatility']
    
    # 检查是否有嵌套的factors列
    if 'factors' in factor_data.columns and not factor_data.empty:
        # 展开factors字典为独立列
        factors_expanded = pd.json_normalize(factor_data['factors'])
        factor_data = pd.concat([factor_data.drop('factors', axis=1), factors_expanded], axis=1)
    
    # 检查哪些因子列存在
    available_factors = [col for col in factor_cols if col in factor_data.columns]
    
    if available_factors:
        correlation_matrix = factor_data[available_factors].corr()
        
        print(f"\n因子相关性矩阵 (基于 {len(available_factors)} 个因子):")
        print(correlation_matrix.round(3))
    else:
        print("\n未找到标准因子数据，跳过相关性分析")
        correlation_matrix = pd.DataFrame()
    
    # 分析综合信号的分布
    if 'combined_signal' in factor_data.columns:
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
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('ETHUSDT 因子分析报告', fontsize=16,
                 bbox=dict(facecolor='none', edgecolor='none', pad=0))
    
    # 价格和综合信号
    ax1 = axes[0, 0]
    ax1.plot(factor_data.index, factor_data['price'], label='ETH Price', color='black')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(factor_data.index, factor_data['combined_signal'], 
                  label='Combined Signal', color='red', alpha=0.7)
    ax1.set_title('价格 vs 综合信号',
                  bbox=dict(facecolor='none', edgecolor='none', pad=0))
    ax1.set_ylabel('价格 (USDT)')
    ax1_twin.set_ylabel('综合信号')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # RSI
    ax2 = axes[0, 1]
    ax2.plot(factor_data.index, factor_data['rsi'], label='RSI', color='blue')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    ax2.set_title('RSI 相对强弱指标',
                  bbox=dict(facecolor='none', edgecolor='none', pad=0))
    ax2.set_ylabel('RSI')
    ax2.legend()
    
    # MACD
    ax3 = axes[1, 0]
    ax3.plot(factor_data.index, factor_data['macd'], label='MACD', color='purple')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('MACD 指标',
                  bbox=dict(facecolor='none', edgecolor='none', pad=0))
    ax3.set_ylabel('MACD 值')
    ax3.legend()
    
    # 布林带
    ax4 = axes[1, 1]
    ax4.plot(factor_data.index, factor_data['price'], label='Price', color='black')
    ax4.plot(factor_data.index, factor_data['bb_upper'], label='BB Upper', color='red', alpha=0.7)
    ax4.plot(factor_data.index, factor_data['bb_lower'], label='BB Lower', color='green', alpha=0.7)
    ax4.fill_between(factor_data.index, factor_data['bb_lower'], factor_data['bb_upper'], 
                     alpha=0.1, color='gray')
    ax4.set_title('布林带',
                  bbox=dict(facecolor='none', edgecolor='none', pad=0))
    ax4.set_ylabel('价格 (USDT)')
    ax4.legend()
    
    # 动量指标
    ax5 = axes[2, 0]
    ax5.plot(factor_data.index, factor_data['momentum'], label='Momentum', color='orange')
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax5.set_title('动量指标',
                  bbox=dict(facecolor='none', edgecolor='none', pad=0))
    ax5.set_ylabel('动量')
    ax5.legend()
    
    # 仓位变化
    ax6 = axes[2, 1]
    ax6.plot(factor_data.index, factor_data['position'], label='Position', color='brown')
    ax6.set_title('持仓变化',
                  bbox=dict(facecolor='none', edgecolor='none', pad=0))
    ax6.set_ylabel('持仓数量')
    ax6.legend()
    
    plt.tight_layout()
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(str(OUTPUT_DIR / 'factor_analysis_charts.png'), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("图表已保存为 output/factor_analysis_charts.png")


def compare_strategies(data):
    """对比三种策略"""
    print("\n=== 三种策略对比测试 ===")
    
    # 运行传统策略
    print("\n1. 运行传统固定权重策略...")
    traditional_analyzer = FactorAnalyzer(data)
    traditional_results = traditional_analyzer.run_backtest(initial_cash=100000.0)
    
    # 运行自适应策略（高相关因子）
    print("\n2. 运行自适应因子策略（高相关因子组合）...")
    adaptive_analyzer = AdaptiveFactorAnalyzer(data)
    adaptive_results = adaptive_analyzer.run_backtest(initial_cash=100000.0)
    
    # 运行增强版策略（低相关因子）
    print("\n3. 运行增强版策略（低相关性因子组合）...")
    enhanced_analyzer = EnhancedFactorAnalyzer(data)
    enhanced_results = enhanced_analyzer.run_backtest(initial_cash=100000.0)
    
    # 对比结果
    print("\n=== 策略对比结果 ===")
    print(f"{'指标':<15} {'传统策略':<15} {'自适应策略':<15} {'增强策略':<15} {'最佳':<10}")
    print("-" * 75)
    
    strategies = {
        '传统': traditional_results,
        '自适应': adaptive_results,
        '增强': enhanced_results
    }
    
    metrics = [
        ('收益率 (%)', 'return_pct'),
        ('夏普比率', 'sharpe_ratio'),
        ('最大回撤 (%)', 'max_drawdown'),
        ('总交易次数', 'total_trades')
    ]
    
    for metric_name, metric_key in metrics:
        trad_val = traditional_results.get(metric_key, 0) or 0
        adapt_val = adaptive_results.get(metric_key, 0) or 0
        enhanced_val = enhanced_results.get(metric_key, 0) or 0
        
        # 找出最佳表现
        values = [trad_val, adapt_val, enhanced_val]
        names = ['传统', '自适应', '增强']
        
        if metric_key == 'max_drawdown':
            best_idx = values.index(min(values))
        else:
            best_idx = values.index(max(values))
        
        best_strategy = names[best_idx]
        
        print(f"{metric_name:<15} {trad_val:<15.2f} {adapt_val:<15.2f} {enhanced_val:<15.2f} {best_strategy:<10}")
    
    return traditional_results, adaptive_results, enhanced_results, enhanced_analyzer


def plot_adaptive_analysis(analysis_data):
    """绘制自适应策略分析图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('自适应因子策略分析报告', fontsize=16)
    
    factor_data = analysis_data['factors']
    weights_data = analysis_data.get('weights', pd.DataFrame())
    
    # 设置日期索引
    if 'date' in factor_data.columns:
        factor_data['date'] = pd.to_datetime(factor_data['date'])
        factor_data.set_index('date', inplace=True)
    
    if not weights_data.empty and 'date' in weights_data.columns:
        weights_data['date'] = pd.to_datetime(weights_data['date'])
        weights_data.set_index('date', inplace=True)
    
    # 价格和综合信号
    ax1 = axes[0, 0]
    ax1.plot(factor_data.index, factor_data['price'], label='ETH Price', color='black')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(factor_data.index, factor_data['combined_signal'], 
                  label='Combined Signal', color='red', alpha=0.7)
    ax1.set_title('价格 vs 自适应信号')
    ax1.set_ylabel('价格 (USDT)')
    ax1_twin.set_ylabel('综合信号')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # 权重变化
    ax2 = axes[0, 1]
    if not weights_data.empty:
        factor_names = ['rsi', 'macd', 'momentum', 'volatility']
        colors = ['blue', 'red', 'green', 'orange']
        for factor, color in zip(factor_names, colors):
            if factor in weights_data.columns:
                ax2.plot(weights_data.index, weights_data[factor], 
                        label=factor.upper(), color=color)
        ax2.set_title('因子权重动态变化')
        ax2.set_ylabel('权重')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, '权重数据不可用', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('因子权重动态变化')
    
    # 持仓变化
    ax3 = axes[1, 0]
    ax3.plot(factor_data.index, factor_data['position'], label='Position', color='brown')
    ax3.set_title('持仓变化')
    ax3.set_ylabel('持仓数量')
    ax3.legend()
    
    # 信号分布
    ax4 = axes[1, 1]
    ax4.hist(factor_data['combined_signal'], bins=50, alpha=0.7, color='purple')
    ax4.axvline(x=0.3, color='red', linestyle='--', label='买入阈值')
    ax4.axvline(x=-0.3, color='green', linestyle='--', label='卖出阈值')
    ax4.set_title('综合信号分布')
    ax4.set_xlabel('信号值')
    ax4.set_ylabel('频次')
    ax4.legend()
    
    plt.tight_layout()
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(str(OUTPUT_DIR / 'adaptive_strategy_analysis.png'), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ 自适应策略分析图表已保存为 output/adaptive_strategy_analysis.png")


def main():
    """主函数"""
    symbol = 'ETH/USDT'
    timeframe = '4h'
    days = 730
    print(f"=== 币安 {symbol} {timeframe} 因子分析系统 ===")
    
    # 1. 获取数据
    print(f"\n1. 正在获取 {symbol} 数据...")
    fetcher = BinanceDataFetcher()
    
    # 获取最近 730 天的4小时数据
    data = fetcher.fetch_recent_with_cache(symbol=symbol, timeframe=timeframe, days=days)
    
    if data.empty:
        print("❌ 数据获取失败，请检查网络连接或API配置")
        return
    
    print(f"✅ 成功获取 {len(data)} 条数据")
    
    # 2. 策略对比
    traditional_results, adaptive_results, enhanced_results, enhanced_analyzer = compare_strategies(data)
    
    # 3. 详细分析增强版策略
    print("\n3. 详细分析增强版策略...")
    analysis_data = enhanced_analyzer.get_analysis_data()
    
    if 'factors' in analysis_data and not analysis_data['factors'].empty:
        factor_data = analysis_data['factors']
        
        # 分析因子表现
        correlation_matrix = analyze_factor_performance(factor_data)
        
        # 4. 绘制分析图表
        print("\n4. 正在生成分析图表...")
        plot_adaptive_analysis(analysis_data)
        
        # 5. 保存结果
        print("\n5. 正在保存分析结果...")
        
        # 确保输出目录存在
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 保存自适应策略结果
        factor_data.to_csv(OUTPUT_DIR / 'adaptive_factor_results.csv')
        print("✅ 自适应因子结果已保存到 output/adaptive_factor_results.csv")
        
        # 保存权重变化
        if 'weights' in analysis_data and not analysis_data['weights'].empty:
            analysis_data['weights'].to_csv(OUTPUT_DIR / 'factor_weights_history.csv')
            print("✅ 因子权重历史已保存到 output/factor_weights_history.csv")
        
        # 保存策略对比结果
        comparison_df = pd.DataFrame({
            '传统策略': traditional_results,
            '自适应策略': adaptive_results,
            '增强策略': enhanced_results
        }).T
        comparison_df.to_csv(OUTPUT_DIR / 'strategy_comparison.csv')
        print("✅ 策略对比结果已保存到 output/strategy_comparison.csv")
        
        # 保存相关性矩阵
        correlation_matrix.to_csv(OUTPUT_DIR / 'factor_correlation_matrix.csv')
        print("✅ 因子相关性矩阵已保存到 output/factor_correlation_matrix.csv")
    
    print("\n=== 分析完成 ===")
    print("📊 查看 output/ 目录中的CSV文件获取详细数据")
    print("📈 查看生成的图表文件")
    print("\n输出文件位置:")
    print("  - output/adaptive_factor_results.csv")
    print("  - output/factor_weights_history.csv")
    print("  - output/strategy_comparison.csv")
    print("  - output/factor_correlation_matrix.csv")
    print("  - output/adaptive_strategy_analysis.png")


if __name__ == "__main__":
    main()
