"""
主程序 - 币安 ETHUSDT 因子分析示例
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from typing import Dict, Any
warnings.filterwarnings('ignore')

from data_fetcher import BinanceDataFetcher
from strategies.factor_strategy import FactorAnalyzer
from strategies.adaptive_factor_strategy import AdaptiveFactorAnalyzer
from strategies.enhanced_factor_strategy import EnhancedFactorAnalyzer
from output_manager import RunOutputManager
from metrics_extractor import PerformanceMetricsExtractor, FactorAnalysisExtractor


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


def plot_factor_analysis(factor_data: pd.DataFrame, output_manager: RunOutputManager):
    """绘制因子分析图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    
    # 检查可用的列
    available_cols = factor_data.columns.tolist()
    print(f"可用列: {available_cols}")
    
    # 检查是否有嵌套的factors列
    if 'factors' in factor_data.columns and not factor_data.empty:
        # 展开factors字典为独立列
        try:
            # 重置索引以避免重复索引问题
            factor_data_reset = factor_data.reset_index(drop=True)
            factors_expanded = pd.json_normalize(factor_data_reset['factors'])
            factor_data = pd.concat([factor_data_reset.drop('factors', axis=1), factors_expanded], axis=1)
            # 恢复原来的索引
            factor_data.index = factor_data_reset.index
            print(f"展开factors后的列: {factor_data.columns.tolist()}")
        except Exception as e:
            print(f"展开factors时出错: {e}")
            # 如果展开失败，尝试手动解析
            try:
                import ast
                factor_rows = []
                for idx, row in factor_data.iterrows():
                    row_dict = dict(row)
                    if isinstance(row['factors'], str):
                        factors_dict = ast.literal_eval(row['factors'])
                    else:
                        factors_dict = row['factors']
                    row_dict.update(factors_dict)
                    del row_dict['factors']
                    factor_rows.append(row_dict)
                factor_data = pd.DataFrame(factor_rows)
                print(f"手动展开factors后的列: {factor_data.columns.tolist()}")
            except Exception as e2:
                print(f"手动展开factors也失败: {e2}")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('ETHUSDT 因子分析报告', fontsize=16,
                 bbox=dict(facecolor='none', edgecolor='none', pad=0))
    
    # 价格和综合信号
    ax1 = axes[0, 0]
    if 'price' in factor_data.columns:
        ax1.plot(factor_data.index, factor_data['price'], label='ETH Price', color='black')
    if 'combined_signal' in factor_data.columns:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(factor_data.index, factor_data['combined_signal'], 
                      label='Combined Signal', color='red', alpha=0.7)
        ax1_twin.set_ylabel('综合信号')
        ax1_twin.legend(loc='upper right')
    ax1.set_title('价格 vs 综合信号',
                  bbox=dict(facecolor='none', edgecolor='none', pad=0))
    ax1.set_ylabel('价格 (USDT)')
    ax1.legend(loc='upper left')
    
    # 波动率因子
    ax2 = axes[0, 1]
    if 'volatility' in factor_data.columns:
        ax2.plot(factor_data.index, factor_data['volatility'], label='Volatility', color='blue')
        ax2.set_title('波动率因子',
                      bbox=dict(facecolor='none', edgecolor='none', pad=0))
        ax2.set_ylabel('波动率')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Volatility 数据不可用', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('波动率因子')
    
    # 成交量趋势因子
    ax3 = axes[1, 0]
    if 'volume_trend' in factor_data.columns:
        ax3.plot(factor_data.index, factor_data['volume_trend'], label='Volume Trend', color='purple')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('成交量趋势因子',
                      bbox=dict(facecolor='none', edgecolor='none', pad=0))
        ax3.set_ylabel('成交量趋势')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Volume Trend 数据不可用', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('成交量趋势因子')
    
    # 价格位置因子
    ax4 = axes[1, 1]
    if 'price_position' in factor_data.columns:
        ax4.plot(factor_data.index, factor_data['price_position'], label='Price Position', color='green')
        ax4.axhline(y=0.5, color='black', linestyle='-', alpha=0.3, label='中性位置')
        ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='超买区域')
        ax4.axhline(y=0.2, color='blue', linestyle='--', alpha=0.5, label='超卖区域')
        ax4.set_title('价格位置因子',
                      bbox=dict(facecolor='none', edgecolor='none', pad=0))
        ax4.set_ylabel('价格位置 (0-1)')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Price Position 数据不可用', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('价格位置因子')
    
    # 动量指标
    ax5 = axes[2, 0]
    if 'momentum' in factor_data.columns:
        ax5.plot(factor_data.index, factor_data['momentum'], label='Momentum', color='orange')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.set_title('动量指标',
                      bbox=dict(facecolor='none', edgecolor='none', pad=0))
        ax5.set_ylabel('动量')
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'Momentum 数据不可用', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('动量指标')
    
    # 趋势强度因子
    ax6 = axes[2, 1]
    if 'trend_strength' in factor_data.columns:
        ax6.plot(factor_data.index, factor_data['trend_strength'], label='Trend Strength', color='brown')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax6.set_title('趋势强度因子',
                      bbox=dict(facecolor='none', edgecolor='none', pad=0))
        ax6.set_ylabel('趋势强度')
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'Trend Strength 数据不可用', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('趋势强度因子')
    
    plt.tight_layout()
    output_manager.save_chart('factor_analysis_charts.png')


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
    
    return traditional_results, adaptive_results, enhanced_results, traditional_analyzer, adaptive_analyzer, enhanced_analyzer


def plot_individual_strategy_portfolios(analyzers: Dict[str, Any], output_manager: RunOutputManager):
    """绘制每个策略的单独资产变化图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    
    # 颜色方案
    colors = {'传统策略': '#1f77b4', '自适应策略': '#ff7f0e', '增强策略': '#2ca02c'}
    
    for strategy_name, analyzer in analyzers.items():
        try:
            # 获取资产变化历史
            portfolio_history = analyzer.get_portfolio_history()
            
            if not portfolio_history or not portfolio_history.get('dates') or not portfolio_history.get('portfolio_values'):
                print(f"⚠️ {strategy_name}没有资产变化数据，跳过图表生成")
                continue
            
            dates = portfolio_history['dates']
            values = portfolio_history['portfolio_values']
            
            if not dates or not values or len(dates) != len(values):
                print(f"⚠️ {strategy_name}资产变化数据不完整，跳过图表生成")
                continue
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{strategy_name} - 详细分析报告', fontsize=16, fontweight='bold')
            
            # 1. 资产变化曲线
            ax1 = axes[0, 0]
            ax1.plot(dates, values, color=colors.get(strategy_name, 'blue'), linewidth=2)
            ax1.set_title('资产价值变化', fontweight='bold')
            ax1.set_xlabel('日期')
            ax1.set_ylabel('资产价值 (USDT)')
            ax1.grid(True, alpha=0.3)
            
            # 添加起始和结束值标注
            if len(values) > 0:
                start_value = values[0]
                end_value = values[-1]
                ax1.axhline(y=start_value, color='gray', linestyle='--', alpha=0.5, label=f'起始: {start_value:.0f}')
                ax1.axhline(y=end_value, color='red', linestyle='--', alpha=0.7, label=f'结束: {end_value:.0f}')
                ax1.legend()
            
            # 2. 收益率曲线
            ax2 = axes[0, 1]
            if len(values) > 1:
                returns = [(values[i] / values[0] - 1) * 100 for i in range(len(values))]
                ax2.plot(dates, returns, color=colors.get(strategy_name, 'blue'), linewidth=2)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.set_title('累计收益率变化', fontweight='bold')
                ax2.set_xlabel('日期')
                ax2.set_ylabel('收益率 (%)')
                ax2.grid(True, alpha=0.3)
                
                # 标注最终收益率
                final_return = returns[-1]
                ax2.text(0.02, 0.98, f'最终收益率: {final_return:.1f}%', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 3. 回撤分析
            ax3 = axes[1, 0]
            if len(values) > 1:
                # 计算回撤
                peak = values[0]
                drawdowns = []
                for value in values:
                    if value > peak:
                        peak = value
                    drawdown = (value - peak) / peak * 100
                    drawdowns.append(drawdown)
                
                ax3.fill_between(dates, drawdowns, 0, color='red', alpha=0.3)
                ax3.plot(dates, drawdowns, color='red', linewidth=1)
                ax3.set_title('回撤分析', fontweight='bold')
                ax3.set_xlabel('日期')
                ax3.set_ylabel('回撤 (%)')
                ax3.grid(True, alpha=0.3)
                
                # 标注最大回撤
                max_drawdown = min(drawdowns)
                ax3.text(0.02, 0.02, f'最大回撤: {max_drawdown:.1f}%', 
                        transform=ax3.transAxes, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            
            # 4. 资产增长分布
            ax4 = axes[1, 1]
            if len(values) > 10:
                # 计算日收益率
                daily_returns = []
                for i in range(1, len(values)):
                    daily_return = (values[i] / values[i-1] - 1) * 100
                    daily_returns.append(daily_return)
                
                ax4.hist(daily_returns, bins=30, alpha=0.7, color=colors.get(strategy_name, 'blue'))
                ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax4.set_title('日收益率分布', fontweight='bold')
                ax4.set_xlabel('日收益率 (%)')
                ax4.set_ylabel('频次')
                ax4.grid(True, alpha=0.3)
                
                # 添加统计信息
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                ax4.text(0.02, 0.98, f'均值: {mean_return:.3f}%\n标准差: {std_return:.3f}%', 
                        transform=ax4.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图表
            filename = f'{strategy_name}_portfolio_analysis.png'
            output_manager.save_chart(filename)
            
        except Exception as e:
            print(f"❌ 生成{strategy_name}资产变化图表时出错: {e}")


def plot_portfolio_comparison(strategy_results: Dict[str, Dict[str, Any]], output_manager: RunOutputManager):
    """绘制各策略资产变化对比图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('策略资产变化对比分析', fontsize=16, fontweight='bold')
    
    # 颜色方案
    colors = {'传统策略': '#1f77b4', '自适应策略': '#ff7f0e', '增强策略': '#2ca02c'}
    
    # 1. 资产变化对比
    ax1 = axes[0, 0]
    for strategy_name, results in strategy_results.items():
        portfolio_history = results.get('portfolio_history', {})
        if portfolio_history and 'dates' in portfolio_history and 'portfolio_values' in portfolio_history:
            dates = portfolio_history['dates']
            values = portfolio_history['portfolio_values']
            if dates and values:
                ax1.plot(dates, values, label=strategy_name, color=colors.get(strategy_name, 'gray'), linewidth=2)
    
    ax1.set_title('资产变化对比', fontweight='bold')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('资产价值 (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 收益率对比
    ax2 = axes[0, 1]
    strategy_names = list(strategy_results.keys())
    returns = [results.get('return_pct', 0) for results in strategy_results.values()]
    bars = ax2.bar(strategy_names, returns, color=[colors.get(name, 'gray') for name in strategy_names])
    ax2.set_title('总收益率对比', fontweight='bold')
    ax2.set_ylabel('收益率 (%)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上显示数值
    for bar, return_val in zip(bars, returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{return_val:.1f}%', ha='center', va='bottom')
    
    # 3. 风险指标对比
    ax3 = axes[1, 0]
    sharpe_ratios = [results.get('sharpe_ratio', 0) for results in strategy_results.values()]
    max_drawdowns = [results.get('max_drawdown', 0) for results in strategy_results.values()]
    
    x = np.arange(len(strategy_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, sharpe_ratios, width, label='夏普比率', color='lightblue')
    bars2 = ax3.bar(x + width/2, [-dd for dd in max_drawdowns], width, label='最大回撤 (%)', color='lightcoral')
    
    ax3.set_title('风险收益指标对比', fontweight='bold')
    ax3.set_ylabel('指标值')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategy_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 4. 交易统计对比
    ax4 = axes[1, 1]
    total_trades = [results.get('total_trades', 0) for results in strategy_results.values()]
    win_rates = [results.get('win_rate', 0) * 100 for results in strategy_results.values()]
    
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar([i - 0.2 for i in range(len(strategy_names))], total_trades, 
                    width=0.4, label='交易次数', color='steelblue', alpha=0.7)
    bars2 = ax4_twin.bar([i + 0.2 for i in range(len(strategy_names))], win_rates, 
                         width=0.4, label='胜率 (%)', color='orange', alpha=0.7)
    
    ax4.set_title('交易统计对比', fontweight='bold')
    ax4.set_xlabel('策略')
    ax4.set_ylabel('交易次数')
    ax4_twin.set_ylabel('胜率 (%)')
    ax4.set_xticks(range(len(strategy_names)))
    ax4.set_xticklabels(strategy_names)
    
    # 合并图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_manager.save_chart('portfolio_comparison.png')


def plot_adaptive_analysis(analysis_data, output_manager: RunOutputManager):
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
    output_manager.save_chart('adaptive_strategy_analysis.png')


def main():
    """主函数"""
    symbol = 'ETH/USDT'
    timeframe = '4h'
    days = 730
    print(f"=== 币安 {symbol} {timeframe} 因子分析系统 ===")
    
    # 0. 创建输出管理器
    output_manager = RunOutputManager()
    print(f"📁 本次运行输出目录: {output_manager.run_dir}")
    
    # 1. 获取完整数据（包含完整性检查和缺失数据补充）
    print(f"\n1. 正在获取 {symbol} 数据...")
    fetcher = BinanceDataFetcher()
    
    # 获取最近 730 天的4小时数据，包含完整性检查
    data = fetcher.fetch_complete_data(symbol=symbol, timeframe=timeframe, days=days)
    
    if data.empty:
        print("❌ 数据获取失败，请检查网络连接或API配置")
        return
    
    print(f"✅ 成功获取 {len(data)} 条数据")
    
    # 2. 策略对比
    traditional_results, adaptive_results, enhanced_results, traditional_analyzer, adaptive_analyzer, enhanced_analyzer = compare_strategies(data)
    
    # 3. 提取和整理指标
    print("\n3. 正在提取性能指标...")
    
    # 获取分析数据
    analysis_data = enhanced_analyzer.get_analysis_data()
    factor_data = analysis_data.get('factors', pd.DataFrame())
    
    # 提取各策略的详细指标
    all_strategy_metrics = {}
    
    if not factor_data.empty:
        # 传统策略指标
        traditional_metrics = PerformanceMetricsExtractor.extract_strategy_metrics(
            traditional_results, factor_data, traditional_results.get('strategy_instance')
        )
        all_strategy_metrics['传统策略'] = traditional_metrics
        
        # 自适应策略指标
        adaptive_metrics = PerformanceMetricsExtractor.extract_strategy_metrics(
            adaptive_results, factor_data, adaptive_results.get('strategy_instance')
        )
        all_strategy_metrics['自适应策略'] = adaptive_metrics
        
        # 增强策略指标
        enhanced_metrics = PerformanceMetricsExtractor.extract_strategy_metrics(
            enhanced_results, factor_data, enhanced_results.get('strategy_instance')
        )
        all_strategy_metrics['增强策略'] = enhanced_metrics
        
        # 保存详细指标
        for strategy_name, metrics in all_strategy_metrics.items():
            output_manager.save_strategy_results(strategy_name, metrics)
    
    # 4. 因子分析
    print("\n4. 正在进行因子分析...")
    factor_analysis = {}
    
    if not factor_data.empty:
        # 相关性分析
        correlation_analysis = FactorAnalysisExtractor.analyze_factor_correlation(factor_data)
        factor_analysis['correlation_analysis'] = correlation_analysis
        
        # 信号有效性分析
        signal_analysis = FactorAnalysisExtractor.analyze_signal_effectiveness(factor_data)
        factor_analysis['signal_analysis'] = signal_analysis
        
        # 保存因子分析结果
        output_manager.save_strategy_results('因子分析', factor_analysis)
        
        # 保存因子数据
        output_manager.save_dataframe(factor_data, 'factor_data.csv')
        
        # 保存权重变化（如果有）
        if 'weights' in analysis_data and not analysis_data['weights'].empty:
            output_manager.save_dataframe(analysis_data['weights'], 'factor_weights_history.csv')
    
    # 5. 生成图表
    print("\n5. 正在生成分析图表...")
    
    # 绘制每个策略的单独资产变化图表
    print("正在生成各策略资产变化图表...")
    analyzers = {
        '传统策略': traditional_analyzer,
        '自适应策略': adaptive_analyzer,
        '增强策略': enhanced_analyzer
    }
    plot_individual_strategy_portfolios(analyzers, output_manager)
    
    # 绘制策略资产变化对比图表
    strategy_results_for_plot = {
        '传统策略': traditional_results,
        '自适应策略': adaptive_results,
        '增强策略': enhanced_results
    }
    plot_portfolio_comparison(strategy_results_for_plot, output_manager)
    
    if not factor_data.empty:
        # 绘制自适应策略分析图表
        plot_adaptive_analysis(analysis_data, output_manager)
        
        # 绘制因子分析图表
        plot_factor_analysis(factor_data, output_manager)
    
    # 6. 创建汇总报告
    print("\n6. 正在创建汇总报告...")
    
    # 创建策略对比DataFrame
    if all_strategy_metrics:
        comparison_data = {}
        for strategy_name, metrics in all_strategy_metrics.items():
            comparison_data[strategy_name] = {
                '收益率(%)': metrics.get('return_pct', 0),
                '夏普比率': metrics.get('sharpe_ratio', 0),
                '最大回撤(%)': metrics.get('max_drawdown', 0),
                '总交易次数': metrics.get('total_trades', 0),
                '胜率': metrics.get('win_rate', 0),
                '风险评级': metrics.get('risk_grade', '未知'),
                'Calmar比率': metrics.get('calmar_ratio', 0),
                '交易效率': metrics.get('trade_efficiency', 0)
            }
        
        comparison_df = pd.DataFrame(comparison_data).T
        output_manager.save_dataframe(comparison_df, 'strategy_comparison.csv', 'reports')
        
        # 创建详细汇总报告
        output_manager.create_summary_report(all_strategy_metrics, factor_analysis)
    
    # 7. 显示运行总结
    print("\n=== 分析完成 ===")
    run_info = output_manager.get_run_summary()
    
    print(f"📊 本次运行输出目录: {run_info['run_directory']}")
    print(f"📈 图表文件目录: {run_info['charts_directory']}")
    print(f"📋 数据文件目录: {run_info['data_directory']}")
    print(f"📄 报告文件目录: {run_info['reports_directory']}")
    
    print("\n主要输出文件:")
    print("  📈 图表文件:")
    for chart_file in output_manager.charts_dir.glob("*.png"):
        print(f"    - {chart_file.name}")
    
    print("  📊 数据文件:")
    for data_file in output_manager.data_dir.glob("*"):
        print(f"    - {data_file.name}")
    
    print("  📄 报告文件:")
    for report_file in output_manager.reports_dir.glob("*"):
        print(f"    - {report_file.name}")
    
    print(f"\n🎉 分析完成！所有结果已保存到: {output_manager.run_dir}")
    
    # 显示关键指标摘要
    if all_strategy_metrics:
        print("\n📊 关键指标摘要:")
        print("-" * 60)
        for strategy_name, metrics in all_strategy_metrics.items():
            print(f"{strategy_name}:")
            print(f"  收益率: {metrics.get('return_pct', 0):.2f}%")
            print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
            print(f"  风险评级: {metrics.get('risk_grade', '未知')}")
            print()


if __name__ == "__main__":
    main()
