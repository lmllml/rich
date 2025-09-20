"""
回测入口 - 专注于策略回测与绩效对比，不做因子相关性分析
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any

from data_fetcher import BinanceDataFetcher
from strategies.factor_strategy import FactorAnalyzer
from strategies.adaptive_factor_strategy import AdaptiveFactorAnalyzer
from strategies.enhanced_factor_strategy import EnhancedFactorAnalyzer
from output_manager import RunOutputManager
from metrics_extractor import PerformanceMetricsExtractor


def compare_strategies(data):
    print("\n=== 三种策略对比测试 ===")
    print("\n1. 运行传统固定权重策略...")
    traditional_analyzer = FactorAnalyzer(data)
    traditional_results = traditional_analyzer.run_backtest(initial_cash=100000.0)

    print("\n2. 运行自适应因子策略（高相关因子组合）...")
    adaptive_analyzer = AdaptiveFactorAnalyzer(data)
    adaptive_results = adaptive_analyzer.run_backtest(initial_cash=100000.0)

    print("\n3. 运行增强版策略（低相关性因子组合）...")
    enhanced_analyzer = EnhancedFactorAnalyzer(data)
    enhanced_results = enhanced_analyzer.run_backtest(initial_cash=100000.0)

    print("\n=== 策略对比结果 ===")
    print(f"{'指标':<15} {'传统策略':<15} {'自适应策略':<15} {'增强策略':<15} {'最佳':<10}")
    print("-" * 75)

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

        values = [trad_val, adapt_val, enhanced_val]
        names = ['传统', '自适应', '增强']

        if metric_key == 'max_drawdown':
            best_idx = values.index(min(values))
        else:
            best_idx = values.index(max(values))

        best_strategy = names[best_idx]
        print(f"{metric_name:<15} {trad_val:<15.2f} {adapt_val:<15.2f} {enhanced_val:<15.2f} {best_strategy:<10}")

    return {
        '传统策略': traditional_results,
        '自适应策略': adaptive_results,
        '增强策略': enhanced_results,
    }, {
        'traditional_analyzer': traditional_analyzer,
        'adaptive_analyzer': adaptive_analyzer,
        'enhanced_analyzer': enhanced_analyzer,
    }


def plot_individual_strategy_portfolios(strategy_results: Dict[str, Dict[str, Any]], output_manager: RunOutputManager):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 

    colors = {'传统策略': '#1f77b4', '自适应策略': '#ff7f0e', '增强策略': '#2ca02c'}

    for strategy_name, results in strategy_results.items():
        try:
            portfolio_history = results.get('portfolio_history', {}) if isinstance(results, dict) else {}
            if not portfolio_history or not portfolio_history.get('dates') or not portfolio_history.get('portfolio_values'):
                print(f"⚠️ {strategy_name}没有资产变化数据，跳过图表生成")
                continue

            dates = portfolio_history['dates']
            values = portfolio_history['portfolio_values']
            if not dates or not values or len(dates) != len(values):
                print(f"⚠️ {strategy_name}资产变化数据不完整，跳过图表生成")
                continue

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{strategy_name} - 详细分析报告', fontsize=16, fontweight='bold')

            ax1 = axes[0, 0]
            ax1.plot(dates, values, color=colors.get(strategy_name, 'blue'), linewidth=2)
            ax1.set_title('资产价值变化', fontweight='bold')
            ax1.set_xlabel('日期')
            ax1.set_ylabel('资产价值 (USDT)')
            ax1.grid(True, alpha=0.3)
            if len(values) > 0:
                start_value = values[0]
                end_value = values[-1]
                ax1.axhline(y=start_value, color='gray', linestyle='--', alpha=0.5, label=f'起始: {start_value:.0f}')
                ax1.axhline(y=end_value, color='red', linestyle='--', alpha=0.7, label=f'结束: {end_value:.0f}')
                ax1.legend()

            ax2 = axes[0, 1]
            if len(values) > 1:
                returns = [(values[i] / values[0] - 1) * 100 for i in range(len(values))]
                ax2.plot(dates, returns, color=colors.get(strategy_name, 'blue'), linewidth=2)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.set_title('累计收益率变化', fontweight='bold')
                ax2.set_xlabel('日期')
                ax2.set_ylabel('收益率 (%)')
                ax2.grid(True, alpha=0.3)
                final_return = returns[-1]
                ax2.text(0.02, 0.98, f'最终收益率: {final_return:.1f}%', transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax3 = axes[1, 0]
            if len(values) > 1:
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
                max_drawdown = min(drawdowns)
                ax3.text(0.02, 0.02, f'最大回撤: {max_drawdown:.1f}%', transform=ax3.transAxes, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

            ax4 = axes[1, 1]
            if len(values) > 10:
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
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                ax4.text(0.02, 0.98, f'均值: {mean_return:.3f}%\n标准差: {std_return:.3f}%', transform=ax4.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            plt.tight_layout()
            filename = f'{strategy_name}_portfolio_analysis.png'
            output_manager.save_chart(filename)

        except Exception as e:
            print(f"❌ 生成{strategy_name}资产变化图表时出错: {e}")


def plot_portfolio_comparison(strategy_results: Dict[str, Dict[str, Any]], output_manager: RunOutputManager):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('策略资产变化对比分析', fontsize=16, fontweight='bold')

    colors = {'传统策略': '#1f77b4', '自适应策略': '#ff7f0e', '增强策略': '#2ca02c'}

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

    ax2 = axes[0, 1]
    strategy_names = list(strategy_results.keys())
    returns = [results.get('return_pct', 0) for results in strategy_results.values()]
    bars = ax2.bar(strategy_names, returns, color=[colors.get(name, 'gray') for name in strategy_names])
    ax2.set_title('总收益率对比', fontweight='bold')
    ax2.set_ylabel('收益率 (%)')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, return_val in zip(bars, returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{return_val:.1f}%', ha='center', va='bottom')

    ax3 = axes[1, 0]
    sharpe_ratios = [results.get('sharpe_ratio', 0) for results in strategy_results.values()]
    max_drawdowns = [results.get('max_drawdown', 0) for results in strategy_results.values()]
    x = np.arange(len(strategy_names))
    width = 0.35
    ax3.bar(x - width/2, sharpe_ratios, width, label='夏普比率', color='lightblue')
    ax3.bar(x + width/2, [-dd for dd in max_drawdowns], width, label='最大回撤 (%)', color='lightcoral')
    ax3.set_title('风险收益指标对比', fontweight='bold')
    ax3.set_ylabel('指标值')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategy_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax4 = axes[1, 1]
    total_trades = [results.get('total_trades', 0) for results in strategy_results.values()]
    win_rates = [results.get('win_rate', 0) * 100 for results in strategy_results.values()]
    ax4_twin = ax4.twinx()
    ax4.bar([i - 0.2 for i in range(len(strategy_names))], total_trades, width=0.4, label='交易次数', color='steelblue', alpha=0.7)
    ax4_twin.bar([i + 0.2 for i in range(len(strategy_names))], win_rates, width=0.4, label='胜率 (%)', color='orange', alpha=0.7)
    ax4.set_title('交易统计对比', fontweight='bold')
    ax4.set_xlabel('策略')
    ax4.set_ylabel('交易次数')
    ax4_twin.set_ylabel('胜率 (%)')
    ax4.set_xticks(range(len(strategy_names)))
    ax4.set_xticklabels(strategy_names)
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_manager.save_chart('portfolio_comparison.png')


def main():
    symbol = 'BTC/USDT'
    timeframe = '4h'
    days = 365
    print(f"=== 回测入口 | 币安 {symbol} {timeframe} ===")

    output_manager = RunOutputManager()
    print(f"📁 本次运行输出目录: {output_manager.run_dir}")
    # 保存运行参数
    output_manager.save_run_params({'mode': 'backtest', 'symbol': symbol, 'timeframe': timeframe, 'days': days})

    print(f"\n1. 正在获取 {symbol} 数据...")
    fetcher = BinanceDataFetcher()
    data = fetcher.fetch_complete_data(symbol=symbol, timeframe=timeframe, days=days)
    if data.empty:
        print("❌ 数据获取失败，请检查网络连接或API配置")
        return
    print(f"✅ 成功获取 {len(data)} 条数据")

    strategy_results, analyzers = compare_strategies(data)

    print("\n3. 正在提取性能指标...")
    all_strategy_metrics: Dict[str, Any] = {}
    # 优先从增强策略提取因子数据（若存在）用于额外指标
    enhanced_analyzer = analyzers['enhanced_analyzer']
    analysis_data = enhanced_analyzer.get_analysis_data() if hasattr(enhanced_analyzer, 'get_analysis_data') else {}
    factor_data = analysis_data.get('factors', pd.DataFrame()) if isinstance(analysis_data, dict) else pd.DataFrame()

    for strategy_name, results in strategy_results.items():
        metrics = PerformanceMetricsExtractor.extract_strategy_metrics(
            results, factor_data, results.get('strategy_instance')
        )
        all_strategy_metrics[strategy_name] = metrics
        output_manager.save_strategy_results(strategy_name, metrics)

    print("\n4. 正在生成图表...")
    plot_individual_strategy_portfolios(strategy_results, output_manager)
    plot_portfolio_comparison(strategy_results, output_manager)

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

    print("\n=== 回测完成 ===")
    run_info = output_manager.get_run_summary()
    print(f"📊 本次运行输出目录: {run_info['run_directory']}")
    print(f"📈 图表文件目录: {run_info['charts_directory']}")
    print(f"📄 报告文件目录: {run_info['reports_directory']}")


if __name__ == "__main__":
    main()


