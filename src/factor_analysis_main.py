"""
因子分析入口 - 专注于因子数据展开、相关性与信号有效性分析与图表
不运行多策略对比与回测。
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any

from data_fetcher import BinanceDataFetcher
from strategies.enhanced_factor_strategy import EnhancedFactorAnalyzer
from output_manager import RunOutputManager
from metrics_extractor import FactorAnalysisExtractor
from factor_selector import FactorCorrelationAnalyzer


def _infer_periods_per_year(index: pd.Index) -> float:
    if len(index) < 2:
        return 365.0
    try:
        deltas = (index[1:] - index[:-1]).astype('timedelta64[s]').astype(float)
        median_sec = float(np.median(deltas)) if len(deltas) > 0 else 3600.0
    except Exception:
        # Fallback for non-numpy index
        median_sec = (index[1] - index[0]).total_seconds() if hasattr(index[1] - index[0], 'total_seconds') else 3600.0
    seconds_per_year = 365 * 24 * 3600
    periods = max(seconds_per_year / max(median_sec, 1.0), 1.0)
    return periods


def _compute_perf(returns: pd.Series, periods_per_year: float) -> Dict[str, Any]:
    r = returns.dropna()
    if r.empty:
        return {
            'return_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
        }
    equity = (1.0 + r).cumprod()
    total_return = equity.iloc[-1] - 1.0
    mean = r.mean()
    std = r.std(ddof=0)
    sharpe = (mean / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    peak = equity.cummax()
    dd = (equity / peak - 1.0).min()
    win_rate = float((r > 0).mean()) if len(r) > 0 else 0.0
    return {
        'return_pct': float(total_return * 100.0),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(dd * 100.0),
        'total_trades': 0,  # 占位，调用方覆盖
        'win_rate': float(win_rate),
    }


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0.0, 0.0)
    loss = -delta.where(delta < 0.0, 0.0)
    roll_u = gain.ewm(alpha=1/period, adjust=False).mean()
    roll_d = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_u / roll_d.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_diversified_factors(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)
    ret1 = close.pct_change()

    factors: Dict[str, pd.Series] = {}

    # 1) Returns (6)
    for n in [1, 2, 3, 5, 10, 20]:
        factors[f'ret_{n}'] = close.pct_change(n)

    # 2) Momentum diff (6)
    for n in [1, 2, 3, 5, 10, 20]:
        factors[f'mom_{n}'] = close.diff(n)

    # 3) SMA ratios (6)
    for n in [5, 10, 20, 50, 100, 200]:
        sma = close.rolling(n).mean()
        factors[f'sma_ratio_{n}'] = close / sma.replace(0, np.nan)

    # 4) EMA ratios (4)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    factors['ema_ratio_12_26'] = ema12 / ema26.replace(0, np.nan)
    factors['ema_ratio_20_50'] = ema20 / ema50.replace(0, np.nan)
    factors['ema_ratio_50_200'] = ema50 / ema200.replace(0, np.nan)
    factors['price_ema20_ratio'] = close / ema20.replace(0, np.nan)

    # 5) Volatility of returns (3)
    for n in [10, 20, 50]:
        factors[f'volret_{n}'] = ret1.rolling(n).std()

    # 6) Bollinger position (1)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=0)
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    factors['bb_pos_20'] = (close - lower) / (upper - lower)

    # 7) RSI (4)
    for n in [6, 14, 21, 28]:
        factors[f'rsi_{n}'] = _rsi(close, n)

    # 8) MACD (3)
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    factors['macd'] = macd
    factors['macd_signal'] = macd_signal
    factors['macd_hist'] = macd - macd_signal

    # 9) Stochastic (2)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    k = (close - low14) / (high14 - low14)
    d = k.rolling(3).mean()
    factors['stoch_k14'] = k
    factors['stoch_d3'] = d

    # 10) OBV / Volume features (3)
    obv = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()
    factors['obv'] = obv
    factors['obv_z20'] = (obv - obv.rolling(20).mean()) / obv.rolling(20).std(ddof=0)
    factors['vol_sma_ratio20'] = volume / volume.rolling(20).mean().replace(0, np.nan)

    # 11) Volume momentum (2)
    factors['vol_mom_5'] = volume.pct_change(5)
    factors['vol_mom_10'] = volume.pct_change(10)

    # 12) Price position min-max (3)
    for n in [20, 50, 100]:
        lo = low.rolling(n).min()
        hi = high.rolling(n).max()
        factors[f'price_pos_{n}'] = (close - lo) / (hi - lo)

    # 13) Trend consistency (1)
    factors['trend_consistency_14'] = (close.diff() > 0).astype(float).rolling(14).mean()

    # 14) Z-scores (2)
    for n in [20, 50]:
        factors[f'zscore_{n}'] = (close - close.rolling(n).mean()) / close.rolling(n).std(ddof=0)

    # 15) Range and ATR normalized (2)
    factors['range_pct'] = (high - low) / close.replace(0, np.nan)
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr14 = tr.rolling(14).mean()
    factors['atrn_14'] = atr14 / close.replace(0, np.nan)

    # 16) VWAP deviation (1)
    typical = (high + low + close) / 3.0
    vwap_num = (typical * volume).rolling(20).sum()
    vwap_den = volume.rolling(20).sum()
    vwap = vwap_num / vwap_den.replace(0, np.nan)
    factors['vwap_dev_20'] = (close - vwap) / vwap.replace(0, np.nan)

    # 17) SMA slope (1)
    factors['sma20_slope'] = ma20.diff()

    out = pd.DataFrame(factors, index=df.index)
    return out


def run_single_factor_backtests(ohlcv: pd.DataFrame, output_manager: RunOutputManager, top_chart_count: int = 6) -> pd.DataFrame:
    factors = build_diversified_factors(ohlcv)
    close = ohlcv['close'].astype(float)
    ret1 = close.pct_change()
    periods_per_year = _infer_periods_per_year(ohlcv.index)
    tc = 0.0005

    metrics_rows: list[Dict[str, Any]] = []
    equity_curves: Dict[str, pd.Series] = {}

    for col in factors.columns:
        f = factors[col].astype(float)
        z = (f - f.rolling(60).mean()) / f.rolling(60).std(ddof=0)
        long_sig = (z > 1.0).astype(int)
        short_sig = (z < -1.0).astype(int) * -1
        pos = (long_sig + short_sig).clip(-1, 1)
        pos = pos.reindex_like(ret1).fillna(0)
        strat_ret = pos.shift(1).fillna(0) * ret1
        turn = pos.diff().abs().fillna(0)
        net_ret = strat_ret - turn * tc
        perf = _compute_perf(net_ret, periods_per_year)
        # 估算交易次数：进场次数
        entries = ((pos.shift(1).fillna(0) == 0) & (pos != 0)).sum()
        perf['total_trades'] = int(entries)
        perf['factor'] = col
        metrics_rows.append(perf)
        equity_curves[col] = (1 + net_ret.fillna(0)).cumprod()

    results_df = pd.DataFrame(metrics_rows).sort_values(['sharpe_ratio', 'return_pct'], ascending=[False, False])

    # 绘制前N个因子的权益曲线
    top_names = results_df['factor'].head(top_chart_count).tolist()
    if top_names:
        fig, ax = plt.subplots(figsize=(12, 6))
        for name in top_names:
            eq = equity_curves[name]
            ax.plot(eq.index, eq.values, label=name)
        ax.set_title('单因子策略权益曲线（Top）')
        ax.set_ylabel('累计净值')
        ax.legend()
        plt.tight_layout()
        output_manager.save_chart('single_factor_top_equity.png')

    return results_df.reset_index(drop=True)
def plot_factor_analysis(factor_data: pd.DataFrame, output_manager: RunOutputManager):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 

    available_cols = factor_data.columns.tolist()
    print(f"可用列: {available_cols}")

    if 'factors' in factor_data.columns and not factor_data.empty:
        try:
            factor_data_reset = factor_data.reset_index(drop=True)
            factors_expanded = pd.json_normalize(factor_data_reset['factors'])
            factor_data = pd.concat([factor_data_reset.drop('factors', axis=1), factors_expanded], axis=1)
            factor_data.index = factor_data_reset.index
        except Exception as e:
            print(f"展开factors时出错: {e}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('因子分析报告', fontsize=16)

    ax1 = axes[0, 0]
    if 'price' in factor_data.columns:
        ax1.plot(factor_data.index, factor_data['price'], label='Price', color='black')
    if 'combined_signal' in factor_data.columns:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(factor_data.index, factor_data['combined_signal'], label='Combined Signal', color='red', alpha=0.7)
        ax1_twin.set_ylabel('综合信号')
        ax1_twin.legend(loc='upper right')
    ax1.set_title('价格 vs 综合信号')
    ax1.set_ylabel('价格 (USDT)')
    ax1.legend(loc='upper left')

    ax2 = axes[0, 1]
    for col in ['volatility', 'momentum']:
        if col in factor_data.columns:
            ax2.plot(factor_data.index, factor_data[col], label=col)
    ax2.set_title('核心因子曲线')
    ax2.legend()

    ax3 = axes[1, 0]
    if 'combined_signal' in factor_data.columns:
        ax3.hist(factor_data['combined_signal'], bins=50, alpha=0.7, color='purple')
        ax3.axvline(x=0.3, color='red', linestyle='--', label='买入阈值')
        ax3.axvline(x=-0.3, color='green', linestyle='--', label='卖出阈值')
        ax3.set_title('综合信号分布')
        ax3.legend()

    ax4 = axes[1, 1]
    if 'position' in factor_data.columns:
        ax4.plot(factor_data.index, factor_data['position'], label='Position', color='brown')
        ax4.set_title('持仓变化')
        ax4.legend()

    plt.tight_layout()
    output_manager.save_chart('factor_analysis_charts.png')


def main():
    symbol = 'BNB/USDT'
    timeframe = '4h'
    days = 365
    print(f"=== 因子分析入口 | 币安 {symbol} {timeframe} ===")

    output_manager = RunOutputManager()
    print(f"📁 本次运行输出目录: {output_manager.run_dir}")
    # 保存运行参数
    output_manager.save_run_params({'mode': 'factor', 'symbol': symbol, 'timeframe': timeframe, 'days': days})

    print(f"\n1. 正在获取 {symbol} 数据...")
    fetcher = BinanceDataFetcher()
    data = fetcher.fetch_complete_data(symbol=symbol, timeframe=timeframe, days=days)
    if data.empty:
        print("❌ 数据获取失败，请检查网络连接或API配置")
        return
    print(f"✅ 成功获取 {len(data)} 条数据")

    print("\n2. 生成因子序列（通过增强策略的因子记录，不进行回测对比）...")
    analyzer = EnhancedFactorAnalyzer(data)
    # 运行一次以产生因子时间序列和权重历史，但不关心绩效对比
    _ = analyzer.run_backtest(initial_cash=100000.0)
    analysis_data = analyzer.get_analysis_data()
    factor_data = analysis_data.get('factors', pd.DataFrame())
    if factor_data.empty:
        print("⚠️ 未获取到因子数据，退出")
        return

    # 保存原始因子数据
    output_manager.save_dataframe(factor_data, 'factor_data.csv')

    # 3. 因子分析
    print("\n3. 正在进行因子分析...")
    factor_analysis = {}
    correlation_analysis = FactorAnalysisExtractor.analyze_factor_correlation(factor_data)
    signal_analysis = FactorAnalysisExtractor.analyze_signal_effectiveness(factor_data)
    factor_analysis['correlation_analysis'] = correlation_analysis
    factor_analysis['signal_analysis'] = signal_analysis
    output_manager.save_strategy_results('因子分析', factor_analysis)

    # 4. 相关性选因子（多候选因子）
    ranking_df, ranking_path = FactorCorrelationAnalyzer.analyze_and_save(factor_data, output_manager)
    if not ranking_df.empty:
        print(f"✅ 最相关因子排名已保存: {ranking_path}")

    # 5. 图表
    print("\n4. 正在生成图表...")
    plot_factor_analysis(factor_data, output_manager)

    print("\n=== 因子分析完成 ===")

    # 5. 选择50个全面分散的因子并进行单因子回测
    print("\n5. 构建50个因子并执行单因子回测...")
    ohlcv = data.copy()
    try:
        results_df = run_single_factor_backtests(ohlcv, output_manager, top_chart_count=6)
        output_manager.save_dataframe(results_df, 'single_factor_backtests.csv', 'reports')
        print("✅ 单因子回测汇总已保存: reports/single_factor_backtests.csv")
    except Exception as e:
        print(f"⚠️ 单因子回测阶段出错: {e}")


if __name__ == "__main__":
    main()


