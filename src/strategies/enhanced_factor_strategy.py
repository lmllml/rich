"""
增强因子策略 - 引入多样化的低相关性因子
基于因子相关性分析，优化因子组合
"""
import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class DiversifiedTechnicalFactors(bt.Indicator):
    """多样化技术因子计算器 - 低相关性因子组合"""
    
    lines = ('momentum', 'volatility', 'volume_trend', 'price_position', 'trend_strength')
    
    params = (
        # 动量因子（保留最有效的趋势因子）
        ('momentum_period', 10),
        
        # 波动率因子
        ('volatility_period', 20),
        
        # 成交量趋势因子
        ('volume_sma_period', 20),
        ('volume_trend_period', 10),
        
        # 价格位置因子（类似布林带位置）
        ('price_position_period', 20),
        ('price_position_std', 2),
        
        # 趋势强度因子（ADX类似）
        ('trend_strength_period', 14),
    )
    
    def __init__(self):
        # 1. 动量因子 - 保留作为主要趋势因子
        self.lines.momentum = bt.indicators.Momentum(
            self.data.close,
            period=self.params.momentum_period
        )
        
        # 2. 波动率因子 - 已证明独立性
        self.lines.volatility = bt.indicators.StandardDeviation(
            self.data.close,
            period=self.params.volatility_period
        )
        
        # 3. 成交量趋势因子 - 全新维度
        volume_sma = bt.indicators.SimpleMovingAverage(
            self.data.volume,
            period=self.params.volume_sma_period
        )
        # 当前成交量相对于平均成交量的比率
        volume_ratio = self.data.volume / volume_sma
        # 成交量比率的动量
        self.lines.volume_trend = bt.indicators.Momentum(
            volume_ratio,
            period=self.params.volume_trend_period
        )
        
        # 4. 价格位置因子 - 价格在近期区间中的相对位置
        sma = bt.indicators.SimpleMovingAverage(
            self.data.close,
            period=self.params.price_position_period
        )
        std = bt.indicators.StandardDeviation(
            self.data.close,
            period=self.params.price_position_period
        )
        # 价格在 (均值 ± 2标准差) 区间中的相对位置，范围 [0, 1]
        upper_band = sma + (self.params.price_position_std * std)
        lower_band = sma - (self.params.price_position_std * std)
        self.lines.price_position = (self.data.close - lower_band) / (upper_band - lower_band)
        
        # 5. 趋势强度因子 - 衡量趋势的持续性和强度
        # 使用价格变化的方向一致性
        price_change = self.data.close - self.data.close(-1)
        # 计算近期价格变化方向的一致性
        positive_changes = bt.indicators.SumN(
            price_change > 0,
            period=self.params.trend_strength_period
        )
        self.lines.trend_strength = (positive_changes / self.params.trend_strength_period) - 0.5


class EnhancedFactorEffectivenessAnalyzer:
    """增强版因子有效性分析器"""
    
    def __init__(self, lookback_period: int = 60, rebalance_period: int = 20):
        self.lookback_period = lookback_period
        self.rebalance_period = rebalance_period
        self.factor_history = deque(maxlen=lookback_period)
        self.effectiveness_history = []
        self.current_weights = {}
        self.days_since_rebalance = 0
        
    def add_factor_data(self, factor_data: Dict[str, float], future_return: float = None):
        """添加因子数据"""
        self.factor_history.append({
            'factors': factor_data.copy(),
            'future_return': future_return
        })
        
    def calculate_factor_effectiveness(self) -> Dict[str, float]:
        """计算各因子与未来收益的相关性和预测能力"""
        if len(self.factor_history) < self.lookback_period:
            return self._get_default_weights()
            
        # 提取因子数据和未来收益
        factor_names = list(self.factor_history[0]['factors'].keys())
        factor_matrix = []
        returns = []
        
        for i in range(len(self.factor_history) - 1):
            factors = [self.factor_history[i]['factors'][name] for name in factor_names]
            future_ret = self.factor_history[i + 1]['future_return'] if i + 1 < len(self.factor_history) else 0
            
            if future_ret is not None and not np.isnan(future_ret):
                factor_matrix.append(factors)
                returns.append(future_ret)
        
        if len(factor_matrix) < 10:
            return self._get_default_weights()
            
        factor_df = pd.DataFrame(factor_matrix, columns=factor_names)
        returns_series = pd.Series(returns)
        
        effectiveness = {}
        
        for factor_name in factor_names:
            factor_values = factor_df[factor_name]
            
            # 1. 信息系数 (IC) - 排序相关性
            ic, ic_p_value = stats.spearmanr(factor_values, returns_series)
            
            # 2. 因子稳定性 - IC的标准差
            window_size = min(20, len(factor_values) // 3)
            rolling_ic = []
            for i in range(window_size, len(factor_values)):
                window_corr, _ = stats.spearmanr(
                    factor_values[i-window_size:i], 
                    returns_series[i-window_size:i]
                )
                if not np.isnan(window_corr):
                    rolling_ic.append(window_corr)
            
            ic_std = np.std(rolling_ic) if rolling_ic else 1.0
            ic_stability = 1 / (1 + ic_std)
            
            # 3. 因子单调性 - 因子值分组后的收益单调性
            try:
                # 将因子值分为5组
                factor_quantiles = pd.qcut(factor_values, 5, labels=False, duplicates='drop')
                group_returns = []
                for group in range(int(factor_quantiles.max()) + 1):
                    group_mask = factor_quantiles == group
                    if group_mask.sum() > 0:
                        group_ret = returns_series[group_mask].mean()
                        group_returns.append(group_ret)
                
                # 计算组间收益的单调性
                if len(group_returns) > 2:
                    monotonicity = abs(stats.spearmanr(range(len(group_returns)), group_returns)[0])
                else:
                    monotonicity = 0
            except:
                monotonicity = 0
            
            # 4. 综合评分
            abs_ic = abs(ic) if not np.isnan(ic) else 0
            significance = max(0, 1 - (ic_p_value if not np.isnan(ic_p_value) else 1))
            
            effectiveness[factor_name] = (
                0.4 * abs_ic +           # IC权重40%
                0.3 * ic_stability +     # 稳定性权重30%
                0.3 * monotonicity       # 单调性权重30%
            ) * significance
            
        return effectiveness
    
    def _get_default_weights(self) -> Dict[str, float]:
        """默认权重 - 均匀分配"""
        return {
            'momentum': 0.2,
            'volatility': 0.2,
            'volume_trend': 0.2,
            'price_position': 0.2,
            'trend_strength': 0.2
        }
    
    def update_weights(self) -> Dict[str, float]:
        """更新因子权重"""
        self.days_since_rebalance += 1
        
        if self.days_since_rebalance < self.rebalance_period and self.current_weights:
            return self.current_weights
            
        # 计算因子有效性
        effectiveness = self.calculate_factor_effectiveness()
        
        # 记录历史
        self.effectiveness_history.append({
            'day': len(self.effectiveness_history),
            'effectiveness': effectiveness.copy()
        })
        
        # 转换为权重
        total_effectiveness = sum(effectiveness.values())
        if total_effectiveness > 0:
            weights = {k: v / total_effectiveness for k, v in effectiveness.items()}
        else:
            weights = self._get_default_weights()
            
        # 权重约束：防止单个因子权重过高
        max_weight = 0.5
        for factor in weights:
            if weights[factor] > max_weight:
                excess = weights[factor] - max_weight
                weights[factor] = max_weight
                # 将多余权重分配给其他因子
                other_factors = [f for f in weights if f != factor]
                if other_factors:
                    excess_per_factor = excess / len(other_factors)
                    for other_factor in other_factors:
                        weights[other_factor] += excess_per_factor
        
        # 应用平滑处理
        if self.current_weights:
            smooth_factor = 0.3
            for factor_name in weights:
                if factor_name in self.current_weights:
                    weights[factor_name] = (
                        smooth_factor * weights[factor_name] + 
                        (1 - smooth_factor) * self.current_weights[factor_name]
                    )
        
        self.current_weights = weights
        self.days_since_rebalance = 0
        
        return weights
    
    def get_effectiveness_history(self) -> pd.DataFrame:
        """获取因子有效性历史"""
        if not self.effectiveness_history:
            return pd.DataFrame()
            
        records = []
        for record in self.effectiveness_history:
            row = {'day': record['day']}
            row.update(record['effectiveness'])
            records.append(row)
            
        return pd.DataFrame(records)


class EnhancedAdaptiveFactorStrategy(bt.Strategy):
    """增强版自适应因子策略"""
    
    params = (
        ('lookback_period', 60),
        ('rebalance_period', 20),
        ('signal_threshold', 0.3),
        ('position_size', 0.95),
        ('analyzer_ref', None),
    )
    
    def __init__(self):
        # 多样化技术因子
        self.factors = DiversifiedTechnicalFactors(self.data)
        
        # 增强版因子有效性分析器
        self.effectiveness_analyzer = EnhancedFactorEffectivenessAnalyzer(
            lookback_period=self.params.lookback_period,
            rebalance_period=self.params.rebalance_period
        )
        
        # 历史数据
        self.price_history = deque(maxlen=100)
        self.factor_history = []
        
        # 当前权重和信号
        self.current_weights = self.effectiveness_analyzer._get_default_weights()
        self.combined_signal = 0
        
    def next(self):
        """策略主逻辑"""
        current_date = self.data.datetime.date(0)
        current_price = self.data.close[0]
        
        # 记录价格历史
        self.price_history.append(current_price)
        
        # 计算未来收益
        future_return = None
        if len(self.price_history) >= 2:
            future_return = (current_price - self.price_history[-2]) / self.price_history[-2]
        
        # 获取当前因子值
        current_factors = self._get_current_factors()
        
        # 添加到因子有效性分析器
        if len(self.price_history) >= 2:
            prev_factors = self.factor_history[-1]['factors'] if self.factor_history else current_factors
            self.effectiveness_analyzer.add_factor_data(prev_factors, future_return)
        
        # 更新权重
        self.current_weights = self.effectiveness_analyzer.update_weights()
        
        # 计算综合信号
        self.combined_signal = self._calculate_adaptive_signal(current_factors)
        
        # 记录数据
        self._record_data(current_date, current_price, current_factors)
        
        # 执行交易逻辑
        self._execute_trading_logic()
        
    def _get_current_factors(self) -> Dict[str, float]:
        """获取当前因子值"""
        return {
            'momentum': self.factors.momentum[0] if not np.isnan(self.factors.momentum[0]) else 0.0,
            'volatility': self.factors.volatility[0] if not np.isnan(self.factors.volatility[0]) else 0.02,
            'volume_trend': self.factors.volume_trend[0] if not np.isnan(self.factors.volume_trend[0]) else 0.0,
            'price_position': self.factors.price_position[0] if not np.isnan(self.factors.price_position[0]) else 0.5,
            'trend_strength': self.factors.trend_strength[0] if not np.isnan(self.factors.trend_strength[0]) else 0.0,
        }
    
    def _normalize_factor(self, factor_name: str, value: float) -> float:
        """标准化因子值到[-1, 1]范围"""
        if factor_name == 'momentum':
            return np.tanh(value / 100)
        elif factor_name == 'volatility':
            # 波动率：低波动率为正信号
            normalized_vol = min(value / 0.05, 1.0)
            return 1 - normalized_vol
        elif factor_name == 'volume_trend':
            # 成交量趋势：正值为正信号
            return np.tanh(value)
        elif factor_name == 'price_position':
            # 价格位置：0.5为中性，0为超卖，1为超买
            return 2 * (value - 0.5)  # 转换到[-1, 1]
        elif factor_name == 'trend_strength':
            # 趋势强度：已经在[-0.5, 0.5]范围，扩展到[-1, 1]
            return 2 * value
        
        return 0.0
    
    def _calculate_adaptive_signal(self, factors: Dict[str, float]) -> float:
        """基于动态权重计算综合信号"""
        signal = 0.0
        
        for factor_name, factor_value in factors.items():
            if factor_name in self.current_weights:
                normalized_value = self._normalize_factor(factor_name, factor_value)
                weight = self.current_weights[factor_name]
                signal += weight * normalized_value
                
        return signal
    
    def _record_data(self, date, price, factors):
        """记录数据"""
        record = {
            'date': date,
            'price': price,
            'factors': factors.copy(),
            'weights': self.current_weights.copy(),
            'combined_signal': self.combined_signal,
            'position': self.position.size if self.position else 0
        }
        
        self.factor_history.append(record)
        
        if self.params.analyzer_ref:
            self.params.analyzer_ref.factor_history.append(record)
    
    def _execute_trading_logic(self):
        """执行交易逻辑"""
        if self.combined_signal > self.params.signal_threshold and not self.position:
            size = int(self.broker.getcash() * self.params.position_size / self.data.close[0])
            if size > 0:
                self.buy(size=size)
                
        elif self.combined_signal < -self.params.signal_threshold and self.position:
            self.sell(size=self.position.size)
    
    def get_analysis_data(self) -> Dict[str, pd.DataFrame]:
        """获取分析数据"""
        # 因子历史数据
        factor_df = pd.DataFrame(self.factor_history)
        
        # 权重变化历史
        weight_records = []
        for record in self.factor_history:
            weight_record = {'date': record['date']}
            weight_record.update(record['weights'])
            weight_records.append(weight_record)
        weight_df = pd.DataFrame(weight_records)
        
        # 因子有效性历史
        effectiveness_df = self.effectiveness_analyzer.get_effectiveness_history()
        
        return {
            'factors': factor_df,
            'weights': weight_df,
            'effectiveness': effectiveness_df
        }


class EnhancedFactorAnalyzer:
    """增强版因子分析器"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.cerebro = bt.Cerebro()
        self.strategy_instance = None
        self.factor_history = []
        
    def run_backtest(self, initial_cash: float = 10000.0) -> Dict[str, Any]:
        """运行回测"""
        self.cerebro.broker.setcash(initial_cash)
        self.cerebro.broker.setcommission(commission=0.001)
        
        data_feed = bt.feeds.PandasData(dataname=self.data)
        self.cerebro.adddata(data_feed)
        
        self.cerebro.addstrategy(EnhancedAdaptiveFactorStrategy, analyzer_ref=self)
        
        # 添加分析器
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        print(f'初始资金: {self.cerebro.broker.getvalue():.2f}')
        
        results = self.cerebro.run()
        if results:
            self.strategy_instance = results[0]
        
        final_value = self.cerebro.broker.getvalue()
        print(f'最终资金: {final_value:.2f}')
        print(f'总收益: {final_value - initial_cash:.2f} ({((final_value - initial_cash) / initial_cash * 100):.2f}%)')
        
        analysis_results = {
            'initial_cash': initial_cash,
            'final_value': final_value,
            'total_return': final_value - initial_cash,
            'return_pct': (final_value - initial_cash) / initial_cash * 100,
        }
        
        if results:
            strategy = results[0]
            trade_analysis = strategy.analyzers.trades.get_analysis()
            
            # 计算胜率
            total_closed_trades = trade_analysis.get('total', {}).get('closed', 0)
            won_trades = trade_analysis.get('won', {}).get('total', 0)
            win_rate = won_trades / total_closed_trades if total_closed_trades > 0 else 0
            
            analysis_results.update({
                'sharpe_ratio': strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0),
                'max_drawdown': strategy.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0),
                'total_trades': trade_analysis.get('total', {}).get('total', 0),
                'win_rate': win_rate,
                'strategy_instance': strategy,  # 保存策略实例以供后续分析使用
            })
        else:
            analysis_results.update({
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_trades': 0,
                'win_rate': 0,
                'strategy_instance': None,
            })
        
        return analysis_results
    
    def get_analysis_data(self) -> Dict[str, pd.DataFrame]:
        """获取详细分析数据"""
        if hasattr(self, 'strategy_instance') and self.strategy_instance is not None:
            return self.strategy_instance.get_analysis_data()
        return {}
    
    def plot_results(self):
        """绘制结果"""
        self.cerebro.plot(style='candlestick', volume=False)


if __name__ == "__main__":
    # 测试代码
    from data_fetcher import BinanceDataFetcher
    
    print("获取测试数据...")
    fetcher = BinanceDataFetcher()
    data = fetcher.fetch_recent_with_cache(days=90)
    
    if not data.empty:
        print("运行增强版自适应因子策略回测...")
        analyzer = EnhancedFactorAnalyzer(data)
        results = analyzer.run_backtest()
        
        print("\n=== 回测结果 ===")
        for key, value in results.items():
            print(f"{key}: {value}")
        
        print("\n增强版自适应因子策略测试完成！")
    else:
        print("无法获取测试数据")
