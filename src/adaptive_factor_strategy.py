"""
自适应因子策略 - 两层架构设计
第一层：动态识别有效因子
第二层：基于有效因子进行交易
"""
import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class FactorEffectivenessAnalyzer:
    """因子有效性分析器 - 第一层"""
    
    def __init__(self, lookback_period: int = 60, rebalance_period: int = 20):
        """
        Args:
            lookback_period: 回看期，用于计算因子有效性
            rebalance_period: 重新平衡周期，多少期重新评估因子
        """
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
        
        for i in range(len(self.factor_history) - 1):  # 排除最后一个（没有未来收益）
            factors = [self.factor_history[i]['factors'][name] for name in factor_names]
            future_ret = self.factor_history[i + 1]['future_return'] if i + 1 < len(self.factor_history) else 0
            
            if future_ret is not None and not np.isnan(future_ret):
                factor_matrix.append(factors)
                returns.append(future_ret)
        
        if len(factor_matrix) < 10:  # 数据不足
            return self._get_default_weights()
            
        factor_df = pd.DataFrame(factor_matrix, columns=factor_names)
        returns_series = pd.Series(returns)
        
        effectiveness = {}
        
        for factor_name in factor_names:
            factor_values = factor_df[factor_name]
            
            # 1. 计算相关系数
            correlation, p_value = stats.pearsonr(factor_values, returns_series)
            
            # 2. 计算信息系数 (IC) - 排序相关性
            ic, ic_p_value = stats.spearmanr(factor_values, returns_series)
            
            # 3. 计算因子稳定性 - IC的标准差
            window_size = 20
            rolling_ic = []
            for i in range(window_size, len(factor_values)):
                window_corr, _ = stats.spearmanr(
                    factor_values[i-window_size:i], 
                    returns_series[i-window_size:i]
                )
                if not np.isnan(window_corr):
                    rolling_ic.append(window_corr)
            
            ic_std = np.std(rolling_ic) if rolling_ic else 1.0
            ic_stability = 1 / (1 + ic_std)  # 标准差越小，稳定性越高
            
            # 4. 综合评分
            # 考虑相关性强度、显著性和稳定性
            abs_ic = abs(ic) if not np.isnan(ic) else 0
            significance = 1 - min(ic_p_value, 0.2) / 0.2 if not np.isnan(ic_p_value) else 0
            
            effectiveness[factor_name] = abs_ic * significance * ic_stability
            
        return effectiveness
    
    def _get_default_weights(self) -> Dict[str, float]:
        """默认权重"""
        return {
            'rsi': 0.25,
            'macd': 0.25, 
            'momentum': 0.25,
            'volatility': 0.25
        }
    
    def update_weights(self) -> Dict[str, float]:
        """更新因子权重"""
        self.days_since_rebalance += 1
        
        # 如果还没到重新平衡时间，返回当前权重
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
            
        # 应用平滑处理，避免权重剧烈变化
        if self.current_weights:
            smooth_factor = 0.3  # 新权重占30%，旧权重占70%
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


class AdaptiveTechnicalFactors(bt.Indicator):
    """自适应技术因子计算器"""
    
    lines = ('rsi', 'macd', 'momentum', 'volatility')
    
    params = (
        ('rsi_period', 14),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('momentum_period', 10),
        ('volatility_period', 20),
    )
    
    def __init__(self):
        # RSI
        self.lines.rsi = bt.indicators.RSI(
            self.data.close, 
            period=self.params.rsi_period
        )
        
        # MACD
        macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        self.lines.macd = macd.macd - macd.signal
        
        # 动量
        self.lines.momentum = bt.indicators.Momentum(
            self.data.close,
            period=self.params.momentum_period
        )
        
        # 波动率
        self.lines.volatility = bt.indicators.StandardDeviation(
            self.data.close,
            period=self.params.volatility_period
        )


class AdaptiveFactorStrategy(bt.Strategy):
    """自适应因子策略 - 第二层"""
    
    params = (
        ('lookback_period', 60),    # 因子有效性回看期
        ('rebalance_period', 20),   # 权重重新平衡周期
        ('signal_threshold', 0.3),  # 交易信号阈值
        ('position_size', 0.95),    # 仓位大小
        ('analyzer_ref', None),     # 分析器引用
    )
    
    def __init__(self):
        # 技术因子
        self.factors = AdaptiveTechnicalFactors(self.data)
        
        # 因子有效性分析器
        self.effectiveness_analyzer = FactorEffectivenessAnalyzer(
            lookback_period=self.params.lookback_period,
            rebalance_period=self.params.rebalance_period
        )
        
        # 历史数据
        self.price_history = deque(maxlen=100)
        self.factor_history = []
        self.weight_history = []
        
        # 当前权重和信号
        self.current_weights = self.effectiveness_analyzer._get_default_weights()
        self.combined_signal = 0
        
    def next(self):
        """策略主逻辑"""
        current_date = self.data.datetime.date(0)
        current_price = self.data.close[0]
        
        # 记录价格历史
        self.price_history.append(current_price)
        
        # 计算未来收益（用于训练因子有效性）
        future_return = None
        if len(self.price_history) >= 2:
            future_return = (current_price - self.price_history[-2]) / self.price_history[-2]
        
        # 获取当前因子值
        current_factors = self._get_current_factors()
        
        # 添加到因子有效性分析器
        if len(self.price_history) >= 2:
            prev_factors = self.factor_history[-1]['factors'] if self.factor_history else current_factors
            self.effectiveness_analyzer.add_factor_data(prev_factors, future_return)
        
        # 更新权重（第一层：识别有效因子）
        self.current_weights = self.effectiveness_analyzer.update_weights()
        
        # 计算综合信号（第二层：基于有效因子交易）
        self.combined_signal = self._calculate_adaptive_signal(current_factors)
        
        # 记录数据
        self._record_data(current_date, current_price, current_factors)
        
        # 执行交易逻辑
        self._execute_trading_logic()
        
    def _get_current_factors(self) -> Dict[str, float]:
        """获取当前因子值"""
        return {
            'rsi': self.factors.rsi[0] if not np.isnan(self.factors.rsi[0]) else 50.0,
            'macd': self.factors.macd[0] if not np.isnan(self.factors.macd[0]) else 0.0,
            'momentum': self.factors.momentum[0] if not np.isnan(self.factors.momentum[0]) else 0.0,
            'volatility': self.factors.volatility[0] if not np.isnan(self.factors.volatility[0]) else 0.02,
        }
    
    def _normalize_factor(self, factor_name: str, value: float) -> float:
        """标准化因子值到[-1, 1]范围"""
        if factor_name == 'rsi':
            # RSI: 0-100 -> -1到1
            return (value - 50) / 50
        elif factor_name == 'macd':
            # MACD: 简单标准化
            return np.tanh(value * 10)  # tanh函数限制在[-1,1]
        elif factor_name == 'momentum':
            # 动量: 标准化
            return np.tanh(value / 100)
        elif factor_name == 'volatility':
            # 波动率: 转换为信号（低波动率为正信号）
            normalized_vol = min(value / 0.05, 1.0)  # 0.05为高波动率阈值
            return 1 - normalized_vol  # 反转，低波动率给正信号
        
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
        
        # 如果有分析器引用，也保存到分析器中
        if self.params.analyzer_ref:
            self.params.analyzer_ref.factor_history.append(record)
    
    def _execute_trading_logic(self):
        """执行交易逻辑"""
        # 强买入信号
        if self.combined_signal > self.params.signal_threshold and not self.position:
            size = int(self.broker.getcash() * self.params.position_size / self.data.close[0])
            if size > 0:
                self.buy(size=size)
                
        # 强卖出信号
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


class AdaptiveFactorAnalyzer:
    """自适应因子分析器"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.cerebro = bt.Cerebro()
        self.strategy_instance = None
        self.factor_history = []
        
    def run_backtest(self, initial_cash: float = 10000.0) -> Dict[str, Any]:
        """运行回测"""
        # 设置初始资金
        self.cerebro.broker.setcash(initial_cash)
        
        # 设置手续费
        self.cerebro.broker.setcommission(commission=0.001)
        
        # 添加数据
        data_feed = bt.feeds.PandasData(dataname=self.data)
        self.cerebro.adddata(data_feed)
        
        # 添加策略
        self.cerebro.addstrategy(AdaptiveFactorStrategy, analyzer_ref=self)
        
        # 添加分析器
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        print(f'初始资金: {self.cerebro.broker.getvalue():.2f}')
        
        # 运行回测
        results = self.cerebro.run()
        if results:
            self.strategy_instance = results[0]
        
        final_value = self.cerebro.broker.getvalue()
        print(f'最终资金: {final_value:.2f}')
        print(f'总收益: {final_value - initial_cash:.2f} ({((final_value - initial_cash) / initial_cash * 100):.2f}%)')
        
        # 获取分析结果
        analysis_results = {
            'initial_cash': initial_cash,
            'final_value': final_value,
            'total_return': final_value - initial_cash,
            'return_pct': (final_value - initial_cash) / initial_cash * 100,
        }
        
        if results:
            strategy = results[0]
            analysis_results.update({
                'sharpe_ratio': strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0),
                'max_drawdown': strategy.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0),
                'total_trades': strategy.analyzers.trades.get_analysis().get('total', {}).get('total', 0),
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
        print("运行自适应因子策略回测...")
        analyzer = AdaptiveFactorAnalyzer(data)
        results = analyzer.run_backtest()
        
        print("\n=== 回测结果 ===")
        for key, value in results.items():
            print(f"{key}: {value}")
        
        # 获取分析数据
        analysis_data = analyzer.get_analysis_data()
        
        if 'weights' in analysis_data and not analysis_data['weights'].empty:
            print("\n=== 最终权重分布 ===")
            final_weights = analysis_data['weights'].iloc[-1]
            for factor in ['rsi', 'macd', 'momentum', 'volatility']:
                if factor in final_weights:
                    print(f"{factor}: {final_weights[factor]:.3f}")
        
        print("\n自适应因子策略测试完成！")
    else:
        print("无法获取测试数据")
