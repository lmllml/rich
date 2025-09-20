"""
因子分析策略 - 使用 Backtrader 实现多因子分析
"""
import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, Any


class TechnicalFactors(bt.Indicator):
    """技术因子计算器"""
    
    lines = ('rsi', 'macd', 'bb_upper', 'bb_lower', 'momentum', 'volatility')
    
    params = (
        ('rsi_period', 14),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('bb_period', 20),
        ('bb_std', 2),
        ('momentum_period', 10),
        ('volatility_period', 20),
    )
    
    def __init__(self):
        # RSI 相对强弱指标
        self.lines.rsi = bt.indicators.RSI(
            self.data.close, 
            period=self.params.rsi_period
        )
        
        # MACD 指标
        macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        self.lines.macd = macd.macd - macd.signal
        
        # 布林带
        bb = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_std
        )
        self.lines.bb_upper = bb.top
        self.lines.bb_lower = bb.bot
        
        # 动量因子
        self.lines.momentum = bt.indicators.Momentum(
            self.data.close,
            period=self.params.momentum_period
        )
        
        # 波动率因子
        self.lines.volatility = bt.indicators.StandardDeviation(
            self.data.close,
            period=self.params.volatility_period
        )


class FactorAnalysisStrategy(bt.Strategy):
    """多因子分析策略"""
    
    params = (
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('bb_threshold', 0.8),
        ('momentum_threshold', 0),
        ('volatility_threshold', 0.02),
        ('position_size', 0.95),  # 仓位大小
        ('analyzer_ref', None),  # 分析器引用
    )
    
    def __init__(self):
        # 初始化技术因子
        self.factors = TechnicalFactors(self.data)
        
        # 因子信号
        self.rsi_signal = 0
        self.macd_signal = 0
        self.bb_signal = 0
        self.momentum_signal = 0
        self.volatility_signal = 0
        
        # 综合信号
        self.combined_signal = 0
        
        # 记录因子值用于分析
        self.factor_history = []
        
    def next(self):
        """策略主逻辑"""
        current_date = self.data.datetime.date(0)
        current_price = self.data.close[0]
        
        # 计算各因子信号
        self._calculate_factor_signals()
        
        # 计算综合信号
        self._calculate_combined_signal()
        
        # 记录因子数据
        self._record_factor_data(current_date, current_price)
        
        # 如果有分析器引用，也保存到分析器中
        if self.params.analyzer_ref:
            self.params.analyzer_ref.factor_history.append(self.factor_history[-1] if self.factor_history else {})
        
        # 执行交易逻辑
        self._execute_trading_logic()
        
    def _calculate_factor_signals(self):
        """计算各因子信号"""
        # RSI 信号
        if self.factors.rsi[0] < self.params.rsi_oversold:
            self.rsi_signal = 1  # 超卖，买入信号
        elif self.factors.rsi[0] > self.params.rsi_overbought:
            self.rsi_signal = -1  # 超买，卖出信号
        else:
            self.rsi_signal = 0
            
        # MACD 信号
        if self.factors.macd[0] > 0:
            self.macd_signal = 1  # 金叉，买入信号
        else:
            self.macd_signal = -1  # 死叉，卖出信号
            
        # 布林带信号
        close_price = self.data.close[0]
        bb_position = (close_price - self.factors.bb_lower[0]) / (
            self.factors.bb_upper[0] - self.factors.bb_lower[0]
        )
        
        if bb_position < (1 - self.params.bb_threshold):
            self.bb_signal = 1  # 接近下轨，买入信号
        elif bb_position > self.params.bb_threshold:
            self.bb_signal = -1  # 接近上轨，卖出信号
        else:
            self.bb_signal = 0
            
        # 动量信号
        if self.factors.momentum[0] > self.params.momentum_threshold:
            self.momentum_signal = 1  # 正动量，买入信号
        else:
            self.momentum_signal = -1  # 负动量，卖出信号
            
        # 波动率信号（低波动率有利于趋势延续）
        if self.factors.volatility[0] < self.params.volatility_threshold:
            self.volatility_signal = 1  # 低波动率，有利信号
        else:
            self.volatility_signal = 0  # 高波动率，中性信号
    
    def _calculate_combined_signal(self):
        """计算综合信号"""
        # 简单加权平均
        weights = {
            'rsi': 0.2,
            'macd': 0.3,
            'bb': 0.2,
            'momentum': 0.2,
            'volatility': 0.1
        }
        
        self.combined_signal = (
            weights['rsi'] * self.rsi_signal +
            weights['macd'] * self.macd_signal +
            weights['bb'] * self.bb_signal +
            weights['momentum'] * self.momentum_signal +
            weights['volatility'] * self.volatility_signal
        )
    
    def _record_factor_data(self, date, price):
        """记录因子数据用于后续分析"""
        factor_data = {
            'date': date,
            'price': price,
            'rsi': self.factors.rsi[0],
            'macd': self.factors.macd[0],
            'bb_upper': self.factors.bb_upper[0],
            'bb_lower': self.factors.bb_lower[0],
            'momentum': self.factors.momentum[0],
            'volatility': self.factors.volatility[0],
            'rsi_signal': self.rsi_signal,
            'macd_signal': self.macd_signal,
            'bb_signal': self.bb_signal,
            'momentum_signal': self.momentum_signal,
            'volatility_signal': self.volatility_signal,
            'combined_signal': self.combined_signal,
            'position': self.position.size if self.position else 0
        }
        self.factor_history.append(factor_data)
    
    def _execute_trading_logic(self):
        """执行交易逻辑"""
        # 强买入信号
        if self.combined_signal > 0.3 and not self.position:
            size = int(self.broker.getcash() * self.params.position_size / self.data.close[0])
            self.buy(size=size)
            
        # 强卖出信号
        elif self.combined_signal < -0.3 and self.position:
            self.sell(size=self.position.size)
    
    def get_factor_analysis(self) -> pd.DataFrame:
        """获取因子分析结果"""
        return pd.DataFrame(self.factor_history)


class FactorAnalyzer:
    """因子分析器"""
    
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
        self.cerebro.broker.setcommission(commission=0.001)  # 0.1% 手续费
        
        # 添加数据
        data_feed = bt.feeds.PandasData(dataname=self.data)
        self.cerebro.adddata(data_feed)
        
        # 添加策略，传递分析器引用
        self.cerebro.addstrategy(FactorAnalysisStrategy, analyzer_ref=self)
        
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
    
    def get_factor_data(self) -> pd.DataFrame:
        """获取因子数据"""
        if self.factor_history:
            return pd.DataFrame(self.factor_history)
        return pd.DataFrame()
    
    def plot_results(self):
        """绘制回测结果"""
        self.cerebro.plot(style='candlestick', volume=False)
