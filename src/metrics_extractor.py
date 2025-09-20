"""
指标提取器 - 从策略结果中提取和计算关键指标
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple


class PerformanceMetricsExtractor:
    """性能指标提取器"""
    
    @staticmethod
    def extract_strategy_metrics(results: Dict[str, Any], 
                               factor_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        从策略结果中提取关键指标
        
        Args:
            results: 策略运行结果
            factor_data: 因子数据DataFrame
            
        Returns:
            包含所有关键指标的字典
        """
        metrics = {}
        
        # 基础指标（从results中直接提取）
        basic_metrics = [
            'return_pct', 'sharpe_ratio', 'max_drawdown', 'total_trades',
            'win_rate', 'final_portfolio_value', 'initial_cash'
        ]
        
        for metric in basic_metrics:
            if metric in results:
                metrics[metric] = results[metric]
        
        # 计算额外指标
        if factor_data is not None and not factor_data.empty:
            additional_metrics = PerformanceMetricsExtractor._calculate_additional_metrics(
                factor_data, results
            )
            metrics.update(additional_metrics)
        
        # 计算风险调整后指标
        risk_metrics = PerformanceMetricsExtractor._calculate_risk_metrics(results)
        metrics.update(risk_metrics)
        
        return metrics
    
    @staticmethod
    def _calculate_additional_metrics(factor_data: pd.DataFrame, 
                                   results: Dict[str, Any]) -> Dict[str, Any]:
        """计算额外的性能指标"""
        additional_metrics = {}
        
        try:
            # 交易频率相关指标
            if 'position' in factor_data.columns:
                position_changes = factor_data['position'].diff().abs()
                trade_frequency = position_changes[position_changes > 0].count()
                
                additional_metrics['trade_frequency'] = trade_frequency
                additional_metrics['avg_holding_period'] = len(factor_data) / max(trade_frequency, 1)
            
            # 信号相关指标
            if 'combined_signal' in factor_data.columns:
                signals = factor_data['combined_signal']
                
                # 信号强度统计
                additional_metrics['signal_strength_mean'] = signals.mean()
                additional_metrics['signal_strength_std'] = signals.std()
                additional_metrics['signal_strength_max'] = signals.max()
                additional_metrics['signal_strength_min'] = signals.min()
                
                # 买卖信号统计
                buy_signals = (signals > 0.3).sum()
                sell_signals = (signals < -0.3).sum()
                neutral_signals = ((signals >= -0.3) & (signals <= 0.3)).sum()
                
                total_signals = len(signals)
                additional_metrics['buy_signal_ratio'] = buy_signals / total_signals if total_signals > 0 else 0
                additional_metrics['sell_signal_ratio'] = sell_signals / total_signals if total_signals > 0 else 0
                additional_metrics['neutral_signal_ratio'] = neutral_signals / total_signals if total_signals > 0 else 0
            
            # 价格相关指标
            if 'price' in factor_data.columns:
                prices = factor_data['price']
                returns = prices.pct_change().dropna()
                
                if len(returns) > 0:
                    # 价格波动率
                    additional_metrics['price_volatility'] = returns.std() * np.sqrt(252 * 6)  # 年化波动率（4小时数据）
                    
                    # 最大单日收益和亏损
                    additional_metrics['max_single_return'] = returns.max()
                    additional_metrics['min_single_return'] = returns.min()
                    
                    # 正收益天数比例
                    positive_returns = (returns > 0).sum()
                    additional_metrics['positive_return_ratio'] = positive_returns / len(returns)
            
            # 持仓相关指标
            if 'position' in factor_data.columns:
                positions = factor_data['position']
                
                # 持仓统计
                long_positions = (positions > 0).sum()
                short_positions = (positions < 0).sum()
                flat_positions = (positions == 0).sum()
                
                total_periods = len(positions)
                additional_metrics['long_position_ratio'] = long_positions / total_periods
                additional_metrics['short_position_ratio'] = short_positions / total_periods
                additional_metrics['flat_position_ratio'] = flat_positions / total_periods
                
                # 平均持仓大小
                non_zero_positions = positions[positions != 0]
                if len(non_zero_positions) > 0:
                    additional_metrics['avg_position_size'] = non_zero_positions.abs().mean()
                    additional_metrics['max_position_size'] = non_zero_positions.abs().max()
        
        except Exception as e:
            print(f"⚠️ 计算额外指标时出错: {e}")
        
        return additional_metrics
    
    @staticmethod
    def _calculate_risk_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
        """计算风险相关指标"""
        risk_metrics = {}
        
        try:
            # 基于现有指标计算风险调整指标
            return_pct = results.get('return_pct', 0)
            max_drawdown = results.get('max_drawdown', 0)
            sharpe_ratio = results.get('sharpe_ratio', 0)
            total_trades = results.get('total_trades', 0)
            win_rate = results.get('win_rate', 0)
            
            # Calmar比率 (年化收益率 / 最大回撤)
            if max_drawdown != 0 and max_drawdown is not None:
                risk_metrics['calmar_ratio'] = abs(return_pct) / abs(max_drawdown)
            else:
                risk_metrics['calmar_ratio'] = 0
            
            # 收益风险比
            if max_drawdown != 0 and max_drawdown is not None:
                risk_metrics['return_risk_ratio'] = return_pct / abs(max_drawdown)
            else:
                risk_metrics['return_risk_ratio'] = 0
            
            # 交易效率 (收益率 / 交易次数)
            if total_trades > 0:
                risk_metrics['trade_efficiency'] = return_pct / total_trades
            else:
                risk_metrics['trade_efficiency'] = 0
            
            # 胜率调整收益 (收益率 * 胜率)
            if win_rate is not None:
                risk_metrics['win_rate_adjusted_return'] = return_pct * win_rate
            else:
                risk_metrics['win_rate_adjusted_return'] = 0
            
            # 风险评级
            risk_metrics['risk_grade'] = PerformanceMetricsExtractor._calculate_risk_grade(
                return_pct, max_drawdown, sharpe_ratio, win_rate
            )
        
        except Exception as e:
            print(f"⚠️ 计算风险指标时出错: {e}")
        
        return risk_metrics
    
    @staticmethod
    def _calculate_risk_grade(return_pct: float, max_drawdown: float, 
                           sharpe_ratio: float, win_rate: float) -> str:
        """计算风险评级"""
        try:
            score = 0
            
            # 收益率评分 (30%)
            if return_pct > 20:
                score += 30
            elif return_pct > 10:
                score += 20
            elif return_pct > 0:
                score += 10
            
            # 最大回撤评分 (25%)
            if abs(max_drawdown) < 5:
                score += 25
            elif abs(max_drawdown) < 10:
                score += 20
            elif abs(max_drawdown) < 20:
                score += 15
            elif abs(max_drawdown) < 30:
                score += 10
            
            # 夏普比率评分 (25%)
            if sharpe_ratio > 2:
                score += 25
            elif sharpe_ratio > 1.5:
                score += 20
            elif sharpe_ratio > 1:
                score += 15
            elif sharpe_ratio > 0.5:
                score += 10
            
            # 胜率评分 (20%)
            if win_rate and win_rate > 0.6:
                score += 20
            elif win_rate and win_rate > 0.5:
                score += 15
            elif win_rate and win_rate > 0.4:
                score += 10
            
            # 根据总分确定评级
            if score >= 80:
                return "A+ (优秀)"
            elif score >= 70:
                return "A (良好)"
            elif score >= 60:
                return "B+ (中等偏上)"
            elif score >= 50:
                return "B (中等)"
            elif score >= 40:
                return "C+ (中等偏下)"
            elif score >= 30:
                return "C (较差)"
            else:
                return "D (差)"
        
        except Exception:
            return "未知"
    
    @staticmethod
    def create_metrics_summary(all_strategy_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """创建所有策略的指标汇总"""
        summary = {
            'strategy_count': len(all_strategy_metrics),
            'metrics_comparison': {},
            'best_performers': {},
            'risk_analysis': {}
        }
        
        if not all_strategy_metrics:
            return summary
        
        # 获取所有可用的指标名称
        all_metrics = set()
        for metrics in all_strategy_metrics.values():
            all_metrics.update(metrics.keys())
        
        # 对比各策略的指标
        for metric in all_metrics:
            values = []
            for strategy_name, metrics in all_strategy_metrics.items():
                if metric in metrics and metrics[metric] is not None:
                    try:
                        # 尝试转换为数值
                        value = float(metrics[metric])
                        values.append((strategy_name, value))
                    except (ValueError, TypeError):
                        # 非数值指标跳过
                        pass
            
            if values:
                summary['metrics_comparison'][metric] = {
                    'max': max(values, key=lambda x: x[1]),
                    'min': min(values, key=lambda x: x[1]),
                    'avg': sum(v[1] for v in values) / len(values)
                }
        
        # 找出最佳表现者
        key_metrics = ['return_pct', 'sharpe_ratio', 'calmar_ratio', 'win_rate']
        for metric in key_metrics:
            if metric in summary['metrics_comparison']:
                best = summary['metrics_comparison'][metric]['max']
                summary['best_performers'][metric] = {
                    'strategy': best[0],
                    'value': best[1]
                }
        
        # 风险分析
        risk_grades = {}
        for strategy_name, metrics in all_strategy_metrics.items():
            if 'risk_grade' in metrics:
                risk_grades[strategy_name] = metrics['risk_grade']
        
        summary['risk_analysis'] = risk_grades
        
        return summary


class FactorAnalysisExtractor:
    """因子分析提取器"""
    
    @staticmethod
    def analyze_factor_correlation(factor_data: pd.DataFrame) -> Dict[str, Any]:
        """分析因子相关性"""
        analysis = {
            'correlation_matrix': {},
            'correlation_summary': [],
            'factor_statistics': {}
        }
        
        try:
            # 基础因子列
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
                analysis['correlation_matrix'] = correlation_matrix.to_dict()
                
                # 生成相关性摘要
                for i, factor1 in enumerate(available_factors):
                    for j, factor2 in enumerate(available_factors):
                        if i < j:  # 避免重复
                            corr_value = correlation_matrix.loc[factor1, factor2]
                            if abs(corr_value) > 0.7:
                                analysis['correlation_summary'].append(
                                    f"{factor1} 与 {factor2} 高度相关 (r={corr_value:.3f})"
                                )
                            elif abs(corr_value) < 0.3:
                                analysis['correlation_summary'].append(
                                    f"{factor1} 与 {factor2} 低相关性 (r={corr_value:.3f})"
                                )
                
                # 因子统计信息
                for factor in available_factors:
                    factor_series = factor_data[factor]
                    analysis['factor_statistics'][factor] = {
                        'mean': factor_series.mean(),
                        'std': factor_series.std(),
                        'min': factor_series.min(),
                        'max': factor_series.max(),
                        'skew': factor_series.skew(),
                        'kurtosis': factor_series.kurtosis()
                    }
        
        except Exception as e:
            print(f"⚠️ 因子相关性分析时出错: {e}")
        
        return analysis
    
    @staticmethod
    def analyze_signal_effectiveness(factor_data: pd.DataFrame) -> Dict[str, Any]:
        """分析信号有效性"""
        analysis = {
            'signal_distribution': {},
            'signal_effectiveness': {},
            'trading_signals': {}
        }
        
        try:
            if 'combined_signal' in factor_data.columns:
                signals = factor_data['combined_signal']
                
                # 信号分布
                analysis['signal_distribution'] = {
                    'mean': signals.mean(),
                    'std': signals.std(),
                    'min': signals.min(),
                    'max': signals.max(),
                    'quantiles': {
                        '25%': signals.quantile(0.25),
                        '50%': signals.quantile(0.50),
                        '75%': signals.quantile(0.75)
                    }
                }
                
                # 交易信号统计
                buy_signals = (signals > 0.3).sum()
                sell_signals = (signals < -0.3).sum()
                neutral_signals = ((signals >= -0.3) & (signals <= 0.3)).sum()
                total_signals = len(signals)
                
                analysis['trading_signals'] = {
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'neutral_signals': neutral_signals,
                    'buy_ratio': buy_signals / total_signals,
                    'sell_ratio': sell_signals / total_signals,
                    'neutral_ratio': neutral_signals / total_signals
                }
                
                # 信号有效性（如果有价格数据）
                if 'price' in factor_data.columns:
                    price_returns = factor_data['price'].pct_change().shift(-1)  # 下一期收益
                    
                    # 买入信号的平均收益
                    buy_mask = signals > 0.3
                    if buy_mask.sum() > 0:
                        buy_returns = price_returns[buy_mask].dropna()
                        analysis['signal_effectiveness']['buy_signal_avg_return'] = buy_returns.mean()
                        analysis['signal_effectiveness']['buy_signal_win_rate'] = (buy_returns > 0).mean()
                    
                    # 卖出信号的平均收益
                    sell_mask = signals < -0.3
                    if sell_mask.sum() > 0:
                        sell_returns = price_returns[sell_mask].dropna()
                        analysis['signal_effectiveness']['sell_signal_avg_return'] = sell_returns.mean()
                        analysis['signal_effectiveness']['sell_signal_win_rate'] = (sell_returns < 0).mean()  # 卖出信号希望价格下跌
        
        except Exception as e:
            print(f"⚠️ 信号有效性分析时出错: {e}")
        
        return analysis
