"""
输出管理器 - 为每次运行创建独立的输出文件夹并管理所有结果
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from paths import PROJECT_ROOT


class RunOutputManager:
    """运行输出管理器 - 为每次运行创建带时间戳的输出文件夹"""
    
    def __init__(self, base_output_dir: Optional[Path] = None):
        """
        初始化输出管理器
        
        Args:
            base_output_dir: 基础输出目录，默认为 PROJECT_ROOT / 'outputs'
        """
        if base_output_dir is None:
            base_output_dir = PROJECT_ROOT / 'outputs'
        
        self.base_output_dir = Path(base_output_dir)
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_output_dir / f"run_{self.run_timestamp}"
        
        # 创建运行目录和子目录
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir = self.run_dir / 'charts'
        self.data_dir = self.run_dir / 'data'
        self.reports_dir = self.run_dir / 'reports'
        
        # 创建子目录
        for dir_path in [self.charts_dir, self.data_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"✅ 创建运行输出目录: {self.run_dir}")
    
    def get_chart_path(self, filename: str) -> Path:
        """获取图表文件路径"""
        return self.charts_dir / filename
    
    def get_data_path(self, filename: str) -> Path:
        """获取数据文件路径"""
        return self.data_dir / filename
    
    def get_report_path(self, filename: str) -> Path:
        """获取报告文件路径"""
        return self.reports_dir / filename
    
    def save_strategy_results(self, strategy_name: str, results: Dict[str, Any]) -> Path:
        """保存策略结果到JSON文件"""
        filename = f"{strategy_name}_results.json"
        filepath = self.get_data_path(filename)
        
        # 处理不能序列化的对象
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 保存{strategy_name}策略结果: {filepath}")
        return filepath
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, subdir: str = 'data') -> Path:
        """保存DataFrame到CSV文件"""
        if subdir == 'data':
            filepath = self.get_data_path(filename)
        elif subdir == 'reports':
            filepath = self.get_report_path(filename)
        else:
            filepath = self.run_dir / subdir / filename
            filepath.parent.mkdir(exist_ok=True)
        
        df.to_csv(filepath, encoding='utf-8-sig')
        print(f"✅ 保存数据文件: {filepath}")
        return filepath
    
    def save_chart(self, filename: str, dpi: int = 300, **kwargs) -> Path:
        """保存当前matplotlib图表"""
        filepath = self.get_chart_path(filename)
        plt.savefig(str(filepath), dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', **kwargs)
        print(f"✅ 保存图表: {filepath}")
        return filepath
    
    def create_summary_report(self, strategy_results: Dict[str, Dict[str, Any]], 
                            factor_analysis: Optional[Dict[str, Any]] = None) -> Path:
        """创建汇总报告"""
        summary = {
            'run_info': {
                'timestamp': self.run_timestamp,
                'run_directory': str(self.run_dir),
                'analysis_time': datetime.now().isoformat()
            },
            'strategy_comparison': {},
            'key_metrics': {},
            'factor_analysis': factor_analysis or {}
        }
        
        # 提取关键指标
        key_metrics = ['return_pct', 'sharpe_ratio', 'max_drawdown', 'total_trades', 'win_rate']
        
        for strategy_name, results in strategy_results.items():
            strategy_metrics = {}
            for metric in key_metrics:
                if metric in results:
                    strategy_metrics[metric] = results[metric]
            
            summary['strategy_comparison'][strategy_name] = strategy_metrics
        
        # 找出最佳策略
        best_strategies = self._find_best_strategies(strategy_results)
        summary['best_strategies'] = best_strategies
        
        # 保存JSON格式
        json_path = self.get_report_path('summary_report.json')
        # 确保所有数据都可以序列化
        serializable_summary = self._make_serializable(summary)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, ensure_ascii=False, indent=2)
        
        # 创建文本格式报告
        txt_path = self.create_text_summary_report(summary)
        
        print(f"✅ 创建汇总报告: {json_path}")
        print(f"✅ 创建文本报告: {txt_path}")
        
        return json_path
    
    def create_text_summary_report(self, summary: Dict[str, Any]) -> Path:
        """创建文本格式的汇总报告"""
        txt_path = self.get_report_path('summary_report.txt')
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Rich 因子分析系统 - 运行报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 运行信息
            f.write("📊 运行信息\n")
            f.write("-" * 30 + "\n")
            f.write(f"运行时间: {summary['run_info']['timestamp']}\n")
            f.write(f"输出目录: {summary['run_info']['run_directory']}\n")
            f.write(f"分析完成时间: {summary['run_info']['analysis_time']}\n\n")
            
            # 策略对比
            f.write("📈 策略对比结果\n")
            f.write("-" * 30 + "\n")
            if summary['strategy_comparison']:
                # 表头
                f.write(f"{'策略':<12} {'收益率(%)':<12} {'夏普比率':<12} {'最大回撤(%)':<12} {'交易次数':<12} {'胜率(%)':<12}\n")
                f.write("-" * 75 + "\n")
                
                for strategy_name, metrics in summary['strategy_comparison'].items():
                    return_pct = metrics.get('return_pct', 0)
                    sharpe = metrics.get('sharpe_ratio', 0)
                    drawdown = metrics.get('max_drawdown', 0)
                    trades = metrics.get('total_trades', 0)
                    win_rate = metrics.get('win_rate', 0) * 100 if metrics.get('win_rate') else 0
                    
                    f.write(f"{strategy_name:<12} {return_pct:<12.2f} {sharpe:<12.3f} {drawdown:<12.2f} {trades:<12} {win_rate:<12.1f}\n")
                f.write("\n")
            
            # 最佳策略
            if 'best_strategies' in summary:
                f.write("🏆 最佳策略\n")
                f.write("-" * 30 + "\n")
                for metric, strategy in summary['best_strategies'].items():
                    f.write(f"{metric}: {strategy}\n")
                f.write("\n")
            
            # 因子分析
            if summary.get('factor_analysis'):
                f.write("🔍 因子分析\n")
                f.write("-" * 30 + "\n")
                factor_info = summary['factor_analysis']
                if 'correlation_summary' in factor_info:
                    f.write("因子相关性分析:\n")
                    for info in factor_info['correlation_summary']:
                        f.write(f"  {info}\n")
                f.write("\n")
            
            # 文件清单
            f.write("📁 输出文件清单\n")
            f.write("-" * 30 + "\n")
            f.write("图表文件:\n")
            for chart_file in self.charts_dir.glob("*.png"):
                f.write(f"  - {chart_file.name}\n")
            
            f.write("\n数据文件:\n")
            for data_file in self.data_dir.glob("*.csv"):
                f.write(f"  - {data_file.name}\n")
            for data_file in self.data_dir.glob("*.json"):
                f.write(f"  - {data_file.name}\n")
            
            f.write("\n报告文件:\n")
            for report_file in self.reports_dir.glob("*"):
                f.write(f"  - {report_file.name}\n")
        
        return txt_path
    
    def _find_best_strategies(self, strategy_results: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """找出各指标的最佳策略"""
        best_strategies = {}
        
        metrics = {
            '最高收益率': ('return_pct', 'max'),
            '最高夏普比率': ('sharpe_ratio', 'max'),
            '最小回撤': ('max_drawdown', 'min'),
            '最多交易次数': ('total_trades', 'max'),
            '最高胜率': ('win_rate', 'max')
        }
        
        for metric_name, (metric_key, direction) in metrics.items():
            values = []
            for strategy_name, results in strategy_results.items():
                if metric_key in results and results[metric_key] is not None:
                    values.append((strategy_name, results[metric_key]))
            
            if values:
                if direction == 'max':
                    best_strategy = max(values, key=lambda x: x[1])
                else:
                    best_strategy = min(values, key=lambda x: x[1])
                best_strategies[metric_name] = f"{best_strategy[0]} ({best_strategy[1]:.3f})"
        
        return best_strategies
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可JSON序列化的格式"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)  # 对于复杂对象，转换为字符串
        else:
            try:
                json.dumps(obj)  # 测试是否可序列化
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    def get_run_summary(self) -> Dict[str, Any]:
        """获取当前运行的汇总信息"""
        return {
            'run_timestamp': self.run_timestamp,
            'run_directory': str(self.run_dir),
            'charts_directory': str(self.charts_dir),
            'data_directory': str(self.data_dir),
            'reports_directory': str(self.reports_dir)
        }


def create_run_output_manager() -> RunOutputManager:
    """创建运行输出管理器的工厂函数"""
    return RunOutputManager()
