"""
因子筛选与相关性分析 - 从因子数据中挖掘与未来价格变动最相关的因子
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple


class FactorCorrelationAnalyzer:
    """基于价格序列与候选因子，评估与未来收益的相关性并排序。"""

    @staticmethod
    def prepare_dataframe(raw_factor_data: pd.DataFrame) -> pd.DataFrame:
        """展开嵌套因子列，统一价格列，并补充一批通用技术因子。"""
        if raw_factor_data is None or raw_factor_data.empty:
            return pd.DataFrame()

        df = raw_factor_data.copy()

        # 统一价格列
        if 'price' not in df.columns:
            if 'close' in df.columns:
                df['price'] = df['close']
            else:
                # 无价格无法分析
                return pd.DataFrame()

        # 展开嵌套因子
        if 'factors' in df.columns:
            try:
                expanded = pd.json_normalize(df['factors'])
                df = pd.concat([df.drop(columns=['factors']), expanded], axis=1)
            except Exception:
                pass

        # 计算通用附加因子（仅基于 price）
        price = df['price'].astype(float)

        # 收益与动量类
        df['ret_1'] = price.pct_change(1)
        df['ret_3'] = price.pct_change(3)
        df['ret_6'] = price.pct_change(6)
        df['roc_10'] = price.pct_change(10)
        df['momentum_10'] = price.diff(10)

        # 波动与带宽位置
        roll20 = price.rolling(20)
        ma20 = roll20.mean()
        std20 = roll20.std(ddof=0)
        df['vol_20'] = std20 / ma20.replace(0, np.nan)
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        df['bb_pos_20'] = (price - lower) / (upper - lower)

        # 均线与坡度
        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = macd - signal
        df['ema_ratio_12_26'] = ema12 / ema26.replace(0, np.nan)
        df['sma_slope_20'] = ma20.diff()

        # RSI 14（无依赖实现）
        delta = price.diff()
        gain = delta.where(delta > 0.0, 0.0)
        loss = -delta.where(delta < 0.0, 0.0)
        roll_u = gain.ewm(alpha=1/14, adjust=False).mean()
        roll_d = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = roll_u / roll_d.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # 趋势一致性（近14根上涨比例，映射到[-1,1]）
        up = (price.diff() > 0).astype(float)
        df['trend_consistency_14'] = (up.rolling(14).mean() - 0.5) * 2

        # Z-Score（20）
        df['zscore_20'] = (price - ma20) / std20.replace(0, np.nan)

        return df

    @staticmethod
    def rank_factors(df: pd.DataFrame, horizons: List[int] | None = None,
                     method: str = 'spearman', min_samples: int = 300) -> Dict[str, Any]:
        """计算与未来收益的相关性，并按绝对值排序；附带滚动稳健性评价。"""
        if df is None or df.empty:
            return {'error': 'empty dataframe'}

        if horizons is None:
            horizons = [1, 3, 6]

        work = df.copy()
        price_col = 'price' if 'price' in work.columns else 'close'
        if price_col not in work.columns:
            return {'error': 'missing price'}

        # 构造未来收益目标
        for h in horizons:
            work[f'ret_future_h{h}'] = work[price_col].pct_change().shift(-h)

        # 候选因子列：排除非数值及目标列
        exclude_prefix = tuple(['ret_future_h'])
        numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
        factor_cols = [c for c in numeric_cols if not c.startswith(exclude_prefix) and c not in ['price', 'close']]

        # 清洗
        work = work[factor_cols + [f'ret_future_h{h}' for h in horizons]].dropna()
        if len(work) < min_samples:
            return {'error': f'samples too few: {len(work)}'}

        # 整体相关性
        overall: Dict[str, Dict[str, float]] = {}
        for h in horizons:
            target = work[f'ret_future_h{h}']
            corr = work[factor_cols].corrwith(target, method=method).fillna(0.0)
            overall[f'h{h}'] = corr.to_dict()

        # 滚动稳健性（取最短窗口）
        window = max(100, min(300, len(work) // 5))
        stability: Dict[str, Dict[str, float]] = {}
        for col in factor_cols:
            s_stats: Dict[str, float] = {}
            for h in horizons:
                # 注意：rolling.corr 不支持 method 参数；此处使用 Pearson 评估滚动稳定性
                r = work[col].rolling(window).corr(work[f'ret_future_h{h}'])
                s_stats[f'h{h}_mean_abs'] = float(r.abs().mean()) if len(r.dropna()) > 0 else 0.0
                s_stats[f'h{h}_iqr_abs'] = float(r.abs().quantile(0.75) - r.abs().quantile(0.25)) if len(r.dropna()) > 0 else 0.0
            stability[col] = s_stats

        # 汇总打分：对每个 horizon 的 |corr| 与稳定性均值做加权
        summary_rows: List[Dict[str, Any]] = []
        for col in factor_cols:
            row: Dict[str, Any] = {'factor': col}
            score_components: List[float] = []
            for h in horizons:
                corr_val = abs(overall[f'h{h}'].get(col, 0.0))
                mean_abs = stability[col].get(f'h{h}_mean_abs', 0.0)
                comp = 0.6 * corr_val + 0.4 * mean_abs
                score_components.append(comp)
                row[f'corr_abs_h{h}'] = corr_val
                row[f'stability_mean_abs_h{h}'] = mean_abs
                row[f'stability_iqr_abs_h{h}'] = stability[col].get(f'h{h}_iqr_abs', 0.0)
            row['final_score'] = float(np.mean(score_components)) if score_components else 0.0
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows).sort_values('final_score', ascending=False)

        result: Dict[str, Any] = {
            'overall': overall,
            'stability': stability,
            'ranking': summary_df,
        }
        return result

    @staticmethod
    def analyze_and_save(raw_factor_data: pd.DataFrame, output_manager) -> Tuple[pd.DataFrame, str]:
        """完整流程：准备数据、计算排名、保存报告CSV，返回排名与路径。"""
        df = FactorCorrelationAnalyzer.prepare_dataframe(raw_factor_data)
        result = FactorCorrelationAnalyzer.rank_factors(df)
        if 'error' in result:
            print(f"⚠️ 因子相关性分析未执行: {result['error']}")
            return pd.DataFrame(), ''
        ranking: pd.DataFrame = result['ranking']
        csv_path = output_manager.save_dataframe(ranking, 'top_factors_correlation.csv', 'reports')
        print("🏁 最相关因子（前10）:")
        preview = ranking[['factor', 'final_score']].head(10)
        try:
            # 简洁打印
            for _, r in preview.iterrows():
                print(f"  {r['factor']}: score={r['final_score']:.4f}")
        except Exception:
            pass
        return ranking, str(csv_path)


