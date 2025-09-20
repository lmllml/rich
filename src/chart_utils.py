"""
图表工具模块 - 处理中文字体和图表美化
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
from pathlib import Path


def setup_chinese_fonts():
    """
    设置中文字体，支持多种字体回退
    """
    # 常见的中文字体列表
    chinese_fonts = [
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'SimSun',           # 宋体
        'KaiTi',            # 楷体
        'FangSong',         # 仿宋
        'Arial Unicode MS', # Arial Unicode MS (Mac)
        'PingFang SC',      # 苹方 (Mac)
        'Hiragino Sans GB', # 冬青黑体 (Mac)
        'WenQuanYi Micro Hei', # 文泉驿微米黑 (Linux)
        'Noto Sans CJK SC',    # 思源黑体 (Linux)
    ]
    
    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 找到第一个可用的中文字体
    chinese_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            chinese_font = font
            break
    
    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ 成功设置中文字体: {chinese_font}")
        return True, chinese_font
    else:
        print("⚠️  未找到中文字体，将使用英文标题")
        return False, None


def create_enhanced_factor_chart(factor_data: pd.DataFrame, output_path: str = '../output/factor_analysis_charts.png'):
    """
    创建增强版因子分析图表
    """
    # 设置中文字体
    has_chinese_font, font_name = setup_chinese_fonts()
    
    # 设置图表样式
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # 创建图表
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    # 设置主标题
    if has_chinese_font:
        fig.suptitle('ETHUSDT 因子分析报告', fontsize=18, fontweight='bold', y=0.98)
    else:
        fig.suptitle('ETHUSDT Factor Analysis Report', fontsize=18, fontweight='bold', y=0.98)
    
    # 1. 价格和综合信号
    ax1 = axes[0, 0]
    line1 = ax1.plot(factor_data.index, factor_data['price'], label='ETH Price', 
                     color='#2E86C1', linewidth=2)
    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(factor_data.index, factor_data['combined_signal'], 
                          label='Combined Signal', color='#E74C3C', alpha=0.8, linewidth=1.5)
    
    if has_chinese_font:
        ax1.set_title('价格 vs 综合信号', fontsize=14, pad=15)
        ax1.set_ylabel('价格 (USDT)', fontsize=12)
        ax1_twin.set_ylabel('综合信号', fontsize=12)
    else:
        ax1.set_title('Price vs Combined Signal', fontsize=14, pad=15)
        ax1.set_ylabel('Price (USDT)', fontsize=12)
        ax1_twin.set_ylabel('Combined Signal', fontsize=12)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # 2. RSI 指标
    ax2 = axes[0, 1]
    ax2.plot(factor_data.index, factor_data['rsi'], label='RSI', color='#3498DB', linewidth=2)
    ax2.axhline(y=70, color='#E74C3C', linestyle='--', alpha=0.7, linewidth=2, 
                label='Overbought (70)')
    ax2.axhline(y=30, color='#27AE60', linestyle='--', alpha=0.7, linewidth=2, 
                label='Oversold (30)')
    ax2.fill_between(factor_data.index, 30, 70, alpha=0.1, color='gray')
    
    if has_chinese_font:
        ax2.set_title('RSI 相对强弱指标', fontsize=14, pad=15)
    else:
        ax2.set_title('RSI Indicator', fontsize=14, pad=15)
    
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    # 3. MACD 指标
    ax3 = axes[1, 0]
    colors = ['#27AE60' if x >= 0 else '#E74C3C' for x in factor_data['macd']]
    ax3.bar(factor_data.index, factor_data['macd'], color=colors, alpha=0.7, width=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    if has_chinese_font:
        ax3.set_title('MACD 指标', fontsize=14, pad=15)
        ax3.set_ylabel('MACD 值', fontsize=12)
    else:
        ax3.set_title('MACD Indicator', fontsize=14, pad=15)
        ax3.set_ylabel('MACD Value', fontsize=12)
    
    # 4. 布林带
    ax4 = axes[1, 1]
    ax4.plot(factor_data.index, factor_data['price'], label='Price', 
             color='black', linewidth=2, zorder=3)
    ax4.plot(factor_data.index, factor_data['bb_upper'], label='Upper Band', 
             color='#E74C3C', alpha=0.8, linewidth=1.5)
    ax4.plot(factor_data.index, factor_data['bb_lower'], label='Lower Band', 
             color='#27AE60', alpha=0.8, linewidth=1.5)
    ax4.fill_between(factor_data.index, factor_data['bb_lower'], factor_data['bb_upper'], 
                     alpha=0.1, color='#3498DB')
    
    if has_chinese_font:
        ax4.set_title('布林带指标', fontsize=14, pad=15)
        ax4.set_ylabel('价格 (USDT)', fontsize=12)
    else:
        ax4.set_title('Bollinger Bands', fontsize=14, pad=15)
        ax4.set_ylabel('Price (USDT)', fontsize=12)
    
    ax4.legend()
    
    # 5. 动量指标
    ax5 = axes[2, 0]
    colors = ['#27AE60' if x >= 0 else '#E74C3C' for x in factor_data['momentum']]
    ax5.bar(factor_data.index, factor_data['momentum'], color=colors, alpha=0.7, width=0.8)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    if has_chinese_font:
        ax5.set_title('动量指标', fontsize=14, pad=15)
        ax5.set_ylabel('动量值', fontsize=12)
    else:
        ax5.set_title('Momentum Indicator', fontsize=14, pad=15)
        ax5.set_ylabel('Momentum Value', fontsize=12)
    
    # 6. 仓位变化
    ax6 = axes[2, 1]
    ax6.step(factor_data.index, factor_data['position'], where='post', 
             label='Position', color='#8E44AD', linewidth=2)
    ax6.fill_between(factor_data.index, 0, factor_data['position'], 
                     alpha=0.3, color='#8E44AD', step='post')
    
    if has_chinese_font:
        ax6.set_title('持仓变化', fontsize=14, pad=15)
        ax6.set_ylabel('持仓数量', fontsize=12)
    else:
        ax6.set_title('Position Changes', fontsize=14, pad=15)
        ax6.set_ylabel('Position Size', fontsize=12)
    
    ax6.legend()
    
    # 统一设置x轴标签旋转
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.3, wspace=0.3)
    
    # 保存图表
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', 
                format='png')
    
    print(f"✅ 增强版图表已保存到: {output_path}")
    
    return fig


if __name__ == "__main__":
    # 测试字体设置
    has_font, font_name = setup_chinese_fonts()
    if has_font:
        print(f"当前使用字体: {font_name}")
    else:
        print("将使用默认字体")
