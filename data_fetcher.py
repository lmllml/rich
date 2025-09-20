"""
数据获取模块 - 使用 CCXT 从币安获取 ETHUSDT 数据
"""
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any


class BinanceDataFetcher:
    """币安数据获取器"""
    
    def __init__(self):
        """初始化币安交易所连接"""
        self.exchange = ccxt.binance({
            'apiKey': '',  # 如果需要私有API，在这里填入
            'secret': '',  # 如果需要私有API，在这里填入
            'sandbox': False,  # 设置为True使用测试环境
            'enableRateLimit': True,
        })
    
    def fetch_ohlcv_data(
        self, 
        symbol: str = 'ETH/USDT',
        timeframe: str = '1h',
        limit: int = 500,
        since: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        获取OHLCV数据
        
        Args:
            symbol: 交易对符号，默认 'ETH/USDT'
            timeframe: 时间周期，默认 '1h'
            limit: 数据条数限制，默认 500
            since: 开始时间，默认为None（获取最新数据）
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        try:
            # 如果指定了开始时间，转换为毫秒时间戳
            since_timestamp = None
            if since:
                since_timestamp = int(since.timestamp() * 1000)
            
            # 获取OHLCV数据
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since=since_timestamp
            )
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换时间戳为日期时间
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # 删除原始时间戳列
            df.drop('timestamp', axis=1, inplace=True)
            
            print(f"成功获取 {symbol} 数据，共 {len(df)} 条记录")
            print(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"获取数据时发生错误: {e}")
            return pd.DataFrame()
    
    def fetch_recent_data(
        self, 
        symbol: str = 'ETH/USDT',
        timeframe: str = '1h',
        days: int = 30
    ) -> pd.DataFrame:
        """
        获取最近N天的数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            days: 天数
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        since = datetime.now() - timedelta(days=days)
        return self.fetch_ohlcv_data(symbol, timeframe, limit=days*24, since=since)


if __name__ == "__main__":
    # 测试数据获取
    fetcher = BinanceDataFetcher()
    data = fetcher.fetch_recent_data(days=7)
    
    if not data.empty:
        print("\n数据预览:")
        print(data.head())
        print(f"\n数据统计:")
        print(data.describe())
