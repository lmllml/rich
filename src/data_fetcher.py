"""
数据获取模块 - 使用 CCXT 从币安获取 ETHUSDT 数据
"""
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
import sqlite3
from pathlib import Path
from paths import DB_DIR, DB_PATH


class MarketDataCache:
    """SQLite 缓存：按 (symbol, timeframe) 分表，主键为时间戳毫秒。"""

    def __init__(self, db_path: Path = DB_PATH):
        DB_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self._ensure_meta_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_meta_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tables_meta (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    UNIQUE(symbol, timeframe)
                )
                """
            )

    @staticmethod
    def _table_name(symbol: str, timeframe: str) -> str:
        safe_symbol = symbol.replace('/', '_').replace('-', '_').upper()
        safe_tf = timeframe.replace(' ', '').lower()
        return f"ohlcv_{safe_symbol}_{safe_tf}"

    def _ensure_data_table(self, symbol: str, timeframe: str) -> str:
        table = self._table_name(symbol, timeframe)
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    ts INTEGER PRIMARY KEY,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO tables_meta(symbol, timeframe, table_name) VALUES(?,?,?)",
                (symbol, timeframe, table),
            )
        return table

    def load_range(self, symbol: str, timeframe: str, since_ms: int, limit: int) -> pd.DataFrame:
        table = self._ensure_data_table(symbol, timeframe)
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT ts, open, high, low, close, volume FROM {table} WHERE ts>=? ORDER BY ts ASC LIMIT ?",
                (since_ms, limit),
            )
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        return df

    def upsert(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        table = self._ensure_data_table(symbol, timeframe)
        to_insert = df.copy()
        if 'timestamp' not in to_insert.columns:
            to_insert['timestamp'] = (to_insert.index.astype('int64') // 10**6).astype('int64')
        records = to_insert[['timestamp', 'open', 'high', 'low', 'close', 'volume']].itertuples(index=False)
        with self._connect() as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO {table}(ts, open, high, low, close, volume) VALUES(?,?,?,?,?,?)",
                list(records),
            )


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

    def fetch_recent_with_cache(
        self,
        symbol: str = 'ETH/USDT',
        timeframe: str = '1h',
        days: int = 30
    ) -> pd.DataFrame:
        """优先从本地 SQLite 读取，不足再请求补齐，并写回缓存。"""
        cache = MarketDataCache()
        since = datetime.now() - timedelta(days=days)
        since_ms = int(since.timestamp() * 1000)
        limit = days * (24 if timeframe == '1h' else 500)

        # 1) 先读缓存
        cached = cache.load_range(symbol, timeframe, since_ms, limit)

        # 2) 如果缓存不足，向交易所请求，并与缓存合并
        need_fetch = cached.empty or len(cached) < limit
        if need_fetch:
            fetched = self.fetch_ohlcv_data(symbol, timeframe, limit=limit, since=since)
            if not fetched.empty:
                # 合并并去重
                combined = pd.concat([cached, fetched]).sort_index()
                combined = combined[~combined.index.duplicated(keep='last')]
                cache.upsert(symbol, timeframe, combined)
                return combined
            return cached
        return cached


if __name__ == "__main__":
    # 测试数据获取
    fetcher = BinanceDataFetcher()
    data = fetcher.fetch_recent_data(days=7)
    
    if not data.empty:
        print("\n数据预览:")
        print(data.head())
        print(f"\n数据统计:")
        print(data.describe())
