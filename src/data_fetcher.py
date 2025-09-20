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
        # 根据时间框架计算数据点数量
        if timeframe == '1h':
            limit = days * 24  # 每天24小时
        elif timeframe == '4h':
            limit = days * 6   # 每天6个4小时K线
        elif timeframe == '12h':
            limit = days * 2   # 每天2个12小时K线
        elif timeframe == '1d':
            limit = days * 1   # 每天1个日K线
        elif timeframe == '15m':
            limit = days * 24 * 4  # 每天96个15分钟K线
        elif timeframe == '5m':
            limit = days * 24 * 12  # 每天288个5分钟K线
        elif timeframe == '1m':
            limit = days * 24 * 60  # 每天1440个1分钟K线
        else:
            limit = days * 500  # 其他时间框架的默认值
        return self.fetch_ohlcv_data(symbol, timeframe, limit=limit, since=since)

    def fetch_recent_with_cache(
        self,
        symbol: str = 'ETH/USDT',
        timeframe: str = '1h',
        days: int = 30
    ) -> pd.DataFrame:
        """优先从本地 SQLite 读取，不足再循环请求补齐，并写回缓存。"""
        cache = MarketDataCache()
        since = datetime.now() - timedelta(days=days)
        since_ms = int(since.timestamp() * 1000)
        
        # 根据时间框架计算期望的数据点数量
        if timeframe == '1h':
            expected_records = days * 24  # 每天24小时
        elif timeframe == '4h':
            expected_records = days * 6   # 每天6个4小时K线
        elif timeframe == '12h':
            expected_records = days * 2   # 每天2个12小时K线
        elif timeframe == '1d':
            expected_records = days * 1   # 每天1个日K线
        elif timeframe == '15m':
            expected_records = days * 24 * 4  # 每天96个15分钟K线
        elif timeframe == '5m':
            expected_records = days * 24 * 12  # 每天288个5分钟K线
        elif timeframe == '1m':
            expected_records = days * 24 * 60  # 每天1440个1分钟K线
        else:
            expected_records = days * 500  # 其他时间框架的默认值

        print(f"期望获取 {expected_records} 条 {timeframe} 数据 (最近{days}天)")

        # 1) 先读缓存
        cached = cache.load_range(symbol, timeframe, since_ms, expected_records * 2)  # 多读一些避免边界问题
        cached_in_range = cached[cached.index >= since] if not cached.empty else cached
        print(f"从缓存中读取到 {len(cached_in_range)} 条目标时间范围内的记录")

        # 2) 如果缓存数据足够，直接返回
        if len(cached_in_range) >= expected_records * 0.95:  # 允许5%的误差
            print(f"缓存数据充足，返回 {len(cached_in_range)} 条记录")
            return cached_in_range.head(expected_records)

        # 3) 缓存不足，需要循环获取
        print(f"缓存不足，需要循环获取更多数据...")
        
        all_data = []
        if not cached_in_range.empty:
            all_data.append(cached_in_range)
        
        # 从最早需要的时间开始，分批向前获取
        current_end_time = datetime.now()
        batch_size = 1000  # 每批最多1000条
        total_collected = len(cached_in_range)
        batch_count = 0
        
        import time
        
        while total_collected < expected_records and batch_count < 10:  # 最多10批，避免无限循环
            batch_count += 1
            
            # 计算这批需要获取多少条
            remaining = expected_records - total_collected
            current_batch_size = min(batch_size, remaining + 200)  # 多获取一些避免边界问题
            
            print(f"第{batch_count}批：尝试获取 {current_batch_size} 条记录...")
            
            # 计算这批数据的开始时间
            if timeframe == '4h':
                batch_days = (current_batch_size // 6) + 10  # 4小时线，每天6条，多加10天缓冲
            elif timeframe == '1h':
                batch_days = (current_batch_size // 24) + 5   # 1小时线，每天24条
            elif timeframe == '1d':
                batch_days = current_batch_size + 10          # 日线
            else:
                batch_days = 100  # 其他时间框架的默认值
                
            batch_since = current_end_time - timedelta(days=batch_days)
            
            # 获取这批数据
            batch_data = self.fetch_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=current_batch_size,
                since=batch_since
            )
            
            if batch_data.empty:
                print(f"第{batch_count}批获取失败，停止获取")
                break
            
            # 过滤到目标时间范围
            batch_in_range = batch_data[batch_data.index >= since]
            if not batch_in_range.empty:
                all_data.append(batch_in_range)
                total_collected += len(batch_in_range)
                print(f"第{batch_count}批获取到 {len(batch_in_range)} 条有效记录，总计 {total_collected} 条")
            
            # 更新下次获取的结束时间
            current_end_time = batch_data.index.min() - timedelta(hours=1)
            
            # 如果已经获取到足够早的数据，停止
            if current_end_time <= since:
                break
                
            # 避免请求过快
            time.sleep(0.2)
        
        # 4) 合并所有数据
        if all_data:
            combined = pd.concat(all_data).sort_index()
            # 去重，保留最新的数据
            combined = combined[~combined.index.duplicated(keep='last')]
            # 确保在目标时间范围内
            combined = combined[combined.index >= since]
            
            print(f"最终获得 {len(combined)} 条记录，时间范围：{combined.index.min()} 到 {combined.index.max()}")
            
            # 保存到缓存
            cache.upsert(symbol, timeframe, combined)
            return combined
        else:
            print("未能获取到新数据，返回缓存数据")
            return cached_in_range


if __name__ == "__main__":
    # 测试数据获取
    fetcher = BinanceDataFetcher()
    data = fetcher.fetch_recent_data(days=7)
    
    if not data.empty:
        print("\n数据预览:")
        print(data.head())
        print(f"\n数据统计:")
        print(data.describe())
