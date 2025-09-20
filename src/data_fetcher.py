"""
数据获取模块 - 使用 CCXT 从币安获取 ETHUSDT 数据
"""
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
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

        # 3) 缓存不足，需要循环获取（向更早的时间回补）
        print(f"缓存不足，需要循环获取更多数据...")
        
        all_data = []
        if not cached_in_range.empty:
            all_data.append(cached_in_range)
        
        # 以缓存中最早一根K线为界，向更早方向抓取
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '12h': 720, '1d': 1440,
        }
        interval_delta = timedelta(minutes=timeframe_minutes.get(timeframe, 60))
        earliest_time = cached_in_range.index.min() if not cached_in_range.empty else None
        batch_size = 1000  # 单批最大条数
        batch_count = 0
        
        import time
        
        # 循环条件：尚未覆盖到起始时间，且尚未达到期望条数，且批次数限制内
        while (
            batch_count < 20 and
            (earliest_time is None or earliest_time > since)
        ):
            batch_count += 1
            # 目标批量：尽量满批，多取少补
            current_batch_size = batch_size
            print(f"第{batch_count}批：尝试获取 {current_batch_size} 条记录...")
            
            # 计算 since：为了不留空洞，使用“current_batch_size - overlap”策略
            overlap_records = 50
            if earliest_time is None:
                # 没有缓存，直接从目标起点开始
                batch_since = since
            else:
                step_records = max(current_batch_size - overlap_records, 1)
                batch_since = earliest_time - interval_delta * step_records
            
            # 获取这批数据
            batch_data = self.fetch_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=current_batch_size,
                since=batch_since
            )
            
            if batch_data.empty:
                print(f"第{batch_count}批获取失败或无数据，停止获取")
                break
            
            # 仅保留目标时间范围内且早于当前已知最早边界的数据，避免重复
            if earliest_time is None:
                batch_filtered = batch_data[batch_data.index >= since]
            else:
                batch_filtered = batch_data[(batch_data.index >= since) & (batch_data.index < earliest_time)]
            
            if not batch_filtered.empty:
                all_data.append(batch_filtered)
                # 更新最早边界为“本批最早时间”，确保下次 since 基于真实边界计算
                earliest_time = batch_filtered.index.min()
                print(f"第{batch_count}批获取到 {len(batch_filtered)} 条有效记录，最早时间推进至 {earliest_time}")
            else:
                print(f"第{batch_count}批无新增有效记录（可能为完全重叠），继续向前尝试")
                # 如果没有新的更早数据，尝试直接把 earliest_time 再向前推进一段，避免停滞
                if earliest_time is not None:
                    earliest_time = earliest_time - interval_delta * max((current_batch_size // 4), 1)
            
            # 已覆盖到目标起点即可退出
            if earliest_time is not None and earliest_time <= since:
                break
            
            # 避免请求过快
            time.sleep(0.25)
        
        # 4) 合并所有数据
        if all_data:
            combined = pd.concat(all_data).sort_index()
            # 去重，保留最新的数据
            combined = combined[~combined.index.duplicated(keep='last')]
            # 仅保留目标时间范围
            combined = combined[combined.index >= since]
            
            # 如果数据多于期望数量，截断到期望规模（从最新往回）
            if len(combined) > expected_records:
                combined = combined.tail(expected_records)
            
            print(f"最终获得 {len(combined)} 条记录，时间范围：{combined.index.min()} 到 {combined.index.max()}")
            
            # 保存到缓存
            cache.upsert(symbol, timeframe, combined)
            return combined
        else:
            print("未能获取到新数据，返回缓存数据")
            return cached_in_range
    
    def check_data_integrity(self, data: pd.DataFrame, timeframe: str, expected_days: int, expected_since: Optional[datetime] = None, expected_until: Optional[datetime] = None) -> Tuple[bool, List[Tuple[datetime, datetime]]]:
        """
        检查K线数据的完整性，返回是否完整和缺失的时间段
        
        Args:
            data: K线数据DataFrame，索引为时间
            timeframe: 时间框架 ('1h', '4h', '1d' 等)
            expected_days: 期望的天数
            
        Returns:
            Tuple[bool, List[Tuple[datetime, datetime]]]: (是否完整, 缺失时间段列表)
        """
        if data.empty:
            return False, []
        
        # 计算时间间隔（分钟）
        timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '12h': 720,
            '1d': 1440,
        }
        
        if timeframe not in timeframe_minutes:
            print(f"⚠️ 不支持的时间框架: {timeframe}")
            return True, []  # 对于不支持的时间框架，假设完整
        
        interval_minutes = timeframe_minutes[timeframe]
        interval_delta = timedelta(minutes=interval_minutes)
        
        # 期望覆盖的时间范围：允许指定边界；未指定时使用数据边界
        raw_start = expected_since if expected_since is not None else data.index.min()
        raw_end = expected_until if expected_until is not None else data.index.max()
        
        print(f"📊 数据完整性检查:")
        print(f"   时间框架: {timeframe}")
        
        # 将期望边界对齐到时间框架网格，避免把全部数据误判为缺失
        # 例如 4h 框架，网格为 00:00/04:00/08:00/12:00/16:00/20:00
        import math
        epoch = datetime(1970, 1, 1)
        interval_minutes = timeframe_minutes[timeframe]
        interval_seconds = interval_minutes * 60
        
        def align(dt: datetime, mode: str) -> datetime:
            delta_sec = (dt - epoch).total_seconds()
            if mode == 'floor':
                aligned = math.floor(delta_sec / interval_seconds) * interval_seconds
            else:  # ceil
                aligned = math.ceil(delta_sec / interval_seconds) * interval_seconds
            return epoch + timedelta(seconds=aligned)
        
        start_time = align(raw_start, 'ceil')
        end_time = align(raw_end, 'floor')
        
        print(f"   数据范围(对齐前): {raw_start} 到 {raw_end}")
        print(f"   数据范围(对齐后): {start_time} 到 {end_time}")
        print(f"   实际记录数: {len(data)}")
        
        # 生成期望的时间序列（严格在网格上）
        expected_times = []
        current_time = start_time
        while current_time <= end_time:
            expected_times.append(current_time)
            current_time += interval_delta
        
        expected_count = len(expected_times)
        print(f"   期望记录数: {expected_count}")
        
        # 找出缺失的时间点
        data_times = set(data.index)
        expected_times_set = set(expected_times)
        missing_times = sorted(expected_times_set - data_times)
        
        if not missing_times:
            print(f"✅ 数据完整，无缺失")
            return True, []
        
        # 将连续的缺失时间合并为时间段
        missing_ranges = []
        if missing_times:
            range_start = missing_times[0]
            range_end = missing_times[0]
            
            for i in range(1, len(missing_times)):
                current_time = missing_times[i]
                expected_next = range_end + interval_delta
                
                if current_time == expected_next:
                    # 连续缺失，扩展当前范围
                    range_end = current_time
                else:
                    # 不连续，保存当前范围并开始新范围
                    missing_ranges.append((range_start, range_end))
                    range_start = current_time
                    range_end = current_time
            
            # 添加最后一个范围
            missing_ranges.append((range_start, range_end))
        
        missing_count = len(missing_times)
        completeness = (expected_count - missing_count) / expected_count * 100 if expected_count > 0 else 100.0
        
        print(f"⚠️ 数据不完整:")
        print(f"   缺失记录数: {missing_count}")
        print(f"   完整度: {completeness:.1f}%")
        print(f"   缺失时间段数: {len(missing_ranges)}")
        
        for i, (start, end) in enumerate(missing_ranges[:5]):  # 只显示前5个缺失段
            if start == end:
                print(f"   缺失段 {i+1}: {start}")
            else:
                print(f"   缺失段 {i+1}: {start} 到 {end}")
        
        if len(missing_ranges) > 5:
            print(f"   ... 还有 {len(missing_ranges) - 5} 个缺失段")
        
        # 边界提示：如果原始起点不在网格上，提示但不计入缺口（无实际K线可补）
        if expected_since is not None and raw_start != start_time:
            print(f"   注意: 期望起点 {raw_start} 未对齐到 {timeframe} 网格，已对齐为 {start_time}")
        if expected_until is not None and raw_end != end_time:
            print(f"   注意: 期望终点 {raw_end} 未对齐到 {timeframe} 网格，已对齐为 {end_time}")
        
        return False, missing_ranges
    
    def fill_missing_data(self, data: pd.DataFrame, symbol: str, timeframe: str, 
                         missing_ranges: List[Tuple[datetime, datetime]]) -> pd.DataFrame:
        """
        补充缺失的K线数据
        
        Args:
            data: 现有的K线数据
            symbol: 交易对符号
            timeframe: 时间框架
            missing_ranges: 缺失的时间段列表
            
        Returns:
            补充后的完整数据
        """
        if not missing_ranges:
            return data
        
        print(f"🔄 开始补充缺失数据...")
        cache = MarketDataCache()
        all_data = [data] if not data.empty else []
        
        for i, (start_time, end_time) in enumerate(missing_ranges):
            print(f"   补充缺失段 {i+1}/{len(missing_ranges)}: {start_time} 到 {end_time}")
            
            # 粒度与步长
            timeframe_minutes = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '4h': 240, '12h': 720, '1d': 1440,
            }
            
            if timeframe not in timeframe_minutes:
                print(f"   ⚠️ 跳过不支持的时间框架: {timeframe}")
                continue
            
            interval_minutes = timeframe_minutes[timeframe]
            interval_delta = timedelta(minutes=interval_minutes)
            
            # 分页循环抓取直到覆盖整个缺口
            current_since = start_time - interval_delta * 2  # 小缓冲避免边界遗漏
            page_count = 0
            last_progress_ts = None
            
            import time
            
            while current_since <= end_time and page_count < 50:  # 给出合理上限防止死循环
                page_count += 1
                try:
                    batch = self.fetch_ohlcv_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=1000,
                        since=current_since
                    )
                    
                    if batch.empty:
                        print(f"      第{page_count}页无数据，停止本缺口抓取")
                        break
                    
                    # 过滤到目标缺口范围
                    batch_filtered = batch[(batch.index >= start_time) & (batch.index <= end_time)]
                    if not batch_filtered.empty:
                        all_data.append(batch_filtered)
                        print(f"      第{page_count}页补入 {len(batch_filtered)} 条")
                        # 写回缓存（使用原batch）
                        cache.upsert(symbol, timeframe, batch)
                    
                    # 推进游标
                    max_ts = batch.index.max()
                    if last_progress_ts is not None and max_ts <= last_progress_ts:
                        # 没有前进，强行推进一格防止卡住
                        max_ts = last_progress_ts + interval_delta
                    last_progress_ts = max_ts
                    current_since = max_ts + interval_delta
                    
                    # 已覆盖到缺口末尾则退出
                    if current_since > end_time:
                        break
                    
                    # 限速
                    time.sleep(0.25)
                except Exception as e:
                    print(f"      抓取第{page_count}页出错: {e}")
                    break
            
            # 小间隔避免过于频繁
            time.sleep(0.3)
        
        # 合并所有数据
        if len(all_data) > 1:
            combined_data = pd.concat(all_data, axis=0)
            # 去重并排序
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
            combined_data = combined_data.sort_index()
            
            print(f"✅ 数据补充完成，最终数据量: {len(combined_data)} 条")
            return combined_data
        else:
            return data
    
    def fetch_complete_data(self, symbol: str = 'ETH/USDT', timeframe: str = '4h', 
                           days: int = 730) -> pd.DataFrame:
        """
        获取完整的K线数据，包含完整性检查和缺失数据补充
        
        Args:
            symbol: 交易对符号
            timeframe: 时间框架
            days: 天数
            
        Returns:
            完整的K线数据
        """
        print(f"📈 获取完整的 {symbol} {timeframe} 数据 (最近{days}天)")
        
        # 1. 首先使用现有方法获取数据
        data = self.fetch_recent_with_cache(symbol, timeframe, days)
        
        if data.empty:
            print("❌ 无法获取基础数据")
            return data
        
        # 2. 检查数据完整性（覆盖目标起点到现有末尾）
        expected_since = datetime.now() - timedelta(days=days)
        is_complete, missing_ranges = self.check_data_integrity(data, timeframe, days, expected_since=expected_since)
        
        # 3. 如果数据不完整，补充缺失数据
        if not is_complete and missing_ranges:
            data = self.fill_missing_data(data, symbol, timeframe, missing_ranges)
            
            # 4. 再次检查完整性
            print(f"\n🔍 补充后再次检查数据完整性...")
            final_is_complete, final_missing = self.check_data_integrity(data, timeframe, days)
            
            if final_is_complete:
                print(f"✅ 数据补充成功，现在数据完整")
            else:
                print(f"⚠️ 仍有 {len(final_missing)} 个时间段缺失数据")
        
        return data


if __name__ == "__main__":
    # 测试数据获取
    fetcher = BinanceDataFetcher()
    data = fetcher.fetch_recent_data(days=7)
    
    if not data.empty:
        print("\n数据预览:")
        print(data.head())
        print(f"\n数据统计:")
        print(data.describe())
