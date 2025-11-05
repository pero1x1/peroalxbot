from typing import Tuple
import os, time
import pandas as pd
import yfinance as yf

PREFER_SOURCE = os.getenv("DATA_SOURCE", "auto").lower()  # auto|yahoo|stooq

def _clean_close(df: pd.DataFrame) -> pd.Series:
    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    s = df["Close"].copy()
    s = s.resample("B").ffill().bfill()
    return s

def _try_yahoo(ticker: str, period: str = "2y"):
    # A) period=2y
    for attempt in range(3):
        try:
            data = yf.download(
                ticker, period=period, interval="1d",
                auto_adjust=True, progress=False, threads=False
            )
            if data is not None and not data.empty and "Close" in data.columns:
                return _clean_close(data)
        except Exception:
            pass
        time.sleep(1.2 * (attempt + 1))

    try:
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=730)
        data = yf.download(
            ticker, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False, threads=False
        )
        if data is not None and not data.empty and "Close" in data.columns:
            return _clean_close(data)
    except Exception:
        pass

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval="1d", auto_adjust=True)
        if hist is not None and not hist.empty and "Close" in hist.columns:
            return _clean_close(hist)
    except Exception:
        pass
    return None

def _try_stooq(ticker: str):
    try:
        from pandas_datareader import data as pdr
    except Exception:
        return None
        
    for sym in (ticker, f"{ticker}.US"):
        try:
            st = pdr.DataReader(sym, "stooq")
            if st is None or st.empty or "Close" not in st.columns:
                continue
            st = st.sort_index()
            s = st["Close"].copy()
            cutoff = s.index.max() - pd.Timedelta(days=730)
            s = s.loc[s.index >= cutoff]
            s = s.asfreq("B").ffill().bfill()
            if len(s) > 50:
                return s
        except Exception:
            continue
    return None

def load_close_series(ticker: str, period: str = "2y") -> pd.Series:
    """
    Надёжная загрузка: сначала Stooq, затем Yahoo.
    Можно переопределить через .env: DATA_SOURCE=stooq|yahoo|auto
    """
    if PREFER_SOURCE in ("stooq", "auto"):
        s = _try_stooq(ticker)
        if s is not None:
            return s
        if PREFER_SOURCE == "stooq":
            raise ValueError(f"Stooq не вернул данные для {ticker}")

    s = _try_yahoo(ticker, period)
    if s is not None:
        return s

    # если auto, но Stooq не пробовали попробуем в конце
    if PREFER_SOURCE == "yahoo":
        s = _try_stooq(ticker)
        if s is not None:
            return s

    raise ValueError(f"Не удалось получить котировки для тикера {ticker}")

def train_test_split_by_time(s: pd.Series, test_days: int = 60) -> Tuple[pd.Series, pd.Series]:
    if len(s) < test_days + 50:
        test_days = max(20, min(60, len(s)//4))
    train = s.iloc[:-test_days]
    test = s.iloc[-test_days:]
    return train, test
