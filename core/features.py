import pandas as pd
import numpy as np

def make_lag_features(s: pd.Series, max_lag: int = 30) -> pd.DataFrame:
    """
    Лаги L1..L30 + скользящие средние/стд (7, 14).
    """
    df = pd.DataFrame({"y": s})
    for l in range(1, max_lag + 1):
        df[f"lag_{l}"] = s.shift(l)
    for win in (7, 14):
        df[f"roll_mean_{win}"] = s.shift(1).rolling(win).mean()
        df[f"roll_std_{win}"] = s.shift(1).rolling(win).std()
    df = df.dropna().copy()
    X = df.drop(columns=["y"])
    y = df["y"].copy()
    return X, y

def last_window_for_recursive(s: pd.Series, max_lag: int = 30) -> pd.Series:
    """Берем последние значения для стартового окна рекурсивного прогноза ML."""
    return s.iloc[-(max_lag + 14 + 1):].copy()