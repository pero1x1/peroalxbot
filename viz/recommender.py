import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def make_signals_and_profit(pred: np.ndarray, amount: float, current_price: float):
    # ищем локальные минимумы как BUY и следующие за ними локальные максимумы как SELL.
    # рассчитываем профит по парной свинговой стратегии на количестве акций = amount/current_price.
    # создаем псевдо-индекс дат бизнес-днями от завтра
    idx = pd.bdate_range(pd.Timestamp.today().normalize(), periods=len(pred), freq="B")
    s = pd.Series(pred, index=idx)

    # пики (SELL)
    peaks, _ = find_peaks(s.values)
    # впадины (BUY) 
    troughs, _ = find_peaks(-s.values)

    signals = []
    for i in range(len(s)):
        if i in troughs:
            signals.append(("BUY", s.index[i], s.iloc[i]))
        if i in peaks:
            signals.append(("SELL", s.index[i], s.iloc[i]))

    # сортируем по дате
    signals.sort(key=lambda x: x[1])

    # после BUY ищем первый SELL строго позже
    shares = amount / current_price
    profit = 0.0
    i = 0
    while i < len(signals):
        if signals[i][0] == "BUY":
            buy_price = signals[i][2]
            # ищем следующий SELL
            j = i + 1
            while j < len(signals) and signals[j][0] != "SELL":
                j += 1
            if j < len(signals):
                sell_price = signals[j][2]
                profit += (sell_price - buy_price) * shares
                i = j + 1
            else:
                break
        else:
            i += 1

    df = pd.DataFrame([{"signal": s0, "date": d.strftime("%Y-%m-%d"), "price": float(p)} for s0, d, p in signals])
    # соберём пары для краткой печати
    pairs = []
    i = 0
    while i < len(signals):
        if signals[i][0] == "BUY":
            buy_d, buy_p = signals[i][1], signals[i][2]
            j = i + 1
            while j < len(signals) and signals[j][0] != "SELL":
                j += 1
            if j < len(signals):
                sell_d, sell_p = signals[j][1], signals[j][2]
                pairs.append((buy_d.strftime("%Y-%m-%d"), float(buy_p), sell_d.strftime("%Y-%m-%d"), float(sell_p)))
                i = j + 1
            else:
                break
        else:
            i += 1

    return df, float(profit), pairs
