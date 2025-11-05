from datetime import datetime
from io import BytesIO
import numpy as np
import pandas as pd

from core.data_loader import load_close_series, train_test_split_by_time
from models.models_ml import fit_eval_ml, forecast_ml
from models.models_stats import fit_eval_ets, forecast_ets
from models.models_nn import fit_eval_nn, forecast_nn
from viz.plotting import build_plot
from viz.recommender import make_signals_and_profit
from models.models_stats import fit_eval_ets, forecast_ets, fit_eval_arima, forecast_arima


def run_pipeline(ticker: str, amount: float, user_id: int) -> dict:
    # 1) загрузка
    s = load_close_series(ticker)
    train, test = train_test_split_by_time(s, test_days=60)

    # 2) обучаем 3 модели
    metrics = []

    # ML
    ml_model, ml_rmse, ml_mape = fit_eval_ml(train, test)
    metrics.append(("ML(Ridge)", ml_rmse, ml_mape, ml_model))

    # ETS
    ets_model, ets_rmse, ets_mape = fit_eval_ets(train, test)
    metrics.append(("ETS", ets_rmse, ets_mape, ets_model))

    # ARIMA
    arima_model, arima_rmse, arima_mape = fit_eval_arima(train, test)
    # имя модели возьмём из атрибута, если он есть
    arima_name = getattr(arima_model, "_name_for_report", "ARIMA")
    metrics.append((arima_name, arima_rmse, arima_mape, arima_model))

# 3) выбор лучшей по RMSE
    metrics_sorted = sorted(metrics, key=lambda x: x[1])
    best_name, best_rmse, best_mape, best_model = metrics_sorted[0]

    # 4) прогноз на 30 дней
    horizon = 30
    hist = pd.concat([train, test])

    if best_name.startswith("ML"):
        y_pred = forecast_ml(hist, best_model, horizon)
    elif best_name == "ETS":
        y_pred = forecast_ets(hist, best_model, horizon)
    elif best_name.startswith("ARIMA"):
        y_pred = forecast_arima(hist, best_model, horizon)
    else:
        y_pred = forecast_ets(hist, best_model, horizon)

    # изменение цены относительно текущей
    last_price = float(s.iloc[-1])
    change_pct = (y_pred[-1] - last_price) / last_price * 100.0

    # 5) график + сохранение в examples/
    hist_tail = s.iloc[-180:]
    import os
    os.makedirs("examples", exist_ok=True)
    png_path = os.path.join("examples", f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plot_bytes = build_plot(hist_tail, y_pred, save_path=png_path)

    # 6) рекомендации и условная прибыль
    signals_df, est_profit, pairs = make_signals_and_profit(y_pred, amount, last_price)
    signals_head = signals_df.head(8).copy()
    pairs_head = pairs[:4]

    # 7) лог в csv
    log_row = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "amount": amount,
        "best_model": best_name,
        "rmse": round(best_rmse, 6),
        "mape": round(best_mape, 6),
        "horizon": horizon,
        "est_profit": round(float(est_profit), 6),
        "status": "ok",
        "error_msg": ""
    }
    _append_log(log_row)

    return {
        "plot_bytes": plot_bytes,
        "change_pct": float(change_pct),
        "pairs_head": pairs_head,
        "best_model": best_name,
        "rmse": float(best_rmse),
        "mape": float(best_mape),
        "est_profit": float(est_profit),
        "signals_head": signals_head,
    }

def _append_log(row: dict):
    import csv, os
    path = "logs.csv"
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def append_error_log(user_id: int, ticker: str, amount: float, msg: str):
    import csv, os
    path = "logs.csv"
    file_exists = os.path.exists(path)
    row = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "amount": amount,
        "best_model": "",
        "rmse": "",
        "mape": "",
        "horizon": "",
        "est_profit": "",
        "status": "error",
        "error_msg": msg[:500],
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)