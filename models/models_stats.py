import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

# ETS 
def fit_eval_ets(train: pd.Series, test: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            train, trend="add", seasonal=None, initialization_method="estimated"
        ).fit(optimized=True)
    pred = fit.forecast(len(test))
    return fit, _rmse(test.values, pred.values), _mape(test.values, pred.values)

def forecast_ets(series: pd.Series, fit, horizon: int = 30):
    return fit.forecast(horizon).values.astype(float)

# ARIMA 
def fit_eval_arima(train: pd.Series, test: pd.Series):
    best = None
    y_te = test.values
    for p in (0, 1, 2):
        for d in (0, 1):
            for q in (0, 1, 2):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = SARIMAX(train, order=(p, d, q),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
                        fit = model.fit(disp=False)
                    pred = fit.forecast(len(test)).values
                    rmse = _rmse(y_te, pred)
                    mape = _mape(y_te, pred)
                    name = f"ARIMA({p},{d},{q})"
                    if (best is None) or (rmse < best[2]):
                        fit._name_for_report = name 
                        best = (fit, rmse, mape)
                except Exception:
                    continue
    if best is None:
        return fit_eval_ets(train, test)
    return best

def forecast_arima(series: pd.Series, fit, horizon: int = 30):
    pred = fit.forecast(horizon)
    return pred.values.astype(float)
