import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from core.features import make_lag_features, last_window_for_recursive

def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

def fit_eval_ml(train: pd.Series, test: pd.Series):
    X_tr, y_tr = make_lag_features(train)
    # В тесте делаем признаки из train+части test, но таргет — реальный test, с выравниванием по датам
    joined = pd.concat([train, test])
    X_all, y_all = make_lag_features(joined)
    X_te = X_all.loc[test.index]  # берем строки по датам теста
    y_te = y_all.loc[test.index]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=(0.1, 1.0, 10.0)))
    ])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = _rmse(y_te, pred)
    mape = _mape(y_te, pred)
    return model, rmse, mape

def forecast_ml(series: pd.Series, model, horizon: int = 30) -> np.ndarray:
    s = series.copy()
    max_lag = 30
    window = last_window_for_recursive(s, max_lag)
    y_pred = []
    cur = s.copy()
    for _ in range(horizon):
        X_step, _ = make_lag_features(cur)
        x_last = X_step.iloc[[-1]]  # последняя строка признаков
        y_next = float(model.predict(x_last))
        y_pred.append(y_next)
        # добавим предсказание в конец ряда, чтобы построить лаги дальше
        cur = pd.concat([cur, pd.Series([y_next], index=[cur.index[-1] + pd.tseries.offsets.BDay(1)])])
    return np.array(y_pred)