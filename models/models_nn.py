import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

def _make_windows(s: pd.Series, win: int = 30):
    arr = s.values.astype(float)
    X, y = [], []
    for i in range(win, len(arr)):
        X.append(arr[i - win:i])
        y.append(arr[i])
    return np.array(X), np.array(y)

def fit_eval_nn(train: pd.Series, test: pd.Series):
    # Пытаемся LSTM → если нет TF, используем MLP как fallback
    try:
        import tensorflow as tf  # noqa
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.callbacks import EarlyStopping

        win = 30
        Xtr, ytr = _make_windows(train, win)
        Xte, yte = _make_windows(pd.concat([train.iloc[-win:], test]), win)
        # Xte последние len(test) выборок
        Xte = Xte[-len(test):]

        Xtr = Xtr.reshape((-1, win, 1))
        Xte = Xte.reshape((-1, win, 1))

        model = Sequential([
            LSTM(32, input_shape=(win,1)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(Xtr, ytr, epochs=15, batch_size=32, validation_split=0.2, callbacks=[es], verbose=0)

        pred = model.predict(Xte, verbose=0).reshape(-1)
        rmse = _rmse(test.values, pred)
        mape = _mape(test.values, pred)
        return model, rmse, mape, "LSTM"
    except Exception:
        # Fallback: MLPRegressor
        win = 30
        Xtr, ytr = _make_windows(train, win)
        Xte, yte = _make_windows(pd.concat([train.iloc[-win:], test]), win)
        Xte = Xte[-len(test):]

        mlp = MLPRegressor(hidden_layer_sizes=(64,64), random_state=42, max_iter=1000)
        mlp.fit(Xtr, ytr)
        pred = mlp.predict(Xte)
        rmse = _rmse(test.values, pred)
        mape = _mape(test.values, pred)
        return mlp, rmse, mape, "NN(MLP-fallback)"

def forecast_nn(series: pd.Series, model, horizon: int = 30):
    # Работает и для LSTM, и для MLP — оба принимают окно длиной win
    win = 30
    window = series.values.astype(float)[-win:].tolist()
    preds = []
    try:
        # LSTM-ветка: нужен shape (1, win, 1)
        import tensorflow as tf  # noqa
        for _ in range(horizon):
            x = np.array(window[-win:]).reshape((1, win, 1))
            y_next = float(model.predict(x, verbose=0).reshape(-1)[0])
            preds.append(y_next)
            window.append(y_next)
    except Exception:
        # MLP-ветка
        for _ in range(horizon):
            x = np.array(window[-win:]).reshape(1, -1)
            y_next = float(model.predict(x)[0])
            preds.append(y_next)
            window.append(y_next)
    return np.array(preds)