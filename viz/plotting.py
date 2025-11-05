from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional

def build_plot(history: pd.Series, forecast: np.ndarray, save_path: Optional[str] = None) -> BytesIO:
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(history.index, history.values, label="История")
    future_idx = pd.date_range(start=history.index[-1], periods=len(forecast)+1, freq="B")[1:]
    ax.plot(future_idx, forecast, linestyle="--", label="Прогноз 30д")
    ax.axvline(history.index[-1], color="gray", alpha=0.6)
    ax.set_title("Цена акции: история и прогноз")
    ax.set_xlabel("Дата"); ax.set_ylabel("Цена")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, format="png", dpi=150)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf
