# Telegram-бот прогнозов акций (Time Series)

Учебный проект: бот загружает котировки за 2 года, обучает 3 модели (ML Ridge, ETS, ARIMA; NN — fallback), выбирает лучшую по RMSE, строит прогноз на 30 дней, даёт сигналы BUY/SELL и считает условную прибыль.

> Дисклеймер: бот предназначен только для учебных целей, это **не** инвестиционная рекомендация.

## Быстрый старт

```powershell
# 1) Python 3.10+
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Зависимости
pip install -r requirements.txt

# 3) Настройки
# Скопируй .env.example -> .env и вставь токен своего бота
# (переменные: BOT_TOKEN, DATA_SOURCE=auto|stooq|yahoo)

# 4) Запуск
python -m bot.main
````

В Telegram: найдите бота (например, `@peroalxbot`) и отправьте команды из раздела ниже.

## Команды

* `/start` — краткая справка
* `/help` — как вводить параметры
* `/predict TICKER AMOUNT` — быстрый режим, пример: `/predict AAPL 1000`
* `/predict` — диалог: тикер → сумма
* `/about` — краткое описание и дисклеймер
* `/source` — прислать zip с исходниками проекта

## Как это работает

* **Данные:** `yfinance` c ретраями + fallback на Stooq (`pandas-datareader`). Индекс делаем tz-naive, ресемпл до B-дней с `ffill()`/`bfill()`.
* **Сплит:** последние 60 дней — тест (адаптация при малом количестве данных).
* **Фичи (ML):** лаги `L1..L30`, скользящие mean/std окна 7 и 14; RidgeCV. Прогноз 30 дней — рекурсивно.
* **Статистика:** ETS (`ExponentialSmoothing`) и SARIMAX c маленькой сеткой `(p,d,q) ∈ {0..2}×{0,1}×{0..2}`.
* **Нейросеть:** LSTM при наличии TensorFlow, иначе fallback на `MLPRegressor`.
* **Метрики:** RMSE — основной критерий выбора; показываем ещё MAPE.
* **Визуализация:** последние ~180 дней истории, 30 дней прогноза пунктиром, вертикальная линия «сегодня». PNG уходит в чат и сохраняется в `examples/`.
* **Сигналы:** локальные минимумы → BUY, следующие максимумы → SELL; считаем последовательные пары и условную прибыль по введённой сумме.
* **Логи:** `logs.csv` — одна строка на запрос:
  `user_id,timestamp,ticker,amount,best_model,rmse,mape,horizon,est_profit,status,error_msg`.

## Структура проекта

```
project/
├─ bot/                 # Telegram-логика (handlers, main, utils)
├─ core/                # загрузка данных, фичи, выбор модели
├─ models/              # ML/ETS/ARIMA/NN
├─ viz/                 # графики и сигналы
├─ examples/            # сохраняемые PNG прогнозов
├─ scr/                 # скриншоты для README/отчёта
├─ logs.csv             # логи сессий
├─ .env.example         # образец env
├─ requirements.txt
└─ README.md
```

## Скриншоты

| /start                  | /about                  |
| ----------------------- | ----------------------- |
| ![start](scr/start.png) | ![about](scr/about.png) |

| /predict (диалог)           | /source                   |
| --------------------------- | ------------------------- |
| ![predict](scr/predict.png) | ![source](scr/source.png) |

Примеры графиков сохраняются в `examples/` и выглядят так:

![AAPL](examples/AAPL_20251105_165954.png)

## Требования

* Python 3.10+
* Windows, macOS или Linux
* Интернет-доступ (для загрузки котировок с Yahoo/Stooq)

## Лицензия

MIT (по желанию добавьте LICENSE).

````

---

# 3) requirements.txt (проверь, что в репозитории именно так)

```text
python-telegram-bot==20.7
python-dotenv==1.0.1
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
statsmodels==0.14.2
matplotlib==3.8.4
scipy==1.11.4
yfinance==0.2.38
pandas-datareader==0.10.0
````

---

## Что дальше

1. `git add README.md .gitignore` → `git commit -m "Add README and .gitignore"` → `git push`.
2. Открой репозиторий — проверь, что Markdown отрисовался, картинки видны.
3. Если хочешь, добавим бейджи и ссылку на твоего бота/демо-видео.
