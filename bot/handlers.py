# bot/handlers.py
import os
import shutil
import tempfile
from datetime import datetime
from io import BytesIO

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from bot.utils import validate_ticker, validate_amount
from core.selection import run_pipeline, append_error_log

# состояния диалога 
T_TICKER, T_AMOUNT = range(2)


# команды
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я учебный бот прогноза акций.\n"
        "Команда: /predict — запускает процесс.\n"
        "Пример: /predict AAPL 1000\n"
        "Ещё: /about, /source"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Введи /predict <ТИКЕР> <СУММА>, например: /predict AAPL 1000.\n"
        "Либо используй диалог: /predict → тикер → сумма."
    )


async def about_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Учебный бот прогнозов акций.\n\n"
        "Что делает:\n"
        "• Загружает 2 года котировок (Yahoo/Stooq)\n"
        "• Обучает 3 модели (ML Ridge, ETS, ARIMA; NN — fallback)\n"
        "• Выбирает лучшую по RMSE, строит прогноз на 30 дней\n"
        "• Даёт сигналы BUY/SELL и считает условную прибыль\n\n"
        "Дисклеймер: это не инвестиционная рекомендация."
    )
    await update.message.reply_text(text)


# быстрый режим /predict AAPL 1000
async def predict_short_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) != 2:
        return await update.message.reply_text(
            "Формат: /predict <TICKER> <AMOUNT>\nНапр.: /predict MSFT 1500"
        )

    ticker = args[0].strip().upper()
    amount = validate_amount(args[1])

    if not validate_ticker(ticker) or amount is None:
        return await update.message.reply_text("Пример: /predict NVDA 500")

    context.user_data["ticker"] = ticker
    update.message.text = str(amount)  # чтобы predict_run прочитал сумму
    return await predict_run(update, context)


# диалговый режим (/predict → тикер → сумма)
async def predict_enter_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Введи тикер, например AAPL:")
    return T_TICKER


async def predict_enter_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ticker = update.message.text.strip().upper()
    if not validate_ticker(ticker):
        await update.message.reply_text("Тикер не распознан. Попробуй, например, AAPL, MSFT.")
        return T_TICKER

    context.user_data["ticker"] = ticker
    await update.message.reply_text("Отлично. Теперь введи сумму инвестиций, например 1000:")
    return T_AMOUNT


async def predict_run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    amount_text = update.message.text.strip()
    amount = validate_amount(amount_text)
    if amount is None:
        await update.message.reply_text("Нужно положительное число. Введи сумму ещё раз:")
        return T_AMOUNT

    ticker = context.user_data.get("ticker")
    user_id = update.message.from_user.id

    status_msg = await update.message.reply_text("Загружаю данные…")
    try:
        await status_msg.edit_text("Обучаю 3 модели…")
        result = run_pipeline(ticker=ticker, amount=amount, user_id=user_id)

        await status_msg.edit_text("Рисую прогноз…")
        plot_bytes: BytesIO = result["plot_bytes"]
        plot_bytes.seek(0)
        await update.message.reply_photo(plot_bytes, caption="История и прогноз на 30 дней")

        change_pct = result["change_pct"]
        best_model = result["best_model"]
        rmse = result["rmse"]
        mape = result["mape"]
        est_profit = result["est_profit"]

        summary = (
            f"Тикер: {ticker}\n"
            f"Лучшая модель: {best_model}\n"
            f"RMSE: {rmse:.2f}, MAPE: {mape:.2f}%\n"
            f"Ожидаемое изменение цены за 30 дн: {change_pct:+.2f}%\n"
            f"Ориентировочная прибыль на сумму {amount:.2f}: {est_profit:.2f}"
        )
        await update.message.reply_text(summary)

        pairs_head = result.get("pairs_head", [])
        if pairs_head:
            lines = ["Пары сделок:"]
            for b_d, b_p, s_d, s_p in pairs_head:
                lines.append(f"BUY {b_d} @ {b_p:.2f}  →  SELL {s_d} @ {s_p:.2f}")
            await update.message.reply_text("\n".join(lines))

        # пары сделок, если посчитали
        pairs_head = result.get("pairs_head", [])
        if pairs_head:
            lines = ["Пары сделок:"]
            for b_d, b_p, s_d, s_p in pairs_head:
                lines.append(f"BUY {b_d} @ {b_p:.2f}  →  SELL {s_d} @ {s_p:.2f}")
            await update.message.reply_text("\n".join(lines))

        await status_msg.delete()
        return -1

    except Exception as e:
        append_error_log(user_id=user_id, ticker=ticker, amount=amount, msg=str(e))
        await status_msg.edit_text(
            "Ошибка при построении прогноза. Попробуй другой тикер или чуть позже.\n"
            f"Детали: {str(e)}"
        )
        return -1


async def cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Ок, отменил.")
    return -1


# zip с исходниками
async def source_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Пакуем проект в zip и отправляем. Игнорируем .venv и __pycache__.
    """
    await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)

    root = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    copy_root = os.path.join(tmpdir, "project")
    os.makedirs(copy_root, exist_ok=True)

    def _skip(path: str) -> bool:
        rel = os.path.relpath(path, root)
        parts = rel.split(os.sep)
        return (".venv" in parts) or ("__pycache__" in parts)

    # чистая версия 
    for dirpath, dirnames, filenames in os.walk(root):
        if _skip(dirpath):
            continue
        rel = os.path.relpath(dirpath, root)
        dst_dir = os.path.join(copy_root, "" if rel == "." else rel)
        os.makedirs(dst_dir, exist_ok=True)
        for fn in filenames:
            src = os.path.join(dirpath, fn)
            if not _skip(src):
                shutil.copy2(src, os.path.join(dst_dir, fn))

    zip_base = os.path.join(tmpdir, f"tg_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    zip_path = shutil.make_archive(zip_base, "zip", copy_root)

    with open(zip_path, "rb") as f:
        await update.message.reply_document(
            f, filename=os.path.basename(zip_path), caption="Исходники проекта"
        )

    shutil.rmtree(tmpdir, ignore_errors=True)
