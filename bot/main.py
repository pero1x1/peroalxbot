# bot/main.py
import os
from dotenv import load_dotenv
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ConversationHandler, filters
)
from bot import handlers as h


def main():
    load_dotenv()
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN not found. Create .env based on .env.example")

    app = ApplicationBuilder().token(token).build()

    # команды
    app.add_handler(CommandHandler("start", h.start_cmd))
    app.add_handler(CommandHandler("help", h.help_cmd))
    app.add_handler(CommandHandler("about", h.about_cmd))
    app.add_handler(CommandHandler("source", h.source_cmd))

    # быстрый режим: /predict <TICKER> <AMOUNT>.
    app.add_handler(
        MessageHandler(
            filters.Regex(r"^/predict(\s+\S+){2}\s*$"),
            h.predict_short_cmd,
        )
    )

    # диалоговый режим: /predict → тикер → сумма
    conv = ConversationHandler(
        entry_points=[CommandHandler("predict", h.predict_enter_ticker)],
        states={
            h.T_TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, h.predict_enter_amount)],
            h.T_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, h.predict_run)],
        },
        fallbacks=[CommandHandler("cancel", h.cancel_cmd)],
        name="predict_dialog",
        persistent=False,
    )
    app.add_handler(conv)

    print("Bot is running... Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
