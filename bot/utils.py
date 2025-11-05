import re

def validate_ticker(t: str) -> bool:
    # Разрешаем латиницу, цифры, точки и дефисы
    return bool(re.fullmatch(r"[A-Za-z0-9\.\-]{1,10}", t or ""))

def validate_amount(s: str):
    try:
        val = float(s.replace(",", "."))
        return val if val > 0 else None
    except Exception:
        return None