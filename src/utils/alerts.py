import requests
from src.utils.logger import get_logger

logger = get_logger(__name__)

def send_telegram_alert(token, chat_id, message):
    if not token or not chat_id:
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=5)
        return resp.status_code == 200
    except Exception as e:
        logger.error(f"Telegram failed: {e}")
        return False
