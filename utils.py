# utils.py
import re
from telegram import ParseMode
from fancy_text import font_style

def escape_markdown(text: str) -> str:
    """Escapes markdown v2 characters in text."""
    return re.sub(r'([_*\[\]()~`>\#\+\-\=|\.!])', r'\\\1', text)

def apply_fancy_font(text: str, style: str = 'double_struck') -> str:
    """Applies a fancy font style to text."""
    try:
        # fancy-text library uses a map
        font_map = {
            'double_struck': '𝔻𝕠𝕦𝕓𝕝𝕖-𝕊𝕥𝕣𝕦𝕔𝕜',
            'script': '𝒮𝒸𝓇𝒾𝓅𝓉',
            'fraktur': '𝔉𝔯𝔞𝔨𝔱𝔲𝔯',
            'monospace': '𝙼𝚘𝚗𝚘𝚜𝚙𝚊𝚌𝚎'
        }
        return font_style(text, style=font_map.get(style, 'double_struck'))
    except Exception:
        return text # Fallback if font fails

def mention_html(user_id, name):
    """Creates an HTML mention link."""
    return f'<a href="tg://user?id={user_id}">{name}</a>'
