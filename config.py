# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Core Config ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("FATAL: TELEGRAM_TOKEN not found in .env file")

ADMIN_IDS_STR = os.getenv("ADMIN_IDS", "")
ADMIN_IDS = [int(admin_id) for admin_id in ADMIN_IDS_STR.split(',') if admin_id.strip().isdigit()]

# --- Database Config ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///niyati.db")

# --- Persona & Behavior Config ---
TIMEZONE = os.getenv("TIMEZONE", "Asia/Kolkata")

# --- Feature Flags (can be toggled by admins) ---
# These are the default states. User preferences are stored in the DB.
DEFAULT_FEATURES = {
    "memes": True,
    "shayari": True,
    "geeta": True,
    "fancy_fonts": True
}

# --- Budget & Rate Limiting ---
# Set to True to enable low budget mode (fewer replies, no extras)
LOW_BUDGET_MODE = False 
# You can add logic to toggle this based on API usage.

# --- Niyati's Persona ---
# A small collection of shayari to draw from.
# In a real app, you might load these from a JSON file.
SHAYARI_COLLECTION = [
    "thoda sa tu, thoda sa mai,\naur baaki sab kismat ka khel...",
    "dil ki raahon me tu hi,\nkhwabon ki roshni saath chale ✨",
    "jo tha bikhar sa, teri baat se judne laga,\ntere hone se hi shayad, mera hona bana 😊",
    "nazar mein tum ho,\nkhwabon mein tum ho,\nbas yaaron ab toh,\nhar jagah tum hi ho ❤️",
    "chand taaron se aage,\nek jahaan aur bhi hai,\nbas wahan sirf hum hai,\naur tumhara gumaan 🌙"
]

# Simple, respectful Gita paraphrases.
GEETA_QUOTES = [
    "Karma karo, phal ki chinta mat karo. Bas apna best do. 🙏",
    "Badalte rehna hi jeevan hai. Jo naya hai, usse apna lo. ✨",
    "Mann hi mitra hai aur mann hi shatru hai. Apne mann ko jeeto. 🧘",
    "Jo hua, achha hua. Jo ho raha hai, achha ho raha hai. Jo hoga, woh bhi achha hoga. ✨",
    "Aatma ko na shastra kaat sakte hain, na aag jala sakti hai. 🔥"
]
