# -*- coding: utf-8 -*-
import traceback
import time
import os
import secrets
import pytz
import re
import json
import threading
import asyncio
import logging  # Logging import zaroori hai
import random
import requests
import signal
import sys
import concurrent.futures
from collections import OrderedDict
from dataclasses import asdict
from html import escape as html_escape
from PIL import Image, ImageOps
from db_migrations import run_migrations
from content_identity import (
    can_create_canonical_content,
    is_safe_canonical_title,
    parse_content_identity,
    resolve_raw_content_identity,
)
from telegram import WebAppInfo
from telegram import MenuButtonWebApp, WebAppInfo
import aiohttp
# import anthropic  # Agar zaroorat ho toh uncomment karein
from flask import jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse, quote, unquote, urlencode
from collections import defaultdict
from telegram.error import RetryAfter, TelegramError
from typing import Optional
from psycopg2 import pool
from io import BytesIO

# Naya Lock banaya Auto-Batch ke liye
auto_batch_lock = asyncio.Lock()

# ==================== 1. LOGGING SETUP (SABSE PEHLE YEH AAYEGA) ====================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Change from INFO to DEBUG if needed
)
logger = logging.getLogger(__name__)

# ==================== CACHING ====================
class FastCache:
    def __init__(self, ttl_seconds=3600, max_entries=1024):
        self.cache = OrderedDict()
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.lock = threading.RLock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            data, timestamp = self.cache.pop(key)
            if time.time() - timestamp >= self.ttl:
                return None
            self.cache[key] = (data, timestamp)
            return data

    def set(self, key, value):
        with self.lock:
            self.cache.pop(key, None)
            self.cache[key] = (value, time.time())
            while len(self.cache) > self.max_entries:
                self.cache.popitem(last=False)

    def clear(self):
        with self.lock:
            self.cache.clear()

search_cache = FastCache(ttl_seconds=30)  # 30 Seconds cache for SQL/Fuzzy searches
api_movies_cache = FastCache(ttl_seconds=30) # 30 Seconds cache for Web App Home
poster_cache = FastCache(ttl_seconds=3600)

# Toggle lightweight search-stage timing instrumentation with env var SEARCH_TIMING=1
SEARCH_TIMING_ENABLED = os.environ.get('SEARCH_TIMING', '0') == '1'
GOOGLE_SUGGESTION_TIMEOUT_SECONDS = 1.5

# ==================== 2. AB IMDB CHECK KAREIN (AB YE SAFE HAI) ====================
try:
    from imdb import Cinemagoer
    try:
        ia = Cinemagoer()
    except Exception as e:
        logger.warning(f"Cinemagoer initialization failed: {e}")
        ia = None
except ImportError:
    # Ab logger define ho chuka hai, toh yeh error nahi dega
    logger.warning("imdb (cinemagoer) module not found. Run: pip install cinemagoer")
    ia = None

# ==================== 3. BAAKI IMPORTS ====================
# Third-party imports
from bs4 import BeautifulSoup
import telegram
import psycopg2
from flask import Flask, request, session, g
import google.generativeai as genai
from googleapiclient.discovery import build
from fuzzywuzzy import process, fuzz
from telegram import Update, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
    CallbackQueryHandler
)

def get_safe_font(text, style=None):
    """
    Normal text ko Premium Fonts mein convert karta hai.
    """
    if not text:
        return ""
    
    # 1. Bold Italic (𝑲𝒂𝒍𝒌𝒊 𝟐𝟖𝟗𝟖 𝑨𝑫)
    def to_bold_italic(s):
        result = ""
        for char in s:
            if 'a' <= char <= 'z': result += chr(0x1D482 + ord(char) - ord('a'))
            elif 'A' <= char <= 'Z': result += chr(0x1D468 + ord(char) - ord('A'))
            elif '0' <= char <= '9': result += chr(0x1D7CE + ord(char) - ord('0'))
            else: result += char
        return result

    return to_bold_italic(text)
# ==================== GLOBAL VARIABLES ====================
BATCH_18_SESSION = {'active': False, 'admin_id': None, 'files': []}

background_tasks = set()

DEFAULT_POSTER = os.environ.get(
    "DEFAULT_POSTER",
    "https://i.imgur.com/6XK4F6K.png"  # fallback placeholder
)
# ==================== CONVERSATION STATES (YEH MISSING HAI) ====================
WAITING_FOR_NAME, CONFIRMATION = range(2)
SEARCHING, REQUESTING, MAIN_MENU, REQUESTING_FROM_BUTTON = range(2, 6)
# ================= CONFIGURATION =================
# ================= CONFIGURATION =================
ANIME_CHANNEL_ID = "-1003523910286"
# =================================================

async def post_to_topic_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Forum topic par post karo + DB mein save karo (Restore ke liye)
    """
    user_id = update.effective_user.id
    if not is_admin(user_id):
        return

    # --- 1. MOVIE SEARCH ---
    movie_search_name = " ".join(context.args).strip() if context.args else ""

    conn = get_db_connection()
    if not conn:
        await update.message.reply_text("❌ Database connection failed.")
        return

    cursor = conn.cursor()

    query = """
        SELECT id, title, year, rating, genre, 
               poster_url, description, category, seasons_data 
        FROM movies
    """

    if movie_search_name:
        cursor.execute(
            query + " WHERE title ILIKE %s LIMIT 1",
            (f"%{movie_search_name}%",)
        )
    elif BATCH_SESSION.get('active'):
        cursor.execute(
            query + " WHERE id = %s",
            (BATCH_SESSION['movie_id'],)
        )
    else:
        await update.message.reply_text(
            "❌ Naam batao!\nExample: `/post Pushpa`",
            parse_mode='Markdown'
        )
        cursor.close()
        close_db_connection(conn)
        return

    movie_data = cursor.fetchone()
    cursor.close()
    close_db_connection(conn)

    if not movie_data:
        await update.message.reply_text("❌ Movie nahi mili database mein.")
        return

    # --- 2. DATA UNPACK ---
    movie_id, title, year, rating, genre, poster_url, description, category, seasons_data = movie_data
    
    import re
    if movie_search_name and seasons_data:
        season_match = re.search(r'(?i)season\s*(\d+)|s(\d+)', movie_search_name)
        if season_match:
            s_num = season_match.group(1) or season_match.group(2)
            s_num_str = str(int(s_num))
            if s_num_str in seasons_data:
                s_info = seasons_data[s_num_str]
                if s_info.get("year"): year = s_info["year"]
                if s_info.get("poster"): poster_url = s_info["poster"]
                title = f"{title} (Season {s_num_str})"

    # --- 3. TARGET CHANNEL SELECTION ---
    cat_lower = str(category or "").lower()
    
    target_channels = []
    if "anime" in cat_lower or "cartoon" in cat_lower or "animation" in cat_lower:
        target_channels = [ANIME_CHANNEL_ID]
    else:
        target_channels = [ch.strip() for ch in os.environ.get('BROADCAST_CHANNELS', '').split(',') if ch.strip()]

    if not target_channels:
        await update.message.reply_text("❌ No channels configured for posting.")
        return

    # --- 4. MISSING DATA HANDLE ---
    final_photo = (
        poster_url
        if poster_url and poster_url != 'N/A'
        else DEFAULT_POSTER
    )
    short_desc = (
        (description[:150] + "...")
        if description
        else "Plot details unavailable."
    )

    # --- 5. CAPTION ---
    caption = (
        f"🎬 **{title} ({year})**\n\n"
        f"⭐️ **Rating:** {rating}/10\n"
        f"🎭 **Genre:** {genre}\n"
        f"📝 **Plot:** {short_desc}\n\n"
        f"👇 **Download via the buttons below:** 👇"
    )

    # --- 6. KEYBOARD BUTTONS ---
    secure_url = f"https://flimfybox-bot-yht0.onrender.com/watch/{movie_id}"
    
    keyboard_data = {
        "inline_keyboard": [
            [
                {"text": "Get Now", "url": secure_url}
            ],
            [
                {"text": "Join Channel", "url": FILMFYBOX_CHANNEL_URL}
            ]
        ]
    }

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Get Now", url=secure_url)
        ],
        [
            InlineKeyboardButton("Join Channel", url=FILMFYBOX_CHANNEL_URL)
        ]
    ])

    # --- 7. POST SEND (Anti-Block Mode) ---
    # 👇 GLOBAL DUPLICATE CHECK — 7 din me kahi bhi post hui ho to skip
    if is_movie_posted_recently(movie_id, days=7):
        await update.message.reply_text(
            f"⏭️ **{title}** pehle se 7 din ke andar post ho chuki hai. Skipping.",
            parse_mode='Markdown'
        )
        return

    try:
        # Pehle image download karne ki koshish karo
        downloaded_poster = await get_poster_bytes(final_photo)

        # Agar download fail ho jaye, tabhi URL use karo (Fallback)
        photo_source = downloaded_poster if downloaded_poster else final_photo
        photo_to_send = await make_landscape_poster(photo_source)

        sent_msg = None
        for chat_id in target_channels:
            try:
                if hasattr(photo_to_send, 'read'):
                    photo_to_send.seek(0)
                sent = await context.bot.send_photo(
                    chat_id            = chat_id,
                    photo              = photo_to_send,
                    caption            = caption,
                    parse_mode         = 'Markdown',
                    reply_markup       = keyboard
                )
                if not sent_msg:
                    sent_msg = sent
            except Exception as e:
                logger.error(f"Failed to post to {chat_id}: {e}")

        # --- 8. DB SAVE (Restore ke liye) ---
        try:
            bot_info = await context.bot.get_me()
            if sent_msg:
                save_post_to_db(
                    movie_id      = movie_id,
                    channel_id    = target_channels[0],
                    message_id    = sent_msg.message_id,
                    bot_username  = bot_info.username,
                    caption       = caption,
                    media_file_id = final_photo,
                    media_type    = "photo",
                    keyboard_data = keyboard_data,
                    topic_id      = None,
                    content_type  = (
                        "adult"  if "adult"    in cat_lower else
                        "series" if "series"   in cat_lower else
                        "anime"  if "anime"    in cat_lower else
                        "movies"
                    )
                )
            save_status = "💾 DB mein save hua ✅"
        except Exception as save_err:
            logger.warning(f"Post DB save failed (non-critical): {save_err}")
            save_status = "⚠️ DB save nahi hua"

        await update.message.reply_text(
            f"✅ **{title}** posted in Topic `{topic_id}`\n"
            f"{save_status}",
            parse_mode='Markdown'
        )

    except Exception as e:
        logger.error(f"Post failed: {e}")
        await update.message.reply_text(f"❌ Post Error: {e}")

# ==================== ENVIRONMENT VARIABLES ====================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DATABASE_URL = os.environ.get('DATABASE_URL')
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
# Keep the Telegram Web App endpoint configurable. The old Render service was
# still hard-coded in several buttons, so Telegram opened the retired Mini App.
def normalize_mini_app_url(value: str) -> str:
    """Keep Telegram buttons on the current Mini App host and route."""
    fallback = 'https://flimfybox-bot-yht0.onrender.com/webapp'
    candidate = (value or '').strip()
    if not candidate:
        return fallback
    parsed = urlparse(candidate)
    if not parsed.scheme or not parsed.netloc:
        return fallback
    hostname = (parsed.hostname or '').lower()
    if hostname == 'flimfybox-bot-yht0.onrender.com':
        parsed = parsed._replace(netloc='flimfybox-bot-yht0.onrender.com')
    if not parsed.path or parsed.path == '/':
        parsed = parsed._replace(path='/webapp')
    return urlunparse(parsed).rstrip('/')


WEB_APP_URL = normalize_mini_app_url(
    os.environ.get('WEB_APP_URL', 'https://flimfybox-bot-yht0.onrender.com/webapp')
)
    # 👇👇👇 START COPY HERE 👇👇👇
db_pool = None
_startup_complete = threading.Event()
_shutdown_requested = threading.Event()
try:
    # Pool create kar rahe hain taki baar baar connection na banana pade
    pool_url = DATABASE_URL
    if pool_url:
        db_pool = psycopg2.pool.ThreadedConnectionPool(
            2, 8,  # Supabase Free Tier (60 connections) ke liye optimized
            dsn=pool_url
        )
        logger.info("✅ Database Connection Pool Created (Thread-Safe)!")
except Exception as e:
    logger.error(f"❌ Error creating pool: {e}")
# 👆👆👆 END COPY HERE 👆👆👆
BLOGGER_API_KEY = os.environ.get('BLOGGER_API_KEY')
BLOG_ID = os.environ.get('BLOG_ID')
UPDATE_SECRET_CODE = os.environ.get('UPDATE_SECRET_CODE')

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN environment variable is not set")
    raise ValueError("TELEGRAM_BOT_TOKEN is not set.")

if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is not set")
    raise ValueError("DATABASE_URL is not set.")

if not TMDB_API_KEY:
    logger.error("TMDB_API_KEY environment variable is not set")
    raise ValueError("TMDB_API_KEY is not set.")

if not UPDATE_SECRET_CODE:
    logger.error("UPDATE_SECRET_CODE environment variable is not set")
    raise ValueError("UPDATE_SECRET_CODE is not set.")
_admin_id = os.environ.get('ADMIN_USER_ID', '8675088364')
ADMIN_USER_ID = int(_admin_id) if _admin_id.isdigit() else 8675088364

# Dono accounts — main bot owner + userbot — dono ko full admin access
ADMIN_IDS = [ADMIN_USER_ID, 8438574164]
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'Ownermahi')  # Admin ka Telegram username

def is_admin(user_id: int) -> bool:
    """Check karo ki user owner/admin hai ya nahi (dono accounts)"""
    return user_id in ADMIN_IDS

GROUP_CHAT_ID = os.environ.get('GROUP_CHAT_ID')
ADMIN_CHANNEL_ID = os.environ.get('ADMIN_CHANNEL_ID')

# ==================== DYNAMIC FSUB SYSTEM ====================
ACTIVE_FSUB = {
    'id': os.environ.get('REQUIRED_CHANNEL_ID', '-1003916450868'),
    'url': 'https://t.me/FlimfyBoxx' 
}
BACKUP_FSUB_LIST = [
    {'id': '-1003916450868', 'url': 'https://t.me/FlimfyBoxx'}, 
    {'id': '-1002222222222', 'url': 'https://t.me/BackupChannel2'}  
]
# =============================================================

# 👇👇 YAHAN YE EK LINE PASTE KAR DO 👇👇
FILMFYBOX_CHANNEL_URL = ACTIVE_FSUB['url']

# 📢 Update/Backup Channel (search results, not-found message, etc. me use hoga)
UPDATE_CHANNEL_URL = "https://t.me/FlimfyBoxBackUp"

REQUIRED_GROUP_ID = os.environ.get('REQUIRED_GROUP_ID', '-1003930961567')
FILMFYBOX_GROUP_URL = 'https://t.me/+dxaCr_cMmGpkYTFl'
REQUEST_CHANNEL_ID = os.environ.get('REQUEST_CHANNEL_ID', '-1003078990647')
DUMP_CHANNEL_ID = os.environ.get('DUMP_CHANNEL_ID', '-1003893346701')
START_GIF_CHANNEL_ID = -1003893346701
START_GIF_MESSAGE_ID = 62
DUMP_CHANNEL_IDS = tuple(
    int(value.strip())
    for value in DUMP_CHANNEL_ID.split(',')
    if value.strip().lstrip('-').isdigit()
)

def get_primary_dump_channel_id() -> int:
    """Return the first configured dump channel for source-message copies."""
    if not DUMP_CHANNEL_IDS:
        raise ValueError("DUMP_CHANNEL_ID must contain at least one numeric Telegram chat ID")
    return DUMP_CHANNEL_IDS[0]
FORCE_JOIN_ENABLED = False

# ✅ NEW ENVIRONMENT VARIABLES FOR MULTI-CHANNEL & AI
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")  # ✅ NEW: Claude API Key
STORAGE_CHANNELS = os.environ.get("STORAGE_CHANNELS", "-1003823464401")  # ✅ NEW: Backup Channels List

# Verified users cache (Taaki baar baar API call na ho)
verified_users = {}
VERIFICATION_CACHE_TIME = 3600  # 1 Hour

# --- Random GIF IDs for Search Failure ---
SEARCH_ERROR_GIFS = [
    'https://media.giphy.com/media/26hkhKd2Cp5WMWU1O/giphy.gif',
    'https://media.giphy.com/media/3o7aTskHEUdgCQAXde/giphy.gif',
    'https://media.giphy.com/media/l2JhkHg5y5tW3wO3u/giphy.gif',
    'https://media.giphy.com/media/14uQ3cOFteDaU/giphy.gif',
    'https://media.giphy.com/media/xT9IgG50Fb7Mi0prBC/giphy.gif',
    'https://media.giphy.com/media/3o7abB06u9bNzA8lu8/giphy.gif',
    'https://media.giphy.com/media/3o7qDP7gNY08v4wYLy/giphy.gif',
]

# Rate limiting dictionary
user_last_request = defaultdict(lambda: datetime.min)

# ===== Configurable rate-limiting and fuzzy settings =====
REQUEST_COOLDOWN_MINUTES = int(os.environ.get('REQUEST_COOLDOWN_MINUTES', '10'))
SIMILARITY_THRESHOLD = int(os.environ.get('SIMILARITY_THRESHOLD', '80'))
MAX_REQUESTS_PER_MINUTE = int(os.environ.get('MAX_REQUESTS_PER_MINUTE', '10'))

# Auto-delete tracking
messages_to_auto_delete = defaultdict(list)

# ✅ NEW GLOBAL VARIABLES FOR BATCH SESSION
BATCH_SESSION = {'active': False, 'movie_id': None, 'movie_title': None, 'file_count': 0, 'admin_id': None}
SUPER_BATCH_SESSION = {'active': False, 'admin_id': None, 'files': []}

# Validate required environment variables
if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN environment variable is not set")
    raise ValueError("TELEGRAM_BOT_TOKEN is not set.")

if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is not set")
    raise ValueError("DATABASE_URL is not set.")


# 👇👇👇 START COPY HERE (Line 290 ke aas-paas paste karein) 👇👇👇
import functools

async def run_async(func, *args, **kwargs):
    """
    Ye function blocking code (jaise Database/Fuzzy search) ko
    background thread me chalata hai taaki bot hang na ho.
    """
    func_partial = functools.partial(func, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(None, func_partial)


def schedule_recommendation_event(*, user_id, event_type, source, movie_id=None, metadata=None):
    """Persist search telemetry without delaying the user-facing response."""
    task = asyncio.create_task(run_async(
        record_recommendation_event_safely,
        user_id,
        event_type,
        source,
        movie_id=movie_id,
        metadata=metadata,
    ))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
# 👆👆👆 END COPY HERE 👆👆👆


# ==================== 🛡️ SAFE_SEND — Global Anti-FloodWait Wrapper ====================
_send_semaphore = asyncio.Semaphore(25)  # Max 25 concurrent outgoing messages
_last_send_time = 0

async def safe_send(coro, max_retries=3):
    """
    🛡️ Global Anti-FloodWait Shield.
    Har high-risk outgoing message isse guzrega.
    - Semaphore se max 25 concurrent sends
    - Min 40ms gap (≈25 msg/sec)
    - RetryAfter auto-catch + wait + retry
    """
    global _last_send_time
    for attempt in range(max_retries):
        async with _send_semaphore:
            now = time.time()
            gap = now - _last_send_time
            if gap < 0.04:
                await asyncio.sleep(0.04 - gap)
            _last_send_time = time.time()
            try:
                return await coro
            except RetryAfter as e:
                wait = e.retry_after + 1
                logger.warning(f"⏳ FloodWait! Waiting {wait}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            except (TelegramError, Exception) as e:
                if 'flood' in str(e).lower():
                    logger.warning(f"⏳ Possible flood error, waiting 5s: {e}")
                    await asyncio.sleep(5)
                else:
                    logger.error(f"safe_send error: {e}")
                    if attempt == max_retries - 1:
                        raise
    return None


# ==================== 🧹 STRIP_CAPTION_JUNK — Caption Cleaner ====================
def strip_caption_junk(text):
    """
    File caption se third-party promotions, links, usernames strip karta hai.
    Hindi/Regional characters aur Anime/Series ki multi-line info ko SAFE rakhta hai.
    Call karo BEFORE generate_quality_label().
    """
    if not text:
        return text

    cleaned_lines = []
    lines = text.split('\n')
    
    # Protection Keywords (Quality aur Season/Episode info)
    protection_pattern = r'(?i)\b(1080p|720p|480p|360p|2160p|4k|s\d+|e\d+|season|episode|ep|hindi|english|dual audio|multi|sub|dub|bluray|web-dl|webrip)\b'
    
    # Promotional Keywords
    promo_pattern = r'(?i)\b(join\s*now|join|subscribe|channel|visit|powered\s*by|telegram|premium|group|owner|main\s*channel|movie\s*channel|backup)\b'

    for line in lines:
        original_line = line
        
        # Line-by-Line Filter: Kachra links remove karo
        line = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', line) # Markdown links
        line = re.sub(r'https?://\S+', '', line) # HTTP/HTTPS URLs
        line = re.sub(r'(?i)t\.me/\S+', '', line) # t.me/ links
        line = re.sub(r'@[a-zA-Z][a-zA-Z0-9_]{2,}', '', line) # @username (but file info like @480p safe)
        
        line = line.strip()
        
        if not line:
            continue
            
        # Smart Promo Drop: Agar line me promo words hain aur protection words NAHI hain, tabhi delete karo
        has_promo = re.search(promo_pattern, original_line)
        has_protection = re.search(protection_pattern, original_line)
        
        if has_promo and not has_protection:
            continue # Is line ko chhod do (delete)
            
        cleaned_lines.append(line)

    # Wapas lines join karo (taki multi-line structure safe rahe)
    text = '\n'.join(cleaned_lines)
    
    # Extra trailing spaces clean karo
    text = re.sub(r'[ \t]+', ' ', text).strip()

    return text


# ==================== UTILITY FUNCTIONS ====================

def extract_season_name(extra_info):
    """File ke extra_info se 'Season 1', 'Season 2' nikalta hai"""
    if not extra_info: 
        return "Extra Files"
    
    import re
    # S01, S1, Season 1, etc. ko dhoondhne ka regex
    match = re.search(r'(?i)(s\d{1,2}|season\s*\d+)', extra_info)
    if match:
        s = match.group().upper()
        # Number nikal lo (e.g., '01' se '1')
        num = re.search(r'\d+', s).group()
        return f"Season {int(num)}"
    return "Extra Files"
    
    import re
    # S01, S1, Season 1, etc. ko dhoondhne ka regex
    match = re.search(r'(?i)(s\d{1,2}|season\s*\d+)', extra_info)
    if match:
        s = match.group().upper()
        # Number nikal lo (e.g., '01' se '1')
        num = re.search(r'\d+', s).group()
        return f"Season {int(num)}"
    return "Extra Files"

async def get_poster_bytes(url):
    """
    Amazon/IMDb se fake browser (User-Agent) ban kar image download karta hai,
    taaki 'Region Block' wala error na aaye.
    """
    if not url or url == 'N/A':
        return None
        
    try:
        # Fake browser details taaki Amazon block na kare
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Referer': 'https://www.imdb.com/'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    image_data = await response.read()
                    return BytesIO(image_data) # Image ko bytes me convert kar diya
        return None
    except Exception as e:
        logger.error(f"Error downloading poster: {e}")
        return None

import re
import unicodedata

def preprocess_query(query):
    """Clean and normalize user query"""
    query = re.sub(r'[^\w\s-]', '', query)
    return query

def clean_telegram_text(text):
    """Removes emojis and converts fancy fonts to normal text"""
    if not text: return ""
    
    # 👇 NAYA FIX: Sabse pehle ye faltu text hatao (taaki fancy font normal hone se pehle hi cut jaye)
    text = text.replace("@BuLMoviee 𝗝𝗼𝗶𝗻 𝗨𝘀 𝗢𝗻 𝗧𝗲𝗹𝗲𝗴𝗿𝗮𝗺", "")
    
    fancy = {'ᴀ':'a','ʙ':'b','ᴄ':'c','ᴅ':'d','ᴇ':'e','ғ':'f','ɢ':'g','ʜ':'h','ɪ':'i','ᴊ':'j','ᴋ':'k','ʟ':'l','ᴍ':'m','ɴ':'n','ᴏ':'o','ᴘ':'p','ǫ':'q','ʀ':'r','s':'s','ᴛ':'t','ᴜ':'u','ᴠ':'v','ᴡ':'w','x':'x','ʏ':'y','ᴢ':'z'}
    for k, v in fancy.items(): 
        text = text.replace(k, v)
    
    text = unicodedata.normalize('NFKC', text)
    
    text = re.sub(r'[^\w\s\.\-\'\[\]\(\)@:]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    text = re.sub(r'^[\.\-\s]+', '', text)
    
    # "Name:", "Title:", "File Name:" jaise words ko shuruat se hata dega
    text = re.sub(r'(?i)^(name|title|file\s*name|movie)\s*:\s*', '', text).strip()
    
    # 👇 Ek aur safety check: Agar normalize hone ke baad simple text me bach gaya ho toh wo bhi hata dega
    text = text.replace("@BuLMoviee Join Us On Telegram", "").strip()
    
    return text
def _process_poster_sync(image_data):
    """Build the square blurred-background poster used by example.py."""
    from PIL import Image, ImageFilter, ImageOps

    img = Image.open(BytesIO(image_data)).convert("RGB")
    target_w = target_h = 800

    bg = ImageOps.fit(
        img,
        (target_w, target_h),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    bg = bg.filter(ImageFilter.GaussianBlur(radius=40))

    fg_h = int(target_h * 0.95)
    fg_w = int(img.width * (fg_h / img.height))
    fg = img.resize((fg_w, fg_h), Image.Resampling.LANCZOS)
    bg.paste(fg, ((target_w - fg_w) // 2, (target_h - fg_h) // 2))

    output = BytesIO()
    output.name = "square_poster.jpg"
    bg.save(output, format="JPEG", quality=95)
    output.seek(0)
    return output


async def make_landscape_poster(url_or_bytes):
    """Download and transform a poster without blocking Telegram handlers."""
    if not url_or_bytes:
        return url_or_bytes

    if isinstance(url_or_bytes, str) and url_or_bytes.startswith("http"):
        cached = poster_cache.get(url_or_bytes)
        if cached is not None:
            return BytesIO(cached)
        downloaded = await get_poster_bytes(url_or_bytes)
        if not downloaded:
            return url_or_bytes
        image_data = downloaded.getvalue()
    elif isinstance(url_or_bytes, bytes):
        image_data = url_or_bytes
    elif hasattr(url_or_bytes, "getvalue"):
        image_data = url_or_bytes.getvalue()
    else:
        return url_or_bytes

    try:
        processed = await run_async(_process_poster_sync, image_data)
        poster_bytes = processed.getvalue()
        if isinstance(url_or_bytes, str):
            poster_cache.set(url_or_bytes, poster_bytes)
        return BytesIO(poster_bytes)
    except Exception as exc:
        logger.warning("Cinematic poster processing failed: %s", exc)
        return url_or_bytes


async def check_rate_limit(user_id):
    """Check if user is rate limited"""
    now = datetime.now()
    last_request = user_last_request[user_id]

    if now - last_request < timedelta(seconds=2):
        return False

    user_last_request[user_id] = now
    return True

async def upload_image_to_telegraph(bot, file_id):
    """Downloads photo from Telegram and uploads to Telegra.ph"""
    try:
        # File download karo
        file = await bot.get_file(file_id)
        byte_array = await file.download_as_bytearray()
        
        # Telegraph par upload karo
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('file', byte_array, filename='poster.jpg', content_type='image/jpeg')
            
            async with session.post('https://telegra.ph/upload', data=data) as resp:
                res = await resp.json()
                if isinstance(res, list) and 'src' in res[0]:
                    return f"https://telegra.ph{res[0]['src']}"
        return None
    except Exception as e:
        logger.error(f"❌ Telegraph upload failed: {e}")
        return None

# 👇 NAYA HELPER FUNCTION: Yeh aapki saari keys .env se nikal lega
def get_gemini_keys():
    keys = []
    # Purani standard key check karein
    std_key = os.environ.get("GEMINI_API_KEY")
    if std_key: keys.append(std_key)
    
    # Nayi numbered keys check karein (1 se 5 tak)
    for i in range(1, 6):
        k = os.environ.get(f"GEMINI_API_KEY_{i}")
        if k and k not in keys:
            keys.append(k)
    return keys


# 👇 UPDATED FUNCTION 1: Name Extraction (With Multi-Key Rotation)
async def get_movie_name_from_caption(caption_text, image_bytes=None):
    """
    🎯 FULLY AI-POWERED EXTRACTION (MULTIMODAL WITH AUTO-KEY ROTATION)
    🔧 FIXED: Ab first 5 lines bheji jaati hain + better prompt + retry logic
    """
    if not caption_text or len(caption_text.strip()) < 2:
        return {"title": "UNKNOWN", "year": "", "language": "", "extra_info": "", "category": ""}
    
    # 🔧 FIX: Pehle sirf first_line jaati thi — ab first 5 lines bheji jaayengi
    # Kyunki bahut se captions mein pehli line promo/group name hoti hai
    caption_lines = caption_text.strip().split('\n')
    # First 5 lines clean karke bhejo (ya jitni bhi hain)
    cleaned_lines = []
    for line in caption_lines[:5]:
        cleaned = clean_telegram_text(line.strip())
        if cleaned and len(cleaned) > 1:
            cleaned_lines.append(cleaned)
    
    caption_for_ai = '\n'.join(cleaned_lines) if cleaned_lines else clean_telegram_text(caption_lines[0].strip())
    first_line = cleaned_lines[0] if cleaned_lines else clean_telegram_text(caption_lines[0].strip())
    
    logger.info(f"📝 Processing caption ({len(cleaned_lines)} lines): {first_line[:100]}...")

    gemini_keys = get_gemini_keys()

    if gemini_keys:
        # 🔧 FIX: Enhanced prompt — explicitly tells AI to ignore promo/group text
        prompt = f"""Extract movie/series info from this file caption. Return ONLY JSON.

Caption:
\"\"\"
{caption_for_ai}
\"\"\"

IMPORTANT Rules:
- title: The ACTUAL movie/series name. Remove S01, E01, group tags, quality info, file extensions.
- IGNORE channel names, group promotions, @usernames, "Join" text — these are NOT the movie name.
- If multiple lines, the movie name is usually the line with quality tags (720p, 1080p) or file extension (.mkv, .mp4).
- year: 4-digit year if present (like 2023, 2024)
- language: Audio languages mentioned (Hindi, English, Multi Audio, Dual Audio, etc.)
- extra_info: Season/episode info (e.g., "S01 E01-12 COMBINED")
- category: 'Web Series' if season/episode found, 'Anime' if anime, else 'Movies'

Example 1:
Input: "A Gatherer's Adventure In Isekai S01 [E01-12] COMBiNED 720p AMZN WEB-DL HEVC Multi DDP2.0 MSub"
Output: {{"title": "A Gatherer's Adventure In Isekai", "year": "", "language": "Multi Audio", "extra_info": "S01 E01-12 COMBINED", "category": "Web Series"}}

Example 2:
Input: "@MovieChannel Join Now\\nPushpa 2 The Rule 2024 1080p WEB-DL Hindi DD5.1"
Output: {{"title": "Pushpa 2 The Rule", "year": "2024", "language": "Hindi", "extra_info": "", "category": "Movies"}}

JSON:"""

        contents = [prompt]
        if image_bytes:
            contents.append({"mime_type": "image/jpeg", "data": image_bytes})

        # 🚀 KEY ROTATION LOOP
        for key in gemini_keys:
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel('gemini-flash-latest')
                response = await run_async(model.generate_content, contents)
                
                if response and response.text:
                    text = response.text.strip()
                    text = re.sub(r'```json|```', '', text)
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        if data.get("title") and len(data["title"]) > 2:
                            logger.info(f"✅ Gemini Success (Key used: {key[:5]}...): {data['title']}")
                            return data
                break # Agar response mila par JSON galat hai, toh aage wali key waste mat karo
                
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"🛑 Asli Gemini Error Key {key[:5]} par: {str(e)}")
                
                if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                    logger.warning(f"⚠️ Key {key[:5]}... limit reached. Shifting to next key...")
                    continue
                else:
                    logger.error(f"❌ Gemini Error on key {key[:5]}...: {e}")
                    break

    # FALLBACK: Improved version
    logger.info("⚠️ Keys exhausted or failed. Using fallback extraction...")
    
    # 🔧 FIX: Try each cleaned line for fallback (not just first line)
    # Sometimes the actual filename is on a different line
    for line in cleaned_lines:
        result = await fallback_extraction(line)
        if result.get("title") and result["title"] != "UNKNOWN" and len(result["title"]) > 2:
            return result
    
    # Last resort: try first line
    return await fallback_extraction(first_line)


# 👇 UPDATED FUNCTION 2: Alias Generation (With Multi-Key Rotation)
def generate_aliases_gemini(movie_title, year="", category=""):
    """
    🎯 AI se 50 search aliases generate karta hai (WITH AUTO-KEY ROTATION)
    """
    logger.info(f"🚀 Generating aliases for: '{movie_title}' ({year}) [{category}]")
    
    if not movie_title or movie_title == "UNKNOWN":
        return []
    
    gemini_keys = get_gemini_keys()
    if not gemini_keys:
        logger.error("❌ No GEMINI_API_KEY found!")
        return generate_basic_aliases(movie_title, year)

    prompt = f"""Generate 50 search aliases for the movie/show: "{movie_title}"
Year: {year if year else "N/A"}
Category: {category if category else "N/A"}

Include these types of variations:
1. Common misspellings (typos people make)
2. With and without year
3. Hindi transliterations if applicable
4. Short forms and abbreviations
5. With "movie", "film", "download" keywords
6. Without spaces, with hyphens
7. Regional language spellings

IMPORTANT: Return ONLY comma-separated aliases, nothing else.
Example format: alias1, alias2, alias3, alias4"""

    safety_settings = {
        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
    }

    # 🚀 KEY ROTATION LOOP
    for key in gemini_keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-flash-latest')
            response = model.generate_content(prompt, safety_settings=safety_settings)
            
            if not response or not response.parts:
                logger.warning("Gemini response was empty or blocked. Trying basic.")
                return generate_basic_aliases(movie_title, year)
            
            ai_text = response.text.strip()
            aliases = []
            
            # 👇 FIX 1: Ab bot comma (,) aur New Line (\n) dono ko split kar lega
            raw_items = re.split(r',|\n', ai_text)
            
            for item in raw_items:
                alias = item.strip().lower()
                # Numbers, bullets (*, -) sab hata dega
                alias = re.sub(r'^[\d\.\-\*\)]+\s*', '', alias).strip('"\'').strip()
                if alias and len(alias) >= 2 and len(alias) <= 100:
                    aliases.append(alias)
            
            aliases = list(dict.fromkeys(aliases))[:50]
            
            # 👇 FIX 2: Agar AI ne ajeeb format diya aur 0 alias bache, toh Basic aliases (Fallback) use kar lo
            if not aliases:
                logger.warning("AI returned bad format. Using fallback aliases.")
                return generate_basic_aliases(movie_title, year)
                
            logger.info(f"✅ Generated {len(aliases)} aliases (Key used: {key[:5]}...)")
            return aliases

        except Exception as e:
            error_msg = str(e).lower()
            # 👇 Yahan ek print statement add karo taaki actual error log me dikhe
            logger.error(f"🛑 Asli Gemini Error Key {key[:5]} par: {str(e)}")
            
            if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                logger.warning(f"⚠️ Key {key[:5]}... limit reached. Shifting to next key...")
                continue
            else:
                logger.warning(f"❌ Alias Gemini error on key {key[:5]}...: {e}")
                break

    return generate_basic_aliases(movie_title, year)

def generate_basic_aliases(title, year=""):
    """
    Fallback function to generate simple search aliases without AI.
    """
    aliases = set()
    title_lower = title.lower().strip()
    aliases.add(title_lower)
    aliases.add(title_lower.replace(" ", ""))
    if year:
        aliases.add(f"{title_lower} {year}")
        aliases.add(f"{title_lower}{year}")
    # Remove leading 'the' variations
    if title_lower.startswith("the "):
        base = title_lower[4:]
        aliases.add(base)
        if year:
            aliases.add(f"{base} {year}")
            aliases.add(f"{base}{year}")
    return list(aliases)

def normalize_episodes(text):
    # 1. E12 22 -> E12-22
    text = re.sub(r'(?i)\b(e|ep|episode)\s*(\d{1,3})\s+(\d{1,3})\b(?!\s*p)', r'\1\2-\3', text)
    
    # 2. E12 e22 / E12 ep22 -> E12-22
    text = re.sub(r'(?i)\b(e|ep|episode)\s*(\d{1,3})\s+(?:e|ep|episode)\s*(\d{1,3})\b', r'\1\2-\3', text)
    
    # 3. E12 to 22 / E12 to22 -> E12-22
    text = re.sub(r'(?i)\b(e|ep|episode)\s*(\d{1,3})\s*to\s*(?:e|ep|episode)?\s*(\d{1,3})\b', r'\1\2-\3', text)
    
    return text
    
# =================================================================================
# EVIDENCE ENGINE - PHASE 1 FOUNDATION
# =================================================================================

_EVIDENCE_KEYS = ("title", "year", "language", "extra_info", "category")


def _valid_evidence_title(value):
    value = str(value or "").strip()
    if not value or len(value) < 2:
        return False
    upper_val = value.upper()
    if upper_val in {"UNKNOWN", "UNKNOWN_MOVIE", "FILE", "DOCUMENT", "VIDEO"}:
        return False

    junk_pattern = re.compile(
        r'^(?:\W|episode\s*\W*\s*\d+|ep\s*\W*\s*\d+|e\d+|season\s*\W*\s*\d+|s\d+|quality\s*\W*\s*|1080p|720p|480p|2160p|4k|web-dl|webrip|bluray|hdrip|camrip|language\s*\W*\s*|hindi dub|dub by\s*\W*\s*|@\w+|join now|subscribe|download|powered by|hindi|tamil|telugu|malayalam|kannada|english|dual audio)+$',
        re.IGNORECASE
    )
    if junk_pattern.match(value):
        return False

    return True


def _normalize_evidence_dict(data):
    """Gemini/local parser output ko stable five-field dictionary mein normalize karta hai."""
    data = data if isinstance(data, dict) else {}
    normalized = {}
    for key in _EVIDENCE_KEYS:
        value = data.get(key, "")
        if value is None:
            value = ""
        normalized[key] = str(value).strip()
    if not normalized["category"]:
        normalized["category"] = "Movies"
    return normalized


def _merge_csv_values(primary, secondary):
    """Languages jaise comma/plus separated values ko order preserve karke merge karta hai."""
    items = []
    seen = set()
    for raw in (primary, secondary):
        if not raw:
            continue
        for item in re.split(r'\s*(?:,|\+|\||/)\s*', str(raw)):
            item = item.strip()
            if not item:
                continue
            key = item.casefold()
            if key not in seen:
                seen.add(key)
                items.append(item)
    return ", ".join(items)


def _merge_extra_info(primary, secondary):
    first = str(primary or "").strip()
    second = str(secondary or "").strip()
    if not first:
        return second
    if not second:
        return first
    if first.casefold() == second.casefold() or second.casefold() in first.casefold():
        return first
    if first.casefold() in second.casefold():
        return second
    return f"{first} {second}".strip()


def _local_evidence_fallback(caption_evidence, filename_evidence, forward_source=None):
    """Gemini unavailable/invalid ho to deterministic, field-wise safe merge."""
    cap = _normalize_evidence_dict(caption_evidence)
    fn = _normalize_evidence_dict(filename_evidence)

    cap_title_ok = _valid_evidence_title(cap.get("title"))
    fn_title_ok = _valid_evidence_title(fn.get("title"))
    
    fwd_title_ok = False
    fwd_title = ""
    if forward_source and forward_source.get("available"):
        t = forward_source.get("title", "")
        fwd_title_ok = _valid_evidence_title(t)
        fwd_title = t

    title = cap["title"] if cap_title_ok else fn["title"] if fn_title_ok else fwd_title if fwd_title_ok else "UNKNOWN"

    category = cap.get("category") or fn.get("category") or "Movies"
    for candidate in (cap.get("category", ""), fn.get("category", "")):
        if str(candidate).casefold() in {"web series", "series", "anime"}:
            category = candidate
            break

    return {
        "title": title,
        "year": cap.get("year") or fn.get("year") or "",
        "language": _merge_csv_values(cap.get("language"), fn.get("language")),
        "extra_info": _merge_extra_info(cap.get("extra_info"), fn.get("extra_info")),
        "category": category,
    }


def _get_message_filename(message):
    """Raw Telegram message object se original media filename nikalta hai."""
    for attr in ("document", "video", "audio", "animation"):
        media = getattr(message, attr, None)
        if media:
            return getattr(media, "file_name", "") or ""
    return ""


async def extract_same_file_evidence(message):
    """Same Telegram file ka raw caption, raw filename aur dono local parser results."""
    caption_raw = (getattr(message, "caption", None) or getattr(message, "text", None) or "").strip()
    filename_raw = _get_message_filename(message).strip()

    # EXTRACT FORWARD INFO
    forward_title = ""
    forward_username = ""
    forward_id = ""
    is_forward = False
    
    if getattr(message, "forward_origin", None):
        is_forward = True
        origin = message.forward_origin
        if getattr(origin, "chat", None):
            forward_title = getattr(origin.chat, "title", "") or ""
            forward_username = getattr(origin.chat, "username", "") or ""
            forward_id = str(getattr(origin.chat, "id", "")) or ""
    elif getattr(message, "forward_from_chat", None):
        is_forward = True
        forward_title = getattr(message.forward_from_chat, "title", "") or ""
        forward_username = getattr(message.forward_from_chat, "username", "") or ""
        forward_id = str(getattr(message.forward_from_chat, "id", "")) or ""

    caption_evidence = await fallback_extraction(caption_raw) if caption_raw else {}
    filename_evidence = await fallback_extraction(filename_raw) if filename_raw else {}

    return {
        "caption_raw": caption_raw,
        "filename_raw": filename_raw,
        "caption_evidence": _normalize_evidence_dict(caption_evidence),
        "filename_evidence": _normalize_evidence_dict(filename_evidence),
        "forward_source": {
            "title": forward_title,
            "username": forward_username,
            "chat_id": forward_id,
            "available": is_forward
        }
    }


async def reconcile_evidence_with_gemini(
    caption_evidence: dict,
    filename_evidence: dict,
    caption_raw: str = "",
    filename_raw: str = "",
    forward_source: dict = None,
) -> dict:
    """
    Caption aur raw Telegram filename SAME file ke do evidence sources hain.
    Gemini sirf identity reconcile karta hai; TMDB/IMDb lookup baad mein code karta hai.
    """
    fallback = _local_evidence_fallback(caption_evidence, filename_evidence, forward_source)
    
    fwd_log = (forward_source or {}).get('title', 'UNKNOWN')
    cap_log = caption_evidence.get('title', 'UNKNOWN')
    fn_log = filename_evidence.get('title', 'UNKNOWN')
    logger.info("Evidence: forward_title='%s', caption_title='%s', filename_title='%s'", fwd_log, cap_log, fn_log)

    gemini_keys = get_gemini_keys()
    if not gemini_keys:
        return fallback

    evidence_bundle = {
        "source_context": (
            "You are receiving evidence from a Telegram media file. "
            "You have forward metadata (if forwarded), the caption, and the filename."
        ),
        "forward_source": forward_source or {
            "title": "",
            "username": "",
            "chat_id": "",
            "available": False
        },
        "caption_source": {
            "raw_text": caption_raw or "",
            "locally_extracted": _normalize_evidence_dict(caption_evidence),
        },
        "filename_source": {
            "raw_text": filename_raw or "",
            "locally_extracted": _normalize_evidence_dict(filename_evidence),
        },
    }

    prompt = f"""You are a movie/series identity reconciliation engine.

You are receiving ALL available evidence for a single Telegram media file:
1. The forwarded source channel metadata (if available).
2. The raw message caption and its locally extracted fields.
3. The raw Telegram filename and its locally extracted fields.

Your goal is to RECONCILE the evidence into ONE correct movie/series identity.
Local extraction is only a hint and may incorrectly extract metadata (e.g. 'Episode 12' or 'Hindi Dub') as the title.
Forwarded channel titles are strong evidence but NOT blind absolute truth (e.g., 'Latest Anime Updates' is generic, not a show name).

Rules:
- Return ONLY one valid JSON object; no markdown or explanation.
- Output keys must be exactly: title, year, language, extra_info, category.
- title: determine the most plausible official movie/series title. Reject metadata-only junk ('Episode 12'). Do not assume generic channel names are titles. If conflicts exist, pick the most plausible title.
- year: four digits only when supported by evidence; otherwise empty string.
- language: merge supported audio languages (e.g. 'Hindi Dub').
- extra_info: only season, episode, part, combined/complete, or edition information (e.g. 'Episode 12').
- category: Movies, Web Series, or Anime.
- If all sources contain only junk or generic promotional text, return UNKNOWN for title.
- Never fabricate a title from episode/quality/language metadata.

Same-file evidence bundle:
{json.dumps(evidence_bundle, ensure_ascii=False, indent=2)}

Required JSON example:
{{"title":"Movie Name","year":"2026","language":"Hindi Dub, English","extra_info":"Episode 12","category":"Web Series"}}
"""

    last_error = None
    for key in gemini_keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-flash-latest')
            response = await run_async(model.generate_content, prompt)
            response_text = (getattr(response, "text", "") or "").strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("Gemini returned no JSON object")

            parsed = json.loads(json_match.group())
            if not isinstance(parsed, dict):
                raise ValueError("Gemini JSON was not an object")

            final_data = _normalize_evidence_dict(parsed)
            for field in _EVIDENCE_KEYS:
                if not final_data.get(field):
                    final_data[field] = fallback.get(field, "")
            if not _valid_evidence_title(final_data.get("title")):
                final_data["title"] = fallback.get("title", "UNKNOWN")

            logger.info(
                "✅ Evidence reconciliation success. Final identity: '%s' (%s)",
                final_data.get("title"),
                final_data.get("year") or "no year",
            )
            return final_data
        except Exception as exc:
            last_error = exc
            error_msg = str(exc).lower()
            if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                logger.warning("⚠️ Evidence Gemini key quota exhausted; trying next key")
            else:
                logger.warning("⚠️ Evidence Gemini key failed; trying next key: %s", exc)

    logger.error("Evidence Engine reconciliation failed on all keys: %s", last_error)
    return fallback


async def process_file_with_evidence_engine(message) -> dict:
    """Normal auto-batch ke liye local caption+filename extraction, phir one Gemini reconciliation."""
    evidence = await extract_same_file_evidence(message)
    return await reconcile_evidence_with_gemini(
        evidence["caption_evidence"],
        evidence["filename_evidence"],
        caption_raw=evidence["caption_raw"],
        filename_raw=evidence["filename_raw"],
        forward_source=evidence.get("forward_source"),
    )


def _canonical_evidence_title(title):
    """Superbatch grouping ke liye conservative canonical title."""
    value = clean_telegram_text(str(title or "")).casefold()
    value = re.sub(r'\b(19|20)\d{2}\b', ' ', value)
    value = re.sub(r'[^\w\s]', ' ', value)
    value = re.sub(r'\s+', ' ', value).strip()
    return value


def _evidence_record_score(record):
    """Best representative file choose karne ke liye completeness score."""
    cap = record.get("caption_evidence", {}) or {}
    fn = record.get("filename_evidence", {}) or {}
    score = 0
    if _valid_evidence_title(cap.get("title")): score += 35
    if cap.get("year"): score += 15
    if cap.get("language"): score += 12
    if cap.get("extra_info"): score += 10
    if record.get("caption"): score += min(len(str(record.get("caption"))), 220) / 20
    if _valid_evidence_title(fn.get("title")): score += 18
    if fn.get("year"): score += 8
    if record.get("file_name"): score += 4
    return score


def _best_local_identity(record):
    merged = _local_evidence_fallback(
        record.get("caption_evidence", {}),
        record.get("filename_evidence", {}),
        forward_source=record.get("forward_source"),
    )
    raw_title = merged.get("title") or "Unknown_Movie"
    parsed = parse_content_identity(raw_title)
    title = (
        parsed.canonical_title
        if is_safe_canonical_title(parsed.canonical_title)
        else raw_title
    )
    year = str(merged.get("year") or parsed.year or "").strip()
    return title, year, _canonical_evidence_title(title)


def _build_superbatch_groups(files):
    """
    Conservative local grouping:
    exact canonical title + compatible year first; fuzzy merge only when one
    unambiguous very-high-confidence match exists. Ambiguous remakes stay separate.
    """
    prepared = []
    for index, record in enumerate(files):
        title, year, canonical = _best_local_identity(record)
        record["display_title"] = f"{title} ({year})" if year else title
        prepared.append((record, title, year, canonical, index))

    prepared.sort(key=lambda item: (bool(item[2]), _evidence_record_score(item[0])), reverse=True)
    groups = []

    for record, title, year, canonical, original_index in prepared:
        compatible = []
        for group in groups:
            group_year = group["year"]
            years_ok = not year or not group_year or year == group_year
            if not years_ok:
                continue
            if canonical and canonical == group["canonical"]:
                compatible.append((100, group))
            elif canonical and group["canonical"]:
                similarity = fuzz.token_set_ratio(canonical, group["canonical"])
                if similarity >= 96:
                    compatible.append((similarity, group))

        compatible.sort(key=lambda item: item[0], reverse=True)
        selected = None
        if len(compatible) == 1:
            selected = compatible[0][1]
        elif len(compatible) > 1 and compatible[0][0] >= compatible[1][0] + 3:
            selected = compatible[0][1]

        if selected is None:
            selected = {
                "canonical": canonical or f"unknown-{original_index}",
                "year": year,
                "display_title": record.get("display_title") or title,
                "files": [],
            }
            groups.append(selected)
        elif not selected["year"] and year:
            selected["year"] = year
            selected["display_title"] = f"{title} ({year})"

        selected["files"].append(record)

    grouped = {}
    for idx, group in enumerate(groups, 1):
        key = f"{group['canonical']}_{group['year'] or 'unknown'}_{idx}"
        grouped[key] = group["files"]
    return grouped


def _select_representative_file(movie_files):
    return max(movie_files, key=_evidence_record_score)


def _split_quality_label(label):
    text = str(label or "").strip()
    lower = text.casefold()
    resolution = ""
    if "4k" in lower or "2160p" in lower:
        resolution = "4K"
    else:
        match = re.search(r'\b(1080p|720p|576p|480p|360p)\b', text, re.IGNORECASE)
        if match:
            resolution = match.group(1).lower()

    source = ""
    for candidate in (
        "WEB-DL", "BluRay", "Remux", "WEBRip", "HDRip", "HDTV",
        "HDTC", "HDTS", "PreDVD", "DVDScr", "HDCAM", "CAMRip",
    ):
        if candidate.casefold() in lower:
            source = candidate
            break
    return resolution, source


def _merge_quality_labels(caption_label, filename_label):
    """Resolution/source separately merge; explicit caption field has priority."""
    cap_res, cap_source = _split_quality_label(caption_label)
    fn_res, fn_source = _split_quality_label(filename_label)
    resolution = cap_res or fn_res or "HD"
    source = cap_source or fn_source

    if cap_res and fn_res and cap_res.casefold() != fn_res.casefold():
        logger.warning("⚠️ Caption/filename resolution conflict: %s vs %s; caption preferred", cap_res, fn_res)
    if cap_source and fn_source and cap_source.casefold() != fn_source.casefold():
        logger.warning("⚠️ Caption/filename source conflict: %s vs %s; caption preferred", cap_source, fn_source)

    return f"{resolution}{' ' + source if source else ''}".strip()


async def fallback_extraction(caption_text):
    """
    SMART FALLBACK: Improved regex-based extraction for both movies and web series.
    """
    try:
        text = clean_telegram_text(caption_text.strip())
        original = text

        # 1. Remove obvious group prefixes and promotional words
        text = re.sub(r'(?i)^(join\s+)?@\w+\s*', '', text)  # JOIN @channel ko udayega
        text = re.sub(r'(?i)^join\s+', '', text)            # Sirf JOIN likha ho toh udayega
        text = re.sub(r'^\{[^}]+\}\s*', '', text)           # {@Royal_Backup2} ko udayega
        text = re.sub(r'^@\w+\s+', '', text)                # @MRKUPDATES4U6 ko udayega
        text = re.sub(r'^\[[^\]]+\]\s*', '', text)          # [Group] ko udayega
        
        # 2. Detect if it's a web series (contains season/episode indicators)
        season_pattern = re.compile(r'\b(S\d{1,2}|Season\s*\d+|S\d{1,2}E\d{1,3}|\[?E\d{1,3}\s*(?:[-~_]|to)\s*(?:e|ep)?\d{1,3}\]?|EP\s*\d{1,3}(?:\s*(?:[-~_]|to)\s*(?:e|ep)?\d{1,3})?|Episode\s*\d+|Part\s*\d+|P\d+)\b', re.IGNORECASE)
        season_match = season_pattern.search(text)
        if season_match:
            # Use existing web series logic (kept from original)
            return await _extract_web_series(text, original)

        # 3. MOVIE EXTRACTION
        # Try to find year
        year_match = re.search(r'[\(\[]?(19|20)\d{2}[\)\]]?', text)
        year = year_match.group() if year_match else ""
        # Clean year to just digits
        year_clean = re.sub(r'[^0-9]', '', year) if year else ""

        # Determine split point
        split_pos = None
        if year_match:
            split_pos = year_match.start()
        else:
            # Look for first quality/resolution tag
            quality_patterns = [
                r'\b\d{3,4}p\b',                     # 480p, 720p, 1080p
                r'\b(HDRip|WEB-DL|BluRay|DVDRip|BRRip|HDTV|WEBRip|DS4K)\b',
                r'\.(mkv|mp4|avi|m4v)$'              # file extension at end
            ]
            for pat in quality_patterns:
                q_match = re.search(pat, text, re.IGNORECASE)
                if q_match:
                    split_pos = q_match.start()
                    break

        # Extract title
        if split_pos is not None:
            title_part = text[:split_pos].strip()
        else:
            title_part = text

        # Clean title
        title = title_part
        # Replace separators with space
        title = re.sub(r'[._\-]+', ' ', title)
        # Remove any remaining brackets and their content (often group names)
        title = re.sub(r'[\[\(].*?[\]\)]', '', title)
        # Remove URLs, mentions, hashtags
        title = re.sub(r'https?://\S+', '', title)
        title = re.sub(r'@\w+', '', title)
        title = re.sub(r'#\w+', '', title)
        # Collapse multiple spaces and strip
        title = re.sub(r'\s+', ' ', title).strip()
        # Remove trailing noise words (common tech tags)
        junk_words = [
            'hindi', 'english', 'tamil', 'telugu', 'malayalam', 'kannada', 'bengali',
            'dubbed', 'multi', 'audio', 'ddp', 'web', 'dl', 'bluray', 'amzn', 'hevc',
            'x264', 'x265', 'mkv', 'mp4', 'avi', '480p', '720p', '1080p', '2160p',
            'hdrip', 'webdl', 'webrip', 'hdtv', 'ds4k', 'uncut', 'extended', 'directors',
            'cut', 'edition', 'repack', 'proper', 'internal', 'nf', 'hulu', 'hotstar',
            'sony', 'zee5', 'mubi', 'esub', 'sub', 'aac', 'ac3', 'dd5', 'dd2', 'ddp5',
            'ddp2', 'xvid', 'divx', 'remux', 'bdrip', 'brrip', 'dvdrip', 'dvdr', 'pal',
            'ntsc', 'region', 'free', 'watch', 'online', 'download', 'movies', 'series',
            'show', 'south', 'movie', 'org', 'dual', 'truehd', 'atmos', 'dts', 'mp3',
            'flac', 'opus', 'aac2', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'xvid', 'hd', 'full', 'half', 'brrip', 'bdrip', 'web', 'dl', 'hdr'
        ]
        words = title.split()
        if words:
            # Remove trailing junk words
            while words and words[-1].lower() in junk_words:
                words.pop()
            # Also remove leading junk? (rare, but possible)
            while words and words[0].lower() in junk_words:
                words.pop(0)
            title = ' '.join(words)

        # If title is too short after cleaning, reject it
        if len(title) < 3:
            title = "UNKNOWN"

        # 4. Language extraction (same as original)
        languages = []
        lang_map = {
            'japanese|日本語': 'Japanese', 'english': 'English',
            'hindi|हिन्दी': 'Hindi', 'tamil|தமிழ்': 'Tamil',
            'telugu|తెలుగు': 'Telugu', 'malayalam': 'Malayalam',
            'korean': 'Korean', 'dual.*audio': 'Dual Audio',
            'multi.*audio': 'Multi Audio'
        }
        for pattern, name in lang_map.items():
            if re.search(pattern, text, re.IGNORECASE):
                languages.append(name)
        language = ', '.join(dict.fromkeys(languages)) if languages else ""

        # 5. Extra info (for movies, we might capture edition like UNCUT, EXTENDED)
        extra_info = ""
        edition_match = re.search(r'\b(UNCUT|EXTENDED|DIRECTOR\'?S?\s*CUT|THEATRICAL|UNRATED|REMASTERED)\b', text, re.IGNORECASE)
        if edition_match:
            extra_info = edition_match.group(0).upper()

        # 6. Category
        category = "Movies"

        logger.info(f"✅ Movie Fallback: '{title}' | Year: {year_clean} | Lang: {language} | Extra: {extra_info} | Cat: {category}")

        return {
            "title": title,
            "year": year_clean,
            "language": language,
            "extra_info": extra_info,
            "category": category
        }

    except Exception as e:
        logger.error(f"❌ Fallback error: {e}")
        return {"title": "UNKNOWN", "year": "", "language": "", "extra_info": "", "category": ""}


async def _extract_web_series(text, original):
    try:
        # 1. Episode formats ko normalize karo (to22, ep22, space etc.)
        text = normalize_episodes(text)
        
        # 2. Remove language indicators line if present
        text = re.sub(r'🔊.*?(?:\n|$)', '', text, flags=re.DOTALL)

        # 2. Find season/episode/part position to split title
        split_pos = None
        season_patterns = [
            r'\bPart\s*\d+\b', r'\bP\d+\b',
            r'\bS\d{1,2}\b', r'\bSeason\s*\d+\b',
            r'\bS\d{1,2}E\d{1,3}\b', r'\[?E\d{1,3}[-~_]\d{1,3}\]?',
            r'\bEP\s*\d{1,3}(?:[-~_]\d{1,3})?\b', r'\bEpisode\s*\d+\b'
        ]
        for pattern in season_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                split_pos = match.start()
                break

        # 3. Extract title
        if split_pos is not None:
            title = text[:split_pos].strip()
        else:
            title = text

        # 4. Clean title (similar to movie cleaning)
        title = re.sub(r'[A-ZА-Я]{2,}\s*!+\s*\w+$', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\[.*?\]', '', title)
        title = re.sub(r'\(.*?\)', '', title)
        title = re.sub(r'by\s+\w+$', '', title, flags=re.IGNORECASE)
        title = re.sub(r'https?://\S+', '', title)
        title = re.sub(r'@\w+', '', title)
        title = re.sub(r'#\w+', '', title)
        title = re.sub(r'[_\.\-]+', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()

        # Remove trailing junk words (common in web series too)
        junk = ['hindi', 'english', 'tamil', 'telugu', 'dubbed', 'multi', 'audio',
                'ddp', 'web', 'dl', 'bluray', 'amzn', 'hevc', 'x264', 'x265', 'mkv']
        words = title.split()
        while words and words[-1].lower() in junk:
            words.pop()
        title = ' '.join(words)

        # 5. Extract metadata
        year_match = re.search(r'[\(\[]?(19|20)\d{2}[\)\]]?', text)
        year = re.search(r'(19|20)\d{2}', year_match.group()) if year_match else ""
        year = year.group() if year else ""

        # Languages
        languages = []
        lang_map = {
            'japanese|日本語': 'Japanese', 'english': 'English',
            'hindi|हिन्दी': 'Hindi', 'tamil|தமிழ்': 'Tamil',
            'telugu|తెలుగు': 'Telugu', 'malayalam': 'Malayalam',
            'korean': 'Korean', 'dual.*audio': 'Dual Audio',
            'multi.*audio': 'Multi Audio'
        }
        for pattern, name in lang_map.items():
            if re.search(pattern, text, re.IGNORECASE):
                languages.append(name)
        language = ', '.join(dict.fromkeys(languages)) if languages else ""

        # Extra info (season/episodes/parts)
        extra_parts = []
        
        # Pehle Part dhoondo (e.g., P1, Part 1)
        p_match = re.search(r'(?i)\b(Part\s*\d+|P\d+)\b', text)
        if p_match:
            extra_parts.append(p_match.group().upper())
            
        s_match = re.search(r'(?i)(s\d{1,2}|season\s*\d+)', text)
        if s_match:
            extra_parts.append(s_match.group().upper())
        
        # Episode detection — S04E01 combined format + standalone E01/EP01
        e_match = re.search(
            r'(?i)(?:'
            r'S\d{1,2}(E\d{1,3}(?:\s*[-~_]\s*E?\d{1,3})?)'  # S04E01 or S04E01-E03
            r'|(\[?(?:ep|e|episode)\s*\d{1,3}\s*(?:[-~_]|to)\s*(?:e|ep)?\s*\d{1,3}\]?)'  # E01-E03, EP1 to 5
            r'|\b((?:ep|e|episode)\s*\d{1,3})\b'  # Standalone E01, EP01, Episode 1
            r')', text)
        if e_match:
            ep = (e_match.group(1) or e_match.group(2) or e_match.group(3) or '').strip()
            ep = re.sub(r'[\[\]]', '', ep).upper()
            if ep:
                extra_parts.append(ep)
            
        if re.search(r'(?i)(combined|complete|batch)', text):
            extra_parts.append('COMBINED')
            
        extra_info = ' '.join(extra_parts)

        # Category
        category = "Web Series"

        # Final check
        if not title or len(title) < 2:
            title = "UNKNOWN"

        logger.info(f"✅ Web Series Fallback: '{title}' | Year: {year} | Lang: {language} | Extra: {extra_info} | Cat: {category}")
        return {
            "title": title,
            "year": year,
            "language": language,
            "extra_info": extra_info,
            "category": category
        }
    except Exception as e:
        logger.error(f"❌ Web series fallback error: {e}")
        return {"title": "UNKNOWN", "year": "", "language": "", "extra_info": "", "category": ""}
# ==================== MEMBERSHIP CHECK LOGIC ====================
async def is_user_member(context, user_id: int, force_fresh: bool = False):
    """Check if user is member of channel and group (Smart Auto-Switch Logic)"""
    global ACTIVE_FSUB, BACKUP_FSUB_LIST
    
    if not FORCE_JOIN_ENABLED:
        return {'is_member': True, 'channel': True, 'group': True, 'error': None}
    
    current_time = datetime.now()
    if not force_fresh and user_id in verified_users:
        last_checked, cached = verified_users[user_id]
        if (current_time - last_checked).total_seconds() < VERIFICATION_CACHE_TIME:
            return cached
    
    result = {'is_member': False, 'channel': False, 'group': False, 'error': None}
    VALID_STATUSES = ['member', 'administrator', 'creator']
    
    # Start both membership checks together. Telegram API round trips are
    # independent, so waiting for the channel check before starting the group
    # check needlessly adds both network latencies to every uncached message.
    async def check_required_group():
        try:
            member = await context.bot.get_chat_member(
                chat_id=REQUIRED_GROUP_ID,
                user_id=user_id,
            )
            return member.status in VALID_STATUSES
        except Exception as e:
            logger.error(f"Group Check Error: {e}")
            return False

    group_check_task = asyncio.create_task(check_required_group())

    # --- 1. SMART CHANNEL CHECK (WITH AUTO-SWITCH) ---
    try:
        channel_member = await context.bot.get_chat_member(chat_id=ACTIVE_FSUB['id'], user_id=user_id)
        if channel_member.status in VALID_STATUSES:
            result['channel'] = True
            
    except telegram.error.Forbidden as e:
        # 🚨 ERROR 1: Bot ko channel se nikal diya gaya hai!
        logger.error(f"🚨 Bot banned from channel! Switching FSub...")
        if BACKUP_FSUB_LIST:
            next_backup = BACKUP_FSUB_LIST.pop(0)
            ACTIVE_FSUB['id'] = next_backup['id']
            ACTIVE_FSUB['url'] = next_backup['url']
            
            # Admin ko SOS Alert bhejo!
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_USER_ID, 
                    text=f"🚨 **URGENT ALARM!** 🚨\n\nTumhara Main Channel ban ho gaya hai ya bot ko admin se hata diya gaya hai!\n\n✅ Maine automatically FSub ko naye channel par shift kar diya hai: {ACTIVE_FSUB['url']}", 
                    parse_mode='Markdown'
                )
            except: pass
            
            # Naye channel ke sath wapas check karo
            if not group_check_task.done():
                group_check_task.cancel()
            return await is_user_member(context, user_id, force_fresh)
        else:
            result['channel'] = True # Agar saare backup khatam, toh FSub bypass kar do taaki bot chalta rahe
            
    except telegram.error.BadRequest as e:
        if "chat not found" in str(e).lower():
            # 🚨 ERROR 2: Channel Telegram ne uda diya (Delete ho gaya)
            logger.error(f"🚨 Channel Deleted! Switching FSub...")
            if BACKUP_FSUB_LIST:
                next_backup = BACKUP_FSUB_LIST.pop(0)
                ACTIVE_FSUB['id'] = next_backup['id']
                ACTIVE_FSUB['url'] = next_backup['url']
                
                try:
                    await context.bot.send_message(
                        chat_id=ADMIN_USER_ID, 
                        text=f"🚨 **URGENT ALARM!** 🚨\n\nMain Channel Telegram dwara Delete/Ban kar diya gaya hai!\n\n✅ Maine traffic backup par shift kar diya hai: {ACTIVE_FSUB['url']}", 
                        parse_mode='Markdown'
                    )
                except: pass
                
                if not group_check_task.done():
                    group_check_task.cancel()
                return await is_user_member(context, user_id, force_fresh)
            else:
                result['channel'] = True
        else:
            # Koi chhota mota network error, channel active hai
            result['channel'] = False 
            
    except Exception as e:
        # Temporary glitch (ignore and allow/deny gracefully without switching)
        logger.error(f"Temporary Channel Check Error: {e}")
        result['channel'] = False 

    # --- 2. GROUP CHECK ---
    result['group'] = await group_check_task

    result['is_member'] = result['channel'] and result['group']
    verified_users[user_id] = (current_time, result)
    
    return result

def get_join_keyboard():
    """Join buttons keyboard"""
    global ACTIVE_FSUB
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📢 Join Channel", url=ACTIVE_FSUB['url']),
            InlineKeyboardButton("💬 Join Group", url=FILMFYBOX_GROUP_URL)
        ],
        [InlineKeyboardButton("✅ Joined Both - Verify", callback_data="verify")]
    ])

def get_join_message(channel_status, group_status):
    """Generate message based on what is missing"""
    if not channel_status and not group_status:
        missing = "Channel and Group both"
    elif not channel_status:
        missing = "Channel"
    else:
        missing = "Group"
    
    return (
        f"📂 **Your File is Ready!**\n\n"
        f"🚫 **But Access Denied**\n\n"
        f"You haven't joined {missing}!\n\n"
        f"📢 Channel: {'✅' if channel_status else '❌'}\n"
        f"💬 Group: {'✅' if group_status else '❌'}\n\n"
        f"Join both, then click **Verify** button 👇"
    )

def is_valid_url(url):
    """Check if a URL is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def normalize_url(url):
    """Normalize and clean URLs"""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        if 'blogspot.com' in url and 'import-urlhttpsfonts' in url:
            url = url.replace('import-urlhttpsfonts', 'import-url-https-fonts')

        if '#' in url:
            base, anchor = url.split('#', 1)
            parsed = urlparse(base)
            normalized_base = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                ''
            ))
            url = f"{normalized_base}#{anchor}"
        else:
            parsed = urlparse(url)
            url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))

        return url
    except:
        return url

def _normalize_title_for_match(title: str) -> str:
    """Normalize title for fuzzy matching"""
    if not title:
        return ""
    t = re.sub(r'[^\w\s]', ' ', title)
    t = re.sub(r'\s+', ' ', t).strip()
    return t.lower()

# NEW: Function to safely escape characters for Admin Notification
def escape_markdown_v2(text: str) -> str:
    """Escapes special characters for Markdown V2 formatting."""
    # Use the simplest escape for characters that commonly break parsing
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', text)

async def send_multi_bot_message(target_user_id, text_message, parse_mode='HTML'):
    """Teeno bots se message bhejkar try karega, jo chal jaye wahi sahi."""
    # Apne .env wale tokens yahan laayein
    tokens = [
        os.environ.get("TELEGRAM_BOT_TOKEN"),
        os.environ.get("BOT_TOKEN_2"),
        os.environ.get("BOT_TOKEN_3")
    ]
    tokens = [t for t in tokens if t] # Khali tokens hata do
    
    for token in tokens:
        try:
            # Temporary bot instance banayega aur message bhejega
            temp_bot = telegram.Bot(token=token)
            await temp_bot.send_message(chat_id=target_user_id, text=text_message, parse_mode=parse_mode)
            return True # Success ho gaya, function khatam
        except telegram.error.Forbidden:
            continue # User ne ye bot block kiya hai, agle bot par jao
        except Exception as e:
            logger.error(f"Multi-bot send error: {e}")
            continue
            
    return False # Teeno bots se fail ho gaya

def get_last_similar_request_for_user(user_id: int, title: str, minutes_window: int = REQUEST_COOLDOWN_MINUTES):
    """Look up the user's most recent request that is sufficiently similar to title"""
    conn = get_db_connection()
    if not conn:
        return None

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT movie_title, requested_at
            FROM user_requests
            WHERE user_id = %s
            ORDER BY requested_at DESC
            LIMIT 200
        """, (user_id,))
        rows = cur.fetchall()
        cur.close()
        close_db_connection(conn)

        if not rows:
            return None

        now = datetime.now()
        cutoff = now - timedelta(minutes=minutes_window)
        norm_target = _normalize_title_for_match(title)

        for stored_title, requested_at in rows:
            if not stored_title or not requested_at:
                continue
            try:
                if isinstance(requested_at, datetime):
                    requested_time = requested_at
                else:
                    requested_time = datetime.strptime(str(requested_at), '%Y-%m-%d %H:%M:%S')
            except Exception:
                requested_time = requested_at

            if requested_time < cutoff:
                break

            norm_stored = _normalize_title_for_match(stored_title)
            score = fuzz.token_sort_ratio(norm_target, norm_stored)
            if score >= SIMILARITY_THRESHOLD:
                return {
                    "stored_title": stored_title,
                    "requested_at": requested_time,
                    "score": score
                }

        return None
    except Exception as e:
        logger.error(f"Error checking last similar request for user {user_id}: {e}")
        try:
            close_db_connection(conn)
        except:
            pass
        return None

def user_burst_count(user_id: int, window_seconds: int = 60):
    """Count how many requests this user made in the last window_seconds"""
    conn = get_db_connection()
    if not conn:
        return 0
    try:
        cur = conn.cursor()
        since = datetime.now() - timedelta(seconds=window_seconds)
        cur.execute("SELECT COUNT(*) FROM user_requests WHERE user_id = %s AND requested_at >= %s", (user_id, since))
        
        result = cur.fetchone()
        cnt = result[0] if result else 0 
        
        cur.close()
        close_db_connection(conn)
        return cnt
    except Exception as e:
        logger.error(f"Error counting burst requests for user {user_id}: {e}")
        try:
            close_db_connection(conn)
        except:
            pass
        return 0

# ==================== DATABASE-BACKED AUTO-DELETE FUNCTIONS ====================
USER_TEXT_DELETE_SECONDS = 5 * 60
USER_FILE_DELETE_SECONDS = 2 * 60

async def add_messages_to_db_queue(context, chat_id, message_ids, delay):
    """Messages ko DB me save karta hai taaki restart hone par bhi yaad rahe"""
    try:
        bot_info = await context.bot.get_me()
        bot_username = bot_info.username
        
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                for msg_id in message_ids:
                    cur.execute(
                        """
                        INSERT INTO auto_delete_queue
                            (bot_username, chat_id, message_id, delete_at)
                        VALUES (%s, %s, %s, NOW() + (%s * INTERVAL '1 second'))
                        """,
                        (bot_username, chat_id, msg_id, delay)
                    )
                conn.commit()
                cur.close()
                logger.info(
                    "Auto-delete queued for chat=%s messages=%s delay=%ss bot=%s",
                    chat_id, message_ids, delay, bot_username
                )
            except Exception as e:
                logger.error(f"Error saving to delete queue: {e}")
            finally:
                close_db_connection(conn)
    except Exception as e:
        logger.error(f"Failed to get bot info for delete queue: {e}")

async def delete_messages_after_delay(context, chat_id, message_ids, delay=USER_TEXT_DELETE_SECONDS):
    """Persist messages for deletion after restarts."""
    await add_messages_to_db_queue(context, chat_id, message_ids, delay)

async def delete_message_directly_after_delay(context, chat_id, message_id, delay):
    """Delete the message in-process as a fast path; the DB queue remains the durable backup."""
    await asyncio.sleep(delay)
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except telegram.error.BadRequest as exc:
        message = str(exc).lower()
        if "not found" not in message and "message to delete not found" not in message:
            logger.warning(
                "Direct auto-delete failed for chat=%s message=%s: %s",
                chat_id, message_id, exc
            )
    except Exception as exc:
        logger.warning(
            "Direct auto-delete failed for chat=%s message=%s: %s",
            chat_id, message_id, exc
        )

def track_message_for_deletion(context, chat_id, message_id, delay=USER_TEXT_DELETE_SECONDS):
    """Queue a message durably and schedule an in-process deletion fallback."""
    if not message_id:
        return
    logger.info(
        "Tracking message for auto-delete: chat=%s message=%s delay=%ss",
        chat_id, message_id, delay
    )
    
    queue_task = asyncio.create_task(
        add_messages_to_db_queue(context, chat_id, [message_id], delay)
    )
    direct_task = asyncio.create_task(
        delete_message_directly_after_delay(
            context, chat_id, message_id, delay
        )
    )
    for task in (queue_task, direct_task):
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

def track_user_message_for_deletion(context, chat_id, message, is_file=False):
    """Use the user-facing retention policy for text and downloadable files."""
    if not message:
        return
    delay = USER_FILE_DELETE_SECONDS if is_file else USER_TEXT_DELETE_SECONDS
    track_message_for_deletion(context, chat_id, message.message_id, delay)

# ==================== DATABASE FUNCTIONS ====================

def save_post_to_db(
    movie_id, channel_id, message_id, bot_username, caption,
    media_file_id=None, media_type="photo", keyboard_data=None, topic_id=None, content_type="movies",
    movie_name=None, imdb_id=None, tmdb_id=None, channel_name=None
):
    """
    Post ka full data save karo.
    content_type = 'movies' / 'adult' / 'series' / 'anime'
    """
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        
        if not movie_name or not imdb_id:
            cur.execute("SELECT title, imdb_id FROM movies WHERE id = %s", (movie_id,))
            res = cur.fetchone()
            if res:
                if not movie_name: movie_name = res[0]
                if not imdb_id: imdb_id = res[1]

        cur.execute("""
            INSERT INTO channel_posts 
                (movie_id, channel_id, message_id, bot_username,
                 caption, media_file_id, media_type, 
                 keyboard_data, topic_id, content_type,
                 movie_name, imdb_id, tmdb_id, channel_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (channel_id, message_id) DO UPDATE SET
                caption       = EXCLUDED.caption,
                media_file_id = EXCLUDED.media_file_id,
                media_type    = EXCLUDED.media_type,
                keyboard_data = EXCLUDED.keyboard_data,
                topic_id      = EXCLUDED.topic_id,
                content_type  = EXCLUDED.content_type,
                movie_name    = COALESCE(EXCLUDED.movie_name, channel_posts.movie_name),
                imdb_id       = COALESCE(EXCLUDED.imdb_id, channel_posts.imdb_id),
                tmdb_id       = COALESCE(EXCLUDED.tmdb_id, channel_posts.tmdb_id),
                channel_name  = COALESCE(EXCLUDED.channel_name, channel_posts.channel_name)
        """, (
            movie_id, channel_id, message_id, bot_username,
            caption, media_file_id, media_type,
            json.dumps(keyboard_data) if keyboard_data else None,
            topic_id, content_type,
            movie_name, imdb_id, tmdb_id, channel_name
        ))
        conn.commit()
        cur.close()
        close_db_connection(conn)
        return True
    except Exception as e:
        logger.error(f"Save post error: {e}")
        if conn:
            conn.rollback()
            close_db_connection(conn)
        return False


# ==================== GLOBAL DUPLICATE POST CHECK ====================
def is_movie_posted_recently(movie_id, days=7):
    """Check if movie was posted to ANY channel within last N days.
    Ye function globally check karta hai — kisi bhi channel me post hui ho to True return karega.
    """
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM channel_posts WHERE movie_id = %s AND posted_at >= NOW() - INTERVAL '7 days' LIMIT 1",
            (movie_id,)
        )
        result = cur.fetchone()
        cur.close()
        return result is not None
    except Exception:
        return False
    finally:
        close_db_connection(conn)


# 👇👇👇 START COPY HERE (New Function) 👇👇👇
def get_db_connection():
    """Pool se connection lene wala naya function"""
    if not db_pool:
        logger.error("Database pool is not ready.")
        return None
    try:
        conn = db_pool.getconn()
        # A previous query can leave a pooled connection in an aborted
        # transaction. Reusing it makes catalogue searches silently fail and
        # incorrectly fall back to TMDB request results.
        try:
            conn.rollback()
        except Exception:
            pass
        return conn
    except Exception as e:
        logger.error(f"Error getting connection from pool: {e}")
        return None

def close_db_connection(conn):
    """Connection ko wapas pool me dalne ke liye helper"""
    if db_pool and conn:
        try:
            # Never return an aborted transaction to the shared pool.
            try:
                conn.rollback()
            except Exception:
                pass
            db_pool.putconn(conn)
        except Exception:
            pass


def close_db_pool():
    """Close all idle pooled connections during process shutdown."""
    global db_pool
    if db_pool:
        try:
            db_pool.closeall()
            logger.info("✅ Database connection pool closed.")
        except Exception:
            logger.exception("❌ Failed to close database connection pool.")
        finally:
            db_pool = None
# 👆👆👆 END COPY HERE 👆👆👆

def update_movies_in_db():
    """Update movies from Blogger API"""
    logger.info("Starting movie update process...")

    conn = None
    cur = None
    new_movies_added = 0

    try:
        conn = get_db_connection()
        if not conn:
            return "Database connection failed"

        cur = conn.cursor()

        cur.execute("SELECT last_sync FROM sync_info ORDER BY id DESC LIMIT 1;")
        last_sync_result = cur.fetchone()
        last_sync_time = last_sync_result[0] if last_sync_result else None

        cur.execute("SELECT title FROM movies;")
        existing_movies = {row[0] for row in cur.fetchall()}

        if not BLOGGER_API_KEY or not BLOG_ID:
            return "Blogger API keys not configured"

        service = build('blogger', 'v3', developerKey=BLOGGER_API_KEY)
        all_items = []

        posts_request = service.posts().list(blogId=BLOG_ID, maxResults=500)
        while posts_request is not None:
            posts_response = posts_request.execute()
            all_items.extend(posts_response.get('items', []))
            posts_request = service.posts().list_next(posts_request, posts_response)

        pages_request = service.pages().list(blogId=BLOG_ID)
        pages_response = pages_request.execute()
        all_items.extend(pages_response.get('items', []))

        unique_titles = set()
        for item in all_items:
            title = item.get('title')
            url = item.get('url')
            if title and not is_safe_canonical_title(title):
                logger.warning("Skipping Blogger title with media noise: %r", title)
                continue

            if last_sync_time and 'published' in item:
                try:
                    published_time = datetime.strptime(item['published'], '%Y-%m-%dT%H:%M:%S.%fZ')
                    if published_time < last_sync_time:
                        continue
                except:
                    pass

            if title and url and title.strip() not in existing_movies and title.strip() not in unique_titles:
                try:
                    cur.execute("INSERT INTO movies (title, url) VALUES (%s, %s);", (title.strip(), url.strip()))
                    new_movies_added += 1
                    unique_titles.add(title.strip())
                except psycopg2.Error as e:
                    logger.error(f"Error inserting movie {title}: {e}")
                    conn.rollback()
                    continue

        cur.execute("INSERT INTO sync_info (last_sync) VALUES (CURRENT_TIMESTAMP);")

        conn.commit()
        return f"Update complete. Added {new_movies_added} new items."

    except Exception as e:
        logger.error(f"Error during movie update: {e}")
        if conn:
            conn.rollback()
        return f"An error occurred during update: {e}"

    finally:
        if cur: cur.close()
        if conn: close_db_connection(conn)


def _normalize_search_text(text: str) -> str:
    """
    Search query aur DB title dono ko sirf lowercase letters/numbers tak todta hai
    (spaces, hyphens, colons, punctuation sab hata deta hai).
    Isse "spider man", "spiderman", "Spider-Man" — teeno ek hi cheez maane jaate hain,
    chahe DB me title kaise bhi likha ho.
    """
    normalized = re.sub(r"['\u2019]s\b", '', (text or '').lower())
    return re.sub(r'[^a-z0-9]', '', normalized)


def _select_single_search_result(query, movies):
    """Return one confident match; return None when the user should choose."""
    if not movies:
        return None

    normalized_query = _normalize_search_text(query)
    unique_movies = []
    seen_ids = set()
    for movie in movies:
        movie_id = movie[0] if movie else None
        if movie_id in seen_ids:
            continue
        seen_ids.add(movie_id)
        unique_movies.append(movie)

    exact_matches = [
        movie for movie in unique_movies
        if len(movie) > 1 and _normalize_search_text(movie[1]) == normalized_query
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        return None

    scored_matches = sorted(
        (
            (fuzz.WRatio(normalized_query, _normalize_search_text(movie[1])), movie)
            for movie in unique_movies
            if len(movie) > 1 and _normalize_search_text(movie[1])
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    strong_matches = [item for item in scored_matches if item[0] >= 62]
    if not strong_matches:
        return None
    if len(strong_matches) == 1:
        return strong_matches[0][1]

    top_score, top_movie = strong_matches[0]
    second_score = strong_matches[1][0]
    return top_movie if top_score - second_score >= 12 else None


def get_google_title_suggestions(query: str, limit: int = 3):
    """Resolve common misspellings server-side so Telegram clients need no JSONP."""
    cache_key = f"google_title_suggestions_{query.lower()}"
    cached = search_cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        response = requests.get(
            'https://suggestqueries.google.com/complete/search',
            params={'client': 'firefox', 'q': f'{query} movie'},
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=GOOGLE_SUGGESTION_TIMEOUT_SECONDS,
        )
        data = response.json()
        suggestions = data[1] if isinstance(data, list) and len(data) > 1 else []
        clean = []
        for item in suggestions:
            # Google commonly suggests searches such as "Reacher movie cast"
            # and "Reacher movie 2026".  Those are queries, not title choices.
            title = re.sub(
                r'\s+(?:movie|film|series|web\s+series)(?:\s+(?:cast|trailer|release\s+date|review|episodes?|season\s*\d+|\d{4}))*\s*$',
                '', str(item), flags=re.I
            ).strip()
            # Do not show unrelated Google search phrases as a movie button.
            if (
                title and title.lower() != query.lower()
                and fuzz.WRatio(query, title) >= 60
                and title.lower() not in {saved.lower() for saved in clean}
            ):
                clean.append(title)
        result = clean[:limit]
    except Exception as e:
        logger.info(f"Google suggestion lookup skipped for '{query}': {e}")
        result = []
    search_cache.set(cache_key, result)
    return result


async def get_google_title_suggestions_with_timeout(query: str, limit: int = 3):
    """Run the optional Google fallback without holding the search response open."""
    try:
        return await asyncio.wait_for(
            run_async(get_google_title_suggestions, query, limit=limit),
            timeout=GOOGLE_SUGGESTION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.info("Google suggestion lookup timed out")
        return []


def get_movies_from_db(user_query, limit=10):
    cache_key = f"db_fuzzy_{user_query}_{limit}"
    cached = search_cache.get(cache_key)
    if cached is not None:
        return cached
    result = _get_movies_from_db_nocache(user_query, limit)
    search_cache.set(cache_key, result)
    return result

def _get_movies_from_db_nocache(user_query, limit=10):
    """Search for MULTIPLE movies in database with fuzzy matching"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            raise RuntimeError("catalog search connection is unavailable")

        cur = conn.cursor()

        logger.info(f"Searching for: '{user_query}'")

        # 🔧 FIX: query ko normalize karo (lowercase + sirf letters/numbers).
        # Isse "spider man", "spiderman", "Spider-Man" sab same ban jaate hain,
        # aur neeche wali query DB me title chahe space se ho ya hyphen se, dono match karegi.
        norm_query = _normalize_search_text(user_query)

        if not norm_query:
            # Sirf symbols/spaces type kiye the — kuch bhi search karne layak nahi hai
            cur.close()
            close_db_connection(conn)
            return []

        # ✅ Updated to include new columns
        # Title ko bhi query jaisa hi normalize karke compare karte hain (DB-side),
        # taaki alag-alag spacing/hyphen/case wale titles bhi pakde jaayein.
        # SPEED FIX: Removed regexp_replace to avoid full table scans.
        # Fallback to ILIKE.
        cur.execute(
            """SELECT id, title, url, file_id, imdb_id, poster_url, year, genre 
               FROM movies
               WHERE title ILIKE %s
               ORDER BY title LIMIT %s""",
            (f'%{user_query}%', limit)
        )
        exact_matches = cur.fetchall()

        if exact_matches:
            logger.info(f"Found {len(exact_matches)} exact matches")
            cur.close()
            close_db_connection(conn)
            return exact_matches

        # SPEED FIX: Removed regexp_replace to avoid full table scans.
        cur.execute("""
            SELECT DISTINCT m.id, m.title, m.url, m.file_id, m.imdb_id, m.poster_url, m.year, m.genre
            FROM movies m
            JOIN movie_aliases ma ON m.id = ma.movie_id
            WHERE ma.alias ILIKE %s
            ORDER BY m.title
            LIMIT %s
        """, (f'%{user_query}%', limit))
        alias_matches = cur.fetchall()

        if alias_matches:
            logger.info(f"Found {len(alias_matches)} alias matches")
            cur.close()
            close_db_connection(conn)
            return alias_matches

        cur.execute("SELECT id, title, url, file_id, imdb_id, poster_url, year, genre FROM movies")
        all_movies = cur.fetchall()

        if not all_movies:
            cur.close()
            close_db_connection(conn)
            return []

        movie_titles = [movie[1] for movie in all_movies]
        movie_dict = {movie[1]: movie for movie in all_movies}

        # 🔧 FIX: pre-filter pool ko result 'limit' se bada rakha hai (kam se kam 50),
        # taaki score >= 65 filter lagne se PEHLE hi koi sahi match discard na ho jaaye
        # (jaise pehle "spiderman" jaisi short query ke liye ho raha tha).
        # Speed par asar nahi: fuzzywuzzy already sabhi titles ko score karta hai,
        # sirf top-N return karta hai — N badhane se extra compute nahi lagta.
        pool_size = max(limit * 5, 50)
        # WRatio is resilient to transposed/missing letters: "rechar" → "Reacher".
        matches = process.extract(user_query, movie_titles, scorer=fuzz.WRatio, limit=pool_size)

        filtered_movies = [movie_dict[title] for title, score, index in matches if score >= 58]

        logger.info(f"Found {len(filtered_movies)} fuzzy matches")

        cur.close()
        close_db_connection(conn)
        return filtered_movies[:limit]

    except Exception as e:
        logger.error(f"Database query error: {e}")
        return []
    finally:
        if conn:
            try:
                close_db_connection(conn)
            except:
                pass


def get_movies_fast_sql(query: str, limit: int = 5):
    """Shared, bounded-cache search entry point for all Telegram search routes."""
    return _get_movies_fast_sql_nocache(query, limit)


def _get_movies_fast_sql_nocache(query: str, limit: int = 5):
    """Search exact normalized titles, then prefixes, then fuzzy matches."""
    total_start = time.perf_counter()
    cache_start = total_start
    normalized_query = _normalize_search_text(query)
    normalization_ms = (time.perf_counter() - cache_start) * 1000
    if not normalized_query:
        return []
    cache_key = ("catalog_search", normalized_query, int(limit))
    cache_start = time.perf_counter()
    cached = search_cache.get(cache_key)
    cache_ms = (time.perf_counter() - cache_start) * 1000
    if cached is not None:
        if SEARCH_TIMING_ENABLED:
            logger.info(
                "Search timing query=%r total_ms=%.2f normalize_ms=%.2f "
                "cache_ms=%.2f connection_acquire_ms=0 exact_prefix_sql_ms=0 "
                "fuzzy_title_sql_ms=0 fuzzy_alias_sql_ms=0 python_ms=0 "
                "fallback_ms=0 "
                "result=cache",
                query[:80],
                (time.perf_counter() - total_start) * 1000,
                normalization_ms,
                cache_ms,
            )
        return list(cached)

    conn = None
    connection_acquire_ms = 0.0
    exact_prefix_sql_ms = 0.0
    fuzzy_title_sql_ms = 0.0
    fuzzy_alias_sql_ms = 0.0
    python_start = time.perf_counter()
    stage = "exact"
    try:
        acquire_start = time.perf_counter()
        conn = get_db_connection()
        connection_acquire_ms = (time.perf_counter() - acquire_start) * 1000
        if not conn:
            raise RuntimeError("catalog search connection is unavailable")

        cur = conn.cursor()
        exact_prefix_sql = """
            SELECT m.id, m.title, m.url, m.file_id, m.imdb_id, m.poster_url, m.year, m.genre
            FROM movies m
            WHERE regexp_replace(LOWER(m.title), '[^a-z0-9]', '', 'g') = %s
               OR regexp_replace(LOWER(m.title), '[^a-z0-9]', '', 'g') LIKE %s
            ORDER BY CASE
                WHEN regexp_replace(LOWER(m.title), '[^a-z0-9]', '', 'g') = %s
                THEN 0 ELSE 1
            END, m.title, m.id
            LIMIT %s
        """
        exact_start = time.perf_counter()
        cur.execute(
            exact_prefix_sql,
            (normalized_query, normalized_query + "%", normalized_query, limit),
        )
        results = cur.fetchall()
        exact_prefix_sql_ms = (time.perf_counter() - exact_start) * 1000

        if not results:
            stage = "fuzzy"
            fuzzy_start = time.perf_counter()
            cur.execute(
                """
                SELECT m.id, m.title, m.url, m.file_id, m.imdb_id, m.poster_url, m.year, m.genre
                FROM movies m
                WHERE SIMILARITY(
                    regexp_replace(LOWER(m.title), '[^a-z0-9]', '', 'g'),
                    %s
                ) >= 0.30
                ORDER BY SIMILARITY(
                    regexp_replace(LOWER(m.title), '[^a-z0-9]', '', 'g'),
                    %s
                ) DESC, m.id
                LIMIT %s
                """,
                (normalized_query, normalized_query, limit),
            )
            results = cur.fetchall()
            fuzzy_title_sql_ms = (time.perf_counter() - fuzzy_start) * 1000
            if not results:
                fuzzy_alias_start = time.perf_counter()
                cur.execute(
                    """
                    SELECT DISTINCT m.id, m.title, m.url, m.file_id,
                                    m.imdb_id, m.poster_url, m.year, m.genre
                    FROM movies m
                    JOIN movie_aliases ma ON ma.movie_id = m.id
                    WHERE SIMILARITY(
                        regexp_replace(LOWER(ma.alias), '[^a-z0-9]', '', 'g'),
                        %s
                    ) >= 0.30
                    ORDER BY m.title, m.id
                    LIMIT %s
                    """,
                    (normalized_query, limit),
                )
                results = cur.fetchall()
                fuzzy_alias_sql_ms = (
                    time.perf_counter() - fuzzy_alias_start
                ) * 1000

        final_results = tuple(tuple(row[:8]) for row in results)
        cur.close()
        search_cache.set(cache_key, final_results)
        return list(final_results)

    except Exception:
        logger.exception("Catalog search failed at %s stage", stage)
        raise
    finally:
        if conn:
            try:
                close_db_connection(conn)
            except Exception:
                logger.exception("Could not return catalog search connection to pool")
        if SEARCH_TIMING_ENABLED:
            python_ms = max(
                0.0,
                (time.perf_counter() - python_start) * 1000
                - connection_acquire_ms
                - exact_prefix_sql_ms
                - fuzzy_title_sql_ms
                - fuzzy_alias_sql_ms,
            )
            logger.info(
                "Search timing query=%r total_ms=%.2f normalize_ms=%.2f "
            "cache_ms=%.2f connection_acquire_ms=%.2f "
            "exact_prefix_sql_ms=%.2f fuzzy_title_sql_ms=%.2f "
                "fuzzy_alias_sql_ms=%.2f python_ms=%.2f fallback_ms=0 result=%s",
                query[:80],
                (time.perf_counter() - total_start) * 1000,
                normalization_ms,
                cache_ms,
                connection_acquire_ms,
                exact_prefix_sql_ms,
                fuzzy_title_sql_ms,
                fuzzy_alias_sql_ms,
                python_ms,
                stage,
            )


def get_movie_by_imdb_id(imdb_id: str):
    """Get movie from database by IMDb ID"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return None

        cur = conn.cursor()
        cur.execute(
            """SELECT id, title, url, file_id, imdb_id, poster_url, year, genre 
               FROM movies WHERE imdb_id = %s LIMIT 1""",
            (imdb_id,)
        )
        result = cur.fetchone()
        cur.close()
        close_db_connection(conn)
        return result

    except Exception as e:
        logger.error(f"Error fetching movie by IMDb ID: {e}")
        return None
    finally:
        if conn:
            try:
                close_db_connection(conn)
            except:
                pass


def update_movie_metadata(
    movie_id: int,
    imdb_id: str = None,
    poster_url: str = None,
    year: int = None,
    genre: str = None,
    rating: str = None,
    description: str = None,
    category: str = None,
    content_type: str = None,
    seasons_data: dict = None
):
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return False

        cur = conn.cursor()
        updates, values = [], []

        def add(field, val):
            updates.append(f"{field} = %s")
            values.append(val)

        if imdb_id: add("imdb_id", imdb_id)
        if poster_url: add("poster_url", poster_url)
        if year is not None: add("year", year)
        if genre: add("genre", genre)
        if rating: add("rating", rating)
        if description: add("description", description)
        if category: add("category", category)
        if content_type: add("content_type", content_type)
        if seasons_data is not None:
            import json
            add("seasons_data", json.dumps(seasons_data))

        if not updates:
            return False

        values.append(movie_id)
        cur.execute(f"UPDATE movies SET {', '.join(updates)} WHERE id = %s", values)
        conn.commit()
        cur.close()
        return True

    except Exception as e:
        logger.error(f"Error updating movie metadata: {e}", exc_info=True)
        try:
            if conn: conn.rollback()
        except Exception:
            pass
        return False
    finally:
        if conn:
            close_db_connection(conn)


def store_user_request(user_id, username, first_name, movie_title, group_id=None, message_id=None):
    """Store user request in database"""
    try:
        conn = get_db_connection()
        if not conn:
            return False

        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_requests (user_id, username, first_name, movie_title, group_id, message_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT ON CONSTRAINT user_requests_unique_constraint DO UPDATE
                SET requested_at = EXCLUDED.requested_at
        """, (user_id, username, first_name, movie_title, group_id, message_id))
        conn.commit()
        cur.close()
        close_db_connection(conn)
        return True
    except Exception as e:
        logger.error(f"Error storing user request: {e}")
        try:
            conn.rollback()
            close_db_connection(conn)
        except:
            pass
        return False


def record_telegram_user(user, chat_id=None):
    """Create/update the user's passwordless Mini App profile from Telegram."""
    if not user:
        return
    conn = get_db_connection()
    if not conn:
        return
    try:
        user_id = user.id
        username = user.username or ''
        first_name = user.first_name or ''
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO miniapp_users (user_id, username, first_name, last_seen)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id) DO UPDATE SET
                username = EXCLUDED.username,
                first_name = EXCLUDED.first_name,
                last_seen = CURRENT_TIMESTAMP
        """, (user_id, username, first_name))
        cur.execute("""
            INSERT INTO user_activity (user_id, username, first_name, chat_id, last_seen)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id) DO UPDATE SET
                username = EXCLUDED.username,
                first_name = EXCLUDED.first_name,
                chat_id = EXCLUDED.chat_id,
                last_seen = CURRENT_TIMESTAMP
        """, (user_id, username, first_name, chat_id))
        conn.commit()
        cur.close()
    except Exception as e:
        logger.warning(f"Could not record Telegram user {getattr(user, 'id', 'unknown')}: {e}")
        conn.rollback()
    finally:
        close_db_connection(conn)


RECOMMENDATION_EVENT_TYPES = {
    'pm_search',
    'pm_exact_match',
    'pm_file_request',
    'group_search',
    'group_selection',
    'miniapp_open_details',
    'miniapp_download',
    'watchlist_add',
    'watchlist_remove',
    'rating_submitted',
    'surprise_impression',
    'surprise_click',
    'surprise_skip',
}


def record_recommendation_event(user_id, event_type, source, movie_id=None, metadata=None):
    """Persist one normalized interaction for recommendation ranking."""
    if not user_id:
        raise ValueError("Recommendation events require a user_id")
    if event_type not in RECOMMENDATION_EVENT_TYPES:
        raise ValueError(f"Unsupported recommendation event type: {event_type}")
    if not source:
        raise ValueError("Recommendation events require a source")

    conn = get_db_connection()
    if not conn:
        raise RuntimeError("Database connection failed while recording recommendation event")
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO miniapp_users (user_id)
            VALUES (%s)
            ON CONFLICT (user_id) DO NOTHING
        """, (user_id,))
        cur.execute("""
            INSERT INTO user_recommendation_events
                (user_id, movie_id, event_type, source, metadata)
            VALUES (%s, %s, %s, %s, %s::jsonb)
        """, (
            user_id,
            movie_id,
            event_type,
            source,
            json.dumps(metadata or {}, ensure_ascii=True),
        ))
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception(
            "Failed to record recommendation event user_id=%s event_type=%s",
            user_id,
            event_type,
        )
        raise
    finally:
        close_db_connection(conn)


def record_recommendation_event_safely(user_id, event_type, source, movie_id=None, metadata=None):
    """Record telemetry without interrupting the user-facing bot flow."""
    try:
        record_recommendation_event(
            user_id=user_id,
            event_type=event_type,
            source=source,
            movie_id=movie_id,
            metadata=metadata,
        )
    except Exception:
        logger.exception(
            "Recommendation telemetry failed user_id=%s event_type=%s",
            user_id,
            event_type,
        )


# ==================== METADATA FUNCTIONS ====================

def is_valid_imdb_id(imdb_id: str) -> bool:
    """Validate IMDb ID format (tt1234567 or tt12345678)"""
    if not imdb_id:
        return False
    return bool(re.match(r'^tt\d{7,8}$', imdb_id.strip()))


def normalize_catalog_labels(category="", content_type=None, language="", extra_info="", genre="", title=""):
    """Return stable regional category and media content type for catalogue rows."""
    category_text = str(category or "").strip()
    signal = " ".join(
        str(value or "").lower()
        for value in (category_text, content_type, language, extra_info, genre, title)
    )
    if any(token in signal for token in ("anime", "cartoon", "animation")):
        media_type = "Anime"
    elif any(token in signal for token in ("web series", "tv series", "television", "season", "episode", " s01", " s02")):
        media_type = "Web Series"
    else:
        media_type = "Movie"

    category_lower = category_text.lower()
    if "bollywood" in category_lower or "hindi" in category_lower or "hindi" in signal:
        region = "Bollywood"
    elif "hollywood" in category_lower or "english" in category_lower or "english" in signal:
        region = "Hollywood"
    elif any(token in signal for token in ("anime", "cartoon", "animation")):
        region = "Anime"
    elif category_text and category_lower not in {"movies", "movie", "web series", "tv series"}:
        region = category_text
    else:
        region = "Hollywood" if media_type == "Movie" else "Bollywood"
    return region, media_type


def normalize_seasons_data(value):
    """Return seasons metadata as a mapping regardless of DB/API encoding."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def auto_fetch_and_update_metadata(movie_id: int, movie_title: str):
    """Automatically fetch and update metadata for a movie"""
    try:
        metadata = fetch_movie_metadata(movie_title)
        if metadata:
            # 🔧 FIX: 8 values unpack (pehle 6 thi — CRASH hoti thi!)
            title, year, poster_url, genre, imdb_id, rating, plot, category, seasons_data = metadata
            category, content_type = normalize_catalog_labels(
                category=category,
                language="",
                extra_info="",
                genre=genre,
                title=title,
            )
            update_movie_metadata(
                movie_id=movie_id,
                imdb_id=imdb_id if imdb_id else None,
                poster_url=poster_url if poster_url else None,
                year=year if year else None,
                genre=genre if genre else None,
                rating=rating if rating and rating != 'N/A' else None,
                description=plot if plot else None,      # 🔧 NAYA: Plot bhi save karo
                category=category if category else None,
                content_type=content_type,
                seasons_data=seasons_data if seasons_data else {}
            )
            logger.info(f"✅ Metadata updated for movie {movie_id}: {title}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error in auto_fetch_and_update_metadata: {e}")
        return False

# ============================================================================
# 🔍 GOOGLE SEARCH METADATA FETCHER (Premium Edition)
# ============================================================================

async def fetch_metadata_from_google(query: str, search_year: str = ""):
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    CX_ID = os.environ.get("GOOGLE_CX_ID")
    
    if not API_KEY or not CX_ID:
        return None
    
    search_query = f"{query} {search_year} poster plot".strip()
    
    try:
        encoded = quote(search_query)
        
        base_url = "https://www.googleapis.com/customsearch/v1"
        
        # ---------- IMAGE SEARCH ----------
        img_url = f"{base_url}?key={API_KEY}&cx={CX_ID}&q={encoded}&num=5&searchType=image"
        response = await run_async(requests.get, img_url, timeout=10)
        data = response.json()
        
        items = data.get("items", [])
        
        # ---------- FALLBACK TO TEXT ----------
        if not items:
            txt_url = f"{base_url}?key={API_KEY}&cx={CX_ID}&q={encoded}&num=5"
            response = await run_async(requests.get, txt_url, timeout=10)
            data = response.json()
            items = data.get("items", [])
        
        if not items:
            return None
        
        # ---------- PICK BEST RESULT ----------
        best_item = items[0]
        
        title = clean_title(best_item.get("title", query))
        snippet = best_item.get("snippet", "")
        
        # ---------- IMAGE EXTRACTION ----------
        image_url = None
        pagemap = best_item.get("pagemap", {})
        
        if "cse_image" in pagemap:
            image_url = pagemap["cse_image"][0].get("src")
        elif "cse_thumbnail" in pagemap:
            image_url = pagemap["cse_thumbnail"][0].get("src")
        
        # fallback: direct link image
        if not image_url:
            link = best_item.get("link", "")
            if link.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_url = link
        
        # ---------- EXTRA CLEANUPS ----------
        plot = snippet[:300] if snippet else "Premium content available."
        
        # better genre detection (thoda smart banaya 😏)
        q_lower = query.lower()
        if any(x in q_lower for x in ['bhabhi', 'unrated', 'adult', 'hot']):
            genre = "Adult"
        elif any(x in q_lower for x in ['crime', 'murder', 'thriller']):
            genre = "Crime/Thriller"
        else:
            genre = "Drama"
        
        return {
            "title": title,
            "poster": image_url or DEFAULT_POSTER,
            "plot": plot,
            "year": search_year or "2024-2026",
            "genre": genre,
            "category": "Web Series"
        }
        
    except Exception as e:
        logger.error(f"Google Search Error: {e}")
        return None

# ============================================================================
# 🔧 HELPER FUNCTIONS
# ============================================================================

def clean_google_title(raw_title: str) -> str:
    """Google title se junk hatao"""
    # Common patterns remove karo
    junk_patterns = [
        r' - IMDb$', r' - Wikipedia$', r' - Rotten Tomatoes$',
        r' \| Netflix$', r' - Prime Video$', r' \| .*?Official',
        r'Watch ', r' Online', r' Full Movie', r' Download'
    ]
    
    title = raw_title
    for pattern in junk_patterns:
        title = re.sub(pattern, '', title, flags=re.IGNORECASE)
    
    # Year hatao agar title me hai
    title = re.sub(r'\s*\(\d{4}\)\s*', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    
    return title


def clean_plot(raw_snippet: str) -> str:
    """Google snippet ko clean plot banao"""
    # Ellipsis hatao
    plot = raw_snippet.replace('...', ' ')
    
    # URLs hatao
    plot = re.sub(r'https?://\S+', '', plot)
    
    # Extra spaces clean karo
    plot = re.sub(r'\s+', ' ', plot).strip()
    
    # Limit karo
    if len(plot) > 300:
        plot = plot[:297] + "..."
    
    return plot if plot else "Premium content available on FlimfyBox."


async def extract_imdb_poster(imdb_url: str) -> Optional[str]:
    """IMDb page se poster nikalo (Fallback)"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0'
        }
        response = await run_async(requests.get, imdb_url, headers=headers, timeout=8)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Meta tag se poster
        meta_img = soup.find('meta', property='og:image')
        if meta_img:
            return meta_img.get('content')
        
        # JSON-LD se poster
        script = soup.find('script', type='application/ld+json')
        if script:
            import json
            data = json.loads(script.string)
            if 'image' in data:
                return data['image']
                
    except Exception as e:
        logger.warning(f"IMDb poster extraction failed: {e}")
    
    return None


def extract_tmdb_poster_from_url(tmdb_url: str) -> Optional[str]:
    """TMDB URL se poster ID nikalo"""
    try:
        # Pattern: themoviedb.org/movie/12345-movie-name
        match = re.search(r'/(movie|tv)/(\d+)', tmdb_url)
        if match:
            media_type, tmdb_id = match.groups()
            # TMDB poster URL construct karo
            return f"https://image.tmdb.org/t/p/w500/{tmdb_id}.jpg"  # Simplified
    except:
        pass
    return None

def fetch_cast_from_imdb(imdb_id: str, limit: int = 5) -> str:
    """Fetch cast list from TMDB using IMDb ID, return comma-separated string."""
    try:
        api_key = "9fa44f5e9fbd41415df930ce5b81c4d7"
        find_url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_key}&external_source=imdb_id"
        resp = requests.get(find_url, timeout=10).json()
        tmdb_results = resp.get('movie_results', [])
        if not tmdb_results:
            tmdb_results = resp.get('tv_results', [])
        if not tmdb_results:
            return ""
        tmdb_id = tmdb_results[0]['id']
        media_type = 'movie' if resp.get('movie_results') else 'tv'
        credits_url = f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}/credits?api_key={api_key}"
        credits = requests.get(credits_url, timeout=10).json()
        cast = credits.get('cast', [])[:limit]
        if cast:
            return ', '.join([c['name'] for c in cast])
    except Exception as e:
        logger.error(f"Failed to fetch cast for {imdb_id}: {e}")
    return ""


def fetch_tmdb_trailer_key(title: str, year: str = "", imdb_id: str = None, category: str = "") -> Optional[str]:
    """Resolve and return a YouTube trailer key while metadata is being saved."""
    api_key = TMDB_API_KEY
    media_type = 'tv' if any(token in str(category).lower() for token in ('tv', 'series', 'web')) else 'movie'
    candidates = []
    try:
        if imdb_id and str(imdb_id).startswith('tt'):
            find_response = requests.get(
                f"https://api.themoviedb.org/3/find/{quote(str(imdb_id))}",
                params={'api_key': api_key, 'external_source': 'imdb_id'},
                timeout=8
            ).json()
            candidates = [
                (item, 'movie') for item in find_response.get('movie_results', [])
            ] + [
                (item, 'tv') for item in find_response.get('tv_results', [])
            ]
        if not candidates and title:
            search_response = requests.get(
                "https://api.themoviedb.org/3/search/multi",
                params={'api_key': api_key, 'query': title, 'include_adult': 'true'},
                timeout=8
            ).json()
            candidates = [
                (item, item.get('media_type'))
                for item in search_response.get('results', [])
                if item.get('media_type') in {'movie', 'tv'}
            ]
            if year:
                dated = [
                    item for item in candidates
                    if str(item[0].get('release_date') or item[0].get('first_air_date') or '')[:4] == str(year)[:4]
                ]
                if dated:
                    candidates = dated
            candidates = [item for item in candidates if item[1] == media_type] or candidates

        for candidate, candidate_type in candidates[:3]:
            videos = requests.get(
                f"https://api.themoviedb.org/3/{candidate_type}/{candidate['id']}/videos",
                params={'api_key': api_key, 'language': 'en-US'},
                timeout=8
            ).json()
            youtube_videos = [
                video for video in videos.get('results', [])
                if video.get('site') == 'YouTube' and video.get('key')
            ]
            preferred = [
                video for video in youtube_videos
                if video.get('type') == 'Trailer' and (
                    video.get('official') or 'official' in (video.get('name') or '').lower()
                )
            ] or [video for video in youtube_videos if video.get('type') == 'Trailer']
            if preferred:
                return preferred[0]['key']
    except Exception as exc:
        logger.warning("TMDb trailer lookup failed for '%s': %s", title, exc)
    return None


def resolve_trailer_key(
    title: str,
    year: str = "",
    imdb_id: str = None,
    category: str = "",
    fallback_title: str = "",
) -> Optional[str]:
    """Resolve a trailer using canonical metadata, then original file identity."""
    trailer_key = fetch_tmdb_trailer_key(title, year, imdb_id, category)
    if trailer_key or not fallback_title or fallback_title.strip().casefold() == str(title or "").strip().casefold():
        return trailer_key
    return fetch_tmdb_trailer_key(fallback_title, year, imdb_id, category)

def fetch_tmdb_artwork(
    title: str,
    year: str = "",
    imdb_id: str = None,
    category: str = "",
) -> tuple:
    """Resolve poster and landscape backdrop URLs for a stored movie."""
    api_key = TMDB_API_KEY
    if not api_key or not title:
        return None, None

    try:
        candidates = []
        if imdb_id and str(imdb_id).startswith("tt"):
            found = requests.get(
                f"https://api.themoviedb.org/3/find/{quote(str(imdb_id))}",
                params={"api_key": api_key, "external_source": "imdb_id"},
                timeout=8,
            ).json()
            candidates = [
                (item, "movie") for item in found.get("movie_results", [])
            ] + [
                (item, "tv") for item in found.get("tv_results", [])
            ]

        if candidates:
            match = candidates[0][0]
            media_type = candidates[0][1]
        else:
            found = requests.get(
                "https://api.themoviedb.org/3/search/multi",
                params={"api_key": api_key, "query": title, "include_adult": "true"},
                timeout=8,
            ).json()
            results = [
                item for item in found.get("results", [])
                if item.get("media_type") in {"movie", "tv"}
            ]
            match = _find_best_tmdb_match(results, title, str(year or ""))
            if not match:
                return None, None
            media_type = match.get("media_type") or (
                "tv" if any(token in str(category).lower() for token in ("tv", "series", "web")) else "movie"
            )

        tmdb_id = match.get("id")
        if tmdb_id and not match.get("backdrop_path"):
            details = requests.get(
                f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}",
                params={"api_key": api_key},
                timeout=8,
            ).json()
            match = {**match, **details}

        poster_path = match.get("poster_path")
        backdrop_path = match.get("backdrop_path")
        poster_url = (
            f"https://image.tmdb.org/t/p/original{poster_path}"
            if poster_path else None
        )
        backdrop_url = (
            f"https://image.tmdb.org/t/p/original{backdrop_path}"
            if backdrop_path else None
        )
        return poster_url, backdrop_url
    except Exception as exc:
        logger.warning("TMDb artwork lookup failed for '%s': %s", title, exc)
        return None, None

# ==================== NEW METADATA HELPER FUNCTIONS ====================

def get_tmdb_backdrop(query, search_year=""):
    """TMDB API se HD Original Poster (Vertical with Text) nikalta hai — run_async se call karo"""
    api_key = "9fa44f5e9fbd41415df930ce5b81c4d7" 
    try:
        url = f"https://api.themoviedb.org/3/search/multi?api_key={api_key}&query={quote(query)}"
        resp = requests.get(url, timeout=10).json()
        
        if resp.get('results'):
            for item in resp['results']:
                item_year = str(item.get('release_date', item.get('first_air_date', '')))[:4]
                if search_year and str(search_year) != item_year:
                    continue
                
                # 🛑 NAYA: Ab pehle Original Poster (Jisme Text hota hai) dhundega
                if item.get('poster_path'):
                    return f"https://image.tmdb.org/t/p/original{item['poster_path']}"
                elif item.get('backdrop_path'):
                    return f"https://image.tmdb.org/t/p/original{item['backdrop_path']}"
            
            first = resp['results'][0]
            if first.get('poster_path'):
                return f"https://image.tmdb.org/t/p/original{first['poster_path']}"
            elif first.get('backdrop_path'):
                return f"https://image.tmdb.org/t/p/original{first['backdrop_path']}"
    except Exception as e:
        logger.error(f"TMDB Error: {e}")
    return None
def _find_best_tmdb_match(tmdb_results: list, search_query: str, search_year: str = ""):
    """
    🎯 TMDb results mein se BEST match choose karta hai — blindly first nahi.
    Scoring: Title similarity + Year match + Popularity
    Ye galat movie aane ka sabse bada fix hai.
    """
    if not tmdb_results:
        return None
    
    search_lower = search_query.lower().strip()
    best_match = None
    best_score = -1
    
    for item in tmdb_results:
        score = 0
        
        # 1. Title similarity score (0-100)
        item_title = (item.get('title') or item.get('name') or '').lower().strip()
        item_original = (item.get('original_title') or item.get('original_name') or '').lower().strip()
        
        # Best of title vs original_title (Hindi movies ka original_title alag hota hai)
        title_sim = fuzz.token_set_ratio(search_lower, item_title)
        original_sim = fuzz.token_set_ratio(search_lower, item_original)
        similarity = max(title_sim, original_sim)
        score += similarity  # 0-100 points
        
        # 2. Year match bonus (+50 points — bahut strong signal)
        if search_year and str(search_year).strip().isdigit():
            item_year = str(item.get('release_date', item.get('first_air_date', '')))[:4]
            if item_year == str(search_year).strip():
                score += 50  # Year match — strong boost
            elif item_year and abs(int(item_year) - int(search_year)) <= 1:
                score += 20  # Off by 1 year — small boost
        
        # 3. Popularity bonus (popular items are usually correct)
        popularity = item.get('popularity', 0)
        if popularity > 50:
            score += 10
        elif popularity > 10:
            score += 5
        
        # 4. Has poster bonus (real movies usually have posters)
        if item.get('poster_path'):
            score += 5
        
        logger.debug(f"TMDb scoring: '{item_title}' | sim={similarity} | total={score}")
        
        if score > best_score:
            best_score = score
            best_match = item
    
    # Minimum threshold — agar score bahut low hai toh reject karo
    if best_score < 55:
        logger.warning(f"⚠️ TMDb: Best match score {best_score} too low for '{search_query}', rejecting")
        return None
    
    logger.info(f"✅ TMDb Best Match: '{best_match.get('title') or best_match.get('name')}' (score: {best_score})")
    return best_match

def resolve_tmdb_id_from_imdb(imdb_id: str, hint_category: str = ""):
    """Resolve the canonical TMDB identity without relying on a display title."""
    if not imdb_id or not re.match(r"^tt\d{7,8}$", str(imdb_id).strip()):
        return None

    tmdb_api_key = os.environ.get(
        "TMDB_API_KEY", "9fa44f5e9fbd41415df930ce5b81c4d7"
    )
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/find/{str(imdb_id).strip()}",
            params={
                "api_key": tmdb_api_key,
                "external_source": "imdb_id",
            },
            timeout=10,
        ).json()
        preferred = (
            ["tv_results", "movie_results"]
            if "series" in str(hint_category).lower()
            else ["movie_results", "tv_results"]
        )
        for result_type in preferred:
            results = response.get(result_type) or []
            if results and results[0].get("id") is not None:
                return int(results[0]["id"])
    except Exception as exc:
        logger.warning("TMDB identity lookup failed for %s: %s", imdb_id, exc)
    return None


def _find_movie_by_provider_identity(cur, imdb_id=None, tmdb_id=None):
    """Return the existing movie ID for provider IDs, never for title alone."""
    imdb_id = str(imdb_id).strip() if imdb_id else None
    tmdb_id = int(tmdb_id) if tmdb_id is not None else None
    if not imdb_id and tmdb_id is None:
        return None

    conditions = []
    params = []
    if imdb_id:
        conditions.append("imdb_id = %s")
        params.append(imdb_id)
    if tmdb_id is not None:
        conditions.append("tmdb_id = %s")
        params.append(tmdb_id)

    cur.execute(
        f"SELECT id, imdb_id, tmdb_id FROM movies WHERE {' OR '.join(conditions)}",
        tuple(params),
    )
    matches = cur.fetchall()
    if not matches:
        return None

    ids = {row[0] for row in matches}
    if len(ids) > 1:
        raise ValueError(
            f"IMDb/TMDB identities point to different movies: imdb_id={imdb_id!r}, "
            f"tmdb_id={tmdb_id!r}"
        )
    return matches[0][0]


def _find_movie_by_canonical_identity(
    cur, title, year=None, require_series=False, content_type=None
):
    """Resolve exact canonical names only when the remaining identity is unique."""
    conditions = ["LOWER(BTRIM(title)) = LOWER(BTRIM(%s))"]
    params = [title]
    if year:
        conditions.append("year = %s")
        params.append(int(year))
    if require_series:
        conditions.append(
            "LOWER(COALESCE(content_type, '')) IN "
            "('web series', 'tv series', 'tv show', 'series', 'anime')"
        )
    elif content_type:
        conditions.append("LOWER(COALESCE(content_type, '')) = LOWER(%s)")
        params.append(content_type)
    cur.execute(
        "SELECT id FROM movies WHERE {} ORDER BY id".format(" AND ".join(conditions)),
        tuple(params),
    )
    rows = cur.fetchall()
    if len(rows) > 1:
        return None, True
    return (rows[0][0], False) if rows else (None, False)


def _record_ingestion_evidence(
    conn,
    *,
    file_unique_id,
    raw_caption,
    raw_filename,
    evidence,
    parsed_identity,
    provider_ids,
    movie_id,
    movie_file_id,
    resolver_method,
    confidence,
    status,
    warnings=(),
):
    """Upsert the latest evidence for a Telegram file without losing its history."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO ingestion_evidence
            (telegram_file_unique_id, raw_caption, raw_filename, evidence,
             parsed_identity, provider_ids, resolved_movie_id, movie_file_id,
             resolver_method, confidence, status, warnings)
        VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (telegram_file_unique_id) DO UPDATE SET
            raw_caption = EXCLUDED.raw_caption,
            raw_filename = EXCLUDED.raw_filename,
            evidence = EXCLUDED.evidence,
            parsed_identity = EXCLUDED.parsed_identity,
            provider_ids = EXCLUDED.provider_ids,
            resolved_movie_id = EXCLUDED.resolved_movie_id,
            movie_file_id = EXCLUDED.movie_file_id,
            resolver_method = EXCLUDED.resolver_method,
            confidence = EXCLUDED.confidence,
            status = EXCLUDED.status,
            warnings = EXCLUDED.warnings,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            file_unique_id,
            raw_caption or "",
            raw_filename or "",
            json.dumps(evidence or {}, ensure_ascii=False),
            json.dumps(parsed_identity or {}, ensure_ascii=False),
            json.dumps(provider_ids or {}, ensure_ascii=False),
            movie_id,
            movie_file_id,
            resolver_method or "",
            max(0.0, min(float(confidence or 0), 1.0)),
            status,
            json.dumps(list(warnings or ()), ensure_ascii=False),
        ),
    )
    cur.close()


def _record_pending_identity(
    file_unique_id,
    raw_caption,
    raw_filename,
    evidence,
    parsed_identity,
    warning,
):
    """Persist identity decisions that are intentionally held before movie creation."""
    conn = get_db_connection()
    if not conn:
        logger.error("Could not persist pending identity: database connection unavailable")
        return
    try:
        _record_ingestion_evidence(
            conn,
            file_unique_id=file_unique_id,
            raw_caption=raw_caption,
            raw_filename=raw_filename,
            evidence=evidence,
            parsed_identity=parsed_identity,
            provider_ids={},
            movie_id=None,
            movie_file_id=None,
            resolver_method="no_safe_canonical_identity",
            confidence=0,
            status="pending_review",
            warnings=(warning,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("Could not persist pending identity evidence")
    finally:
        close_db_connection(conn)


def _link_movie_file_episodes(conn, movie_file_id, movie_id, parsed_identity):
    """Create normalized season/episode records and link this file to each episode."""
    season_number = parsed_identity.get("season_number")
    episode_numbers = parsed_identity.get("episode_numbers") or ()
    if season_number is None:
        return

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO seasons (movie_id, season_number)
        VALUES (%s, %s)
        ON CONFLICT (movie_id, season_number)
        DO UPDATE SET season_number = EXCLUDED.season_number
        RETURNING id
        """,
        (movie_id, int(season_number)),
    )
    season_id = cur.fetchone()[0]
    cur.execute(
        """
        INSERT INTO file_seasons (movie_file_id, season_id)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING
        """,
        (movie_file_id, season_id),
    )
    for episode_number in episode_numbers:
        cur.execute(
            """
            INSERT INTO episodes (season_id, episode_number)
            VALUES (%s, %s)
            ON CONFLICT (season_id, episode_number)
            DO UPDATE SET episode_number = EXCLUDED.episode_number
            RETURNING id
            """,
            (season_id, int(episode_number)),
        )
        episode_id = cur.fetchone()[0]
        cur.execute(
            """
            INSERT INTO file_episodes (movie_file_id, episode_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            (movie_file_id, episode_id),
        )
    conn.commit()
    cur.close()


def fetch_movie_metadata(query: str, search_year: str = "", search_lang: str = "", adult_mode: bool = False, hint_category: str = ""):
    """
    IMDb से डेटा और TMDb से सिर्फ Lamba (Portrait) पोस्टर निकालने वाला इंजन
    adult_mode=True होने पर TMDb सर्च में include_adult=true भेजेगा और OMDb को बायपास करेगा।
    """
    omdb_api_key = os.environ.get("OMDB_API_KEY")
    tmdb_api_key = "9fa44f5e9fbd41415df930ce5b81c4d7"

    search_query = query.strip()
    is_imdb_id = bool(re.match(r'^tt\d{7,8}$', search_query))

    logger.info(f"🔍 Metadata fetch for: '{search_query}' | year={search_year} | category={hint_category}")

    # ----- एडल्ट मोड: OMDb का उपयोग न करें (क्योंकि उसमें एडल्ट डेटा नहीं) -----
    if adult_mode:
        try:
            tmdb_search = f"https://api.themoviedb.org/3/search/multi?api_key={tmdb_api_key}&query={quote(search_query)}&include_adult=true"
            if search_year and search_year.strip().isdigit():
                tmdb_search += f"&year={search_year.strip()}"
            t_resp = requests.get(tmdb_search, timeout=10).json()
            if not t_resp.get('results'):
                return None

            # 🔧 FIX: Smart match instead of blindly first
            best_match = _find_best_tmdb_match(t_resp['results'], search_query, search_year)
            if not best_match:
                best_match = t_resp['results'][0]  # Fallback to first if scoring rejects all

            title = best_match.get('title') or best_match.get('name') or search_query
            year_str = str(best_match.get('release_date', best_match.get('first_air_date', '')))[:4]
            year = int(year_str) if year_str.isdigit() else 0
            plot = best_match.get('overview', 'No story available.')
            rating = str(round(best_match.get('vote_average', 0), 1)) if best_match.get('vote_average') else 'N/A'
            category = "Adult"
            genre = "Romance, Drama"

            path = best_match.get('poster_path')
            poster_url = f"https://image.tmdb.org/t/p/original{path}" if path else None

            imdb_id = None
            try:
                tmdb_id = best_match.get('id')
                media_type = best_match.get('media_type', 'movie')
                ext_url = f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}/external_ids?api_key={tmdb_api_key}"
                imdb_id = requests.get(ext_url, timeout=5).json().get('imdb_id')
            except:
                pass

            return title, year, poster_url, genre, imdb_id, rating, plot, category, {}
        except Exception as e:
            logger.error(f"Adult TMDb Fetch Error: {e}")
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🔧 FIXED: NORMAL MODE — Smart OMDb → TMDb Chain
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    try:
        omdb_resp = None
        
        # ━━━━━ STEP 1: OMDb Search (agar key hai toh) ━━━━━
        if omdb_api_key and not is_imdb_id:
            # 🔧 FIX: Pehle BINA type ke try karo (type galat hone se result miss hota tha)
            url_no_type = f"https://www.omdbapi.com/?t={quote(search_query)}&apikey={omdb_api_key}&plot=full"
            if search_year and str(search_year).strip().isdigit():
                url_no_type += f"&y={str(search_year).strip()}"
            
            resp = requests.get(url_no_type, timeout=10).json()
            
            if resp.get("Response") == "True":
                omdb_resp = resp
            else:
                # 🔧 FIX: Retry WITH type parameter (agar bina type se nahi mila)
                is_series = "series" in hint_category.lower() if hint_category else False
                if is_series:
                    url_with_type = f"https://www.omdbapi.com/?t={quote(search_query)}&type=series&apikey={omdb_api_key}&plot=full"
                    if search_year and str(search_year).strip().isdigit():
                        url_with_type += f"&y={str(search_year).strip()}"
                    resp2 = requests.get(url_with_type, timeout=10).json()
                    if resp2.get("Response") == "True":
                        omdb_resp = resp2
                        
        elif omdb_api_key and is_imdb_id:
            url = f"https://www.omdbapi.com/?i={search_query}&apikey={omdb_api_key}&plot=full"
            resp = requests.get(url, timeout=10).json()
            if resp.get("Response") == "True":
                omdb_resp = resp

        # ━━━━━ STEP 2: OMDb se data mila — process karo ━━━━━
        if omdb_resp:
            title = omdb_resp.get('Title')
            year = int(omdb_resp.get('Year', '0').split('–')[0]) if omdb_resp.get('Year') else 0
            genre = omdb_resp.get('Genre', 'Action, Drama')
            rating = omdb_resp.get('imdbRating', 'N/A')
            plot = omdb_resp.get('Plot', 'No story available.')
            imdb_id = omdb_resp.get('imdbID')
            country = omdb_resp.get('Country', '')
            lang = omdb_resp.get('Language', '').lower()

            # Smart category detection
            category = "Movies"
            omdb_type = omdb_resp.get('Type', '').lower()
            g_low = genre.lower()
            
            if omdb_type == 'series':
                category = "Web Series"
            elif "animation" in g_low or "anime" in g_low:
                category = "Anime"
            elif "india" in country.lower():
                if any(x in lang for x in ['telugu', 'tamil', 'kannada', 'malayalam']):
                    category = "South"
                else:
                    category = "Bollywood"
            else:
                category = "Hollywood"

            # TMDb se HD poster laao
            poster_url = omdb_resp.get('Poster')
            if imdb_id and imdb_id != 'N/A':
                try:
                    tmdb_find = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={tmdb_api_key}&external_source=imdb_id"
                    t_resp = requests.get(tmdb_find, timeout=10).json()
                    results = t_resp.get('movie_results', []) + t_resp.get('tv_results', [])
                    if results:
                        path = results[0].get('poster_path')
                        if path:
                            poster_url = f"https://image.tmdb.org/t/p/original{path}"
                except:
                    pass
            else:
                try:
                    tmdb_search = f"https://api.themoviedb.org/3/search/multi?api_key={tmdb_api_key}&query={quote(title)}"
                    t_resp = requests.get(tmdb_search, timeout=10).json()
                    if t_resp.get('results'):
                        for item in t_resp['results']:
                            item_year = str(item.get('release_date', item.get('first_air_date', '')))[:4]
                            if str(year) == item_year and item.get('poster_path'):
                                poster_url = f"https://image.tmdb.org/t/p/original{item['poster_path']}"
                                break
                        else:
                            path = t_resp['results'][0].get('poster_path')
                            if path:
                                poster_url = f"https://image.tmdb.org/t/p/original{path}"
                except:
                    pass

            
            seasons_data = {}
            try:
                tmdb_id = None
                if category in ["Web Series", "Anime", "Adult"]:
                    if not tmdb_id and imdb_id:
                        tmdb_find = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={tmdb_api_key}&external_source=imdb_id"
                        t_resp = requests.get(tmdb_find, timeout=10).json()
                        tv_res = t_resp.get('tv_results', [])
                        if tv_res: tmdb_id = tv_res[0].get('id')
                    if tmdb_id:
                        tv_details = requests.get(f"https://api.themoviedb.org/3/tv/{tmdb_id}?api_key={tmdb_api_key}", timeout=10).json()
                        for s in tv_details.get('seasons', []):
                            s_num = str(s.get('season_number', ''))
                            if s_num and str(s_num) != "0":
                                s_air_date = str(s.get('air_date', ''))
                                s_year = s_air_date[:4]
                                s_poster = f"https://image.tmdb.org/t/p/original{s.get('poster_path')}" if s.get('poster_path') else None
                                episode_count = s.get('episode_count', 0)
                                episodes_info = {}
                                try:
                                    season_url = f"https://api.themoviedb.org/3/tv/{tmdb_id}/season/{s_num}?api_key={tmdb_api_key}"
                                    season_details = requests.get(season_url, timeout=5).json()
                                    for ep in season_details.get('episodes', []):
                                        ep_num = str(ep.get('episode_number'))
                                        episodes_info[ep_num] = {'air_date': ep.get('air_date', '')}
                                except Exception as ep_e:
                                    logger.error(f"Episode fetch error: {ep_e}")
                                seasons_data[str(s_num)] = {
                                    "year": int(s_year) if s_year.isdigit() else 0,
                                    "poster": s_poster,
                                    "air_date": s_air_date,
                                    "episode_count": episode_count,
                                    "episodes": episodes_info
                                }
            except Exception as e:
                logger.error(f"Seasons Fetch Error: {e}")
                
            logger.info(f"✅ OMDb Success: '{title}' ({year}) [{category}]")
            return title, year, poster_url, genre, imdb_id, rating, plot, category, seasons_data

        # ━━━━━ STEP 3: OMDb fail — TMDb SMART FALLBACK ━━━━━
        logger.info(f"⚠️ OMDb miss for '{search_query}', trying TMDb smart search...")
        
        # 🔧 FIX: search/multi use karo (movie + tv dono milenge) 
        tmdb_search = f"https://api.themoviedb.org/3/search/multi?api_key={tmdb_api_key}&query={quote(search_query)}"
        if search_year and str(search_year).strip().isdigit():
            tmdb_search += f"&year={search_year.strip()}"
        t_resp = requests.get(tmdb_search, timeout=10).json()
        
        if not t_resp.get('results'):
            # Agar multi mein nahi mila, try TV-only search (agar series hint hai)
            is_series = "series" in hint_category.lower() if hint_category else False
            if is_series:
                tmdb_tv = f"https://api.themoviedb.org/3/search/tv?api_key={tmdb_api_key}&query={quote(search_query)}"
                if search_year and str(search_year).strip().isdigit():
                    tmdb_tv += f"&first_air_date_year={search_year.strip()}"
                t_resp = requests.get(tmdb_tv, timeout=10).json()
            
            if not t_resp.get('results'):
                logger.warning(f"❌ TMDb bhi fail for '{search_query}'")
                return None

        # 🔧 FIX: Smart match — blindly first nahi lega!
        best_match = _find_best_tmdb_match(t_resp['results'], search_query, search_year)
        if not best_match:
            # Agar smart match reject kar de, still try first as last resort
            best_match = t_resp['results'][0]
            logger.warning(f"⚠️ TMDb smart match rejected all, using first result as fallback")

        title = best_match.get('title') or best_match.get('name') or search_query
        year_str = str(best_match.get('release_date', best_match.get('first_air_date', '')))[:4]
        year = int(year_str) if year_str.isdigit() else 0
        plot = best_match.get('overview', 'No story available.')
        rating = str(round(best_match.get('vote_average', 0), 1)) if best_match.get('vote_average') else 'N/A'
        
        # Smart category from TMDb media_type
        media_type = best_match.get('media_type', '')
        if media_type == 'tv':
            category = "Web Series"
        elif media_type == 'movie':
            category = "Movies"
        else:
            category = "Web Series" if ("series" in hint_category.lower() if hint_category else False) else "Movies"
        
        genre = "Action, Drama"  # TMDb genre IDs need separate API call, using default
        
        # TMDb poster
        path = best_match.get('poster_path')
        poster_url = f"https://image.tmdb.org/t/p/original{path}" if path else None

        # IMDb ID nikalo TMDb se
        imdb_id = None
        try:
            tmdb_id = best_match.get('id')
            mt = 'tv' if media_type == 'tv' else 'movie'
            ext_url = f"https://api.themoviedb.org/3/{mt}/{tmdb_id}/external_ids?api_key={tmdb_api_key}"
            imdb_id = requests.get(ext_url, timeout=5).json().get('imdb_id')
        except:
            pass

        
        seasons_data = {}
        try:
            if category in ["Web Series", "Anime", "Adult"] and tmdb_id:
                tv_details = requests.get(f"https://api.themoviedb.org/3/tv/{tmdb_id}?api_key={tmdb_api_key}", timeout=10).json()
                for s in tv_details.get('seasons', []):
                    s_num = str(s.get('season_number', ''))
                    if s_num and str(s_num) != "0":
                        s_air_date = str(s.get('air_date', ''))
                        s_year = s_air_date[:4]
                        s_poster = f"https://image.tmdb.org/t/p/original{s.get('poster_path')}" if s.get('poster_path') else None
                        episode_count = s.get('episode_count', 0)
                        episodes_info = {}
                        try:
                            season_url = f"https://api.themoviedb.org/3/tv/{tmdb_id}/season/{s_num}?api_key={tmdb_api_key}"
                            season_details = requests.get(season_url, timeout=5).json()
                            for ep in season_details.get('episodes', []):
                                ep_num = str(ep.get('episode_number'))
                                episodes_info[ep_num] = {'air_date': ep.get('air_date', '')}
                        except Exception as ep_e:
                            logger.error(f"Episode fetch error: {ep_e}")
                        seasons_data[str(s_num)] = {
                            "year": int(s_year) if s_year.isdigit() else 0,
                            "poster": s_poster,
                            "air_date": s_air_date,
                            "episode_count": episode_count,
                            "episodes": episodes_info
                        }
        except Exception as e:
            logger.error(f"Seasons Fetch Error: {e}")
            
        logger.info(f"✅ TMDb Success: '{title}' ({year}) [{category}]")
        return title, year, poster_url, genre, imdb_id, rating, plot, category, seasons_data

    except Exception as e:
        logger.error(f"Metadata Fetch Error: {e}")
        return None

# ==================== AI INTENT ANALYSIS ====================
# 👇👇👇 START COPY HERE 👇👇👇
async def analyze_intent(message_text):
    """
    Bina AI (Gemini) ke message analyze karna.
    Isse API limit waste nahi hogi!
    """
    try:
        text_lower = message_text.lower().strip()
        
        # 1. Agar message bahut lamba hai ya usme Link hai, toh reject kar do
        if len(text_lower) > 60 or "http" in text_lower or "t.me" in text_lower:
            return {"is_request": False, "content_title": None}

        # 2. Agar chota message hai, toh usko direct Movie ka naam maan lo
        # Faltu words hatane ki koshish (Optional)
        words_to_remove = ["please", "plz", "bhai", "movie", "series", "chahiye", "give", "me"]
        clean_name = text_lower
        for word in words_to_remove:
            clean_name = clean_name.replace(word, "").strip()

        if len(clean_name) < 2:
            return {"is_request": False, "content_title": None}

        return {"is_request": True, "content_title": message_text.strip()}

    except Exception as e:
        logger.error(f"Error in intent analysis: {e}")
        return {"is_request": True, "content_title": message_text.strip()}
# 👆👆👆 END COPY HERE 👆👆👆

# ==================== NOTIFICATION FUNCTIONS ====================
async def send_admin_notification(context, user, movie_title, group_info=None):
    """Send notification to admin channel about a new request with Lifetime Buttons"""
    if not REQUEST_CHANNEL_ID: return

    try:
        safe_movie_title = movie_title.replace('<', '&lt;').replace('>', '&gt;')
        safe_username = user.username if user.username else 'N/A'
        safe_first_name = (user.first_name or 'Unknown').replace('<', '&lt;').replace('>', '&gt;')

        # 🌟 Premium Mention
        if user.username:
            user_display = f"<a href='https://t.me/{safe_username}'>{safe_first_name}</a>"
        else:
            user_display = f"<a href='tg://user?id={user.id}'>{safe_first_name}</a>"

        message = f"<b>━━━━━ 🎬 𝗡𝗲𝘄 𝗥𝗲𝗾𝘂𝗲𝘀𝘁! ━━━━━</b>\n\n"
        message += f"◈ Movie: <b>{safe_movie_title}</b>\n"
        message += f"◈ User: {user_display}\n"
        message += f"◈ ID: <code>{user.id}</code>\n"
        message += f"◈ From: {'Group: '+str(group_info) if group_info else 'Private Message'}\n"
        message += f"◈ Time: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}\n"
        message += f"<b>━━━━━━━━━━━━━━━━━━━</b>"

        # ⚡ LIFETIME BUTTONS LOGIC
        # Telegram me button data limit 64 bytes hoti hai, isliye title chota kiya hai
        short_title = safe_movie_title[:15].replace('_', ' ') 
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Movie Add Kar Di Gai Hai", callback_data=f"reqA_{user.id}_{short_title}")],
            [InlineKeyboardButton("❌ Nahi Mili", callback_data=f"reqN_{user.id}_{short_title}")]
        ])

        await context.bot.send_message(
            chat_id=REQUEST_CHANNEL_ID,
            text=message,
            parse_mode='HTML',
            reply_markup=keyboard
        )
    except Exception as e:
        logger.error(f"Error sending admin notification: {e}")

async def notify_users_for_movie(context: ContextTypes.DEFAULT_TYPE, movie_title, movie_url_or_file_id):
    logger.info(f"Attempting to notify users for movie: {movie_title}")
    conn = None
    cur = None
    notified_count = 0

    caption_text = (
        f"🎬 <b>{movie_title}</b>\n\n"
        "➖➖➖➖➖➖➖➖➖➖\n"
        "🔹 <b>Please drop the movie name, and I'll find it for you as soon as possible. 🎬✨👇</b>\n"
        "➖➖➖➖➖➖➖➖➖➖\n"
        "🔹 <b>Support group:</b> https://t.me/+dxaCr_cMmGpkYTFl\n"
    )
    join_keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("➡️ Join Channel", url=FILMFYBOX_CHANNEL_URL)]])

    try:
        conn = get_db_connection()
        if not conn:
            return 0

        cur = conn.cursor()
        cur.execute(
            "SELECT user_id, username, first_name FROM user_requests WHERE movie_title ILIKE %s AND notified = FALSE",
            (f'%{movie_title}%',)
        )
        users_to_notify = cur.fetchall()

        for user_id, username, first_name in users_to_notify:
            try:
                # 🌟 Premium Mention Format
                safe_name = (first_name or username or 'there').replace('<', '&lt;').replace('>', '&gt;')
                if username:
                    user_display = f"<a href='https://t.me/{username}'>{safe_name}</a>"
                else:
                    user_display = f"<a href='tg://user?id={user_id}'>{safe_name}</a>"

                # Optional heads-up text with premium mention
                try:
                    await safe_send(context.bot.send_message(
                        chat_id=user_id,
                        text=(
                            f"<b>━━━━━ 🎉 𝗚𝗼𝗼𝗱 𝗡𝗲𝘄𝘀! ━━━━━</b>\n\n"
                            f"✦ Hey {user_display}!\n\n"
                            f"◈ आपकी requested movie '<b>{movie_title}</b>' अब उपलब्ध है! 🥳\n\n"
                            f"<b>━━━━━━━━━━━━━━━━━━━</b>"
                        ),
                        parse_mode='HTML'
                    ))
                except Exception:
                    pass

                warning_msg = None
                try:
                    warning_msg = await safe_send(context.bot.copy_message(
                        chat_id=user_id,
                        from_chat_id=get_primary_dump_channel_id(),
                        message_id=3384
                    ))
                except Exception:
                    warning_msg = None

                sent_msg = None

                val = str(movie_url_or_file_id or "").strip()

                # Telegram file_id heuristics (your existing logic)
                is_file_id = any(val.startswith(prefix) for prefix in ["BQAC", "BAAC", "CAAC", "AQAC"])

                if is_file_id:
                    # try video then document
                    try:
                        sent_msg = await safe_send(context.bot.send_video(
                            chat_id=user_id, video=val, caption=caption_text,
                            parse_mode='HTML', reply_markup=join_keyboard
                        ))
                    except telegram.error.BadRequest:
                        sent_msg = await safe_send(context.bot.send_document(
                            chat_id=user_id, document=val, caption=caption_text,
                            parse_mode='HTML', reply_markup=join_keyboard
                        ))

                elif val.startswith("https://t.me/c/"):
                    parts = val.split('/')
                    from_chat_id = int("-100" + parts[-2])
                    msg_id = int(parts[-1])
                    sent_msg = await safe_send(context.bot.copy_message(
                        chat_id=user_id,
                        from_chat_id=from_chat_id,
                        message_id=msg_id,
                        caption=caption_text,
                        parse_mode='HTML',
                        reply_markup=join_keyboard
                    ))

                elif val.startswith("http"):
                    sent_msg = await safe_send(context.bot.send_message(
                        chat_id=user_id,
                        text=f"{caption_text}\n\n<b>Link:</b> {val}",
                        parse_mode='HTML',
                        disable_web_page_preview=True,
                        reply_markup=join_keyboard
                    ))

                else:
                    # last fallback: try send as document
                    sent_msg = await safe_send(context.bot.send_document(
                        chat_id=user_id,
                        document=val,
                        caption=caption_text,
                        parse_mode='HTML',
                        reply_markup=join_keyboard
                    ))

                if sent_msg:
                    # copy_message() returns MessageId rather than a media-bearing
                    # Message, so infer file retention from the delivery source too.
                    is_file_message = bool(file_id or (
                        url and "t.me/" in str(url)
                    )) or any(
                        getattr(sent_msg, media_type, None)
                        for media_type in ('document', 'video', 'audio', 'photo')
                    )
                    track_user_message_for_deletion(
                        context, user_id, sent_msg, is_file=is_file_message
                    )
                if warning_msg:
                    track_user_message_for_deletion(
                        context, user_id, warning_msg, is_file=True
                    )

                cur.execute(
                    "UPDATE user_requests SET notified = TRUE WHERE user_id = %s AND movie_title ILIKE %s",
                    (user_id, f'%{movie_title}%')
                )
                conn.commit()
                notified_count += 1

                await asyncio.sleep(0.1)

            except telegram.error.Forbidden:
                logger.error(f"User {user_id} blocked the bot")
                continue
            except Exception as e:
                logger.error(f"Error notifying user {user_id}: {e}", exc_info=True)
                continue

        return notified_count

    except Exception as e:
        logger.error(f"Error in notify_users_for_movie: {e}", exc_info=True)
        return 0
    finally:
        if cur:
            try: cur.close()
            except Exception: pass
        if conn:
            close_db_connection(conn)

async def notify_in_group(context: ContextTypes.DEFAULT_TYPE, movie_title):
    """Notify users in group when a requested movie becomes available"""
    logger.info(f"Attempting to notify users in group for movie: {movie_title}")
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        if not conn:
            return

        cur = conn.cursor()
        cur.execute(
            "SELECT user_id, username, first_name, group_id, message_id FROM user_requests WHERE movie_title ILIKE %s AND notified = FALSE",
            (f'%{movie_title}%',)
        )
        users_to_notify = cur.fetchall()

        if not users_to_notify:
            return

        groups_to_notify = defaultdict(list)
        for user_id, username, first_name, group_id, message_id in users_to_notify:
            if group_id:
                groups_to_notify[group_id].append((user_id, username, first_name, message_id))

        for group_id, users in groups_to_notify.items():
            try:
                notification_text = "<b>━━━━━ 🎉 𝗨𝗽𝗱𝗮𝘁𝗲! ━━━━━</b>\n\n✦ आपकी requested movie अब आ गई है! 🥳\n\n"
                notified_users_ids = []
                user_mentions = []
                for user_id, username, first_name, message_id in users:
                    name_to_show = first_name or username
                    if username:
                        mention = f"[{name_to_show}](https://t.me/{username})"
                    else:
                        mention = f"[{name_to_show}](tg://user?id={user_id})"
                    user_mentions.append(mention)
                    notified_users_ids.append(user_id)

                notification_text += "◈ " + ", ".join(user_mentions)
                notification_text += f"\n\n◈ आपकी फिल्म '{movie_title}' अब उपलब्ध है! इसे पाने के लिए, कृपया मुझे private में नाम भेजें...\n\n**━━━━━━━━━━━━━━━━━━━**"

                await context.bot.send_message(
                    chat_id=group_id,
                    text=notification_text,
                    parse_mode='Markdown'
                )

                for user_id in notified_users_ids:
                    cur.execute(
                        "UPDATE user_requests SET notified = TRUE WHERE user_id = %s AND movie_title ILIKE %s",
                        (user_id, f'%{movie_title}%')
                    )
                conn.commit()

            except Exception as e:
                logger.error(f"Failed to send message to group {group_id}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error in notify_in_group: {e}")
    finally:
        if cur: cur.close()
        if conn: close_db_connection(conn)

# ==================== NEW GENRE FUNCTIONS ====================

def get_all_genres_from_db():
    """Fetch all unique genres from database"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT genre FROM movies WHERE genre IS NOT NULL AND genre != ''")
        results = cur.fetchall()
        
        # Parse comma-separated genres and flatten
        all_genres = []
        for row in results:
            genre_str = row[0]
            if genre_str:
                # Split by comma and strip spaces
                genres = [g.strip() for g in genre_str.split(',')]
                all_genres.extend(genres)
        
        # Remove duplicates and return sorted list
        unique_genres = sorted(set(all_genres))
        cur.close()
        close_db_connection(conn)
        return unique_genres
        
    except Exception as e:
        logger.error(f"Error fetching genres: {e}")
        return []
    finally:
        if conn:
            close_db_connection(conn)


def create_genre_selection_keyboard():
    """Create inline keyboard with genre selection buttons"""
    genres = get_all_genres_from_db()
    
    if not genres:
        return InlineKeyboardMarkup([[InlineKeyboardButton("❌ No Genres Found", callback_data="cancel_genre")]])
    
    keyboard = []
    row = []
    
    for idx, genre in enumerate(genres):
        row.append(InlineKeyboardButton(
            f"📂 {genre}",
            callback_data=f"genre_{genre}"
        ))
        
        # 2 buttons per row
        if (idx + 1) % 2 == 0:
            keyboard.append(row)
            row = []
    
    # Add remaining buttons
    if row:
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="cancel_genre")])
    return InlineKeyboardMarkup(keyboard)


def get_movies_by_genre(genre: str, limit: int = 10):
    """Fetch movies filtered by genre"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cur = conn.cursor()
        # Use ILIKE for case-insensitive search within genre string
        cur.execute("""
            SELECT id, title, url, file_id, poster_url, year 
            FROM movies 
            WHERE genre ILIKE %s
            ORDER BY year DESC NULLS LAST
            LIMIT %s
        """, (f'%{genre}%', limit))
        
        results = cur.fetchall()
        cur.close()
        close_db_connection(conn)
        return results
        
    except Exception as e:
        logger.error(f"Error fetching movies by genre: {e}")
        return []
    finally:
        if conn:
            close_db_connection(conn)


async def show_genre_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle 'Browse by Genre' button click"""
    if update.message:
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        
        # FSub check
        check = await is_user_member(context, user_id)
        if not check['is_member']:
            msg = await update.message.reply_text(
                get_join_message(check['channel'], check['group']),
                reply_markup=get_join_keyboard(),
                parse_mode='Markdown'
            )
            track_message_for_deletion(context, chat_id, msg.message_id, USER_TEXT_DELETE_SECONDS)
            return
        
        # Show genre selection
        keyboard = create_genre_selection_keyboard()
        msg = await update.message.reply_text(
            "📂 **Select a genre to browse movies:**",
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
        track_message_for_deletion(context, chat_id, msg.message_id, USER_TEXT_DELETE_SECONDS)


async def handle_genre_selection(update: Update, context:  ContextTypes.DEFAULT_TYPE):
    """Handle genre selection callback"""
    query = update.callback_query
    await query.answer()
    
    data = query. data
    
    if data == "cancel_genre":
        await query.edit_message_text("❌ Genre browsing cancelled.")
        return
    
    if data.startswith("genre_"):
        genre = data.replace("genre_", "")
        
        # Fetch movies for this genre
        movies = get_movies_by_genre(genre, limit=15)
        
        if not movies:
            await query.edit_message_text(
                f"😕 No movies found for genre: **{genre}**\n\n"
                "Try another genre or use 🔍 Search.",
                parse_mode='Markdown'
            )
            return
        
        # Create movie selection keyboard
        context.user_data['search_results'] = movies
        context.user_data['search_query'] = genre
        
        keyboard = create_movie_selection_keyboard(movies, page=0)  # ✅ Now handles 6-tuple
        
        await query.edit_message_text(
            f"🎬 **Found {len(movies)} movies in '{genre}' genre**\n\n"
            "👇 Select a movie:",
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
# ==================== KEYBOARD MARKUPS ====================
def get_main_keyboard():
    # The Mini App/menu button is now the single navigation surface. Returning
    # this markup removes the old six-button reply keyboard for existing users.
    return ReplyKeyboardRemove()

def get_admin_request_keyboard(user_id, movie_title):
    """Inline keyboard for admin actions on a user request"""
    sanitized_title = movie_title[:30]

    keyboard = [
        [InlineKeyboardButton("✅ FULFILL MOVIE", callback_data=f"admin_fulfill_{user_id}_{sanitized_title}")],
        [InlineKeyboardButton("❌ IGNORE/DELETE", callback_data=f"admin_delete_{user_id}_{sanitized_title}")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_movie_options_keyboard(movie_title, url, movie_id=None, file_info=None):
    keyboard = []

    # Scan info only if movie_id is available
    if movie_id is not None:
        keyboard.append([InlineKeyboardButton("ℹ️ SCAN INFO : AUDIO & SUBS", callback_data=f"scan_{movie_id}")])

    if url:
        keyboard.append([InlineKeyboardButton("🎬 Watch Now", url=url)])

    keyboard.append([InlineKeyboardButton("📥 Download", callback_data=f"download_{movie_title[:50]}")])
    keyboard.append([InlineKeyboardButton("➡️ Join Channel", url=FILMFYBOX_CHANNEL_URL)])

    return InlineKeyboardMarkup(keyboard)

def create_movie_selection_keyboard(movies, page=0, movies_per_page=5, requester_id=None):
    """Movie selection keyboard. requester_id set karo to group me buttons locked honge sirf us user ke liye."""
    start_idx = page * movies_per_page
    end_idx = start_idx + movies_per_page
    current_movies = movies[start_idx:end_idx]

    # Group buttons ke liye user_id suffix
    u_suffix = f"_u{requester_id}" if requester_id else ""

    keyboard = []

    for movie in current_movies:
        # FIX: check 8-tuple before 6-tuple
        if len(movie) >= 8:
            movie_id, title, url, file_id, imdb_id, poster_url, year, genre = movie[:8]
        elif len(movie) >= 6:
            movie_id, title, url, file_id, poster_url, year = movie[:6]
        else:
            movie_id, title = movie[0], movie[1]

        year_text = ""
        if len(movie) >= 7 and movie[6]:
            year_text = str(movie[6])[:4]
        button_text = f"{title}   {year_text}" if year_text else title
        if len(button_text) > 40:
            button_text = button_text[:37] + "..."
        keyboard.append([InlineKeyboardButton(f"🎬 {button_text}", callback_data=f"movie_{movie_id}{u_suffix}")])

    total_pages = (len(movies) + movies_per_page - 1) // movies_per_page
    nav_buttons = []

    if page > 0:
        nav_buttons.append(InlineKeyboardButton("◀️ Previous", callback_data=f"page_{page-1}{u_suffix}"))
    if end_idx < len(movies):
        nav_buttons.append(InlineKeyboardButton("Next ▶️", callback_data=f"page_{page+1}{u_suffix}"))

    if nav_buttons:
        keyboard.append(nav_buttons)

    keyboard.append([InlineKeyboardButton("📢 Update Channel: Join BackUp", url=UPDATE_CHANNEL_URL)])
    keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_selection{u_suffix}")])
    return InlineKeyboardMarkup(keyboard)

def get_all_movie_qualities(movie_id):
    """Fetch all available qualities and their SIZES for a given movie ID"""
    conn = get_db_connection()
    if not conn:
        return []

    try:
        cur = conn.cursor()
        # NAYI QUERY: languages aur extra_info add kiya hai
        cur.execute("""
            SELECT quality, url, file_id, file_size, languages, extra_info
            FROM movie_files
            WHERE movie_id = %s AND (url IS NOT NULL OR file_id IS NOT NULL)
            ORDER BY CASE quality
                WHEN '4K' THEN 1
                WHEN 'HD Quality' THEN 2
                WHEN 'Standart Quality'  THEN 3
                WHEN 'Low Quality'  THEN 4
                ELSE 5
            END DESC
        """, (movie_id,))
        results = cur.fetchall()
        cur.close()
        return results
    except Exception as e:
        logger.error(f"Error fetching movie qualities for {movie_id}: {e}")
        return []
    finally:
        if conn:
            close_db_connection(conn)


def get_movie_delivery_meta(movie_id):
    """Load the small metadata row needed before rendering a file menu."""
    conn = get_db_connection()
    if not conn:
        return "", None
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT category, poster_url FROM movies WHERE id = %s",
            (movie_id,),
        )
        result = cur.fetchone()
        cur.close()
        return (
            result[0] if result else "",
            result[1] if result and len(result) > 1 else None,
        )
    except Exception as e:
        logger.error(f"Error fetching movie metadata for {movie_id}: {e}")
        return "", None
    finally:
        close_db_connection(conn)


def get_movie_delivery_data(movie_id):
    """Fetch file qualities and rendering metadata in one database round-trip."""
    total_start = time.perf_counter()
    acquire_start = total_start
    conn = get_db_connection()
    acquire_ms = (time.perf_counter() - acquire_start) * 1000
    if not conn:
        if SEARCH_TIMING_ENABLED:
            logger.info(
                "Search timing stage=delivery movie_id=%s total_ms=%.2f "
                "connection_acquire_ms=%.2f sql_ms=0 python_ms=0 result=unavailable",
                movie_id,
                (time.perf_counter() - total_start) * 1000,
                acquire_ms,
            )
        return [], ("", None)

    sql_ms = 0.0
    try:
        cur = conn.cursor()
        sql_start = time.perf_counter()
        cur.execute("""
            SELECT m.category, m.poster_url,
                   mf.quality, mf.url, mf.file_id, mf.file_size,
                   mf.languages, mf.extra_info
            FROM movies AS m
            LEFT JOIN movie_files AS mf
              ON mf.movie_id = m.id
             AND (mf.url IS NOT NULL OR mf.file_id IS NOT NULL)
            WHERE m.id = %s
            ORDER BY CASE mf.quality
                WHEN '4K' THEN 1
                WHEN 'HD Quality' THEN 2
                WHEN 'Standart Quality' THEN 3
                WHEN 'Low Quality' THEN 4
                ELSE 5
            END DESC
        """, (movie_id,))
        rows = cur.fetchall()
        sql_ms = (time.perf_counter() - sql_start) * 1000
        cur.close()

        if not rows:
            return [], ("", None)

        category = rows[0][0] or ""
        poster_url = rows[0][1]
        qualities = [
            (row[2], row[3], row[4], row[5], row[6], row[7])
            for row in rows
            if row[2] is not None
        ]
        return qualities, (category, poster_url)
    except Exception as e:
        logger.error(f"Error fetching delivery data for {movie_id}: {e}")
        return [], ("", None)
    finally:
        if conn:
            close_db_connection(conn)
        if SEARCH_TIMING_ENABLED:
            logger.info(
                "Search timing stage=delivery movie_id=%s total_ms=%.2f "
                "connection_acquire_ms=%.2f sql_ms=%.2f python_ms=%.2f",
                movie_id,
                (time.perf_counter() - total_start) * 1000,
                acquire_ms,
                sql_ms,
                max(
                    0.0,
                    (time.perf_counter() - total_start) * 1000 - acquire_ms - sql_ms,
                ),
            )


# create_quality_selection_keyboard function ko isse replace karein ya modify karein:

def create_quality_selection_keyboard(movie_id, view="main", page=1, total_pages=1, current_files=None, season_view=False):
    """नया UI: फाइल्स के लिए बटन्स, फिल्टर्स और पेजिनेशन"""
    keyboard = []
    
    if view == "main":

        # 2. अगर सीजन के अंदर हैं, तो बैक बटन दिखाओ
        if season_view:
            keyboard.append([InlineKeyboardButton("🔙 Back to Seasons", callback_data=f"back_to_seasons_{movie_id}")])

        # 3. Send All, Trending (Row 1)
        keyboard.append([
            InlineKeyboardButton("◆ SEND ALL", callback_data=f"sendall_{movie_id}_{page}"),
            InlineKeyboardButton("TRENDING", url=FILMFYBOX_GROUP_URL)
        ])
        
        # 4. Filters (Row 2)
        keyboard.append([
            InlineKeyboardButton("QUALITY", callback_data=f"v_qual_{movie_id}"),
            InlineKeyboardButton("LANGUAGE", callback_data=f"v_lang_{movie_id}"),
            InlineKeyboardButton("SEASON", callback_data=f"v_seas_{movie_id}")
        ])
        
        # 5. Pagination (Premium look)
        nav_buttons = []
        nav_buttons.append(InlineKeyboardButton("◀️ ᴘʀᴇᴠ" if page > 1 else "ᴘᴀɢᴇ", callback_data=f"vpage_{movie_id}_{page-1}" if page > 1 else "ignore"))
        nav_buttons.append(InlineKeyboardButton(f"{page}/{total_pages}", callback_data="ignore"))
        nav_buttons.append(InlineKeyboardButton("ɴᴇxᴛ ▶️" if page < total_pages else "ɴᴇxᴛ >", callback_data=f"vpage_{movie_id}_{page+1}" if page < total_pages else "ignore"))
        keyboard.append(nav_buttons)

    # ... (बाकी व्यूज जैसे language, quality, season पहले जैसे ही रहेंगे)
    elif view == "language":
                keyboard.append([InlineKeyboardButton("MALAYALAM", callback_data=f"fl_lang_{movie_id}_Malayalam"), InlineKeyboardButton("TAMIL", callback_data=f"fl_lang_{movie_id}_Tamil")])
                keyboard.append([InlineKeyboardButton("ENGLISH", callback_data=f"fl_lang_{movie_id}_English"), InlineKeyboardButton("HINDI", callback_data=f"fl_lang_{movie_id}_Hindi")])
                keyboard.append([InlineKeyboardButton("TELUGU", callback_data=f"fl_lang_{movie_id}_Telugu"), InlineKeyboardButton("KANNADA", callback_data=f"fl_lang_{movie_id}_Kannada")])
                # ✅ NAYA: Gujarati, Marathi aur Punjabi add ho gaye
                keyboard.append([InlineKeyboardButton("GUJARATI", callback_data=f"fl_lang_{movie_id}_Gujarati"), InlineKeyboardButton("MARATHI", callback_data=f"fl_lang_{movie_id}_Marathi")])
                keyboard.append([InlineKeyboardButton("PUNJABI", callback_data=f"fl_lang_{movie_id}_Punjabi")])
                keyboard.append([InlineKeyboardButton("🔄 CLEAR FILTER", callback_data=f"fl_clear_{movie_id}_all")])
                keyboard.append([InlineKeyboardButton("<< BACK TO FILES >>", callback_data=f"v_main_{movie_id}")])

    elif view == "quality":
                keyboard.append([InlineKeyboardButton("360P", callback_data=f"fl_qual_{movie_id}_360p"), InlineKeyboardButton("480P", callback_data=f"fl_qual_{movie_id}_480p")])
                keyboard.append([InlineKeyboardButton("720P", callback_data=f"fl_qual_{movie_id}_720p"), InlineKeyboardButton("1080P", callback_data=f"fl_qual_{movie_id}_1080p")])
                # ✅ NAYA: 1440P aur 2160P (Premium Quality) add ho gaye
                keyboard.append([InlineKeyboardButton("1440P", callback_data=f"fl_qual_{movie_id}_1440p"), InlineKeyboardButton("2160P", callback_data=f"fl_qual_{movie_id}_2160p")])
                keyboard.append([InlineKeyboardButton("4K", callback_data=f"fl_qual_{movie_id}_4K")])
                keyboard.append([InlineKeyboardButton("🔄 CLEAR FILTER", callback_data=f"fl_clear_{movie_id}_all")])
                keyboard.append([InlineKeyboardButton("<< BACK TO FILES >>", callback_data=f"v_main_{movie_id}")])

    elif view == "season":
        # ये डमी है, असली सीजन्स डायनामिकली बनते हैं
        keyboard.append([InlineKeyboardButton("🔄 CLEAR FILTER", callback_data=f"fl_clear_{movie_id}_all")])
        keyboard.append([InlineKeyboardButton("<< BACK TO FILES >>", callback_data=f"v_main_{movie_id}")])

    return InlineKeyboardMarkup(keyboard)


async def deliver_movie_page_on_start(update: Update, context: ContextTypes.DEFAULT_TYPE, movie_id: int, page: int):
    """Deliver the requested Send All page after a user starts the bot in PM."""
    page = max(1, page)
    conn = get_db_connection()
    if not conn:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="❌ System is temporarily unavailable. Please try again."
        )
        return

    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT title, genre, year, language, seasons_data FROM movies WHERE id = %s",
            (movie_id,)
        )
        movie = cur.fetchone()
        cur.close()
    finally:
        close_db_connection(conn)

    if not movie:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="❌ Movie not found or deleted."
        )
        return

    title, genre, year, language, seasons_data = movie
    qualities = get_all_movie_qualities(movie_id)
    start = (page - 1) * 10
    page_files = qualities[start:start + 10]
    if not page_files:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="❌ This file page is no longer available."
        )
        return

    metadata = {
        'genre': genre,
        'year': year,
        'language': language,
        'seasons_data': seasons_data,
    }
    for file_data in page_files:
        metadata['extra_info'] = str(file_data[5]).strip() if len(file_data) > 5 and file_data[5] else ""
        await send_movie_to_user(
            update,
            context,
            movie_id,
            title,
            file_data[1],
            file_data[2],
            send_warning=False,
            pre_fetched_meta=dict(metadata),
            suppress_delete_notice=True,
        )
        await asyncio.sleep(0.3)

    # Send auto-delete notice only once after all files are delivered
    try:
        target_chat_id = update.effective_user.id
        warn_text_msg = await context.bot.send_message(
            chat_id=target_chat_id,
            text=(
                "⚠️ <b>𝗔𝘂𝘁𝗼-𝗗𝗲𝗹𝗲𝘁𝗲 𝗡𝗼𝘁𝗶𝗰𝗲</b>\n\n"
                "◈ ऊपर भेजी गयी file <b>2 minutes</b> बाद auto-delete हो जाएगी।\n"
                "◈ कृपया file को <b>forward/save</b> कर लें। 🔄"
            ),
            parse_mode='HTML'
        )
        track_message_for_deletion(
            context, target_chat_id, warn_text_msg.message_id, USER_FILE_DELETE_SECONDS,
        )
    except:
        pass

def load_movie_selection_data(movie_id):
    """Reload callback state from the database after a file-link click clears memory."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, category FROM movies WHERE id = %s",
            (movie_id,)
        )
        movie = cur.fetchone()
        cur.close()
        if not movie:
            return None
    except Exception as exc:
        logger.error(f"Could not reload movie selection {movie_id}: {exc}")
        return None
    finally:
        close_db_connection(conn)
    return {
        'id': movie[0],
        'title': movie[1] or "Requested Movie",
        'category': movie[2] or "",
        'qualities': get_all_movie_qualities(movie_id)
    }

# ==================== HELPER FUNCTION ====================
async def send_movie_to_user(update: Update, context: ContextTypes.DEFAULT_TYPE, movie_id: int, title: str, url: Optional[str] = None, file_id: Optional[str] = None, send_warning: bool = True, pre_fetched_meta: dict = None, require_exact_file: bool = False, suppress_delete_notice: bool = False):
    """Sends the movie file/link to the user with THUMBNAIL PROTECTION - OPTIMIZED & FIXED"""
    chat_id = update.effective_chat.id

    # --- 1. Fetch movie details (Genre, Year, Language) ---
    genre = ""
    year = ""
    lang_display = ""
    extra_display = "" # NAYA: Info (Ep) dikhane ke liye

    # ✅ OPTIMIZATION: Agar data pehle se diya gaya hai, to DB connect mat karo
    seasons_data_db = None
    if pre_fetched_meta:
        db_genre = pre_fetched_meta.get('genre')
        db_year = pre_fetched_meta.get('year')
        db_lang = pre_fetched_meta.get('language')
        seasons_data_db = pre_fetched_meta.get('seasons_data')
        
        if db_genre and db_genre != 'Unknown': genre = f"🎭 <b>Genre:</b> {db_genre}\n"
        if db_year and db_year > 0: year = f"📅 <b>Year:</b> {db_year}\n"
        if db_lang and db_lang.strip(): lang_display = f"🔊 <b>Language:</b> {db_lang}\n"
    
    # Agar data nahi diya gaya, tabhi DB open karo
    else:
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT genre, year, language, seasons_data FROM movies WHERE id = %s", (movie_id,))
                result = cur.fetchone()
                if result:
                    db_genre, db_year, db_lang, seasons_data_db = result
                    if db_genre and db_genre != 'Unknown': genre = f"🎭 <b>Genre:</b> {db_genre}\n"
                    if db_year and db_year > 0: year = f"📅 <b>Year:</b> {db_year}\n"
                    if db_lang and db_lang.strip(): lang_display = f"🔊 <b>Language:</b> {db_lang}\n"
                cur.close()
            except Exception as e:
                logger.error(f"Error fetching movie info: {e}")
            finally:
                close_db_connection(conn)


    # 👇 NAYA CODE: Yahan hum us ek specific file ka info 'movie_files' table se nikalenge! 👇
    # ⚡ OPTIMIZATION: Agar extra_info pre_fetched_meta mein already hai (Send All se), toh DB call skip karo
    pre_extra = pre_fetched_meta.get('extra_info', '') if pre_fetched_meta else ''
    if pre_extra and pre_extra.strip():
        extra_val = pre_extra.strip()
        ext = extra_val.upper()
        
        edition_keywords = ["UNCUT", "EXTENDED", "CUT", "UNRATED", "REMASTERED", "EDITION"]
        
        if any(word in ext for word in edition_keywords):
            extra_display = f"📌 <b>Edition:</b> {extra_val}\n"
        elif "S" in ext and "E" in ext:
            extra_display = f"📌 <b>Season & Episode:</b> {extra_val}\n"
        elif "S" in ext:
            extra_display = f"📌 <b>Season:</b> {extra_val}\n"
        elif "E" in ext:
            extra_display = f"📌 <b>Episode:</b> {extra_val}\n"
        else:
            extra_display = f"📌 <b>Info:</b> {extra_val}\n"
            
        # SMART FIX: Update year based on seasons_data
        if ("S" in ext or "SEASON" in ext) and seasons_data_db:
            try:
                s_match = re.search(r'(?i)(?:S|SEASON\s*)0*(\d+)', ext)
                e_match = re.search(r'(?i)(?:E|EPISODE\s*)0*(\d+)', ext)
                if s_match:
                    s_num = str(int(s_match.group(1)))
                    if isinstance(seasons_data_db, dict) and s_num in seasons_data_db:
                        s_data = seasons_data_db[s_num]
                        specific_year = s_data.get('year') or (s_data.get('air_date', '')[:4] if s_data.get('air_date') else None)
                        
                        if e_match and 'episodes' in s_data:
                            e_num = str(int(e_match.group(1)))
                            ep_info = s_data['episodes'].get(e_num)
                            if ep_info and ep_info.get('air_date'):
                                specific_year = ep_info['air_date'][:4]
                                
                        if specific_year and str(specific_year).isdigit() and int(specific_year) > 0:
                            year = f"📅 <b>Year:</b> {specific_year}\n"
            except Exception as parse_e:
                logger.error(f"Error parsing season date: {parse_e}")
    
    elif url or file_id:
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                if file_id:
                    cur.execute("SELECT extra_info FROM movie_files WHERE file_id = %s LIMIT 1", (file_id,))
                else:
                    cur.execute("SELECT extra_info FROM movie_files WHERE url = %s LIMIT 1", (url,))
                
                res = cur.fetchone()
                if res and res[0] and res[0].strip():
                    extra_val = res[0].strip()
                    ext = extra_val.upper()
                    
                    # 👇 SMART FIX: Check karega ki kya likhna sahi rahega
                    edition_keywords = ["UNCUT", "EXTENDED", "CUT", "UNRATED", "REMASTERED", "EDITION"]
                    
                    if any(word in ext for word in edition_keywords):
                        extra_display = f"📌 <b>Edition:</b> {extra_val}\n"
                    elif "S" in ext and "E" in ext:
                        extra_display = f"📌 <b>Season & Episode:</b> {extra_val}\n"
                    elif "S" in ext:
                        extra_display = f"📌 <b>Season:</b> {extra_val}\n"
                    elif "E" in ext:
                        extra_display = f"📌 <b>Episode:</b> {extra_val}\n"
                    else:
                        extra_display = f"📌 <b>Info:</b> {extra_val}\n"
                        
                    # SMART FIX: Update year based on seasons_data if specific season/episode year is available
                    if ("S" in ext or "SEASON" in ext) and seasons_data_db:
                        try:
                            s_match = re.search(r'(?i)(?:S|SEASON\s*)0*(\d+)', ext)
                            e_match = re.search(r'(?i)(?:E|EPISODE\s*)0*(\d+)', ext)
                            if s_match:
                                s_num = str(int(s_match.group(1)))
                                if isinstance(seasons_data_db, dict) and s_num in seasons_data_db:
                                    s_data = seasons_data_db[s_num]
                                    specific_year = s_data.get('year') or (s_data.get('air_date', '')[:4] if s_data.get('air_date') else None)
                                    
                                    if e_match and 'episodes' in s_data:
                                        e_num = str(int(e_match.group(1)))
                                        ep_info = s_data['episodes'].get(e_num)
                                        if ep_info and ep_info.get('air_date'):
                                            specific_year = ep_info['air_date'][:4]
                                            
                                    if specific_year and str(specific_year).isdigit() and int(specific_year) > 0:
                                        year = f"📅 <b>Year:</b> {specific_year}\n"
                        except Exception as parse_e:
                            logger.error(f"Error parsing season date: {parse_e}")
                            
                cur.close()
            except Exception:
                pass
            finally:
                close_db_connection(conn)
    # 👆 ---------------------------------------------------- 👆

    # A Mini App quality button must never fall back to the whole movie list.
    # It always carries a concrete movie_files id; if that record has no usable
    # source, fail clearly instead of sending every available quality.
    if require_exact_file and not url and not file_id:
        await context.bot.send_message(
            chat_id=update.effective_user.id,
            text="❌ Selected file is unavailable. Please choose another quality from the Mini App."
        )
        return

    # 1. Multi-Quality Check (Agar direct link/file nahi hai)
    if not url and not file_id:
        all_qualities = get_all_movie_qualities(movie_id)
        if all_qualities:
            context.user_data['selected_movie_data'] = {'id': movie_id, 'title': title, 'qualities': all_qualities}
            context.user_data['active_filter'] = None
            context.user_data.pop('selected_season', None)
            
            limit = 10
            total_pages = (len(all_qualities) + limit - 1) // limit if all_qualities else 1
            current_files = all_qualities[0:limit]
            
            # 👇 YAHAN SE FIX SHURU HOTA HAI (HTML INLINE LINKS KE LIYE) 👇
            bot_username = context.bot.username
            text = (
                f"<b>🎬 {title}</b>\n"
                "<i>Choose your preferred version</i>\n\n"
            )
            
            
            for idx, f_data in enumerate(current_files, start=1):
                q_name = str(f_data[0])
                
                # Kachra saaf kar rahe hain taaki deep links perfect banein
                q_name = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', q_name)
                q_name = re.sub(r'\(https?://[^\)]+\)', '', q_name)
                q_name = re.sub(r'https?://[^\s]+', '', q_name)
                q_name = re.sub(r'(?i)t\.me/[^\s]+', '', q_name)
                q_name = re.sub(r'@[a-zA-Z0-9_]+', '', q_name)
                
                f_size = f_data[3] if len(f_data)>3 else "Unknown"
                
                e_info = str(f_data[5]) if len(f_data)>5 else ""
                e_info = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', e_info)
                e_info = re.sub(r'\(https?://[^\)]+\)', '', e_info)
                e_info = re.sub(r'https?://[^\s]+', '', e_info)
                e_info = re.sub(r'(?i)t\.me/[^\s]+', '', e_info)
                e_info = re.sub(r'@[a-zA-Z0-9_]+', '', e_info)
                
                # ✅ NAYA: HTML wala Neela (Inline) link
                real_idx = all_qualities.index(f_data)
                label = html_escape(
                    _format_file_link_label(f_size, title, e_info, q_name)
                )
                text += f"<b>{idx}.</b> <b><a href='https://t.me/{bot_username}?start=file_{movie_id}_{real_idx}'>{label}</a></b>\n\n"
            
            text += (
                f"\n<b>◆ More updates:</b> <a href='{UPDATE_CHANNEL_URL}'>Join BackUp</a>"
            )

            keyboard = create_quality_selection_keyboard(movie_id, view="main", page=1, total_pages=total_pages, current_files=current_files)
            
            # ✅ NAYA: parse_mode='HTML' kar diya aur link preview off kar diya
            msg = await context.bot.send_message(
                chat_id=chat_id, 
                text=text, 
                reply_markup=keyboard, 
                parse_mode='HTML', 
                disable_web_page_preview=True
            )
            # 👆 FIX KHATAM 👆
            
            track_message_for_deletion(
                context, chat_id, msg.message_id, USER_TEXT_DELETE_SECONDS
            )
            
            if update.callback_query:
                try:
                    await update.callback_query.answer("This file menu expires in 5 minutes.", show_alert=True)
                except:
                    pass
            return

    target_chat_id = update.effective_user.id if (url or file_id) else chat_id

    try:
        warning_msg = None
        # ❌ WARNING STICKER DISABLED — Ab yahan se koi sticker nahi jayega
        # if send_warning:
        #     try:
        #         warning_msg = await safe_send(context.bot.copy_message(
        #             chat_id=target_chat_id,
        #             from_chat_id=-1003893346701,
        #             message_id=3384
        #         ))
        #     except Exception as e:
        #         logger.error(f"Warning file send failed: {e}")
        
        # --- CAPTION UPDATE WITH EXTRA INFO ---
        caption_text = (
            f"<b>━━━━━ 🎬 𝗙𝗶𝗹𝗲 𝗗𝗲𝘁𝗮𝗶𝗹𝘀 ━━━━━</b>\n"
            f"✦ <b>{title}</b>\n"
            f"{extra_display}"
            f"{year}"        
            f"{genre}"       
            f"{lang_display}"  
            f"<b>Update Channel:</b> <a href='{UPDATE_CHANNEL_URL}'>Join BackUp</a>\n"
            f"\n◈ <b>JOIN »</b> <a href='{FILMFYBOX_CHANNEL_URL}'>FilmfyBox</a>\n\n"
            f"◈ <b>Drop the movie name, I'll find it for you 🎬✨👇</b>\n"
            f"◈ <b><a href='https://t.me/+dxaCr_cMmGpkYTFl'>FlimfyBox Chat</a></b>\n"
            f"<b>━━━━━━━━━━━━━━━━━━━</b>"
        )
        
        join_keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("➡️ Join Channel", url=FILMFYBOX_CHANNEL_URL)]])

        sent_msg = None
        if url and ("t.me/c/" in url or "t.me/" in url) and "http" in url:
            try:
                clean_url = url.strip()
                parts = clean_url.rstrip('/').split('/')
                msg_id = int(parts[-1])
                
                if "t.me/c/" in clean_url:
                    from_chat_id = int("-100" + parts[-2])
                else:
                    from_chat_id = f"@{parts[-2]}"

                sent_msg = await safe_send(context.bot.copy_message(
                    chat_id=target_chat_id,
                    from_chat_id=from_chat_id,
                    message_id=msg_id,
                    caption=caption_text,
                    parse_mode='HTML',
                    reply_markup=join_keyboard
                ))
            except Exception as e:
                logger.error(f"Copy link failed: {e}")

        if not sent_msg and file_id:
            clean_file_id = str(file_id).strip()
            try:
                sent_msg = await safe_send(context.bot.send_video(
                    chat_id=target_chat_id,
                    video=clean_file_id,
                    caption=caption_text,
                    parse_mode='HTML',
                    reply_markup=join_keyboard
                ))
            except telegram.error.BadRequest:
                try:
                    sent_msg = await safe_send(context.bot.send_document(
                        chat_id=target_chat_id,
                        document=clean_file_id,
                        caption=caption_text,
                        parse_mode='HTML',
                        reply_markup=join_keyboard
                    ))
                except Exception as e:
                    logger.error(f"Send Document failed: {e}")

        if not sent_msg and url and "http" in url and "t.me" not in url:
             sent_msg = await context.bot.send_message(
                chat_id=target_chat_id,
                text=f"🎬 <b>{title}</b>\n\n🔗 <b>Watch/Download:</b> {url}",
                parse_mode='HTML',
                reply_markup=join_keyboard
            )

        if sent_msg:
            # Downloadable media is retained for 2 minutes; link-only replies
            # are retained for the normal 5-minute user-message window.
            is_file_message = bool(
                file_id or (url and "t.me/" in str(url))
            ) or any(
                getattr(sent_msg, media_type, None)
                for media_type in ('document', 'video', 'audio', 'photo')
            )
            track_user_message_for_deletion(
                context, target_chat_id, sent_msg, is_file=is_file_message
            )
        if warning_msg:
            track_user_message_for_deletion(context, target_chat_id, warning_msg)
        elif not sent_msg:
            err_msg = await context.bot.send_message(chat_id=target_chat_id, text="❌ Error: File not found or Bot needs Admin rights in Source Channel.")
            track_message_for_deletion(context, target_chat_id, err_msg.message_id, USER_TEXT_DELETE_SECONDS)

        if sent_msg and update.callback_query:
            try:
                await update.callback_query.answer("✅ File Sent!\n⚠️ Ye file aur message 2 minutes baad delete ho jayegi.", show_alert=True)
            except:
                pass
        elif sent_msg and not update.callback_query and not suppress_delete_notice:
            # 🛡️ PM search / Deep link — no callback popup available, so text warning bhejo
            # Send All flow suppresses this and sends one notice after the loop.
            try:
                warn_text_msg = await context.bot.send_message(
                    chat_id=target_chat_id,
                    text=(
                        "⚠️ <b>𝗔𝘂𝘁𝗼-𝗗𝗲𝗹𝗲𝘁𝗲 𝗡𝗼𝘁𝗶𝗰𝗲</b>\n\n"
                        "◈ ऊपर भेजी गयी file <b>2 minutes</b> बाद auto-delete हो जाएगी।\n"
                        "◈ कृपया file को <b>forward/save</b> कर लें। 🔄"
                    ),
                    parse_mode='HTML'
                )
                # Keep the notice in the same deletion window as the file it
                # describes, so no stale warning remains after the file is gone.
                track_message_for_deletion(
                    context,
                    target_chat_id,
                    warn_text_msg.message_id,
                    USER_FILE_DELETE_SECONDS,
                )
            except:
                pass

    except telegram.error.Forbidden:
        if update.callback_query:
            try:
                bot_username = context.bot.username
                if bot_username and update.effective_chat.type in ("group", "supergroup"):
                    start_url = f"https://t.me/{bot_username}?start=sendall_{movie_id}_{context.user_data.get('sendall_page', 1)}"
                    # A callback URL opens the bot PM directly. Do not add a
                    # second group message just to explain the same action.
                    await update.callback_query.answer(url=start_url)
                else:
                    await update.callback_query.answer(
                        "⚠️ Please START the bot in PM first to receive files!",
                        show_alert=True
                    )
            except:
                pass
        logger.warning(f"User {target_chat_id} blocked or hasn't started the bot.")
    except Exception as e:
        logger.error(f"Critical Error in send_movie: {e}")
        try: await context.bot.send_message(chat_id=target_chat_id, text="❌ System Error.")
        except: pass

# ==================== TELEGRAM BOT HANDLERS ====================
# ============================================================================
# NEW BACKGROUND SEARCH & START LOGIC
# ============================================================================

async def background_search_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE, query_text: str, status_msg):
    """
    Runs database search in background to prevent blocking the bot.
    """
    chat_id = update.effective_chat.id
    try:
        # 1. PEHLE EXACT MATCH CHECK KAREIN (Ye FAST hai - 0.1 sec)
        # This saves resources if the user clicked a precise link
        conn = get_db_connection()
        exact_movies = []
        if conn:
            try:
                cur = conn.cursor()
                # Keep every exact-title row so duplicate titles can be
                # disambiguated by year in the selection keyboard.
                cur.execute(
                    """
                    SELECT id, title, url, file_id, imdb_id, poster_url, year, genre
                    FROM movies
                    WHERE title ILIKE %s
                    ORDER BY year DESC NULLS LAST, id DESC
                    LIMIT 20
                    """,
                    (query_text.strip(),),
                )
                exact_movies = cur.fetchall()
            except Exception as db_e:
                logger.error(f"Database error in exact match: {db_e}")
            finally:
                if conn: close_db_connection(conn)

        movies_found = []
        if exact_movies:
            movies_found = exact_movies
        else:
            # Agar exact nahi mila to hi Fuzzy Search karein (Slower process)
            # Assuming get_movies_from_db is your existing function
            movies_found = await run_async(get_movies_from_db, query_text, limit=1)

        # 2. Result Handle karein
        if not movies_found:
            try: await status_msg.delete() 
            except: pass
            
            suggestions = await run_async(get_google_title_suggestions, query_text, limit=3)
            keyboard = _not_found_keyboard(query_text, suggestions)
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"😕 Sorry, <b>'{html_escape(query_text)}</b>' not found.\n\n"
                    "Check the spelling and request the title below."
                ),
                reply_markup=keyboard,
                parse_mode='HTML'
            )
            return

        # 3. Movie Mil gayi - Send karein
        if len(movies_found) > 1:
            context.user_data['search_results'] = movies_found
            context.user_data['search_query'] = query_text
            try:
                await status_msg.delete()
            except:
                pass
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"🎬 <b>Multiple results found for</b> "
                    f"<b>{query_text}</b>\n\n"
                    "👇 Select the correct title and year:"
                ),
                reply_markup=create_movie_selection_keyboard(movies_found, page=0),
                parse_mode="HTML",
            )
            return

        movie_id, title, url, file_id = movies_found[0][:4]
        
        # Loading msg delete karein
        try: await status_msg.delete() 
        except: pass

        # Send the movie using your existing helper function
        await send_movie_to_user(update, context, movie_id, title, url, file_id)

    except Exception:
        logger.exception("Background search failed")
        await remove_search_progress(status_msg)
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text="⚠️ Search is temporarily unavailable. Please try again.",
            )
        except Exception:
            logger.exception("Could not send background-search error response")

# ==================== CLEAN LOADING FUNCTION (FIXED) ====================
async def deliver_movie_on_start(update: Update, context: ContextTypes.DEFAULT_TYPE, movie_id: int):
    """
    Fetches and sends a movie with a clean 'Loading' animation.
    No technical details shown to the user.
    """
    chat_id = update.effective_chat.id
    
    # 1. Loading Effect
    status_msg = None
    try:
        status_msg = await context.bot.send_message(chat_id, "⏳ <b>Please wait...</b>", parse_mode='HTML')
        
        # Backup Auto-delete
        track_message_for_deletion(
            context, chat_id, status_msg.message_id, USER_TEXT_DELETE_SECONDS
        )
    except:
        pass

    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            # User ko technical error mat dikhao, bas chupchap delete kar do
            if status_msg: 
                try: 
                    await status_msg.delete() 
                except: 
                    pass
            return

        cur = conn.cursor()
        cur.execute("SELECT title, url, file_id FROM movies WHERE id = %s", (movie_id,))
        movie_data = cur.fetchone()
        cur.close()
        close_db_connection(conn)

        # 2. Movie milne ke baad turant Loading Msg delete karo
        if status_msg:
            try: 
                await status_msg.delete()
            except: 
                pass

        if movie_data:
            title, url, file_id = movie_data
            # Movie bhejo
            await send_movie_to_user(update, context, movie_id, title, url, file_id)
        else:
            # Agar movie nahi mili
            fail_msg = await context.bot.send_message(chat_id, "❌ <b>Movie not found or deleted.</b>", parse_mode='HTML')
            track_message_for_deletion(context, chat_id, fail_msg.message_id, USER_TEXT_DELETE_SECONDS)

    except Exception as e:
        logger.error(f"Error in deliver_movie: {e}")
        if status_msg:
            try: 
                await status_msg.delete()
            except: 
                pass
        if movie_data:
            title, url, file_id = movie_data
            await send_movie_to_user(update, context, movie_id, title, url, file_id)
        else:
            await context.bot.send_message(
                chat_id=chat_id, 
                text="❌ Movie not found. It may have been removed from our database."
            )

    except Exception as e:
        logger.error(f"CRITICAL ERROR in deliver_movie: {e}", exc_info=True)
        error_msg = "❌ Failed to retrieve movie. Please try again or use search."
        if status_msg:
            try:
                await status_msg.edit_text(error_msg)
            except:
                pass
        else:
            await context.bot.send_message(chat_id=chat_id, text=error_msg)
            
    finally:
        if conn:
            try:
                close_db_connection(conn)
            except:
                pass

# Add this at the top level
from asyncio import Lock
from collections import defaultdict

user_processing_locks = defaultdict(Lock)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    # User registration is bookkeeping and must not delay deep-link delivery
    # or the welcome response. Keep failures visible in the background task.
    async def record_user():
        try:
            await run_async(record_telegram_user, update.effective_user, chat_id)
        except Exception as e:
            logger.warning(f"Telegram profile setup failed for {user_id}: {e}")

    task = asyncio.create_task(record_user())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    
    # ✅ FIX 1: Message ko safe tarike se nikalein (Button aur Text dono ke liye)
    message = update.effective_message 

    is_group_chat = update.effective_chat.type in ("group", "supergroup")

    # Membership checks are meaningful for private users only. In groups,
    # Telegram routes /start@<this bot username> to this same handler
    # automatically; no bot username is hardcoded here.
    if not is_group_chat:
        force_check = True if context.args else False
        check = await is_user_member(context, user_id, force_fresh=force_check)

        if not check['is_member']:
            # Agar deep link (args) hain to unhe save kar lo
            if context.args:
                context.user_data['pending_start_args'] = context.args

            msg = await context.bot.send_message(
                chat_id=chat_id,
                text=get_join_message(check['channel'], check['group']),
                reply_markup=get_join_keyboard(),
                parse_mode='Markdown'
            )
            track_message_for_deletion(context, chat_id, msg.message_id, USER_TEXT_DELETE_SECONDS)
            return
    # ==================

    logger.info(f"START called by user {user_id} with args: {context.args}")

    # Purani states clear karein
    context.user_data.clear()
    if hasattr(context, 'conversation') and context.conversation:
        context.conversation = None

    # === DEEP LINK PROCESSING ===
    if context.args and len(context.args) > 0:
        payload = context.args[0]
        
        # Check lock (taaki user spam na kare)
        if user_processing_locks[user_id].locked():
            await context.bot.send_message(
                chat_id=chat_id, 
                text="⏳ Please wait! Your previous request is still processing..."
            )
            return

        async with user_processing_locks[user_id]:
            
            # 🔐 NAYA: ANTI-BOT TEMPORARY LINK SYSTEM (BURN ON READ)
            if payload.startswith("tmp_"):
                if not re.fullmatch(r"tmp_[0-9a-f]{12}", payload):
                    await context.bot.send_message(chat_id, "❌ Link Expired ya Invalid hai!")
                    return
                conn = get_db_connection()
                if not conn:
                    await context.bot.send_message(chat_id, "❌ System Error.")
                    return

                try:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT movie_id, movie_file_id, created_at
                        FROM temp_links
                        WHERE token = %s
                          AND created_at >= NOW() - INTERVAL '1 minute'
                    """, (payload,))
                    res = cur.fetchone()
                    
                    # Token TURANT delete kar do (Single Use)
                    cur.execute("DELETE FROM temp_links WHERE token = %s", (payload,))
                    conn.commit()
                    cur.close()
                    
                    if not res:
                        msg = await context.bot.send_message(chat_id, "❌ <b>Link Expired ya Invalid hai!</b>\nKripya app par jaakar dobara click karein.", parse_mode='HTML')
                        track_message_for_deletion(context, chat_id, msg.message_id, USER_TEXT_DELETE_SECONDS)
                        return
                    
                    movie_id, movie_file_id, created_at = res
                    
                    # A Mini App quality click includes a concrete movie_files
                    # record, so send only that file rather than the whole list.
                    if movie_file_id:
                        cur = conn.cursor()
                        cur.execute("""
                            SELECT m.title, mf.url, mf.file_id
                            FROM movie_files mf
                            JOIN movies m ON m.id = mf.movie_id
                            WHERE mf.id = %s AND mf.movie_id = %s
                        """, (movie_file_id, movie_id))
                        selected_file = cur.fetchone()
                        cur.close()
                        if not selected_file:
                            await context.bot.send_message(chat_id, "❌ Selected file is no longer available.")
                            return
                        title, url, file_id = selected_file
                        await send_movie_to_user(
                            update, context, movie_id, title, url, file_id,
                            send_warning=True, require_exact_file=True
                        )
                    else:
                        # Backward compatibility for previously issued links.
                        await deliver_movie_on_start(update, context, movie_id)
                    logger.info(f"✅ Secure token {payload} used successfully for movie {movie_id}")
                    return

                except Exception as e:
                    logger.error(f"Temp Link Error: {e}")
                    await context.bot.send_message(chat_id, "❌ Processing error.")
                    return
                finally:
                    close_db_connection(conn)

                    
            # --- SEND ALL FROM GROUP: START PM THEN DELIVER THE REQUESTED PAGE ---
            if payload.startswith("sendall_"):
                try:
                    parts = payload.split('_')
                    movie_id = int(parts[1])
                    page = int(parts[2]) if len(parts) > 2 else 1
                    await deliver_movie_page_on_start(update, context, movie_id, page)
                except (TypeError, ValueError, IndexError):
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="❌ Invalid file request. Please search again."
                    )
                except Exception as e:
                    logger.error(f"Send All deep-link error: {e}", exc_info=True)
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="❌ Files could not be sent. Please try again."
                    )
                return

            # --- CASE NAYA: DIRECT FILE CLICK FROM TEXT LINK ---
            if payload.startswith("file_"):
                try:
                    parts = payload.split('_')
                    movie_id = int(parts[1])
                    file_index = int(parts[2])
                    
                    # ✅ STICKER bhejo "Fetching file" text ki jagah
                    status_msg = await safe_send(context.bot.copy_message(
                        chat_id=chat_id,
                        from_chat_id=-1003893346701,
                        message_id=8675
                    ))
                    if status_msg:
                        track_user_message_for_deletion(
                            context, chat_id, status_msg, is_file=True
                        )
                    
                    # File ka data nikalo
                    qualities = get_all_movie_qualities(movie_id)
                    if qualities and len(qualities) > file_index:
                        file_data = qualities[file_index]
                        url = file_data[1]
                        file_id = file_data[2]
                        
                        # Movie ka naam nikalo
                        conn = get_db_connection()
                        cur = conn.cursor()
                        cur.execute("SELECT title FROM movies WHERE id = %s", (movie_id,))
                        res = cur.fetchone()
                        cur.close()
                        close_db_connection(conn)
                        title = res[0] if res else "Requested File"
                        
                        # ✅ FIX: Sticker PEHLE delete karo, phir file bhejo
                        # Taaki user ko lage: "file mil gayi, ab aa rahi hai"
                        try:
                            if status_msg:
                                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
                        except Exception as e:
                            logger.error(f"Failed to delete status message: {e}")
                        
                        # Tera premium thumbnail wala function!
                        await send_movie_to_user(update, context, movie_id, title, url, file_id, send_warning=True)  # Single file → ek GIF
                    else:
                        try:
                            if status_msg:
                                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
                        except Exception:
                            pass
                        await context.bot.send_message(chat_id=chat_id, text="❌ File not found or expired.")
                    return
                except Exception as e:
                    logger.error(f"File click error: {e}")
                    await context.bot.send_message(chat_id=chat_id, text="❌ Invalid File Link")
                    return
                    
            # --- CASE 1: DIRECT MOVIE ID (movie_123) ---
            if payload.startswith("movie_"):
                try:
                    movie_id = int(payload.split('_')[1])
                    
                    # ✅ FIX 3: send_message use karein
                    status_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"🎬 Deep link detected!\nMovie ID: {movie_id}\nFetching... Please wait ⏳"
                    )
                    
                    try:
                        await deliver_movie_on_start(update, context, movie_id)
                        
                        # Success hone par status msg delete karein
                        try: await status_msg.delete() 
                        except: pass
                        
                        logger.info(f"✅ Deep link SUCCESS for user {user_id}, movie {movie_id}")
                        
                    except Exception as e:
                        logger.error(f"❌ Deep link FAILED: {e}")
                        await status_msg.edit_text(f"❌ Error fetching movie: {e}")
                    
                    return # Movie mil gayi, Welcome msg mat dikhao

                except Exception as e:
                    logger.error(f"Invalid movie link: {e}")
                    await context.bot.send_message(chat_id=chat_id, text="❌ Invalid Link Format")
                    return

            # --- CASE 2: AUTO SEARCH (q_kalki) ---
            # ✅ RESTORED: Ye logic maine wapas add kar di hai
            elif payload.startswith("q_"):
                try:
                    query_text = payload[2:].replace("_", " ").strip()
                    
                    # ✅ FIX 4: send_message use karein
                    status_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"🔎 Deep link search detected!\nQuery: '{query_text}'\nSearching... Please wait ⏳"
                    )
                    
                    try:
                        # Background search function call karein
                        await background_search_and_send(update, context, query_text, status_msg)
                        logger.info(f"✅ Deep link SEARCH SUCCESS for user {user_id}, query: {query_text}")
                        
                    except Exception as e:
                        logger.error(f"❌ Deep link SEARCH FAILED: {e}")
                        error_text = f"❌ Search failed for '{query_text}'.\nTry searching manually."
                        try: await status_msg.edit_text(error_text)
                        except: await context.bot.send_message(chat_id=chat_id, text=error_text)
                    
                    return # Search ho gaya, Welcome msg mat dikhao
                    
                except Exception as e:
                    logger.error(f"Deep link search error: {e}")
                    await context.bot.send_message(chat_id=chat_id, text="❌ Error processing search link.")
                    return

    # Group commands should not post the private welcome GIF/menu or run
    # private-user membership UX. Give the group a concise usage response.
    if is_group_chat:
        bot_info = await context.bot.get_me()
        bot_username = html_escape(bot_info.username or "this bot")
        group_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🎬 Open FlimfyBox", url=WEB_APP_URL)]
        ])
        msg = await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"👋 <b>{html_escape(bot_info.first_name or 'FlimfyBox')}</b> group me active hai.\n\n"
                "Movie ya series ka naam isi group me bhejo, "
                "main matching files dikha dunga.\n\n"
                f"Bot ko direct start karne ke liye: <code>/start@{bot_username}</code>"
            ),
            reply_markup=group_keyboard,
            parse_mode='HTML'
        )
        track_user_message_for_deletion(context, chat_id, msg)
        return

    # --- NORMAL WELCOME MESSAGE (WITH GIF & DYNAMIC GREETING) ---
    user = update.effective_user
    user_name = user.first_name
    user_id_val = user.id
    user_uname = user.username  # Telegram username
    
    # 🌟 Mention banao: Clickable Name (Direct Profile Link without Web Preview)
    user_display = f"<a href='tg://user?id={user_id_val}'>{user_name}</a>"
    
    # 🌟 NAYA: Bot ka actual naam aur username nikalo
    bot_info = await context.bot.get_me()
    bot_name = bot_info.first_name  # Ye har bot ka apna alag naam uthayega!
    
    # 1. Dynamic Greeting Logic
    try:
        import pytz
        tz = pytz.timezone('Asia/Kolkata')
        hour = datetime.now(tz).hour
    except ImportError:
        hour = datetime.now().hour # Fallback agar pytz na ho
        
    if 5 <= hour < 12: greeting = "Good Morning ☀️"
    elif 12 <= hour < 17: greeting = "Good Afternoon 🌤️"
    elif 17 <= hour < 21: greeting = "Good Evening 🌆"
    else: greeting = "Good Night 🌙"

    # 2. Premium Caption (Dynamic Bot Name ke sath)
    caption_text = (
        f"<b>━━━━━━━ 🚩 𝐉𝐀𝐈 𝐒𝐇𝐑𝐈 𝐑𝐀𝐌 🚩 ━━━━━━━</b>\n\n"
        f"✦ {greeting}, {user_display}!\n\n"
        f"╭─── ❖ 𝗔𝗕𝗢𝗨𝗧 𝗠𝗘 ❖ ───╮\n"
        f"│\n"
        f"│  🤖 Main hoon <b>{bot_name}</b>\n"
        f"│  𝗧𝗵𝗲 𝗠𝗼𝘀𝘁 𝗣𝗼𝘄𝗲𝗿𝗳𝘂𝗹 𝗔𝘂𝘁𝗼 𝗙𝗶𝗹𝘁𝗲𝗿 𝗕𝗼𝘁\n"
        f"│\n"
        f"╰──────────────────╯\n\n"
        f"<b>⟐ 𝗠𝘆 𝗣𝗿𝗲𝗺𝗶𝘂𝗺 𝗙𝗲𝗮𝘁𝘂𝗿𝗲𝘀:</b>\n"
        f"  ◈ ⚡ 𝗟𝗶𝗴𝗵𝘁𝗻𝗶𝗻𝗴-𝗳𝗮𝘀𝘁 Auto Filtering\n"
        f"  ◈ 🛡️ 𝟮𝟰/𝟳 Premium Uptime\n"
        f"  ◈ 🎬 HD/4K File Processing\n"
        f"  ◈ 🔍 𝗦𝗺𝗮𝗿𝘁 𝗦𝗲𝗮𝗿𝗰𝗵 + AI Matching\n\n"
        f"<b>━━━━━━━━━━━━━━━━━━━━━</b>\n"
        f"👇 <b>𝗧𝗮𝗽 𝘁𝗵𝗲 𝗯𝘂𝘁𝘁𝗼𝗻𝘀 𝗯𝗲𝗹𝗼𝘄 𝘁𝗼 𝗲𝘅𝗽𝗹𝗼𝗿𝗲!</b> 👇"
    )

    # 3. Inline Buttons
    inline_buttons = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔰 ADD ME TO YOUR GROUP 🔰", url=f"https://t.me/{bot_info.username}?startgroup=true")],
        [InlineKeyboardButton("HELP 📢", callback_data="start_help"), InlineKeyboardButton("ABOUT 📖", callback_data="start_about")],
        [InlineKeyboardButton("DONATION 💰", callback_data="start_donate")]
    ])

    try:
        # Web App button set karna
        web_app_url = WEB_APP_URL
        await context.bot.set_chat_menu_button(
            chat_id=chat_id,
            menu_button=MenuButtonWebApp(text="🎬 Web Version", web_app=WebAppInfo(url=web_app_url))
        )
        
        # Bottom Keyboard ('Search', 'Request') lane ke liye ek chhota silent message
        # Bottom keyboard bhej kar turant delete kar do (chat clean rahegi)
        menu_msg = await context.bot.send_message(chat_id=chat_id, text="🔄 Loading Menu...", reply_markup=get_main_keyboard())
        try:
            await menu_msg.delete()
        except: 
            pass

        # This exact channel message is the approved Start GIF source.
        msg = await context.bot.copy_message(
            chat_id=chat_id,
            from_chat_id=START_GIF_CHANNEL_ID,
            message_id=START_GIF_MESSAGE_ID,
            caption=caption_text,
            parse_mode='HTML',
            reply_markup=inline_buttons
        )
        track_message_for_deletion(context, chat_id, msg.message_id, delay=300)
        
    except Exception as e:
        logger.error(f"Start Menu Error: {e}")
        # If the approved GIF cannot be copied, keep the menu text-only.
        msg = await context.bot.send_message(chat_id=chat_id, text=caption_text, parse_mode='HTML', reply_markup=inline_buttons)
        track_message_for_deletion(context, chat_id, msg.message_id, delay=300)
        
    return
async def main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle main menu options"""
    try:
        query = update.message.text

        if query == '🔍 Search Movies':
            msg = await update.message.reply_text("Great! Tell me the name of the movie you want to search for.")
            track_user_message_for_deletion(context, update.effective_chat.id, msg)
            return SEARCHING

        elif query == '🙋 Request Movie':
            msg = await update.message.reply_text("Okay, you've chosen to request a new movie. Please tell me the name of the movie you want me to add.")
            track_message_for_deletion(context, update.effective_chat.id, msg.message_id, USER_TEXT_DELETE_SECONDS)
            return REQUESTING

        elif query == '📊 My Stats':
            user_id = update.effective_user.id
            conn = None
            try:
                conn = get_db_connection()
                if conn:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM user_requests WHERE user_id = %s", (user_id,))
                    request_count = cur.fetchone()

                    cur.execute("SELECT COUNT(*) FROM user_requests WHERE user_id = %s AND notified = TRUE", (user_id,))
                    fulfilled_count = cur.fetchone()

                    stats_text = f"""
📊 Your Stats:
- Total Requests: {request_count}
- Fulfilled Requests: {fulfilled_count}
"""
                    msg = await update.message.reply_text(stats_text)
                    track_message_for_deletion(context, update.effective_chat.id, msg.message_id, USER_TEXT_DELETE_SECONDS)
                else:
                    await update.message.reply_text("Sorry, database connection failed.")
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
                await update.message.reply_text("Sorry, couldn't retrieve your stats at the moment.")
            finally:
                if conn: close_db_connection(conn)

            return MAIN_MENU

        elif query == '❓ Help':
            help_text = """
🤖 How to use FlimfyBox Bot:

🔍 Search Movies: Find movies in our collection
🙋 Request Movie: Request a new movie to be added
📊 My Stats: View your request statistics

Just use the buttons below to navigate!
            """
            msg = await update.message.reply_text(help_text)
            track_message_for_deletion(context, update.effective_chat.id, msg.message_id, USER_TEXT_DELETE_SECONDS)
            return MAIN_MENU
        else:
            return await search_movies(update, context)

    except Exception as e:
        logger.error(f"Error in main menu: {e}")
        return MAIN_MENU

def _format_requested_files_header(title, qualities, user, bot_info):
    """Build the compact branded header used above every file list."""
    languages = []
    for file_data in qualities or []:
        raw_language = str(file_data[4]).strip() if len(file_data) > 4 and file_data[4] else ""
        for language in re.split(r"[,/|]+", raw_language):
            language = re.sub(r"\s+", " ", language).strip()
            if language and language.casefold() not in {item.casefold() for item in languages}:
                languages.append(language)
    language_label = ", ".join(languages)
    requester = (
        getattr(user, "first_name", None)
        or getattr(user, "username", None)
        or "User"
    )
    requester_name = html_escape(str(requester).lstrip("@"))
    requester_id = getattr(user, "id", None)
    requester = (
        f"<a href='tg://user?id={requester_id}'>{requester_name}</a>"
        if requester_id
        else requester_name
    )
    display_title = html_escape(str(title or "Requested Movie").strip().title())
    language_label = html_escape(language_label)
    language_line = (
        f"<b>🧱 𝙻𝚊𝚗𝚐𝚞𝚊ɢᴇ </b><code>{language_label}</code>\n\n"
        if language_label
        else ""
    )
    bot_name = html_escape(
        str(getattr(bot_info, "first_name", None) or getattr(bot_info, "username", None) or "FlimfyBox")
    )
    bot_id = getattr(bot_info, "id", None)
    bot_mention = (
        f"<a href='tg://user?id={bot_id}'>⚡️{bot_name}</a>"
        if bot_id
        else f"⚡️{bot_name}"
    )
    return (
        f"<b>🏷 ᴛɪᴛʟᴇ : </b><code>{display_title}</code>\n"
        f"{language_line}"
        f"📝 ʀᴇǫᴜᴇsᴛᴇᴅ ʙʏ : {requester}\n"
        f"⚜️ ᴘᴏᴡᴇʀᴇᴅ ʙʏ : {bot_mention} 🔍\n\n"
        "Your Requested Files Are Here\n\n"
    )
    return (
        f"<b>🏷 ᴛɪᴛʟᴇ : </b><code>{display_title}</code>\n"
        f"🧱 𝙻𝚊𝚗𝚐𝚞𝚊𝚐𝚎 : <b><code>{language_label}</code></b>\n"
        f"📝 ʀᴇǫᴜᴇsᴛᴇᴅ ʙʏ : {requester}\n"
        f"⚜️ ᴘᴏᴡᴇʀᴇᴅ ʙʏ : {bot_mention} 🔍\n\n"
        "Your Requested Files Are Here\n\n"
    )


def _format_file_link_label(file_size, title, extra_info, quality, language=""):
    """Build consistent, pipe-separated file labels for Telegram links."""
    parts = [
        str(file_size or "Unknown Size").strip(),
        str(title or "Requested Movie").strip(),
    ]
    if language and str(language).strip():
        parts.append(str(language).strip())

    metadata = " ".join(
        value for value in (str(extra_info or ""), str(quality or ""))
        if value.strip()
    )
    episode_pattern = re.compile(
        r"(?i)\b(?:S\d{1,2}E\d{1,3}|S\d{1,2}|E\d{1,3}|EP\d{1,3}|"
        r"Season\s*\d+|Episode\s*\d+)\b"
    )
    episodes = []
    for match in episode_pattern.finditer(metadata):
        token = match.group(0).upper()
        combined = re.fullmatch(r"S(\d{1,2})E(\d{1,3})", token)
        if combined:
            tokens = (f"S{int(combined.group(1)):02}", f"E{int(combined.group(2)):02}")
        elif token.startswith("SEASON"):
            season = re.search(r"\d+", token)
            tokens = (f"S{int(season.group()):02}",) if season else ()
        elif token.startswith("EPISODE") or token.startswith("EP"):
            episode = re.search(r"\d+", token)
            tokens = (f"E{int(episode.group()):02}",) if episode else ()
        else:
            number = re.search(r"\d+", token)
            prefix = "S" if token.startswith("S") else "E"
            tokens = (f"{prefix}{int(number.group()):02}",) if number else ()
        for item in tokens:
            if item not in episodes:
                episodes.append(item)

    parts.extend(episodes)
    quality_text = episode_pattern.sub("", str(quality or ""))
    quality_text = re.sub(r"\s+", " ", quality_text).strip()
    resolution = re.search(
        r"(?i)\b(4K|2160p|1440p|1080p|720p|576p|480p|360p)\b", quality_text
    )
    if resolution:
        parts.append(resolution.group(1))
        source = quality_text[resolution.end():].strip(" -_|")
        if source:
            parts.append(source)
    elif quality_text:
        parts.append(quality_text)

    return " | ".join(part for part in parts if part)


async def process_movie_exact_match(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    movie_id: int,
    title: str,
    timing: Optional[dict] = None,
    status_message=None,
):
    delivery_start = time.perf_counter() if SEARCH_TIMING_ENABLED else None
    if status_message:
        try:
            await status_message.edit_text(
                f"✅ Found <b>{html_escape(title)}</b> — loading available files…",
                parse_mode="HTML",
            )
        except Exception:
            logger.debug("Could not update search progress before file lookup")
    qualities, (category, poster_url) = await run_async(
        get_movie_delivery_data, movie_id
    )
    if timing is not None and delivery_start is not None:
        timing['delivery_data_ms'] = int((time.perf_counter() - delivery_start) * 1000)
    if not qualities:
        no_files_text = f"❌ No files found for <b>{html_escape(title)}</b>."
        if status_message:
            try:
                response_start = time.perf_counter()
                await status_message.edit_text(no_files_text, parse_mode="HTML")
                if SEARCH_TIMING_ENABLED:
                    logger.info(
                        "Search timing stage=result_response movie_id=%s "
                        "telegram_ms=%.2f result=no_files",
                        movie_id,
                        (time.perf_counter() - response_start) * 1000,
                    )
            except Exception:
                logger.exception("Could not update empty search result state")
                await update.message.reply_text(no_files_text, parse_mode="HTML")
        else:
            await update.message.reply_text(no_files_text, parse_mode="HTML")
        return

    format_start = time.perf_counter()
    context.user_data['selected_movie_data'] = {
        'id': movie_id,
        'title': title,
        'category': category,
        'qualities': qualities
    }

    bot_info = context.bot
    get_me_ms = 0.0
    if not bot_info.username:
        get_me_start = time.perf_counter()
        bot_info = await context.bot.get_me()
        get_me_ms = (time.perf_counter() - get_me_start) * 1000
    bot_username = bot_info.username
    file_list_text = _format_requested_files_header(
        title,
        qualities,
        update.effective_user,
        bot_info,
    )
    
    for idx, file_data in enumerate(qualities[:10], start=1):
        quality = file_data[0]
        file_size = file_data[3] if len(file_data) > 3 else "Unknown Size"
        extra_info = file_data[5] if len(file_data) > 5 else ""
        real_idx = qualities.index(file_data)
        label = html_escape(
            _format_file_link_label(file_size, title, extra_info, quality)
        )
        file_list_text += f"<b>{idx}.</b> <b><a href='https://t.me/{bot_username}?start=file_{movie_id}_{real_idx}'>{label}</a></b>\n\n"

    limit = 10
    total_pages = (len(qualities) + limit - 1) // limit if qualities else 1
    context.user_data['active_filter'] = None
    
    current_files = qualities[:limit]
    keyboard_markup = create_quality_selection_keyboard(
        movie_id=movie_id, 
        view="main", 
        page=1, 
        total_pages=total_pages, 
        current_files=current_files
    )
    format_ms = (time.perf_counter() - format_start) * 1000 - get_me_ms
    if SEARCH_TIMING_ENABLED:
        logger.info(
            "Search timing stage=result_format movie_id=%s format_ms=%.2f "
            "telegram_get_me_ms=%.2f file_count=%d",
            movie_id,
            max(0.0, format_ms),
            get_me_ms,
            len(qualities),
        )
    
    response_start = time.perf_counter() if SEARCH_TIMING_ENABLED else None
    if status_message:
        try:
            msg = await status_message.edit_text(
                file_list_text,
                reply_markup=keyboard_markup,
                parse_mode='HTML',
                disable_web_page_preview=True,
            )
        except Exception:
            logger.exception("Could not replace search progress with movie files")
            msg = await update.message.reply_text(
                file_list_text,
                reply_markup=keyboard_markup,
                parse_mode='HTML',
                disable_web_page_preview=True,
            )
    else:
        msg = await update.message.reply_text(
            file_list_text,
            reply_markup=keyboard_markup,
            parse_mode='HTML',
            disable_web_page_preview=True
        )
    if timing is not None and response_start is not None:
        timing['response_ms'] = int((time.perf_counter() - response_start) * 1000)
    if SEARCH_TIMING_ENABLED and response_start is not None:
        logger.info(
            "Search timing stage=result_response movie_id=%s telegram_edit_ms=%.2f",
            movie_id,
            (time.perf_counter() - response_start) * 1000,
        )
    track_message_for_deletion(
        context, update.effective_chat.id, msg.message_id, USER_TEXT_DELETE_SECONDS
    )


async def send_search_progress(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str = None):
    """Send a lightweight temporary search progress message that can be edited/deleted.
       Returns the sent Message object or None on failure. Controlled by SEARCH_TIMING_ENABLED.
    """
    try:
        if not update.message:
            return None
        display = (query or '').strip()
        if len(display) > 120:
            display = display[:117] + '...'
        if display:
            text = f'🔎 Searching for "{display}"...'
        else:
            text = '🔎 Searching...'
        msg = await update.message.reply_text(text)
        return msg
    except Exception as e:
        logger.debug(f"send_search_progress failed: {e}")
        return None


async def _update_search_progress(progress_message, source_message, text, **kwargs):
    """Replace a temporary search status, with a reply fallback if editing fails."""
    if progress_message:
        try:
            return await progress_message.edit_text(text, **kwargs)
        except Exception:
            logger.info("Could not edit search progress; sending result as a reply")
            try:
                await progress_message.delete()
            except Exception:
                logger.debug("Could not remove stale search progress")
    return await source_message.reply_text(text, **kwargs)


async def remove_search_progress(progress_message):
    if not progress_message:
        return
    try:
        await progress_message.delete()
    except Exception as exc:
        logger.debug(f"Search progress message could not be deleted: {exc}")


def _request_portal_url(title=""):
    """Return a usable HTTPS Mini App URL, falling back if configuration is invalid."""
    fallback = "https://flimfybox-bot-yht0.onrender.com/webapp"
    try:
        parsed = urlparse(WEB_APP_URL or "")
        is_configured_portal = (
            parsed.scheme.casefold() == "https"
            and bool(parsed.netloc)
            and parsed.path.rstrip("/") == "/webapp"
        )
    except ValueError:
        is_configured_portal = False
        parsed = urlparse(fallback)
    if not is_configured_portal:
        parsed = urlparse(fallback)
    base_url = urlunparse(parsed._replace(query="", fragment=""))
    query_string = urlencode({"req": str(title or "")[:200]}) if title else ""
    return "{}?{}".format(base_url, query_string) if query_string else base_url


def _not_found_keyboard(query_text, suggestions=()):
    """Build only buttons with valid callback sizes and usable HTTPS destinations."""
    rows = []
    for title in suggestions:
        encoded_title = quote(str(title), safe="")
        callback_data = "retrysearch_" + encoded_title
        if len(callback_data.encode("utf-8")) <= 64:
            display_title = str(title)
            if len(display_title) > 56:
                display_title = display_title[:53] + "..."
            rows.append([
                InlineKeyboardButton(
                    "🔎 Search: {}".format(display_title),
                    callback_data=callback_data,
                )
            ])

    encoded_request = quote(str(query_text or ""), safe="")
    request_callback = "request_prefill_" + encoded_request
    if query_text and len(request_callback.encode("utf-8")) <= 64:
        rows.append([
            InlineKeyboardButton(
                "🙋 Request this title",
                callback_data=request_callback,
            )
        ])

    portal_url = _request_portal_url(query_text)
    if is_valid_url(portal_url) and urlparse(portal_url).scheme.casefold() == "https":
        rows.append([
            InlineKeyboardButton("🌐 Open Request Portal", url=portal_url)
        ])

    if (
        is_valid_url(UPDATE_CHANNEL_URL)
        and urlparse(UPDATE_CHANNEL_URL).scheme.casefold() == "https"
    ):
        rows.append([
            InlineKeyboardButton(
                "📢 Update Channel: Join BackUp",
                url=UPDATE_CHANNEL_URL,
            )
        ])
    return InlineKeyboardMarkup(rows) if rows else None


async def _replace_search_result_message(
    query,
    context,
    text,
    reply_markup=None,
    parse_mode="HTML",
    disable_web_page_preview=True,
):
    """Edit text/caption when supported; otherwise send a replacement message."""
    message = getattr(query, "message", None)
    if message:
        has_caption = any(
            getattr(message, attribute, None)
            for attribute in ("animation", "photo", "video", "document", "audio", "voice")
        )
        try:
            if has_caption:
                return await query.edit_message_caption(
                    caption=text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode,
                )
            return await query.edit_message_text(
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
            )
        except Exception as exc:
            logger.info("Could not edit search result message; sending a replacement: %s", exc)

    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", None) or getattr(query, "chat_id", None)
    if chat_id is None:
        raise ValueError("Search result callback has no target chat")
    return await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=reply_markup,
        parse_mode=parse_mode,
        disable_web_page_preview=disable_web_page_preview,
    )


async def search_movies(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Search for movies in the database"""
    handler_entry = time.perf_counter()
    progress_message = None
    try:
        # Agar ye button click se aya hai (cancel/back)
        if update.callback_query:
            query = update.callback_query
            await query.answer()
            # Yahan hum kuch return nahi kar rahe, bas message bhej rahe hain
            return

        # Agar message text nahi hai
        if not update.message or not update.message.text:
            return 

        query = update.message.text.strip()
        
        # Safety check
        if query in ['🔍 Search Movies', '📊 My Stats', '❓ Help']:
             return await main_menu_or_search(update, context)

        # 👇 NAYA FIX: Search query se Season/Episode tags hata do taaki main show mil jaye 👇
        import re
        clean_query = re.sub(r'(?i)\b(s\d{1,2}|season\s*\d+|ep\s?\d+|e\d{1,2})\b.*', '', query).strip()
        search_term = clean_query if (clean_query and len(clean_query) > 1) else query

        # Lightweight timing instrumentation for debug (disabled by default)
        times = {}
        if SEARCH_TIMING_ENABLED:
            times['handler_start'] = handler_entry
            logger.info(
                "Search timing stage=handler_entry query=%r",
                search_term[:80],
            )

        progress_message = context.user_data.pop('_search_progress_message', None)
        if progress_message is None:
            progress_message = await send_search_progress(update, context, search_term)
        if SEARCH_TIMING_ENABLED:
            times['progress_sent'] = time.perf_counter()

        schedule_recommendation_event(
            user_id=update.effective_user.id if update.effective_user else None,
            event_type='pm_search',
            source='pm',
            metadata={'query': search_term[:200]},
        )

        # 1. Use the indexed similarity query for the common path. The legacy
        # alias/fuzzy search remains a fallback for titles not covered by SQL.
        if SEARCH_TIMING_ENABLED:
            times['before_first_db'] = time.perf_counter()
        movies = await run_async(get_movies_fast_sql, search_term, limit=10)
        if SEARCH_TIMING_ENABLED:
            times['first_db'] = time.perf_counter()
        
        # 2. Not Found
        if not movies:
            # Google runs on the server (not through a WebView JSONP callback),
            # so a spelling such as "rechar" can be retried from Telegram too.
            if SEARCH_TIMING_ENABLED:
                times['google_start'] = time.perf_counter()
            suggestions = await get_google_title_suggestions_with_timeout(
                search_term, limit=3
            )
            if SEARCH_TIMING_ENABLED:
                times['google_end'] = time.perf_counter()

            result_format_start = time.perf_counter()
            keyboard = _not_found_keyboard(query, suggestions)
            not_found_text = (
                "<b>❌ No matching title found</b>\n\n"
                "Check the spelling or add the release year. "
                "You can retry a suggestion or request this title below."
            )
            if SEARCH_TIMING_ENABLED:
                times['result_format_ms'] = (
                    time.perf_counter() - result_format_start
                ) * 1000

            response_start = time.perf_counter() if SEARCH_TIMING_ENABLED else None
            msg = None
            if SEARCH_ERROR_GIFS:
                try:
                    gif = random.choice(SEARCH_ERROR_GIFS)
                    msg = await update.message.reply_animation(
                        animation=gif,
                        caption=not_found_text,
                        reply_markup=keyboard,
                        parse_mode='HTML'
                    )
                except Exception as exc:
                    logger.warning(f"Search failure animation could not be sent: {exc}")

            if msg is None:
                msg = await update.message.reply_text(
                    text=not_found_text,
                    reply_markup=keyboard,
                    parse_mode='HTML',
                    disable_web_page_preview=True
                )
            # Auto Delete Not Found Msg
            track_user_message_for_deletion(context, update.effective_chat.id, msg)
            if SEARCH_TIMING_ENABLED and response_start is not None:
                times['response_ms'] = int(
                    (time.perf_counter() - response_start) * 1000
                )
            return # <--- YAHAN SE MAIN_MENU HATA DIYA HAI

        # 3. Found
        selection_start = time.perf_counter()
        chosen_movie = _select_single_search_result(query, movies)
        if SEARCH_TIMING_ENABLED:
            times['result_format_ms'] = (
                time.perf_counter() - selection_start
            ) * 1000
        
        if chosen_movie:
            movie_id, title, url, file_id = chosen_movie[:4]
            schedule_recommendation_event(
                user_id=update.effective_user.id if update.effective_user else None,
                event_type='pm_exact_match',
                source='pm',
                movie_id=movie_id,
                metadata={'query': search_term[:200], 'title': title},
            )
            # Exact match par qualities menu dikhao
            if SEARCH_TIMING_ENABLED:
                times['exact_match_decision'] = time.perf_counter()
            await process_movie_exact_match(
                update,
                context,
                movie_id,
                title,
                timing=times,
                status_message=progress_message,
            )
            progress_message = None
            return

        result_format_start = time.perf_counter()
        context.user_data['search_results'] = movies
        context.user_data['search_query'] = query

        keyboard = create_movie_selection_keyboard(movies, page=0)
        result_text = (
            f"<b>━━━━━━ 🎬 𝗦𝗲𝗮𝗿𝗰𝗵 𝗥𝗲𝘀𝘂𝗹𝘁𝘀 ━━━━━━</b>\n\n"
            f"✦ 𝗙𝗼𝘂𝗻𝗱 <b>{len(movies)}</b> results for '<b>{query}</b>'\n\n"
            f"👇 <b>𝗦𝗲𝗹𝗲𝗰𝘁 𝘆𝗼𝘂𝗿 𝗺𝗼𝘃𝗶𝗲 𝗯𝗲𝗹𝗼𝘄:</b>"
        )
        if SEARCH_TIMING_ENABLED:
            times['result_format_ms'] = (
                time.perf_counter() - result_format_start
            ) * 1000
            response_start = time.perf_counter()
        msg = await update.message.reply_text(
            result_text,
            reply_markup=keyboard,
            parse_mode='HTML'
        )
        if SEARCH_TIMING_ENABLED:
            times['response_ms'] = (
                time.perf_counter() - response_start
            ) * 1000
        
        track_message_for_deletion(context, update.effective_chat.id, msg.message_id, USER_TEXT_DELETE_SECONDS)
        return # <--- YAHAN SE BHI MAIN_MENU HATA DIYA HAI

    except Exception:
        await remove_search_progress(progress_message)
        logger.exception("Error in search_movies")
        if update.message:
            try:
                await update.message.reply_text(
                    "⚠️ Search is temporarily unavailable. Please try again."
                )
            except Exception:
                logger.exception("Could not send private-search error response")
        return
    finally:
        # The temporary loading GIF must never remain after the search finishes.
        await remove_search_progress(progress_message)
        # Optional: log per-stage timing when enabled
        try:
            if SEARCH_TIMING_ENABLED and 'times' in locals() and times.get('handler_start'):
                handler_start = times.get('handler_start')
                end = time.perf_counter()
                deltas = {}
                deltas['total_ms'] = int((end - handler_start) * 1000)
                if times.get('progress_sent'):
                    deltas['to_progress_ms'] = int((times.get('progress_sent') - handler_start) * 1000)
                if times.get('before_first_db') and times.get('first_db'):
                    deltas['first_db_ms'] = int((times.get('first_db') - times.get('before_first_db')) * 1000)
                if times.get('google_start') and times.get('google_end'):
                    deltas['fallback_ms'] = int((times.get('google_end') - times.get('google_start')) * 1000)
                if times.get('response_ms'):
                    deltas['response_ms'] = times['response_ms']
                if times.get('result_format_ms') is not None:
                    deltas['result_format_ms'] = round(
                        times['result_format_ms'], 2
                    )
                if times.get('google_start') and times.get('google_end'):
                    deltas['google_ms'] = int((times.get('google_end') - times.get('google_start')) * 1000)
                if times.get('exact_match_decision'):
                    deltas['exact_match_decision_ms'] = int(
                        (times.get('exact_match_decision') - handler_start) * 1000
                    )
                timing_data = {
                    key: value for key, value in times.items()
                    if key in (
                        'delivery_data_ms',
                        'poster_ms',
                        'telegram_response_ms',
                    )
                }
                deltas.update(timing_data)
                # Truncate query for privacy in logs
                q_display = (query[:80] + '...') if query and len(query) > 80 else (query or '')
                logger.info(f"Search timing for '{q_display}': {deltas}")
        except Exception as e:
            logger.debug(f"Failed to log search timing: {e}")

async def request_movie(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle movie requests with duplicate detection, fuzzy matching and cooldowns"""
    try:
        user_message = (update.message.text or "").strip()
        user = update.effective_user

        if not user_message:
            await update.message.reply_text("कृपया मूवी का नाम भेजें।")
            return REQUESTING

        burst = user_burst_count(user.id, window_seconds=60)
        if burst >= MAX_REQUESTS_PER_MINUTE:
            msg = await update.message.reply_text(
                "🛑 तुम बहुत जल्दी-जल्दी requests भेज रहे हो। कुछ देर रोकें (कुछ मिनट) और फिर कोशिश करें।\n"
                "बार‑बार भेजने से फ़ायदा नहीं होगा।"
            )
            track_message_for_deletion(context, update.effective_chat.id, msg.message_id, USER_TEXT_DELETE_SECONDS)
            return REQUESTING

        intent = await analyze_intent(user_message)
        if not intent["is_request"]:
            msg = await update.message.reply_text("यह एक मूवी/सीरीज़ का नाम नहीं लग रहा है। कृपया सही नाम भेजें।")
            track_message_for_deletion(context, update.effective_chat.id, msg.message_id, USER_TEXT_DELETE_SECONDS)
            return REQUESTING

        movie_title = intent["content_title"] or user_message

        similar = get_last_similar_request_for_user(user.id, movie_title, minutes_window=REQUEST_COOLDOWN_MINUTES)
        if similar:
            last_time = similar.get("requested_at")
            elapsed = datetime.now() - last_time
            minutes_passed = int(elapsed.total_seconds() / 60)
            minutes_left = max(0, REQUEST_COOLDOWN_MINUTES - minutes_passed)
            if minutes_left > 0:
                strict_text = (
                    "🛑 Ruk jao! Aapne ye request abhi bheji thi.\n\n"
                    "Baar‑baar request karne se movie jaldi nahi aayegi.\n\n"
                    f"Similar previous request: \"{similar.get('stored_title')}\" ({similar.get('score')}% match)\n"
                    f"Kripya {minutes_left} minute baad dobara koshish karein. 🙏"
                )
                msg = await update.message.reply_text(strict_text)
                track_message_for_deletion(context, update.effective_chat.id, msg.message_id, USER_TEXT_DELETE_SECONDS)
                return REQUESTING

        stored = await run_async(store_user_request,
            user.id,
            user.username,
            user.first_name,
            movie_title,
            update.effective_chat.id if update.effective_chat.type != "private" else None,
            update.message.message_id
        )
        if not stored:
            logger.error("Failed to store user request in DB.")
            await update.message.reply_text("Sorry, आपका request store नहीं हो पाया। बाद में कोशिश करें।")
            return REQUESTING

        group_info = update.effective_chat.title if update.effective_chat.type != "private" else None
        await send_admin_notification(context, user, movie_title, group_info)

        msg = await update.message.reply_text(
            f"✅ Got it! Your request for '{movie_title}' has been sent. I'll let you know when it's available.",
            reply_markup=get_main_keyboard()
        )
        track_message_for_deletion(context, update.effective_chat.id, msg.message_id, USER_TEXT_DELETE_SECONDS)

        return MAIN_MENU

    except Exception as e:
        logger.error(f"Error in request_movie: {e}")
        await update.message.reply_text("Sorry, an error occurred while processing your request.")
        return REQUESTING

async def request_movie_from_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle movie request after user sends movie name following button click"""
    try:
        user_message = (update.message.text or "").strip()
        
        # Check for Main Menu Buttons (Emergency Exit)
        menu_buttons = ['🔍 Search Movies', '🙋 Request Movie', '📊 My Stats', '❓ Help', '/start']
        if user_message in menu_buttons:
            if 'awaiting_request' in context.user_data:
                del context.user_data['awaiting_request']
            if 'pending_request' in context.user_data:
                del context.user_data['pending_request']
            return await main_menu(update, context)

        if not user_message:
            await update.message.reply_text("कृपया मूवी का नाम भेजें।")
            return REQUESTING_FROM_BUTTON

        # Store movie name
        context.user_data['pending_request'] = user_message
        
        confirm_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📽️ Confirm 🎬", callback_data=f"confirm_request_{user_message[:40]}")]
        ])
        
        msg = await update.message.reply_text(
            f"✅ आपने '<b>{user_message}</b>' को रिक्वेस्ट करना चाहते हैं?\n\n"
            f"<b>💫 अब बस अपनी मूवी या वेब-सीरीज़ का मूल नाम भेजें और कन्फर्म बटन पर क्लिक करें!</b>\n\n"
            f"कृपया कन्फर्म बटन पर क्लिक करें 👇",
            reply_markup=confirm_keyboard,
            parse_mode='HTML'
        )
        track_message_for_deletion(context, update.effective_chat.id, msg.message_id, USER_TEXT_DELETE_SECONDS)
        
        return MAIN_MENU

    except Exception as e:
        logger.error(f"Error in request_movie_from_button: {e}")
        return MAIN_MENU

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    chat_id = query.message.chat.id
    data = query.data

    # Server-side spelling suggestions shown after a failed Telegram search.
    # Re-run the normal DB fuzzy search with the suggested, corrected title.
    if data.startswith("retrysearch_"):
        try:
            await query.answer()
        except Exception:
            logger.exception("Could not acknowledge retry-search callback")
        suggested_title = unquote(data[len("retrysearch_"):]).strip()
        try:
            if not suggested_title:
                await _replace_search_result_message(
                    query, context, "⚠️ Search suggestion was empty. Please search again."
                )
                return
            movies = await run_async(get_movies_fast_sql, suggested_title, limit=10)
            if not movies:
                movies = await run_async(get_movies_from_db, suggested_title, limit=10)
            if not movies:
                suggestions = await get_google_title_suggestions_with_timeout(
                    suggested_title, limit=3
                )
                await _replace_search_result_message(
                    query,
                    context,
                    (
                        f"❌ <b>{html_escape(suggested_title)}</b> अभी database में available नहीं है.\n\n"
                        "Try another suggestion or request this title."
                    ),
                    reply_markup=_not_found_keyboard(suggested_title, suggestions),
                )
                return

            chosen_movie = _select_single_search_result(suggested_title, movies)
            if chosen_movie:
                movie_id, title, url, file_id = chosen_movie[:4]
                await _replace_search_result_message(
                    query,
                    context,
                    f"🔎 <b>{html_escape(title)}</b> found — getting its files…",
                )
                await send_movie_to_user(update, context, movie_id, title, url, file_id)
                return

            context.user_data['search_results'] = movies
            context.user_data['search_query'] = suggested_title
            await _replace_search_result_message(
                query,
                context,
                (
                    f"<b>🎬 Search results</b>\n\n"
                    f"Found <b>{len(movies)}</b> results for "
                    f"'<b>{html_escape(suggested_title)}</b>'.\n"
                    "Select the correct title below:"
                ),
                reply_markup=create_movie_selection_keyboard(movies, page=0),
            )
        except Exception:
            logger.exception("Retry search failed for callback")
            try:
                await _replace_search_result_message(
                    query,
                    context,
                    "⚠️ Search is temporarily unavailable. Please try again.",
                )
            except Exception:
                logger.exception("Could not show retry-search error state")
        return

    if chat_id < 0 and data.startswith(("movie_", "page_", "cancel_selection")):
        requester_match = re.search(r"_u(\d+)$", data)
        if not requester_match:
            await query.answer(
                "This search result is no longer available. Please search again.",
                show_alert=True,
            )
            return
        if user_id != int(requester_match.group(1)):
            user_name = query.from_user.first_name
            alert_text = (
                f"✋ Hello {user_name}!\n\n"
                "This is not your search result. Please run your own search."
            )
            await query.answer(alert_text, show_alert=True)
            return


    # ✅ NAYA: Video wala Pages Button Popup
    if data == "ignore":
        await query.answer("THIS IS PAGES BUTTON 🔴", show_alert=False)
        return

    if data.startswith("fl_") or data.startswith("v_"):
        parts = query.data.split('_')
        view_type = parts[1] if parts[0] == "v" else "main" 
        
        # ✅ NAYA: Video wale cool popups!
        if view_type in ["lang", "qual", "seas"]:
             await query.answer("Select a filter below 👇", show_alert=False)

    # ==================== NAYA: SINGLE FILE SEND ====================
    if data.startswith("send_single_"):
        # Telegram File IDs mein underscores (_) ho sakte hain, isliye safai se nikalenge
        parts = data.split('_')
        movie_id = int(parts[-1]) # Aakhri hissa hamesha movie_id hota hai
        file_id_to_send = data.replace("send_single_", "").replace(f"_{movie_id}", "")
        
        # Memory se movie ka naam nikal lo
        movie_data = context.user_data.get('selected_movie_data')
        if not movie_data or movie_data.get('id') != movie_id:
            movie_data = load_movie_selection_data(movie_id)
        title = movie_data['title'] if movie_data else "Requested Movie"

        try:
            # 🚀 NAYA: Ab simple text ki jagah tera Premium function use hoga!
            await send_movie_to_user(
                update=update, 
                context=context, 
                movie_id=movie_id, 
                title=title, 
                url=None, 
                file_id=file_id_to_send, 
                send_warning=True  # Single file click → ek baar GIF bhejna zaroori hai
            )
        except Exception as e:
            await query.answer("❌ Error sending file.", show_alert=True)
            logger.error(f"Single file send error: {e}")
        return

    elif data.startswith("back_to_seasons_"):
        movie_id = int(data.split('_')[3])
        context.user_data.pop('active_filter', None)
        context.user_data.pop('selected_season', None)
        movie_data = context.user_data.get('selected_movie_data')
        if not movie_data or movie_data.get('id') != movie_id:
            movie_data = load_movie_selection_data(movie_id)
            if movie_data:
                context.user_data['selected_movie_data'] = movie_data
        if not movie_data:
            await query.answer("❌ Movie data is no longer available.", show_alert=True)
            return
        title = movie_data['title']
        qualities = movie_data['qualities']
        seasons = set()
        for f in qualities:
            extra = f[5] if len(f) > 5 else ""
            if extra:
                s_name = extract_season_name(extra)
                if s_name != "Extra Files": seasons.add(s_name)
        
        keyboard = []
        keyboard.append([InlineKeyboardButton("🎬 Movie", callback_data=f"showseason_{movie_id}_Extra Files")])
        for s in sorted(list(seasons)):
            keyboard.append([InlineKeyboardButton(f"📁 {s}", callback_data=f"showseason_{movie_id}_{s}")])
        keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="cancel_selection")])
        
        await query.edit_message_text(f"📺 **{title}**\n\n👇 **Select Option:**", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        return

    
    
    # === START MENU BUTTONS LOGIC ===
    if data.startswith("start_"):
        await query.answer()

        if data == "start_help":
            text = (
                "<b>━━━━━ 🛠 𝗗𝗲𝗹𝗽 𝗠𝗲𝗻𝘂 ━━━━━</b>\n\n"
                "╭─── ❖ 𝗗𝗼𝘄 𝘁𝗼 𝗨𝘀𝗲 ❖ ───╮\n"
                "│\n"
                "│  ◈ Mujhe apne group me add karo\n"
                "│  ◈ Admin bana do\n"
                "│  ◈ Main auto files filter karunga!\n"
                "│\n"
                "╰─────────────────╯"
            )
            back_btn = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 BACK", callback_data="start_back")]])
            # Purani GIF delete karke naya message bhejenge (sabse safe tareeka)
            try: await query.message.delete()
            except: pass
            msg = await context.bot.send_message(chat_id=chat_id, text=text, parse_mode='HTML', reply_markup=back_btn)
            track_message_for_deletion(context, chat_id, msg.message_id, USER_TEXT_DELETE_SECONDS)
            return

        elif data == "start_about":
            text = (
                f"<b>━━━━━ 📖 𝗔𝗯𝗼𝘂𝘁 𝗠𝗲 ━━━━━</b>\n\n"
                f"╭─── ❖ 𝗗𝗲𝘁𝗮𝗶𝗹𝘀 ❖ ───╮\n"
                f"│\n"
                f"│  ◈ <b>Developer:</b> @{ADMIN_USERNAME}\n"
                f"│  ◈ <b>Language:</b> Python 3\n"
                f"│  ◈ <b>Library:</b> python-telegram-bot\n"
                f"│\n"
                f"╰─────────────────╯"
            )
            back_btn = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 BACK", callback_data="start_back")]])
            try: await query.message.delete()
            except: pass
            msg = await context.bot.send_message(chat_id=chat_id, text=text, parse_mode='HTML', reply_markup=back_btn)
            track_message_for_deletion(context, chat_id, msg.message_id, USER_TEXT_DELETE_SECONDS)
            return
            
        elif data == "start_donate":
            # Yahan se tumhara purana start_donate wala code shuru hoga...
            await query.answer()
            user = update.effective_user
            amount = 10  # Tumhara VIP amount
            upi_id = os.environ.get("UPI_ID", "default_id@ybl")
            
            try:
                import qrcode
                from io import BytesIO
                from urllib.parse import quote
                
                # QR Code Generate karna
                note = f"TG-{user.id}"
                upi_url = f"upi://pay?pa={upi_id}&pn=VIP+Subscription&am={amount}&tn={note}&cu=INR"
                
                qr = qrcode.QRCode(version=1, box_size=10, border=4)
                qr.add_data(upi_url)
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")
                bio = BytesIO()
                img.save(bio, format='PNG')
                bio.seek(0)
                
                text = (
                    f"💎 <b>VIP DONATION - ₹{amount}</b>\n\n"
                    f"📱 <b>Scan QR Code</b> from any UPI app (GPay/PhonePe/Paytm)\n"
                    f"💳 <b>UPI ID:</b> <code>{upi_id}</code>\n\n"
                    f"✅ Payment ke baad:\n"
                    f"1️⃣ <b>Screenshot</b> bhejo yahan\n"
                    f"2️⃣ Phir <b>UTR Number</b> type karke bhejo\n\n"
                    f"📸 <i>Intezaar hai aapke screenshot ka...</i>"
                )
                
                # Bot ko batana ki user ab screenshot bhejega
                context.user_data['payment_step'] = 'screenshot'
                
                # Purana menu delete karke QR bhejna. The callback can arrive
                # after Telegram has already removed the old menu.
                try:
                    await query.message.delete()
                except TelegramError as exc:
                    logger.info(f"Start menu was already unavailable: {exc}")
                donation_msg = await context.bot.send_photo(
                    chat_id=query.message.chat_id,
                    photo=bio,
                    caption=text,
                    parse_mode='HTML',
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="start_back")]])
                )
                track_user_message_for_deletion(context, chat_id, donation_msg)
            except Exception as e:
                # Agar qrcode install nahi hai toh normal text bhejega
                text = f"<b>💰 DONATION</b>\n\nAgar aapko mera kaam pasand aaya, toh aap UPI pe support kar sakte hain: <code>{upi_id}</code>"
                back_btn = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 BACK", callback_data="start_back")]])
                await query.edit_message_caption(caption=text, parse_mode='HTML', reply_markup=back_btn)
            return

    if data == "start_back":
        await query.answer()
        chat_id = query.message.chat_id
        
        # Pehle wala message delete karo (chahe wo Help ho, About ho ya QR code ho)
        try:
            await query.message.delete()
        except: pass

        # Wapas Start Menu Banane ka logic
        user = update.effective_user
        user_name = user.first_name
        user_id_val = user.id
        user_uname = user.username
        
        ## 🌟 Mention banao (Direct Profile Link without Web Preview)
        user_display = f"<a href='tg://user?id={user_id_val}'>{user_name}</a>"
        
        bot_info = await context.bot.get_me()
        bot_name = bot_info.first_name
        
        try:
            import pytz
            tz = pytz.timezone('Asia/Kolkata')
            hour = datetime.now(tz).hour
        except ImportError:
            hour = datetime.now().hour
            
        if 5 <= hour < 12: greeting = "Good Morning ☀️"
        elif 12 <= hour < 17: greeting = "Good Afternoon 🌤️"
        elif 17 <= hour < 21: greeting = "Good Evening 🌆"
        else: greeting = "Good Night 🌙"

        caption_text = (
            f"<b>━━━━━━━ 🚩 𝐉𝐀𝐈 𝐒𝐇𝐑𝐈 𝐑𝐀𝐌 🚩 ━━━━━━━</b>\n\n"
            f"✦ {greeting}, {user_display}!\n\n"
            f"╭─── ❖ 𝗔𝗕𝗢𝗨𝗧 𝗠𝗘 ❖ ───╮\n"
            f"│\n"
            f"│  🤖 Main hoon <b>{bot_name}</b>\n"
            f"│  𝗧𝗵𝗲 𝗠𝗼𝘀𝘁 𝗣𝗼𝘄𝗲𝗿𝗳𝘂𝗹 𝗔𝘂𝘁𝗼 𝗙𝗶𝗹𝘁𝗲𝗿 𝗕𝗼𝘁\n"
            f"│\n"
            f"╰──────────────────╯\n\n"
            f"<b>⟐ 𝗠𝘆 𝗣𝗿𝗲𝗺𝗶𝘂𝗺 𝗙𝗲𝗮𝘁𝘂𝗿𝗲𝘀:</b>\n"
            f"  ◈ ⚡ 𝗟𝗶𝗴𝗵𝘁𝗻𝗶𝗻𝗴-𝗳𝗮𝘀𝘁 Auto Filtering\n"
            f"  ◈ 🛡️ 𝟮𝟰/𝟳 Premium Uptime\n"
            f"  ◈ 🎬 HD/4K File Processing\n"
            f"  ◈ 🔍 𝗦𝗺𝗮𝗿𝘁 𝗦𝗲𝗮𝗿𝗰𝗵 + AI Matching\n\n"
            f"<b>━━━━━━━━━━━━━━━━━━━━━</b>\n"
            f"👇 <b>𝗧𝗮𝗽 𝘁𝗵𝗲 𝗯𝘂𝘁𝘁𝗼𝗻𝘀 𝗯𝗲𝗹𝗼𝘄 𝘁𝗼 𝗲𝘅𝗽𝗹𝗼𝗿𝗲!</b> 👇"
        )

        inline_buttons = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔰 ADD ME TO YOUR GROUP 🔰", url=f"https://t.me/{bot_info.username}?startgroup=true")],
            [InlineKeyboardButton("HELP 📢", callback_data="start_help"), InlineKeyboardButton("ABOUT 📖", callback_data="start_about")],
            [InlineKeyboardButton("DONATION 💰", callback_data="start_donate")]
        ])

        # Restore the exact approved GIF from the requested channel message.
        try:
            msg = await context.bot.copy_message(
                chat_id=chat_id,
                from_chat_id=START_GIF_CHANNEL_ID,
                message_id=START_GIF_MESSAGE_ID,
                caption=caption_text,
                parse_mode='HTML',
                reply_markup=inline_buttons
            )
        except TelegramError as exc:
            logger.error(f"Could not restore start menu media: {exc}")
            msg = await context.bot.send_message(
                chat_id=chat_id,
                text=caption_text,
                parse_mode='HTML',
                reply_markup=inline_buttons
            )
        track_user_message_for_deletion(context, chat_id, msg)
        return
        
    # === ADMIN REQUEST BUTTONS (Add/Not Found) ===
    if data.startswith("reqA_") or data.startswith("reqN_"):
        await query.answer("🔄 Sending message to user...", show_alert=False)
        parts = data.split('_', 2)
        action = parts[0]  # Yahan '_' hat jata hai, sirf 'reqA' ya 'reqN' bachta hai
        target_user_id = int(parts[1])
        movie_title = parts[2]

        # User ka naam DB se nikalo + mention format banao taaki message personal lage
        conn = get_db_connection()
        first_name = "User"
        db_username = None
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT first_name, username FROM user_requests WHERE user_id = %s LIMIT 1", (target_user_id,))
                res = cur.fetchone()
                if res:
                    first_name = res[0] or "User"
                    db_username = res[1] if len(res) > 1 else None
            except: pass
            finally: close_db_connection(conn)

        # 🌟 Premium Mention Format
        if db_username:
            user_mention_full = f"<a href='https://t.me/{db_username}'>{first_name}</a>"
        else:
            user_mention_full = f"<a href='tg://user?id={target_user_id}'>{first_name}</a>"

        # ✅ FIXED: "reqA_" ki jagah "reqA" use karna hai
        if action == "reqA":
            user_msg = (
                f"<b>━━━━━ 🎉 𝗡𝗲𝘄 𝗨𝗽𝗱𝗮𝘁𝗲 𝗙𝗼𝗿 𝗨𝗼𝘂! ━━━━━</b>\n\n"
                f"✦ Hey {user_mention_full}!\n\n"
                f"◈ आपकी Requested Movie अब उपलब्ध है।\n\n"
                f"🎬 File: <b>{movie_title}</b>\n\n"
                f"इसे पाने के लिए अभी बॉट में मूवी का नाम टाइप करें और एन्जॉय करें! 😊\n\n"
                f"<b>━━━━━━━━━━━━━━━━━━━</b>\n"
                f"◈ Regards, <b>@{ADMIN_USERNAME}</b>"
            )
            btn_status = "✅ User Notified: Added"
        else:
            user_msg = (
                f"<b>━━━━━ 😔 𝗨𝗽𝗱𝗮𝘁𝗲 𝗙𝗼𝗿 𝗨𝗼𝘂 ━━━━━</b>\n\n"
                f"✦ Hey {user_mention_full}!\n\n"
                f"◈ आपकी Requested File (<b>{movie_title}</b>) अभी हमें कहीं नहीं मिल पाई है।\n\n"
                f"जैसे ही यह अवेलेबल होगी, हम आपको जरूर बताएंगे।\n\n"
                f"<b>━━━━━━━━━━━━━━━━━━━</b>\n"
                f"◈ Regards, <b>@{ADMIN_USERNAME}</b>"
            )
            btn_status = "❌ User Notified: Not Found"

        # ✅ FIXED: Yahan user ko message send karna hai, taaki request block sahi se band ho jaye!
        success = await send_multi_bot_message(target_user_id, user_msg)
        
        if success:
            # Button hata do aur Admin ko updated status dikhao
            await query.edit_message_text(f"{query.message.text}\n\n{btn_status} 📩", parse_mode='HTML')
        else:
            await query.answer("❌ Failed! User ne sabhi bots block kar diye hain.", show_alert=True)
            
        return # Yahan is block ka kaam khatam!

    

    # =======================================================
    # 🖼️ NEW: ASK POSTER LOGIC (Semi-Auto Post)
    # =======================================================
    if data.startswith("askposter_"):
        if update.effective_user.id not in ADMIN_IDS:
            await query.answer("❌ Admin only!", show_alert=True)
            return

        movie_id = int(data.split("_")[1])
        
        # Bot ko yaad dilao ki ab agli photo is movie ke liye aayegi
        context.user_data['waiting_for_poster'] = movie_id
        
        await query.answer()
        await query.message.reply_text(
            "🖼️ **Please send the Landscape Poster (Image) for this movie now.**\n\n"
            "*(सिर्फ़ फोटो भेजें, कोई कैप्शन लिखने की ज़रूरत नहीं है)*",
            parse_mode='Markdown'
        )
        return
        
    # Iske niche aapke baki ke callback conditions waise hi rahenge (autopost_, cancel_genre aadi...)
    
    # =======================================================
    # 🤖 NEW: AUTO POST LOGIC (Premium Cinematic & Random Styles)
    # =======================================================
    if query.data.startswith("autopost_"):
        await query.answer("⏳ Premium Post Generate ho rahi hai...")
        movie_id = int(query.data.split("_")[1])
        
        # --- 1. DATABASE SE DATA NIKALNA ---
        conn = get_db_connection()
        cur = conn.cursor()
        
        # क्वालिटी निकालें
        cur.execute("SELECT quality FROM movie_files WHERE movie_id = %s", (movie_id,))
        rows = cur.fetchall()
        
        # मूवी की डिटेल्स निकालें (🚀 NAYA: Ab poster_url aur category bhi nikalega)
        cur.execute("""
            SELECT title, genre, language, poster_url, category, year, rating
            FROM movies WHERE id = %s
        """, (movie_id,))
        m_data = cur.fetchone()
        cur.close()
        
        # Connection close (Tera custom function)
        try: close_db_connection(conn) 
        except: db_pool.putconn(conn) 

        if not m_data:
            await query.edit_message_text("❌ Error: Movie DB mein nahi mili!")
            return

        # क्वालिटी फॉर्मेटिंग
        res_list = []
        for r in rows:
            if r and r[0]:
                match = re.search(r'(\d{3,4}p)', str(r[0]))
                if match: res_list.append(match.group(1))
        res_list = sorted(list(set(res_list)), key=lambda x: int(x.replace('p','')), reverse=True)
        dynamic_res = " | ".join(res_list) if res_list else "1080p | 720p | 480p"

        m_title = m_data[0] if m_data[0] else "Unknown Movie"
        m_genre = m_data[1] if m_data[1] else "Action, Drama"
        m_lang = m_data[2] if m_data[2] else "Hindi + English"
        m_poster = m_data[3] if len(m_data) > 3 and m_data[3] else None
        m_category = m_data[4] if len(m_data) > 4 and m_data[4] else ""
        m_year = m_data[5] if len(m_data) > 5 and m_data[5] else "N/A"
        m_rating = m_data[6] if len(m_data) > 6 and m_data[6] else "N/A"

        # --- 2. POSTER PROCESSING (Cinematic Square Effect) ---
        # 🚀 NAYA FIX: Pehle TMDB ka link uthao. Agar TMDB poster nahi hai, tabhi Thumbnail use karo.
        raw_photo = m_poster if (m_poster and m_poster != 'N/A' and m_poster.startswith('http')) else None
        
        if not raw_photo:
            thumb_id = context.bot_data.get(f"auto_thumb_{movie_id}")
            if isinstance(thumb_id, str) and not thumb_id.startswith("http"):
                try:
                    tg_file = await context.bot.get_file(thumb_id)
                    raw_photo = bytes(await tg_file.download_as_bytearray())
                except Exception as e:
                    logger.error(f"Autopost thumb download error: {e}")
                    raw_photo = None
        
        # Yahan hum naya blurred poster banayenge
        if raw_photo:
            photo_to_send = await make_landscape_poster(raw_photo)
        else:
            # Default poster agar kuch na mile
            photo_to_send = "https://i.imgur.com/6XK4F6K.png"

        # --- 3. TRENDING-STYLE POST TEMPLATE ---
        safe_title = m_title.replace('<', '').replace('>', '')
        channel_caption = (
            f"🎬 <b>{safe_title}</b>\n"
            f"➖➖➖➖➖➖➖➖➖➖\n"
            f"📅 <b>Date:</b> {m_year}\n"
            f"⭐ <b>iMDB Rating:</b> {m_rating}/10\n"
            f"🎭 <b>Genre:</b> {m_genre}\n"
            f"🔊 <b>Language:</b> {m_lang}\n"
            f"<b>Quality:</b> V2 HQ-HDTC {dynamic_res}\n"
            f"➖➖➖➖➖➖➖➖➖➖\n"
            f"<b>Update Channel:</b> <a href='https://t.me/FlimfyBoxBackUp'>Join BackUp</a>\n"
            f"👇 <b>Download Below</b> 👇"
        )

        # --- 4. SECURE LINK & BUTTONS ---
        secure_url = f"https://flimfybox-bot-yht0.onrender.com/watch/{movie_id}"
        channel_link = os.environ.get('FILMFYBOX_CHANNEL_URL', 'https://t.me/your_channel')

        post_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Get Now", url=secure_url)],
            [InlineKeyboardButton("Join Channel", url=channel_link)]
        ])

        # --- 5. BROADCASTING TO CHANNELS ---
        cat_lower = str(m_category).lower()
        if "anime" in cat_lower or "cartoon" in cat_lower or "animation" in cat_lower:
            target_channels = [ANIME_CHANNEL_ID]
        else:
            channels_str = os.environ.get('BROADCAST_CHANNELS', '')
            target_channels = [ch.strip() for ch in channels_str.split(',') if ch.strip()]

        if not target_channels:
            await query.edit_message_text(f"{query.message.text}\n\n❌ Error: No BROADCAST_CHANNELS found in env.")
            return

        # 👇 GLOBAL DUPLICATE CHECK — 7 din me kahi bhi post hui ho to skip
        if is_movie_posted_recently(movie_id, days=7):
            await query.edit_message_text(f"⏭️ **{m_title}** pehle se 7 din ke andar post ho chuki hai. Skipping.", parse_mode='Markdown')
            return

        sent_count = 0
        last_error = ""
        telegram_photo_id = None 

        for chat_id_str in target_channels:
            try:
                chat_id = int(chat_id_str)
                
                # Fast posting ke liye Telegram File ID use karo
                if telegram_photo_id:
                    sent_msg = await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=telegram_photo_id,
                        caption=channel_caption,
                        parse_mode='HTML',
                        reply_markup=post_keyboard
                    )
                else:
                    # BytesIO pointer ko start par reset karna zaroori hai
                    if hasattr(photo_to_send, 'seek'):
                        photo_to_send.seek(0)
                        
                    sent_msg = await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=photo_to_send,
                        caption=channel_caption,
                        parse_mode='HTML',
                        reply_markup=post_keyboard
                    )
                    # Agle channel ke liye File ID save kar lo
                    if sent_msg and sent_msg.photo:
                        telegram_photo_id = sent_msg.photo[-1].file_id

                # DB mein save karne wala tera purana logic
                if sent_msg:
                    try:
                        save_post_to_db(movie_id, chat_id, sent_msg.message_id, "bot3", channel_caption, telegram_photo_id, "photo", post_keyboard.to_dict(), None, "movies")
                    except Exception as db_err:
                        logger.error(f"Save to DB Error: {db_err}")
                        
                sent_count += 1
                await asyncio.sleep(1) # Flood se bachne ke liye chhota delay
                
            except Exception as e:
                logger.error(f"Auto-post failed for {chat_id_str}: {e}")
                last_error = str(e)

        # --- 6. SUCCESS MESSAGE ---
        result_msg = f"✅ <b>Auto-Posted (VIP Square Poster) to {sent_count} channels!</b>"
        if sent_count == 0 and last_error: 
            result_msg += f"\n❌ <b>Failed Reason:</b> <code>{last_error}</code>"

        await query.edit_message_text(result_msg, parse_mode='HTML')
        
        # Memory saaf karo
        context.bot_data.pop(f"auto_thumb_{movie_id}", None)
        return
    
    # === NEW: GENRE CALLBACK HANDLER ===
    if data.startswith(("genre_", "cancel_genre")):
        await handle_genre_selection(update, context)
        return
    
    # === SEND ALL FILES LOGIC (CURRENT PAGE ONLY) ===
    if query.data.startswith("sendall_"):
        parts = query.data.split("_")
        movie_id = int(parts[1])
        # Page number callback data se lo, nahi mila toh 1 assume karo
        current_page = int(parts[2]) if len(parts) > 2 else 1
        chat_id = update.effective_chat.id

        # Send All must always be delivered in PM. A group callback cannot
        # reliably tell whether this user has started or blocked the bot
        # without attempting a private Telegram action, so hand the request
        # to /start first. The deep-link handler then delivers the page only
        # after Telegram has accepted the user's PM update.
        if update.effective_chat.type in ("group", "supergroup"):
            bot_username = context.bot.username
            if not bot_username:
                bot_username = (await context.bot.get_me()).username
            if bot_username:
                start_url = (
                    f"https://t.me/{bot_username}"
                    f"?start=sendall_{movie_id}_{current_page}"
                )
                await query.answer(url=start_url)
            else:
                await query.answer(
                    "⚠️ Please START the bot in PM first to receive files!",
                    show_alert=True,
                )
            return

        # ✅ FAST FETCH: Ek hi bar mein sab nikal lo (seasons_data bhi le lo extra DB calls bachane ke liye)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT title, genre, year, language, seasons_data FROM movies WHERE id = %s", (movie_id,))
        res = cur.fetchone()
        cur.close()
        close_db_connection(conn)

        if res:
            title, db_genre, db_year, db_lang, db_seasons = res
            pre_fetched_meta = {'genre': db_genre, 'year': db_year, 'language': db_lang, 'seasons_data': db_seasons}
        else:
            title = "Movie"
            pre_fetched_meta = {}

        qualities = get_all_movie_qualities(movie_id)
        
        # NAYA: Filter apply karo taaki Send All sirf filter ki hui files bheje
        active_filter = context.user_data.get('active_filter')
        if active_filter:
            f_type = active_filter['type']
            f_val = active_filter['value'].lower()
            temp_list = []
            for q in qualities:
                q_name = str(q[0]).lower()
                lang_name = str(q[4]).lower() if len(q) > 4 else ""
                extra = str(q[5]).lower() if len(q) > 5 else ""
                if f_type == "lang" and f_val in lang_name: temp_list.append(q)
                elif f_type == "qual" and f_val in q_name: temp_list.append(q)
                elif f_type == "seas" and f_val in extract_season_name(extra).lower(): temp_list.append(q)
            qualities = temp_list

        if not qualities:
            await query.answer("❌ No files found!", show_alert=True)
            return

        # 🚀 CURRENT PAGE KI FILES NIKALO (limit = 10 per page)
        limit = 10
        start_idx = (current_page - 1) * limit
        end_idx = start_idx + limit
        page_files = qualities[start_idx:end_idx]

        if not page_files:
            await query.answer("❌ Is page par koi file nahi hai!", show_alert=True)
            return

        context.user_data['sendall_page'] = current_page
        await query.answer(f"🚀 Sending {len(page_files)} files (Page {current_page})...")
        status_msg = await query.message.reply_text(f"🚀 **Sending {len(page_files)} files (Page {current_page})...**", parse_mode='Markdown')
        
        # 1. LOOP: SIRF CURRENT PAGE KI FILES BHEJO
        # ⚡ SPEED FIX: extra_info bhi pre_fetched_meta mein pass karo + sleep kam kiya
        count = 0
        for file_data in page_files:
            url = file_data[1]
            file_id = file_data[2]
            
            # Har file ka extra_info directly qualities tuple se nikal lo (DB call nahi lagegi)
            file_extra_info = str(file_data[5]).strip() if len(file_data) > 5 and file_data[5] else ""
            file_meta = dict(pre_fetched_meta)  # Copy banao taaki original change na ho
            file_meta['extra_info'] = file_extra_info  # Ye pass karo taaki send_movie_to_user DB na hit kare
            
            try:
                await send_movie_to_user(
                    update, context, movie_id, title, url, file_id, 
                    send_warning=False,
                    pre_fetched_meta=file_meta,
                    suppress_delete_notice=True,
                )
                await asyncio.sleep(0.3)  # ⚡ 1.2s → 0.3s (safe_send mein flood protection already hai)
                count += 1
            except Exception as e:
                logger.error(f"Send All Error: {e}")

        # Send auto-delete notice only once after all files are delivered
        if count > 0:
            try:
                warn_text_msg = await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "⚠️ <b>𝗔𝘂𝘁𝗼-𝗗𝗲𝗹𝗲𝘁𝗲 𝗡𝗼𝘁𝗶𝗰𝗲</b>\n\n"
                        "◈ ऊपर भेजी गयी file <b>2 minutes</b> बाद auto-delete हो जाएगी।\n"
                        "◈ कृपया file को <b>forward/save</b> कर लें। 🔄"
                    ),
                    parse_mode='HTML'
                )
                track_message_for_deletion(
                    context, chat_id, warn_text_msg.message_id, USER_FILE_DELETE_SECONDS,
                )
            except:
                pass

        await status_msg.edit_text(f"✅ **Sent {count}/{len(page_files)} Files (Page {current_page})!**", parse_mode='Markdown')
        track_message_for_deletion(context, chat_id, status_msg.message_id, USER_TEXT_DELETE_SECONDS)
        return
    
    # === NEW: SCAN INFO POPUP ===
    if data.startswith("scan_"):
        m_id = int(data.split("_")[1])
        
        # Database se details nikalo
        conn = get_db_connection()
        cur = conn.cursor()
        # Maan lo tumhare DB mein 'language' aur 'subtitle' column hain, ya tum file name se guess karoge
        cur.execute("SELECT title, year, genre FROM movies WHERE id = %s", (m_id,))
        res = cur.fetchone()
        cur.close()
        close_db_connection(conn)

        if res:
            title, year, genre = res
            # Ye wo text hai jo Popup mein dikhega
            popup_text = (
                f"📂 File Info:\n"
                f"🎬 Movie: {title}\n"
                f"📅 Year: {year}\n"
                f"🎭 Genre: {genre}\n"
                f"🔊 Audio: Hindi, English (Dual)\n" # Ise DB se dynamic bana sakte ho
                f"📝 Subs: English, Hindi"
            )
            # show_alert=True ka matlab hai Screen par bada popup aayega!
            await query.answer(popup_text, show_alert=True)
        else:
            await query.answer("❌ Info not found", show_alert=True)
        return
    
    # ===================================
    
    # 👇👇👇 YE NAYA CODE ADD KARO 👇👇👇
    if query.data.startswith("clearfiles_"):
        if update.effective_user.id not in ADMIN_IDS:
            await query.answer("❌ Sirf Admin ke liye!", show_alert=True)
            return

        movie_id = int(query.data.split("_")[1])
        
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("DELETE FROM movie_files WHERE movie_id = %s", (movie_id,))
                deleted_count = cur.rowcount # Kitni delete hui
                conn.commit()
                cur.close()
                close_db_connection(conn)
                
                # 👇 NAYA: BATCH_SESSION ke counter ko bhi zero (0) kar do
                if BATCH_SESSION.get('movie_id') == movie_id:
                    BATCH_SESSION['file_count'] = 0
                
                await query.answer(f"✅ {deleted_count} purani files delete ho gayi!", show_alert=True)
                await query.edit_message_text(
                    f"🗑️ **Deleted {deleted_count} old files.**\n\n"
                    f"✅ **Clean Slate!** Ab nayi files upload karo.",
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Delete Error: {e}")
                await query.answer("❌ Error deleting files", show_alert=True)
        return
    
    
    # === CANCEL BATCH LOGIC ===
    if query.data == "cancel_batch":
        if update.effective_user.id not in ADMIN_IDS:
            await query.answer("❌ Sirf Admin ke liye!", show_alert=True)
            return

        movie_id = BATCH_SESSION.get('movie_id')

        # 👇 NAYA LOGIC: Agar koi file save nahi hui thi, toh galat naam DB se uda do
        if movie_id:
            conn = get_db_connection()
            if conn:
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM movie_files WHERE movie_id = %s", (movie_id,))
                    file_count = cur.fetchone()[0]
                    
                    # Agar movie khali hai, toh delete maar do!
                    if file_count == 0:
                        cur.execute("DELETE FROM movies WHERE id = %s", (movie_id,))
                        conn.commit()
                    cur.close()
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
                finally:
                    close_db_connection(conn)

        # Session ko off kar do taaki aur files save na hon
        BATCH_SESSION.update({
            'active': False, 'movie_id': None, 'movie_title': None,
            'file_count': 0, 'admin_id': None, 'year': '', 'category': ''
        })

        await query.answer("🛑 Batch Stopped & Cleaned!", show_alert=True)
        await query.edit_message_text(
            "❌ **Batch Cancelled & Junk Data Removed.**\n\n"
            "Aap chaho to manually sahi naam dekar naya batch start kar sakte ho:\n"
            "`/batch Sahi Movie Name, 2024`",
            parse_mode='Markdown'
        )
        return
        
    # === CANCEL 18+ BATCH LOGIC ===
    if query.data == "cancel_batch18":
        if update.effective_user.id not in ADMIN_IDS:
            await query.answer("❌ Sirf Admin ke liye!", show_alert=True)
            return

        movie_id = BATCH_18_SESSION.get('movie_id')

        # 👇 NAYA LOGIC: 18+ wale kachre ko bhi uda do
        if movie_id:
            conn = get_db_connection()
            if conn:
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM movie_files WHERE movie_id = %s", (movie_id,))
                    if cur.fetchone()[0] == 0:
                        cur.execute("DELETE FROM movies WHERE id = %s", (movie_id,))
                        conn.commit()
                    cur.close()
                except Exception as e:
                    pass
                finally:
                    close_db_connection(conn)

        BATCH_18_SESSION.update({
            'active': False, 'movie_id': None, 'movie_title': None,
            'file_count': 0, 'admin_id': None, 'year': '', 'category': ''
        })

        await query.answer("🛑 18+ Batch Stopped!", show_alert=True)
        await query.edit_message_text(
            "❌ **18+ Batch Stopped & Junk Removed.**\n\n"
            "Aap chaho to manually naya batch start kar sakte ho.",
            parse_mode='Markdown'
        )
        return
    
    # === 1. VERIFY BUTTON LOGIC (UPDATED) ===
    if data == "verify":
        await query.answer("🔍 Checking membership...", show_alert=False) # Alert False rakha taki user disturb na ho
        
        # Force Fresh Check
        check = await is_user_member(context, user_id, force_fresh=True)
        
        if check['is_member']:
            # ✅ SCENARIO 1: Agar koi Deep Link pending tha (e.g. start=movie_123)
            if 'pending_start_args' in context.user_data:
                saved_args = context.user_data.pop('pending_start_args')
                
                # "Verified" wala msg delete kar do taaki clean lage
                try: await query.message.delete()
                except: pass
                
                # Start function ko manually call karo saved args ke saath
                context.args = saved_args
                await start(update, context)
                return

            # ✅ SCENARIO 2: Agar koi Text Search pending tha (e.g. "Kalki")
            elif 'pending_search_query' in context.user_data:
                saved_query = context.user_data.pop('pending_search_query')
                
                # "Verified" wala msg delete kar do
                try: await query.message.delete()
                except: pass
                
                # Search Movies ko call karne ke liye update object ko modify karein
                # Hum current query message ko use karenge par text replace kar denge
                update.message = query.message 
                update.message.text = saved_query
                
                # User ko feedback do ki search shuru ho gaya
                await search_movies(update, context)
                return

            # ✅ SCENARIO 3: Agar koi pending request nahi thi (Normal Verify)
            else:
                await query.edit_message_text(
                    "✅ **Verified Successfully!**\n\n"
                    "You can now use the bot! 🎬\n"
                    "Click /start or search any movie.",
                    parse_mode='Markdown'
                )
                track_message_for_deletion(context, chat_id, query.message.message_id, USER_TEXT_DELETE_SECONDS)
        else:
            # Agar abhi bhi join nahi kiya
            try:
                await query.edit_message_text(
                    get_join_message(check['channel'], check['group']),
                    reply_markup=get_join_keyboard(),
                    parse_mode='Markdown'
                )
            except telegram.error.BadRequest:
                await query.answer("❌ You haven't joined yet!", show_alert=True)
        return
    # ==============================

    # === 2. OTHER BUTTONS PROTECTION (Optional but Recommended) ===
    # Agar user 'download', 'movie', 'request' dabaye to bhi check karo
    if data.startswith(("movie_", "download_", "quality_", "request_")):
        check = await is_user_member(context, user_id) # Cache use karega
        if not check['is_member']:
            await query.answer("❌ Please join channels first!", show_alert=True)
            await query.edit_message_text(
                get_join_message(check['channel'], check['group']),
                reply_markup=get_join_keyboard(),
                parse_mode='Markdown'
            )
            return
    # ==============================================================

    try:
        # ==================== MOVIE SELECTION ====================
        if query.data.startswith("movie_"):
            # Strip _u{user_id} suffix if present (group buttons)
            movie_data_part = re.sub(r'_u\d+$', '', query.data)
            movie_id = int(movie_data_part.replace("movie_", ""))

            conn = get_db_connection()
            cur = conn.cursor()
            # 🚀 FIX: Yahan 'category' bhi nikal rahe hain taaki pata chale Web Series hai ya nahi
            cur.execute("SELECT id, title, category, poster_url FROM movies WHERE id = %s", (movie_id,))
            movie = cur.fetchone()
            cur.close()
            close_db_connection(conn)

            if not movie:
                await query.edit_message_text("❌ Movie not found in database.")
                return

            movie_id, title, category, poster_url = movie
            qualities = get_all_movie_qualities(movie_id)

            if not qualities:
                await query.answer("❌ No files found!", show_alert=True)
                return

            # Data context mein save karo aage ke liye
            context.user_data['selected_movie_data'] = {
                'id': movie_id,
                'title': title,
                'category': category,
                'qualities': qualities
            }


            # Agar normal Movie hai (ya Series ka season logic fail hua), toh direct qualities dikhao
            bot_username = context.bot.username
            bot_info = await context.bot.get_me()
            file_list_text = _format_requested_files_header(
                title,
                qualities,
                query.from_user,
                bot_info,
            )
            
            for idx, file_data in enumerate(qualities[:10], start=1):
                quality = file_data[0]
                file_size = file_data[3] if len(file_data) > 3 else "Unknown Size"
                extra_info = file_data[5] if len(file_data) > 5 else ""
                # ✅ CLEAN HTML LINK: Naruto bot jaisa neela text!
                real_idx = qualities.index(file_data)
                label = html_escape(
                    _format_file_link_label(file_size, title, extra_info, quality)
                )
                file_list_text += f"<b>{idx}.</b> <b><a href='https://t.me/{bot_username}?start=file_{movie_id}_{real_idx}'>{label}</a></b>\n\n"

            selection_text = file_list_text
            
            # Pagination calculate karo pehli baar ke liye
            limit = 10
            total_pages = (len(qualities) + limit - 1) // limit if qualities else 1
            
            # CLEAR PREVIOUS FILTERS
            context.user_data['active_filter'] = None
            
            # ✅ NAYA: Function ko call karo taaki 1, 2, 3 wale buttons aa jayein!
            current_files = qualities[:limit]
            keyboard_markup = create_quality_selection_keyboard(
                movie_id=movie_id, 
                view="main", 
                page=1, 
                total_pages=total_pages, 
                current_files=current_files
            )
            
            # Message update aur link preview disable
            if poster_url:
                try:
                    await query.message.delete()
                except:
                    pass
                try:
                    processed_poster = await make_landscape_poster(poster_url)
                    msg = await context.bot.send_photo(
                        chat_id=update.effective_chat.id,
                        photo=processed_poster,
                        caption=selection_text,
                        reply_markup=keyboard_markup,
                        parse_mode='HTML'
                    )
                    track_message_for_deletion(context, update.effective_chat.id, msg.message_id, USER_TEXT_DELETE_SECONDS)
                except Exception as e:
                    logger.error(f"Failed to send photo: {e}")
                    # send as text if photo fails
                    msg = await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=selection_text,
                        reply_markup=keyboard_markup,
                        parse_mode='HTML',
                        disable_web_page_preview=True
                    )
                    track_message_for_deletion(context, update.effective_chat.id, msg.message_id, USER_TEXT_DELETE_SECONDS)
            else:
                await query.edit_message_text(
                    selection_text,
                    reply_markup=keyboard_markup,
                    parse_mode='HTML',
                    disable_web_page_preview=True
                )
                track_message_for_deletion(context, update.effective_chat.id, query.message.message_id, USER_TEXT_DELETE_SECONDS)
            
            return


        # ==================== SEASON SELECTION (NEW) ====================
        elif query.data.startswith("showseason_"):
            parts = query.data.split('_', 2)
            movie_id = int(parts[1])
            selected_season = parts[2]
            
            # Context me season save karo
            context.user_data['selected_season'] = selected_season
            context.user_data['active_filter'] = None
            
            # 🚀 FIX: `query.data` read-only hai, usko badalna allowed nahi hai. 
            # Iski jagah sidha update.callback_query_data object modify nahi karke
            # manually call karte hain ya redirect code yahi execute karte hain.
            
            # Naye UI logic ki taraf redirect
            # Hum data ko sidha bhej rahe hain taaki button_callback khud ise handle kare, bina modify kiye
            class FakeQuery:
                def __init__(self, from_user, message, data):
                    self.from_user = from_user
                    self.message = message
                    self.data = data
                    
                async def answer(self, *args, **kwargs):
                    pass # Silent ignore
                    
                async def edit_message_text(self, *args, **kwargs):
                    return await query.edit_message_text(*args, **kwargs)

                async def edit_message_caption(self, *args, **kwargs):
                    return await query.edit_message_caption(*args, **kwargs)

            # Ek naya fake query object banaya taki read-only error na aaye
            update._callback_query = FakeQuery(query.from_user, query.message, f"v_main_{movie_id}")
            
            await button_callback(update, context)
            return
                
            title = movie_data['title']
            all_qualities = movie_data['qualities']
            
            # Sirf wahi files filter karo jo is selected season ki hain
            filtered_qualities = []
            for file_data in all_qualities:
                extra_info = file_data[5] if len(file_data) > 5 else ""
                if extract_season_name(extra_info) == selected_season:
                    filtered_qualities.append(file_data)
                    
            if not filtered_qualities:
                await query.answer("❌ No files found for this season!", show_alert=True)
                return
                
            # Ab sirf is Season ki files list karo
            # Video jaisa Text List format banana
            bot_info = await context.bot.get_me()
            file_list_text = _format_requested_files_header(
                f"{title} - {selected_season}",
                filtered_qualities,
                query.from_user,
                bot_info,
            )
            
            for idx, file_data in enumerate(filtered_qualities[:10], start=1):
                quality = file_data[0]
                file_size = file_data[3] if len(file_data) > 3 else "Unknown Size"
                extra_info = file_data[5] if len(file_data) > 5 else ""
                label = _format_file_link_label(file_size, title, extra_info, quality)
                file_list_text += f"**{idx}.** 💾 {label}\n\n"

            selection_text = file_list_text
            keyboard_markup = create_quality_selection_keyboard(movie_id, title, filtered_qualities, page=0, season=selected_season, view="main")
            
            # Hum wahi purana keyboard function use kar rahe hain, bas list chhoti bhej rahe hain
            keyboard_markup = create_quality_selection_keyboard(movie_id, title, filtered_qualities, page=0, season=selected_season)
            
            # ✅ FIX: InlineKeyboardMarkup ke andar list 'inline_keyboard' ek tuple ki tarah return hoti hai naye python-telegram-bot versions me.
            # Isliye humein pehle usko list mein badalna padega, tab usme Naya button daalna hoga.
            
            keyboard_list = list(keyboard_markup.inline_keyboard)
            keyboard_list.insert(0, [InlineKeyboardButton("🔙 Back to Seasons", callback_data=f"movie_{movie_id}")])
            
            new_keyboard = InlineKeyboardMarkup(keyboard_list)
            
            await query.edit_message_text(
                selection_text,
                reply_markup=keyboard_markup,
                parse_mode='HTML',
                disable_web_page_preview=True
            )
            return
            

        # ==================== ADMIN ACTIONS ====================
        
        # ==================== NAYA UI VIEWS, FILTERS & PAGINATION ====================
        elif query.data.startswith("v_") or query.data.startswith("fl_") or query.data.startswith("vpage_"):
            movie_data = context.user_data.get('selected_movie_data')
            callback_parts = query.data.split('_')
            try:
                movie_id_from_callback = int(
                    callback_parts[2] if callback_parts[0] != "vpage" else callback_parts[1]
                )
            except (IndexError, ValueError):
                movie_id_from_callback = None
            if movie_id_from_callback is not None and (
                not movie_data or movie_data.get('id') != movie_id_from_callback
            ):
                movie_data = load_movie_selection_data(movie_id_from_callback)
                if movie_data:
                    context.user_data['selected_movie_data'] = movie_data
            if not movie_data:
                await query.answer("❌ Movie data is no longer available.", show_alert=True)
                return

            movie_id = movie_data['id']
            title = movie_data['title']
            all_qualities = movie_data['qualities']

            if 'active_filter' not in context.user_data:
                context.user_data['active_filter'] = None

            # Filter Handle Karna
            if query.data.startswith("fl_"):
                parts = query.data.split('_', 3)
                f_type = parts[1]
                if f_type == "clear":
                    context.user_data['active_filter'] = None
                    await query.answer("✅ Filters Cleared!")
                else:
                    f_val = parts[3]
                    context.user_data['active_filter'] = {'type': f_type, 'value': f_val}
                    await query.answer(f"✅ Filter Applied: {f_val}")
                view_type = "main"
                page = 1
                
            # Pagination Handle Karna
            elif query.data.startswith("vpage_"):
                parts = query.data.split('_')
                page = int(parts[2])
                view_type = "main"
                
            # Menu Navigation
            else:
                parts = query.data.split('_')
                view_type = parts[1]
                page = 1 
                
                # ✅ NAYA: Video wale cool popups! (Removed duplicate answer call)

            # ==========================================
            # 🚀 SMART FILTER LOGIC (Seasons + Lang + Qual)
            # ==========================================
            filtered_qualities = all_qualities
            active_filter = context.user_data.get('active_filter')
            
            if active_filter:
                f_type = active_filter.get('type')
                f_val = active_filter.get('value').lower()
                temp_list = []
                
                for f in all_qualities:
                    # File ki saari details combine kar rahe hain
                    quality_str = str(f[0]).lower()
                    lang_name = str(f[4]).lower() if len(f) > 4 else ""
                    extra_info = str(f[5]).lower() if len(f) > 5 else ""
                    combined_text = f"{quality_str} {lang_name} {extra_info}"
                    
                    if f_type == 'seas':
                        s_name = extract_season_name(f[5] if len(f) > 5 else "").lower()
                        if s_name == f_val:
                            temp_list.append(f)
                            
                    elif f_type == 'lang':
                        if f_val in combined_text:
                            temp_list.append(f)
                            
                    elif f_type == 'qual':
                        if f_val in combined_text:
                            temp_list.append(f)
                            
                # ✅ NAYA POP-UP LOGIC: Agar is filter ki koi file nahi mili
                if not temp_list:
                    # 1. Telegram ka in-built Popup dikhao
                    await query.answer(f"❌ {active_filter['value'].upper()} format me file abhi available nahi hai!", show_alert=True)
                    # 2. Galat filter ko history se uda do taaki bot aage na atke
                    context.user_data['active_filter'] = None 
                    # 3. Yahi se waapis bhej do (UI change nahi hoga, waisa hi rahega)
                    return
                            
                filtered_qualities = temp_list

            # ==========================================
            # Pagination Logic (10 files per page)
            # ==========================================
            limit = 10
            total_pages = (len(filtered_qualities) + limit - 1) // limit if filtered_qualities else 1
            if page > total_pages: page = total_pages
            if page < 1: page = 1
            
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            current_page_files = filtered_qualities[start_idx:end_idx]

            # UI Text Banana
            if view_type == "main" or view_type == "seas":
                bot_info = await context.bot.get_me()
                text = _format_requested_files_header(
                    title,
                    filtered_qualities,
                    query.from_user,
                    bot_info,
                )
                
                # 🚀 NAYA FIX: Season ko alag se bada aur highlight dikhane ke liye
                if 'selected_season' in context.user_data and context.user_data['selected_season']:
                    s_name = context.user_data['selected_season'].upper()
                    text += f"━━━━━━━━━━━━━━━━━━━━\n"
                    text += f" <b>[ {s_name} ]</b> \n"
                    text += f"━━━━━━━━━━━━━━━━━━━━\n"
                    
                if active_filter:
                    text += f"🔍 Filter: <b>{active_filter['value']}</b>\n"
                if not filtered_qualities:
                    text += "❌ No files found for this filter.\n"
                else:
                    bot_username = context.bot.username
                    
                    for idx, file_data in enumerate(current_page_files, start=start_idx + 1):
                        quality = str(file_data[0])
                        
                        # 🚀 NAYA FIX: Doosre Bot (Manvi Bot) ke links ko hamesha ke liye uda do
                        quality = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', quality)
                        quality = re.sub(r'\(https?://[^\)]+\)', '', quality)
                        quality = re.sub(r'https?://[^\s]+', '', quality)
                        # 👇 Ye 2 lines nayi add karni hain: t.me aur @usernames udane ke liye
                        quality = re.sub(r'(?i)t\.me/[^\s]+', '', quality)
                        quality = re.sub(r'@[a-zA-Z0-9_]+', '', quality)
                        
                        file_size = file_data[3] if len(file_data) > 3 else "Unknown"
                        
                        # Extra Info (Episodes) se bhi link saaf karo
                        extra_info = str(file_data[5]) if len(file_data) > 5 else ""
                        extra_info = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', extra_info)
                        extra_info = re.sub(r'\(https?://[^\)]+\)', '', extra_info)
                        extra_info = re.sub(r'https?://[^\s]+', '', extra_info)
                        # 👇 Ye 2 lines yahan bhi add karni hain
                        extra_info = re.sub(r'(?i)t\.me/[^\s]+', '', extra_info)
                        extra_info = re.sub(r'@[a-zA-Z0-9_]+', '', extra_info)
                        
                        lang_name = str(file_data[4]).strip() if len(file_data) > 4 and file_data[4] else ""
                        real_idx = all_qualities.index(file_data)
                        label = html_escape(
                            _format_file_link_label(
                                file_size, title, extra_info, quality, lang_name
                            )
                        )
                        text += f"<b>{idx}.</b> <b><a href='https://t.me/{bot_username}?start=file_{movie_id}_{real_idx}'>{label}</a></b>\n\n"

            elif view_type in ["lang", "qual"]:
                text = f"📁 <b>{title}</b>\n\n👇 <b>Select {view_type.upper()} Filter:</b>\n\n"

            # Keyboard Banana
            keyboard = []
            
            # 1. MAIN MENU: Yahan normal buttons dikhenge
            if view_type == "main":
                if filtered_qualities:
                    keyboard.append([
                        InlineKeyboardButton("🔶 Sᴇɴᴅ Aʟʟ 🔶", callback_data=f"sendall_{movie_id}_{page}"),
                        InlineKeyboardButton("⚡ Tʀᴇɴᴅɪɴɢ", url=FILMFYBOX_GROUP_URL)
                    ])
                else:
                    keyboard.append([
                        InlineKeyboardButton("⚡ Tʀᴇɴᴅɪɴɢ", url=FILMFYBOX_GROUP_URL)
                    ])
                
                keyboard.append([
                    InlineKeyboardButton("📍 Qᴜᴀʟɪᴛʏ", callback_data=f"v_qual_{movie_id}"),
                    InlineKeyboardButton("🔊 Lᴀɴɢᴜᴀɢᴇ", callback_data=f"v_lang_{movie_id}"),
                    InlineKeyboardButton("🏷️ Sᴇᴀsᴏɴ", callback_data=f"v_seas_{movie_id}")
                ])
                
                nav_buttons = []
                nav_buttons.append(InlineKeyboardButton("◀️ ᴘʀᴇᴠ" if page > 1 else "ᴘᴀɢᴇ", callback_data=f"vpage_{movie_id}_{page-1}" if page > 1 else "ignore"))
                nav_buttons.append(InlineKeyboardButton(f"{page}/{total_pages}", callback_data="ignore"))
                nav_buttons.append(InlineKeyboardButton("ɴᴇxᴛ ▶️" if page < total_pages else "ɴᴇxᴛ >", callback_data=f"vpage_{movie_id}_{page+1}" if page < total_pages else "ignore"))
                keyboard.append(nav_buttons)

            # 2. SEASON MENU: 🚀 NAYA FIX - Yahan baaki kachra gayab, sirf Seasons!
            elif view_type == "seas":
                keyboard.append([InlineKeyboardButton("⬇ SELECT SEASON ⬇", callback_data="ignore")])
                
                seasons = set()
                for f in all_qualities:
                    extra = f[5] if len(f) > 5 else ""
                    if extra:
                        s = extract_season_name(extra)
                        if s != "Extra Files": seasons.add(s)
                        
                s_list = sorted(list(seasons))
                row = []
                for s in s_list:
                    btn_text = s.upper()
                    if btn_text.startswith("SEASON ") and len(btn_text.split(" ")[1]) == 1:
                        btn_text = btn_text.replace("SEASON ", "SEASON 0")
                        
                    row.append(InlineKeyboardButton(btn_text, callback_data=f"fl_seas_{movie_id}_{s}"))
                    if len(row) == 2:
                        keyboard.append(row)
                        row = []
                if row: keyboard.append(row)
                
                keyboard.append([
                    InlineKeyboardButton("🔄 CLEAR FILTER", callback_data=f"fl_clear_{movie_id}_all"),
                    InlineKeyboardButton("🔼 BACK TO MENU", callback_data=f"v_main_{movie_id}")
                ])

            # 3. LANGUAGE MENU
            elif view_type == "lang":
                keyboard.append([InlineKeyboardButton("⬇ SELECT LANGUAGE ⬇", callback_data="ignore")])
                
                languages = set()
                for f in all_qualities:
                    lang = f[4] if len(f) > 4 and f[4] else ""
                    if lang and lang.strip():
                        for l in lang.split(','):
                            languages.add(l.strip())
                            
                l_list = sorted(list(languages))
                row = []
                for l in l_list:
                    row.append(InlineKeyboardButton(l.upper(), callback_data=f"fl_lang_{movie_id}_{l}"))
                    if len(row) == 2:
                        keyboard.append(row)
                        row = []
                if row: keyboard.append(row)
                
                keyboard.append([
                    InlineKeyboardButton("🔄 CLEAR FILTER", callback_data=f"fl_clear_{movie_id}_all"),
                    InlineKeyboardButton("🔼 BACK TO MENU", callback_data=f"v_main_{movie_id}")
                ])

            # 4. QUALITY MENU
            elif view_type == "qual":
                keyboard.append([InlineKeyboardButton("⬇ SELECT QUALITY ⬇", callback_data="ignore")])
                
                quals = set()
                for f in all_qualities:
                    q = f[0] if len(f) > 0 and f[0] else ""
                    if q and q.strip():
                        quals.add(q.strip())
                        
                q_list = sorted(list(quals))
                row = []
                for q in q_list:
                    row.append(InlineKeyboardButton(q.upper(), callback_data=f"fl_qual_{movie_id}_{q}"))
                    if len(row) == 2:
                        keyboard.append(row)
                        row = []
                if row: keyboard.append(row)
                
                keyboard.append([
                    InlineKeyboardButton("🔄 CLEAR FILTER", callback_data=f"fl_clear_{movie_id}_all"),
                    InlineKeyboardButton("🔼 BACK TO MENU", callback_data=f"v_main_{movie_id}")
                ])

            # 👇 YAHAN disable_web_page_preview=True ADD KAR DIYA HAI 👇
            if query.message.photo:
                try:
                    await query.edit_message_caption(
                        caption=text,
                        reply_markup=InlineKeyboardMarkup(keyboard),
                        parse_mode='HTML'
                    )
                except Exception as e:
                    pass
            else:
                await query.edit_message_text(
                    text=text, 
                    reply_markup=InlineKeyboardMarkup(keyboard), 
                    parse_mode='HTML',
                    disable_web_page_preview=True 
                )
            return
        
        # ==================== QUALITY PAGINATION (NEXT/BACK) ====================
        elif query.data.startswith("qualpage_"):
            # FIX: Split up to 3 times to get the season name safely
            parts = query.data.split('_', 3)
            movie_id = int(parts[1])
            page = int(parts[2])
            selected_season = parts[3] if len(parts) > 3 else None

            # Try fetching data from user_data first (Fast)
            movie_data = context.user_data.get('selected_movie_data')
            
            # Agar data expire ho gaya ho ya ID match na kare, to DB se nikalo
            if not movie_data or movie_data.get('id') != movie_id:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("SELECT title FROM movies WHERE id = %s", (movie_id,))
                res = cur.fetchone()
                cur.close()
                close_db_connection(conn)
                
                title = res[0] if res else "Movie"
                qualities = get_all_movie_qualities(movie_id)
                
                # Context update karo
                context.user_data['selected_movie_data'] = {
                    'id': movie_id,
                    'title': title,
                    'qualities': qualities
                }
            else:
                title = movie_data['title']
                qualities = movie_data['qualities']

            # 👇 FIX: Agar Season select kiya tha, toh pehle wapas files filter karo page badalne se pehle
            if selected_season:
                filtered_qualities = []
                for file_data in qualities:
                    extra_info = file_data[5] if len(file_data) > 5 else ""
                    if extract_season_name(extra_info) == selected_season:
                        filtered_qualities.append(file_data)
                
                keyboard_markup = create_quality_selection_keyboard(movie_id, title, filtered_qualities, page=page, season=selected_season)
                
                # Season wale Next/Back mein bhi Upar "Back to Seasons" daalna zaroori hai
                keyboard_list = list(keyboard_markup.inline_keyboard)
                keyboard_list.insert(0, [InlineKeyboardButton("🔙 Back to Seasons", callback_data=f"movie_{movie_id}")])
                keyboard = InlineKeyboardMarkup(keyboard_list)
            else:
                # Normal Movie Pagination
                keyboard = create_quality_selection_keyboard(movie_id, title, qualities, page=page)
            
            # Sirf buttons update karein (Text same rahega)
            await query.edit_message_reply_markup(reply_markup=keyboard)
            return
        
        elif query.data.startswith("admin_fulfill_"):
            parts = query.data.split('_', 3)
            user_id = int(parts[2])
            movie_title = parts[3]

            conn = get_db_connection()
            if conn:
                cur = conn.cursor()
                cur.execute("SELECT id, url, file_id FROM movies WHERE title = %s LIMIT 1", (movie_title,))
                movie_data = cur.fetchone()

                if movie_data:
                    movie_id, url, file_id = movie_data
                    value_to_send = file_id if file_id else url
                    num_notified = await notify_users_for_movie(context, movie_title, value_to_send)

                    await query.edit_message_text(
                        f"✅ FULFILLED: Movie '{movie_title}' updated and user (ID: {user_id}) notified ({num_notified} total users).",
                        parse_mode='Markdown'
                    )
                else:
                    await query.edit_message_text(f"❌ ERROR: Movie '{movie_title}' not found in the `movies` table. Please add it first.", parse_mode='Markdown')

                cur.close()
                close_db_connection(conn)
            else:
                await query.edit_message_text("❌ Database error during fulfillment.")

        elif query.data.startswith("admin_delete_"):
            parts = query.data.split('_', 3)
            user_id = int(parts[2])
            movie_title = parts[3]

            conn = get_db_connection()
            if conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM user_requests WHERE user_id = %s AND movie_title = %s", (user_id, movie_title))
                conn.commit()
                cur.close()
                close_db_connection(conn)
                await query.edit_message_text(f"❌ DELETED: Request for '{movie_title}' from User ID {user_id} removed.", parse_mode='Markdown')
            else:
                await query.edit_message_text("❌ Database error during deletion.")

        # ==================== QUALITY SELECTION ====================
        # ==================== QUALITY SELECTION ====================
        elif query.data.startswith("quality_"):
            parts = query.data.split('_')
            movie_id = int(parts[1])
            selected_quality = parts[2]

            movie_data = context.user_data.get('selected_movie_data')

            if not movie_data or movie_data.get('id') != movie_id:
                qualities = get_all_movie_qualities(movie_id)
                movie_data = {'id': movie_id, 'title': 'Movie', 'qualities': qualities}

            if not movie_data or 'qualities' not in movie_data:
                await query.edit_message_text("❌ Error: Could not retrieve movie data. Please search again.")
                return

            chosen_file = None
            
            # 👇 NAYA BULLETPROOF CODE 👇
            for file_data in movie_data['qualities']:
                quality = file_data[0]
                url = file_data[1]
                file_id = file_data[2]
                
                if quality == selected_quality:
                    chosen_file = {'url': url, 'file_id': file_id}
                    break

            if not chosen_file:
                await query.edit_message_text("❌ Error fetching the file for that quality.")
                return

            title = movie_data['title']
            await query.edit_message_text(f"Sending **{title}**...", parse_mode='Markdown')

            await send_movie_to_user(
                update,
                context,
                movie_id,
                title,
                chosen_file['url'],
                chosen_file['file_id']
            )

            if 'selected_movie_data' in context.user_data:
                del context.user_data['selected_movie_data']

        # ==================== PAGINATION ====================
        elif query.data.startswith("page_"):
            # Strip _u{user_id} suffix if present (group buttons)
            page_data_part = re.sub(r'_u\d+$', '', query.data)
            page = int(page_data_part.replace("page_", ""))

            if 'search_results' not in context.user_data:
                await query.edit_message_text("❌ Search results expired. Please search again.")
                return

            movies = context.user_data['search_results']
            search_query = context.user_data.get('search_query', 'your search')

            # Group me user_id wapas pass karo taaki naye page ke buttons bhi locked rahen
            requester_id = None
            if "_u" in query.data:
                try:
                    requester_id = int(query.data.split("_u")[-1])
                except (ValueError, IndexError):
                    pass

            selection_text = f"🎬 **Found {len(movies)} movies matching '{search_query}'**\n\nPlease select the movie you want:"
            keyboard = create_movie_selection_keyboard(movies, page=page, requester_id=requester_id)

            await query.edit_message_text(
                selection_text,
                reply_markup=keyboard,
                parse_mode='Markdown'
            )

        elif query.data.startswith("cancel_selection"):
            await query.edit_message_text("❌ Selection cancelled.")
            keys_to_clear = ['search_results', 'search_query', 'selected_movie_data', 'awaiting_request', 'pending_request']
            for key in keys_to_clear:
                if key in context.user_data:
                    del context.user_data[key]

        
        # ==================== DOWNLOAD SHORTCUT ====================
        elif query.data.startswith("download_"):
            movie_title = query.data.replace("download_", "")

            conn = get_db_connection()
            if not conn:
                await query.answer("❌ Database connection failed.", show_alert=True)
                return

            cur = conn.cursor()
            cur.execute("SELECT id, title, url, file_id FROM movies WHERE title ILIKE %s LIMIT 1", (f'%{movie_title}%',))
            movie = cur.fetchone()
            cur.close()
            close_db_connection(conn)

            if movie:
                movie_id, title, url, file_id = movie
                await send_movie_to_user(update, context, movie_id, title, url, file_id)
            else:
                await query.answer("❌ Movie not found.", show_alert=True)

    except Exception as e:
        logger.error(f"Error in button callback: {e}")
        try:
            await query.answer(f"❌ Error: {str(e)}", show_alert=True)
        except:
            pass

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel the current operation"""
    msg = await update.message.reply_text("Operation cancelled.", reply_markup=get_main_keyboard())
    track_message_for_deletion(context, update.effective_chat.id, msg.message_id, USER_TEXT_DELETE_SECONDS)
    return MAIN_MENU

# ==================== NEW MULTI-CHANNEL BACKUP FUNCTIONS ====================

def get_storage_channels():
    """Load channel list from .env"""
    channels_str = os.environ.get('STORAGE_CHANNELS', '')
    return [int(c.strip()) for c in channels_str.split(',') if c.strip()]

# ==================== 🔄 THEATER PRINT AUTO-UPGRADE SYSTEM ====================

# 4-Level Hierarchy: Higher level aane par lower level auto-delete ho jayega
_SOURCE_LEVELS = {
    # Level 1 — Camera Prints (सबसे घटिया)
    1: ['cam', 'camrip', 'hdcam', 'hd-cam', 'hqcam', 'hq-cam',
        'telecine', 'tc', 'telesync', 'ts'],
    # Level 2 — Theater Prints with Better Audio
    2: ['hdts', 'hd-ts', 'predvd', 'pre-dvd', 'dvdscr', 'dvdscreener',
        'scr', 'screener', 'line', 'line audio', 'hdtc', 'hd-tc', 'hq-hdtc'],
    # Level 3 — Good Digital but Compressed/TV
    3: ['hdrip', 'webrip', 'web-rip', 'hc-webrip', 'hdtv'],
    # Level 4 — Ultimate OTT / Disc Quality
    4: ['web-dl', 'webdl', 'bluray', 'blu-ray', 'bdrip', 'brrip',
        'ds4k', 'remux'],
}

# Reverse lookup: keyword → level (for fast detection)
_KEYWORD_TO_LEVEL = {}
for _lvl, _keywords in _SOURCE_LEVELS.items():
    for _kw in _keywords:
        _KEYWORD_TO_LEVEL[_kw] = _lvl


def get_source_level(text):
    """
    File name ya quality label se source level detect karta hai.
    Level 1 = CamRip (सबसे घटिया)
    Level 2 = HDTS/PreDVD (थोड़ा अच्छा theater print)
    Level 3 = HDRip/WEBRip (Good digital, compressed)
    Level 4 = WEB-DL/BluRay (Ultimate OTT/Disc)
    Returns: 0 (unknown), 1, 2, 3, or 4
    """
    if not text:
        return 0
    text_lower = text.lower()

    # Longer keywords pehle check karo (e.g., 'web-dl' before 'web')
    # Sorted by length descending for greedy matching
    for kw in sorted(_KEYWORD_TO_LEVEL.keys(), key=len, reverse=True):
        # Word boundary check using regex for accuracy
        if re.search(r'(?:^|[\s._\-\[\(])' + re.escape(kw) + r'(?:$|[\s._\-\]\)])', text_lower):
            return _KEYWORD_TO_LEVEL[kw]

    return 0  # Unknown source


def get_resolution(text):
    """
    File name ya quality label se resolution extract karta hai.
    Returns: '2160p', '1080p', '720p', '576p', '480p', '360p', ya 'unknown'
    """
    if not text:
        return 'unknown'
    text_lower = text.lower()

    # 4K / 2160p check
    if '2160p' in text_lower or '4k' in text_lower:
        return '2160p'
    # Standard resolutions (higher to lower)
    for res in ['1080p', '720p', '576p', '480p', '360p']:
        if res in text_lower:
            return res

    return 'unknown'


def get_file_content_scope(extra_info):
    """Return the exact movie/season/episode unit represented by a file row."""
    text = str(extra_info or '').upper()
    season_match = re.search(r'\b(?:S|SEASON)\s*0*(\d{1,2})(?=\s|E|EP|$)', text)
    episode_match = re.search(
        r'(?:\b(?:E|EP|EPISODE)|(?<=\d)E)\s*0*(\d{1,3})(?:\s*(?:-|~|TO)\s*(?:E|EP|EPISODE)?\s*0*(\d{1,3}))?\b',
        text
    )
    part_match = re.search(r'\b(?:P|PART)\s*0*(\d{1,3})\b', text)

    if season_match and episode_match:
        start = int(episode_match.group(1))
        end = int(episode_match.group(2) or start)
        return f"series:s{int(season_match.group(1)):02d}:e{start:03d}-{end:03d}"
    if season_match:
        return f"series:s{int(season_match.group(1)):02d}:season-pack"
    if part_match:
        return f"part:{int(part_match.group(1)):03d}"
    return "movie"


def is_downgrade(movie_id, new_quality_label, new_extra_info, conn):
    """
    Theatre-print shield, scoped to the exact movie/episode.
    Digital sources (HDRip, WEBRip, WEB-DL, BluRay) are additive; a theatre
    print is rejected only when a better theatre/digital source exists for the
    same content scope.
    """
    new_level = get_source_level(new_quality_label)
    # Unknown and digital files are additive; never block them.
    if new_level == 0 or new_level >= 3:
        return False, None

    new_scope = get_file_content_scope(new_extra_info)

    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT quality, extra_info FROM movie_files WHERE movie_id = %s",
            (movie_id,)
        )
        existing_files = cur.fetchall()
        cur.close()

        for row in existing_files:
            old_label, old_extra_info = row
            if get_file_content_scope(old_extra_info) != new_scope:
                continue
            old_level = get_source_level(old_label)

            # Digital makes theatre prints obsolete. Before that, don't add a
            # worse theatre print after a better theatre print for this episode.
            if old_level >= 3 or old_level > new_level:
                logger.info(
                    f"🛡️ Anti-Downgrade BLOCKED: movie_id={movie_id} | "
                    f"Tried='{new_quality_label}' (L{new_level}) | "
                    f"DB has='{old_label}' (L{old_level}) | scope={new_scope}"
                )
                return True, old_label

        return False, None

    except Exception as e:
        logger.error(f"❌ Anti-Downgrade check error: {e}")
        return False, None  # Error par allow kar do (safe side)


def auto_upgrade_delete(movie_id, new_quality_label, new_extra_info, conn):
    """
    Digital-release cleanup, scoped to the exact movie/episode.
    When HDRip/WEBRip/WEB-DL/BluRay arrives, remove only CAM/HDTC/HDTS-style
    theatre prints for that same scope. Digital qualities always coexist.
    Returns: (deleted_count, deleted_labels)
    """
    new_level = get_source_level(new_quality_label)

    if new_level < 3:
        return 0, []

    new_scope = get_file_content_scope(new_extra_info)

    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, quality, extra_info FROM movie_files WHERE movie_id = %s",
            (movie_id,)
        )
        existing_files = cur.fetchall()

        ids_to_delete = []
        labels_to_delete = []
        for row in existing_files:
            file_row_id, old_label, old_extra_info = row
            if get_file_content_scope(old_extra_info) != new_scope:
                continue
            old_level = get_source_level(old_label)

            # Theatre prints are disposable only after a digital release for
            # this exact movie/season/episode has been saved.
            if old_level in (1, 2):
                ids_to_delete.append(file_row_id)
                labels_to_delete.append(old_label)

        deleted_count = 0
        if ids_to_delete:
            placeholders = ','.join(['%s'] * len(ids_to_delete))
            cur.execute(
                f"DELETE FROM movie_files WHERE movie_id = %s AND id IN ({placeholders})",
                [movie_id] + ids_to_delete
            )
            deleted_count = cur.rowcount
            conn.commit()
            logger.info(
                f"🔄 Auto-Upgrade: movie_id={movie_id} | "
                f"New='{new_quality_label}' (L{new_level}) | scope={new_scope} | "
                f"Deleted {deleted_count} theatre print(s): {labels_to_delete}"
            )

        cur.close()
        return deleted_count, labels_to_delete

    except Exception as e:
        logger.error(f"❌ Auto-Upgrade Error for movie_id={movie_id}: {e}")
        return 0, []


def upsert_movie_file(conn, movie_id, label, file_size_str, main_url, backup_map_json, f_lang, f_extra, file_unique_id):
    """Idempotently upsert a Telegram file without ever changing its parent movie."""
    cur = conn.cursor()
    try:
        if file_unique_id:
            cur.execute(
                """
                INSERT INTO movie_files
                    (movie_id, quality, file_size, url, backup_map, languages, extra_info, file_unique_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (file_unique_id) DO UPDATE SET
                    quality = EXCLUDED.quality,
                    file_id = NULL,
                    file_size = EXCLUDED.file_size,
                    url = EXCLUDED.url,
                    backup_map = EXCLUDED.backup_map,
                    languages = EXCLUDED.languages,
                    extra_info = EXCLUDED.extra_info
                WHERE movie_files.movie_id = EXCLUDED.movie_id
                RETURNING id
                """,
                (movie_id, label, file_size_str, main_url, backup_map_json, f_lang, f_extra, file_unique_id),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(
                    "Telegram file identity is already attached to a different movie"
                )
        else:
            cur.execute(
                """
                INSERT INTO movie_files
                    (movie_id, quality, file_size, url, backup_map, languages, extra_info, file_unique_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NULL)
                RETURNING id
                """,
                (movie_id, label, file_size_str, main_url, backup_map_json, f_lang, f_extra),
            )
            row = cur.fetchone()
        conn.commit()
        return row[0]
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def generate_quality_label(file_name, file_size_str="", ai_language=""):
    """
    File name se CLEAN quality label generate karta hai.
    Returns ONLY: resolution + source + episode info.
    Example: '1080p WEB-DL', '720p CAMRip', 'S01E03 480p HDRip'
    
    NOTE: file_size_str aur ai_language params backward compat ke liye hain,
    but ye quality label me INCLUDE nahi hote. Ye apne dedicated DB columns
    (file_size, languages) me separately store hone chahiye.
    """
    # Pehle episode format ko hamesha ke liye theek karo (S07E12 22 -> S07E12-22)
    name_lower = normalize_episodes(file_name.lower())
    quality = "HD"

    # 1. Detect Quality (576p bhi add kiya)
    if "4k" in name_lower or "2160p" in name_lower: quality = "4K"
    elif "1080p" in name_lower: quality = "1080p"
    elif "720p" in name_lower:  quality = "720p"
    elif "576p" in name_lower:  quality = "576p"
    elif "480p" in name_lower:  quality = "480p"
    elif "360p" in name_lower:  quality = "360p"
    elif "cam" in name_lower or "rip" in name_lower: quality = "CamRip"

    # 2. Detect Source (ALL levels — Level 4 to Level 1, longest keywords first)
    source_tag = ""
    # Level 4 — Ultimate OTT / Disc
    if "web-dl" in name_lower or "webdl" in name_lower:   source_tag = " WEB-DL"
    elif "bluray" in name_lower or "blu-ray" in name_lower: source_tag = " BluRay"
    elif "bdrip" in name_lower or "brrip" in name_lower:  source_tag = " BluRay"
    elif "remux" in name_lower:                            source_tag = " Remux"
    # Level 3 — Good Digital
    elif "webrip" in name_lower or "web-rip" in name_lower: source_tag = " WEBRip"
    elif "hc-webrip" in name_lower:                        source_tag = " WEBRip"
    elif "hdrip" in name_lower:                            source_tag = " HDRip"
    elif "hdtv" in name_lower:                             source_tag = " HDTV"
    # Level 2 — Theater Print with Better Audio (check BEFORE Level 1)
    elif "hq-hdtc" in name_lower or "hqhdtc" in name_lower: source_tag = " HDTC"
    elif "hdtc" in name_lower or "hd-tc" in name_lower:   source_tag = " HDTC"
    elif "hdts" in name_lower or "hd-ts" in name_lower:   source_tag = " HDTS"
    elif "predvd" in name_lower or "pre-dvd" in name_lower: source_tag = " PreDVD"
    elif "dvdscr" in name_lower or "screener" in name_lower: source_tag = " DVDScr"
    # Level 1 — Camera Prints (सबसे घटिया)
    elif "hdcam" in name_lower or "hd-cam" in name_lower: source_tag = " HDCAM"
    elif "hqcam" in name_lower or "hq-cam" in name_lower: source_tag = " HDCAM"
    elif "camrip" in name_lower:                           source_tag = " CAMRip"
    elif "telecine" in name_lower or "tc" in name_lower.split(): source_tag = " CAMRip"
    elif "telesync" in name_lower or "ts" in name_lower.split(): source_tag = " CAMRip"
    elif "cam" in name_lower.split():                      source_tag = " CAMRip"

    # 3. Detect Series (S01, S02, S01E01, S01P01, Season 1, etc.)
    # \b used taaki 1080p 10bit ka 'p 10' E10 na ban jaye
    season_match = re.search(
        r'(?i)(\bs\d{1,2}\s*(?:[ep]\d{1,3})?'
        r'|\bs\d{1,2}\s*\[?(?:e|ep|episode|p|part)\s*\d{1,3}'
        r'|\b\[?(?:e|ep|episode|p|part)\s*\d{1,3}(?:\s*(?:[-~_]|to)\s*(?:e|ep|episode|p|part)?\s*\d{1,3})?\]?'
        r'|\bseason\s?\d+\b)',
        name_lower
    )

    if season_match:
        episode_tag = season_match.group(0).upper().replace("P", "E").strip()
        # Episode/Season should NOT go into quality string, it goes to extra_info.
        pass

    # ✅ CLEAN: Resolution + Source ONLY
    return f"{quality}{source_tag}".strip()

def get_readable_file_size(size_in_bytes):
    """Converts bytes to readable format (MB, GB)"""
    try:
        if not size_in_bytes: return "N/A"
        size = int(size_in_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
    except Exception:
        return "Unknown"
    return "Unknown"

# ============================================================================
# 🎬 BATCH ID COMMAND (Fully Automatic via TMDB/IMDb)
# ============================================================================
async def batch_id_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    if not context.args:
        await update.message.reply_text("❌ Usage: `/batchid tt1234567`")
        return
        
    imdb_id = context.args[0].strip()
    status_msg = await update.message.reply_text(f"⏳ Extracting all details for {imdb_id}...")
    
    try:
        # 1. Metadata + Poster
        data = await run_async(fetch_movie_metadata, imdb_id)
        if not data:
            await status_msg.edit_text("❌ IMDb से डेटा नहीं मिला। API Key चेक करें।")
            return
        
        title, year, poster, genre, imdb_id_f, rating, plot, category, seasons_data = data
        if not is_safe_canonical_title(title):
            await status_msg.edit_text(
                "❌ Provider returned an invalid content title; no movie row was created."
            )
            return
        seasons_data = normalize_seasons_data(seasons_data)
        tmdb_id = await run_async(
            resolve_tmdb_id_from_imdb, imdb_id_f, category
        )
        category, content_type = normalize_catalog_labels(
            category=category,
            language="Hindi",
            extra_info=" ".join(seasons_data.keys()),
            genre=genre,
            title=title,
        )
        
        # 2. Cast/Stars लाना
        cast_str = await run_async(fetch_cast_from_imdb, imdb_id_f, 5)
        
        # 3. DB Insertion (All Fields)
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 🛑 "cast" quoted and year is integer
        # 🎯 NAYA LOGIC: Title ki jagah IMDb ID par conflict check karega
        import json
        trailer_key = await run_async(
            resolve_trailer_key, title, year, imdb_id_f, category, title
        )
        movie_values = (
            title, imdb_id_f, tmdb_id, poster, year, genre, rating, plot, category,
            content_type, "Hindi", cast_str,
            json.dumps(seasons_data) if seasons_data else '{}', trailer_key,
        )
        existing_movie_id = _find_movie_by_provider_identity(
            cur, imdb_id_f, tmdb_id
        )
        if existing_movie_id:
            cur.execute("""
                UPDATE movies
                SET title = %s, imdb_id = %s, tmdb_id = %s, poster_url = %s,
                    year = %s, genre = %s, rating = %s, description = %s,
                    category = %s, content_type = %s, language = %s,
                    "cast" = %s, seasons_data = %s,
                    trailer_key = COALESCE(%s, trailer_key)
                WHERE id = %s
                RETURNING id
            """, (*movie_values, existing_movie_id))
        else:
            cur.execute("""
                INSERT INTO movies
                    (title, url, imdb_id, tmdb_id, poster_url, year, genre, rating,
                     description, category, content_type, language, "cast",
                     seasons_data, trailer_key)
                VALUES (%s, '', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, movie_values)
        
        movie_id = cur.fetchone()[0]
        
        # 👇 NAYA: Database se check karein ki kya pehle se files hain
        cur.execute("SELECT COUNT(*) FROM movie_files WHERE movie_id = %s", (movie_id,))
        file_count = cur.fetchone()[0]
        
        conn.commit()
        cur.close()
        close_db_connection(conn)

        # 4. Start Batch Session
        BATCH_SESSION.update({
            'active': True, 'movie_id': movie_id, 'movie_title': title, 
            'file_count': file_count, 'admin_id': update.effective_user.id, 
            'language': 'Hindi', 'category': category,
            'parsed_identity': {},
            'resolver_method': 'provider_identity',
            'confidence': 1,
            'imdb_id': imdb_id_f,
            'tmdb_id': tmdb_id,
        })

        # 5. Success Message with Details
        success_msg = (
            f"✅ **Dada! Metadata Fetched Successfully**\n\n"
            f"🎬 **Title:** `{title}`\n"
            f"📅 **Year:** {year}\n"
            f"🎭 **Genre:** {genre}\n"
            f"⭐️ **Rating:** {rating}\n"
            f"🏷️ **Category:** {category}\n"
            f"👥 **Cast:** {cast_str}\n\n"
        )
        
        if file_count > 0:
            success_msg += f"⚠️ **Old Files Found:** {file_count} (Aap inhe delete kar sakte hain ya nayi add kar sakte hain)\n\n"
            
        success_msg += f"🚀 **अब फाइल्स भेजें, फिर /done लिखें।**"
        
        # 👇 NAYA: Button Add Karein (Agar files hain tabhi delete button aayega)
        # 👇 NAYA: Button Add Karein (Agar files hain tabhi delete button aayega)
        keyboard = []
        if file_count > 0:
            keyboard.append([InlineKeyboardButton("🗑️ Delete OLD Files", callback_data=f"clearfiles_{movie_id}")])
        keyboard.append([InlineKeyboardButton("❌ Cancel Batch", callback_data="cancel_batch")])
        
        await status_msg.edit_text(success_msg, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))

    # ✅ BAS YE 3 LINES YAHAN ADD KARNI HAIN 👇
    except Exception as e:
        print(f"Error in batch_id_command: {e}")
        await status_msg.edit_text(f"❌ Kuch galat ho gaya: {e}")


# ============================================================================
# ✍️ BATCH MANUAL COMMAND (For Custom Names & Details)
# ============================================================================

async def batch_add_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id): return

    if not context.args: 
        await update.message.reply_text(
            "❌ **Galat Format!** Aise use karein:\n\n"
            "`/batch Movie Name, Year, Language, Genre, Category`\n\n"
            "**Example:**\n"
            "`/batch Pink Bra, 2023, Hindi, Adult, Web Series`", 
            parse_mode='Markdown'
        )
        return

    # 1. Parsing Custom Format
    raw_text = " ".join(context.args)
    parts = [p.strip() for p in raw_text.split(',')]
    
    title = parts[0] if len(parts) > 0 else "Unknown Title"
    if not is_safe_canonical_title(title):
        await update.message.reply_text(
            "❌ Use a canonical movie/series title without episode or release metadata."
        )
        return
    year = parts[1] if len(parts) > 1 else ""
    language = parts[2] if len(parts) > 2 else "Hindi"
    genre = parts[3] if len(parts) > 3 else "Adult, Drama"
    category = parts[4] if len(parts) > 4 else "Web Series"
    category, content_type = normalize_catalog_labels(
        category=category,
        language=language,
        extra_info="",
        genre=genre,
        title=title,
    )
    
    rating = "N/A"
    plot = "Watch exclusive content on FlimfyBox Premium."
    poster_url = None
    
    # Retrieve stored IMDb ID and cast from user_data (if batchid was used)
    imdb_id = context.user_data.pop('batch_imdb_id', None)
    cast_str = context.user_data.pop('batch_cast', None)

    status_msg = await update.message.reply_text(f"⏳ Saving '{title}' to Database...", parse_mode='Markdown')

    conn = get_db_connection()
    if not conn: return
    
    try:
        cur = conn.cursor()
        
        # ✅ FIXED: Quote "cast" because it's a reserved keyword
        trailer_key = await run_async(
            resolve_trailer_key, title, year, imdb_id, category, title
        )
        tmdb_id = await run_async(resolve_tmdb_id_from_imdb, imdb_id, category)
        existing_id = _find_movie_by_provider_identity(cur, imdb_id, tmdb_id)
        values = (
            title, imdb_id, tmdb_id, poster_url, year, genre, rating, plot,
            category, content_type, language, cast_str, trailer_key,
        )
        if existing_id:
            cur.execute(
                """
                UPDATE movies
                SET title = %s, imdb_id = COALESCE(%s, movies.imdb_id),
                    tmdb_id = COALESCE(%s, movies.tmdb_id),
                    poster_url = COALESCE(%s, poster_url), year = %s,
                    genre = %s, rating = %s, description = %s,
                    category = %s, content_type = %s, language = %s,
                    "cast" = COALESCE(%s, "cast"),
                    trailer_key = COALESCE(%s, trailer_key)
                WHERE id = %s
                RETURNING id
                """,
                (*values, existing_id),
            )
        else:
            cur.execute(
                """
                INSERT INTO movies
                    (title, url, imdb_id, tmdb_id, poster_url, year, genre,
                     rating, description, category, content_type, language,
                     "cast", trailer_key)
                VALUES (%s, '', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                values,
            )
        movie_id = cur.fetchone()[0]
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM movie_files WHERE movie_id = %s", (movie_id,))
        file_count = cur.fetchone()[0]
        cur.close()

        BATCH_SESSION.update({
            'active': True,
            'movie_id': movie_id,
            'movie_title': title,
            'file_count': 0,
            'admin_id': user_id,
            'language': language,
            'category': category,
            'parsed_identity': {},
            'resolver_method': 'manual_admin_title',
            'confidence': 1,
            'imdb_id': imdb_id,
            'tmdb_id': tmdb_id,
        })

        # Show cast in confirmation message (if any)
        cast_display = f"👥 **Cast:** {cast_str}\n" if cast_str else ""
        msg_text = (
            f"✅ **Batch Custom Mode Started!**\n\n"
            f"🎬 **Title:** {title}\n"
            f"📅 **Year:** {year}\n"
            f"🎭 **Genre:** {genre}\n"
            f"🗣️ **Language:** {language}\n"
            f"🏷️ **Category:** {category}\n"
            f"{cast_display}"
            f"🚀 **Step 1:** Ab movie/series ki Files (Video/Doc) bhejo.\n"
            f"🖼️ **Step 2:** Poster ke liye koi bhi ek Image bhej do.\n"
            f"✅ **Step 3:** Jab sab ho jaye to `/done` bhejo."
        )

        keyboard = []
        if file_count > 0:
            keyboard.append([InlineKeyboardButton("🗑️ Delete OLD Files", callback_data=f"clearfiles_{movie_id}")])
        keyboard.append([InlineKeyboardButton("❌ Cancel Batch", callback_data="cancel_batch")])
        
        await status_msg.edit_text(msg_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))

    except Exception as e:
        logger.error(f"Batch Error: {e}")
        await status_msg.edit_text(f"❌ DB Error: {e}")
    finally:
        if conn: close_db_connection(conn)

# ============================================================================
# 🚀 SUPER BATCH SYSTEM (Smart Grouping + Auto Post)
# ============================================================================

async def superbatch_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Super Batch shuru karega"""
    if update.effective_user.id not in ADMIN_IDS: return
    
    SUPER_BATCH_SESSION['active'] = True
    SUPER_BATCH_SESSION['admin_id'] = update.effective_user.id
    SUPER_BATCH_SESSION['files'] = []
    
    await update.message.reply_text(
        "🚀 **SUPER BATCH MODE ON!**\n\n"
        "👉 Ab aap ek sath 50-100 files (alag-alag movies ki) yahan forward kar dein.\n"
        "👉 Bot khud unhe movies ke hisaab se group karega.\n"
        "👉 Jab sab bhej dein, to type karein: `/superdone`",
        parse_mode='Markdown'
    )

# 🎵 SONG FILTER: Superbatch me kabhi-kabhi movie ke saath "Song.mkv" jaisi
# standalone video-song files bhi aa jaati hain, jo galti se movie/episode
# samajh ke DB me save aur channel pe auto-post ho jaati hain. Neeche wala
# helper aisi files ko pehchan ke superbatch collection se hi bahar rakh deta hai.
_SONG_FILENAME_PATTERN = re.compile(
    r'(?i)(?<![a-z0-9])('
    r'video[\s._-]*song|full[\s._-]*song|song[\s._-]*video|title[\s._-]*track|'
    r'lyrical(?:[\s._-]*video)?|jukebox|ost|soundtrack|item[\s._-]*song|'
    r'audio[\s._-]*song|music[\s._-]*video|song'
    r')(?![a-z0-9])'
)
_SONG_MAX_DURATION_SECONDS = 480  # 8 min — extended/unplugged songs bhi cover, real movie/episode kabhi itni chhoti nahi hoti


def _looks_like_song_file(message, record) -> bool:
    """
    True agar filename/caption me koi song-indicator keyword mile ("Video Song",
    "Jukebox", "OST", etc.) YA file bahut chhoti duration (< 5 min) ki video/audio ho —
    dono hi cases me ye asli movie/episode nahi, balki ek standalone song lagti hai.
    """
    text = f"{record.get('file_name') or ''} {record.get('caption') or ''}"
    if _SONG_FILENAME_PATTERN.search(text):
        return True

    media_with_duration = getattr(message, "video", None) or getattr(message, "audio", None)
    duration = getattr(media_with_duration, "duration", 0) or 0
    if duration and duration < _SONG_MAX_DURATION_SECONDS:
        return True

    return False


async def _collect_superbatch_file(message):
    """Telegram file ka raw data + local evidence ek consistent record mein collect karta hai."""
    if not message or not (message.document or message.video or message.audio):
        return None

    media = message.document or message.video or message.audio
    evidence = await extract_same_file_evidence(message)
    thumb = getattr(media, "thumbnail", None) or getattr(media, "thumb", None)

    return {
        "file_id": getattr(media, "file_id", None),
        "file_unique_id": getattr(media, "file_unique_id", None),
        "file_name": evidence["filename_raw"] or getattr(media, "file_name", None) or "File",
        "file_size": getattr(media, "file_size", 0) or 0,
        "caption": evidence["caption_raw"],
        "thumb_id": getattr(thumb, "file_id", None) if thumb else None,
        "message_obj": message,
        "caption_evidence": evidence["caption_evidence"],
        "filename_evidence": evidence["filename_evidence"],
        "forward_source": evidence.get("forward_source"),
    }


async def superbatch_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Compatibility listener; active wiring pm_file_listener ko use karti hai."""
    if not SUPER_BATCH_SESSION.get('active') or update.effective_user.id != SUPER_BATCH_SESSION.get('admin_id'):
        return
    record = await _collect_superbatch_file(update.effective_message)
    if not record:
        return
    SUPER_BATCH_SESSION['files'].append(record)
    count = len(SUPER_BATCH_SESSION['files'])
    if count % 10 == 0:
        await update.effective_message.reply_text(f"📥 Received {count} files so far...")


async def superbatch_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Superbatch ka kaam:
      1. Files ko movie ke hisaab se group karo
      2. Har movie ke liye _core_movie_processor (Phase 1) → BATCH_SESSION set karo
      3. Har file ke liye _pm_save_file (Phase 2) — pm_file_listener ka exact same code
      4. Post karo channel pe
    """
    if not SUPER_BATCH_SESSION['active'] or update.effective_user.id != SUPER_BATCH_SESSION['admin_id']:
        return

    SUPER_BATCH_SESSION['active'] = False
    files = SUPER_BATCH_SESSION['files']
    SUPER_BATCH_SESSION['files'] = []

    if not files:
        await update.message.reply_text("❌ Koi file nahi mili!")
        return

    status_msg = await update.message.reply_text(
        f"🔄 **{len(files)} files group ho rahi hain...**", parse_mode='Markdown'
    )

    # ── STEP 1: CONSERVATIVE LOCAL GROUPING ─────────────────────────────
    # Gemini grouping ke baad identity reconcile karega. Isliye grouping yahan
    # exact/very-high-confidence aur ambiguity-safe rules se hoti hai.
    grouped_movies = _build_superbatch_groups(files)

    total_movies = len(grouped_movies)
    await status_msg.edit_text(f"✅ **Files grouped into {total_movies} unique movies!**\n\n🚀 Auto-Processing & Posting starts now...", parse_mode='Markdown')

    success_movies = 0
    total_files_saved = 0           # 👈 NAYA: Kitni files save hui uski ginti
    movies_posted_list = []         # 👈 NAYA: Jo movies post hui unki list
    channels = get_storage_channels()

    for i, (group_key, movie_files) in enumerate(grouped_movies.items(), 1):
        try:
            representative = _select_representative_file(movie_files)
            display_name = representative.get('display_title', group_key)
            await status_msg.edit_text(f"⚙️ Processing Movie {i}/{total_movies}...\n🎬 Name: `{display_name}`")

            image_bytes = None
            if representative.get('thumb_id'):
                try:
                    # Thumbnail multimodal analysis abhi intentionally disabled hai.
                    image_bytes = None
                except Exception:
                    image_bytes = None

            # Exactly ONE Gemini reconciliation per finalized group, using the
            # most complete same-file caption+filename evidence packet.
            reconciled_data = await reconcile_evidence_with_gemini(
                representative.get('caption_evidence', {}),
                representative.get('filename_evidence', {}),
                caption_raw=representative.get('caption', ''),
                filename_raw=representative.get('file_name', ''),
                forward_source=representative.get('forward_source'),
            )

            result = await _core_movie_processor(
                representative.get('caption') or representative.get('file_name') or display_name,
                image_bytes,
                reconciled_data=reconciled_data,
                raw_caption=representative.get('caption', ''),
                raw_filename=representative.get('file_name', ''),
                file_unique_id=representative.get('file_unique_id'),
            )

            if not result:
                logger.warning(f"Superbatch: '{display_name}' process nahi ho paya, skip kar raha hoon.")
                continue

            movie_id   = result['movie_id']
            title      = result['title']
            year       = result['year']
            genre      = result['genre']
            rating     = result['rating']
            plot       = result['plot']
            category   = result['category']
            movie_lang = result['movie_lang']
            poster_url = result['poster_url']
            imdb_id    = result['imdb_id']

            # ── STEP 2: BATCH_SESSION set karo (Phase 1 jaisa) ──────────────────
            BATCH_SESSION.update({
                'active':      True,
                'movie_id':    movie_id,
                'movie_title': title,
                'file_count':  0,
                'admin_id':    ADMIN_USER_ID,
                'year':        str(year) if year else '',
                'category':    category,
                'language':    movie_lang,
                'parsed_identity': result.get('parsed_identity', {}),
                'resolver_method': result.get('resolver_method', ''),
                'confidence': result.get('confidence', 0),
                'imdb_id': result.get('imdb_id'),
                'tmdb_id': result.get('tmdb_id'),
            })

            # ── STEP 3: Har file ke liye unified Phase 2 saver ───────────────
            saved_labels = []
            try:
                for f in movie_files:
                    label = await _pm_save_file(f['message_obj'], context)
                    if label:
                        saved_labels.append(label)
                    await asyncio.sleep(0.5)
            finally:
                # Koi file error kare tab bhi next movie ke liye session leak na ho.
                BATCH_SESSION.update({'active': False, 'movie_id': None, 'movie_title': None,
                                      'file_count': 0, 'admin_id': None, 'year': '',
                                      'category': '', 'language': ''})

            if not saved_labels:
                logger.warning(f"Superbatch: '{display_name}' — koi file save nahi ho paya")
                continue
            total_files_saved += len(saved_labels)

            # 🚫 AI Alias Generation OFF — Flask Web App mein Google Suggest + pg_trgm handles typos
            # generate_basic_aliases() bhi hata diya — DB clean rahega
            aliases = []
            alias_count = 0
            
            # --- POSTER PROCESSING (Landscape Blur Effect) ---
            raw_photo = poster_url if (poster_url and poster_url != 'N/A' and poster_url.startswith('http')) else None
            
            if not raw_photo and image_bytes:
                raw_photo = image_bytes
                
            # 👇 NAYA LOGIC: अगर ओरिजिनल इमेज (poster) नहीं मिली, तो इस मूवी को पोस्ट मत करो
            if not raw_photo:
                logger.warning(f"⚠️ Post Skipped: '{title}' के लिए कोई इमेज नहीं मिली।")
                continue  # 'continue' का मतलब है ये चैनल/फोरम में पोस्ट किए बिना अगली मूवी पर चला जायेगा
                
            # अगर असली इमेज है, तभी लैंडस्केप पोस्टर बनाओ
            photo_to_send = await make_landscape_poster(raw_photo)

            # 🛑 100% SAFE HTML CAPTION + RANDOM STYLES
            safe_rating = rating if rating else "N/A"
            safe_genre = genre if genre else "Unknown"

            res_set = set()
            for lbl in saved_labels:
                match = re.search(r'(\d{3,4}p)', lbl)
                if match:
                    res_set.add(match.group(1))
            # file_names se bhi try karo agar label mein nahi mila
            if not res_set:
                for f in movie_files:
                    match = re.search(r'(\d{3,4}p)', str(f.get('file_name', '')).lower())
                    if match:
                        res_set.add(match.group(1))
            res_list = sorted(list(res_set), key=lambda x: int(x.replace('p','')), reverse=True)
            dynamic_res = " | ".join(res_list) if res_list else "HD"
            
            safe_title = title.replace('<', '').replace('>', '')
            unicode_title = get_safe_font(safe_title)

            caption = (
                f"🎬 <b>{safe_title}</b>\n"
                f"➖➖➖➖➖➖➖➖➖➖\n"
                f"📅 <b>Date:</b> {year or 'N/A'}\n"
                f"⭐ <b>iMDB Rating:</b> {safe_rating}/10\n"
                f"🎭 <b>Genre:</b> {safe_genre}\n"
                f"🔊 <b>Language:</b> {movie_lang if movie_lang else 'Hindi'}\n"
                f"<b>Quality:</b> V2 HQ-HDTC {dynamic_res}\n"
                f"➖➖➖➖➖➖➖➖➖➖\n"
                f"<b>Update Channel:</b> <a href='https://t.me/FlimfyBoxBackUp'>Join BackUp</a>\n"
                f"👇 <b>Download Below</b> 👇"
            )

            # --- SECURE LINK & BUTTONS (As it was) ---
            secure_url = f"https://flimfybox-bot-yht0.onrender.com/watch/{movie_id}"

            post_keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("Get Now", url=secure_url)],
                [InlineKeyboardButton("Join Channel", url=FILMFYBOX_CHANNEL_URL)]
            ])

            # --- TARGET CHANNEL SELECTION (New System) ---
            cat_lower = str(category or "").lower()
            if "anime" in cat_lower or "cartoon" in cat_lower or "animation" in cat_lower:
                target_channels = [ANIME_CHANNEL_ID]
            else:
                target_channels = [ch.strip() for ch in os.environ.get('BROADCAST_CHANNELS', '').split(',') if ch.strip()]

            # 👇 YAHAN SE MAIN CHANNEL PAR BHEJNE KA ASLI LOGIC SHURU HOTA HAI 👇
            
            # --- THE "NINJA FIX" --- 
            # Pehle decide karte hain ki photo kya bhejna hai
            is_bytes = hasattr(photo_to_send, 'read')
            current_media = photo_to_send
            
            uploaded_file_id = None # Isme Telegram ki File ID store hogi

            # 👇 GLOBAL DUPLICATE CHECK — Agar kisi bhi channel me 7 din me post ho chuki hai to skip 👇
            if is_movie_posted_recently(movie_id, days=7):
                logger.info(f"⏭️ Skipping '{title}' (already posted within the last 7 days in some channel).")
                continue
            
            if target_channels:
                for chat_id_str in target_channels: 
                    try:
                        chat_id = int(chat_id_str)

                        sent_msg = None
                        
                        # Agar humare pass pehle se ID hai, toh file upload nahi karni
                        if uploaded_file_id:
                            sent_msg = await context.bot.send_photo(
                                chat_id=chat_id,
                                photo=uploaded_file_id, # 👈 Direct ID
                                caption=caption,
                                parse_mode='HTML',
                                reply_markup=post_keyboard
                            )
                        else:
                            # Pehli baar upload karna hai (Bytes se)
                            if is_bytes:
                                current_media.seek(0) # File pointer ko shuru me laao
                                
                            sent_msg = await context.bot.send_photo(
                                chat_id=chat_id,
                                photo=current_media, # 👈 Actual bytes
                                caption=caption,
                                parse_mode='HTML',
                                reply_markup=post_keyboard
                            )
                            # Ek baar upload hone ke baad, Telegram se permanent File ID save karlo
                            if sent_msg and sent_msg.photo:
                                uploaded_file_id = sent_msg.photo[-1].file_id 
                                
                        # ✅ DB me save karna zaroori hai taaki baad me /restore kaam kare
                        if sent_msg:
                            ch_name = sent_msg.chat.title if sent_msg.chat else "Unknown"
                            save_post_to_db(
                                movie_id, chat_id, sent_msg.message_id, "FlimfyBoxBot", caption, 
                                uploaded_file_id or poster_url, "photo", post_keyboard.to_dict(), None, "movies",
                                movie_name=title, imdb_id=imdb_id, tmdb_id=None, channel_name=ch_name
                            )
                            await asyncio.sleep(1.5)
                            
                    except Exception as e:
                        logger.error(f"❌ Failed to post in channel {chat_id_str}: {e}")

            success_movies += 1
            movies_posted_list.append(title) # 👈 NAYA: List me Title add kiya
            await asyncio.sleep(2) # Flood limit se bachne ke liye delay

        except Exception as e:
            logger.error(f"SuperBatch Movie Error: {e}")
            continue

    # 📝 NAYA: List format banana
    if movies_posted_list:
        posted_names = "\n".join([f"🔹 {name}" for name in movies_posted_list])
    else:
        posted_names = "Koyi nayi movie post nahi hui."

    # 🎉 NAYA: Final Message (HTML format me)
    final_text = (
        f"🎉 <b>SUPER BATCH COMPLETED!</b>\n\n"
        f"💾 <b>Total Files Saved in DB:</b> {total_files_saved}\n"
        f"🚀 <b>Movies/Series Auto-Posted:</b> {len(movies_posted_list)}/{total_movies}\n\n"
        f"<b>📑 Posted List:</b>\n{posted_names}"
    )

    await status_msg.edit_text(final_text, parse_mode='HTML')

    SUPER_BATCH_SESSION.update({'active': False, 'admin_id': None, 'files': []})


# ==============================================================================
# 🎯 CORE MOVIE PROCESSOR — PM FILE LISTENER KA DIL
# ==============================================================================
# Yeh function ek "engine" hai.
# pm_file_listener aur superbatch_done DONO isko call karte hain.
# Iska matlab: superbatch ko wahi accuracy milegi jo pm_file_listener ko milti hai.
#
# Flow: raw_text + image_bytes
#         → Gemini AI  (title, year, language, category extract)
#         → TMDB       (HD poster, genre, rating, plot)
#         → IMDb       (cast)
#         → DB INSERT  (pm_file_listener wala COMPLETE ON CONFLICT logic)
#         → Returns dict with movie_id aur saari details
# ==============================================================================
async def _core_movie_processor(
    raw_text: str,
    image_bytes: Optional[bytes] = None,
    reconciled_data: Optional[dict] = None,
    raw_caption: str = "",
    raw_filename: str = "",
    file_unique_id: Optional[str] = None,
) -> Optional[dict]:
    """
    Ek jagah se sab kuch. Returns movie dict ya None agar fail ho.
    """
    # --- STEP 1: Reconciled identity (ya legacy Gemini fallback) ---
    if reconciled_data:
        ai_data = reconciled_data
    else:
        ai_data = await get_movie_name_from_caption(raw_text, image_bytes)
        
    source, parsed_identity, identity_error = resolve_raw_content_identity(
        raw_caption, raw_filename, raw_text
    )
    if not parsed_identity:
        logger.warning("Ingestion held: no safe canonical title in supplied evidence")
        _record_pending_identity(
            file_unique_id,
            raw_caption or raw_text,
            raw_filename,
            ai_data,
            asdict(parse_content_identity(raw_caption or raw_filename or raw_text)),
            identity_error or "No plausible canonical content title was found",
        )
        return None

    movie_name = parsed_identity.canonical_title
    movie_year = ai_data.get("year", "") or parsed_identity.year or ""
    if not str(movie_year).isdigit():
        movie_year = parsed_identity.year or ""
    movie_lang = ai_data.get("language", "")
    extra_info = ai_data.get("extra_info", "")
    gemini_category = ai_data.get("category", "")
    if not is_safe_canonical_title(movie_name):
        return None

    if parsed_identity.has_episode_marker or parsed_identity.season_number is not None:
        extra_info = ""

    # --- STEP 2: TMDB + IMDb METADATA ---
    metadata = await run_async(fetch_movie_metadata, movie_name, movie_year, movie_lang, False, gemini_category)
    if metadata:
        title, year, poster_url, genre, imdb_id, rating, plot, category, seasons_data = metadata
    else:
        title      = movie_name
        year       = int(movie_year) if movie_year and str(movie_year).isdigit() else 0
        poster_url = None
        imdb_id    = None
        genre      = "Unknown"
        rating     = "N/A"
        plot       = "Auto Added"
        category   = gemini_category if gemini_category else "Movies"

    if not is_safe_canonical_title(title):
        logger.warning("Ingestion held: metadata returned unsafe canonical title %r", title)
        _record_pending_identity(
            file_unique_id,
            raw_caption or raw_text,
            raw_filename,
            ai_data,
            asdict(parsed_identity),
            "Provider metadata returned an unsafe title",
        )
        return None
    is_episode = parsed_identity.has_episode_marker or parsed_identity.season_number is not None
    if is_episode:
        provider_title = parse_content_identity(title).canonical_title
        if provider_title.casefold() != movie_name.casefold():
            _record_pending_identity(
                file_unique_id,
                raw_caption or raw_text,
                raw_filename,
                ai_data,
                asdict(parsed_identity),
                "Provider title does not confirm the raw episode's canonical series",
            )
            return None
        title = movie_name

    # 👇 NAYA LOGIC: Gemini Category Priority for Anime 👇
    cat_lower = str(gemini_category or "").lower()
    genre_lower = str(genre or "").lower()
    if "anime" in cat_lower or "cartoon" in cat_lower or "animation" in cat_lower or "anime" in genre_lower or "animation" in genre_lower:
        category = "Anime"
    if parsed_identity.has_episode_marker or parsed_identity.season_number is not None:
        category = "Anime" if "anime" in str(gemini_category).casefold() else "Web Series"
    category, content_type = normalize_catalog_labels(
        category=category,
        language=movie_lang,
        extra_info=extra_info,
        genre=genre,
        title=title,
    )

    # --- STEP 4: DB INSERT (pm_file_listener ka EXACT ON CONFLICT logic) ---
    if not imdb_id:  # Fix for empty string violating unique constraint
        imdb_id = None
    tmdb_id = await run_async(
        resolve_tmdb_id_from_imdb, imdb_id, category
    )

    conn = get_db_connection()
    if not conn:
        return None

    try:
        cur = conn.cursor()
        try:
            existing_id = _find_movie_by_provider_identity(cur, imdb_id, tmdb_id)
        except ValueError as exc:
            _record_ingestion_evidence(
                conn,
                file_unique_id=file_unique_id,
                raw_caption=raw_caption or raw_text,
                raw_filename=raw_filename,
                evidence=ai_data,
                parsed_identity=asdict(parsed_identity),
                provider_ids={"imdb_id": imdb_id, "tmdb_id": tmdb_id},
                movie_id=None,
                movie_file_id=None,
                resolver_method="conflicting_provider_identity",
                confidence=parsed_identity.confidence,
                status="pending_review",
                warnings=parsed_identity.parse_warnings + (str(exc),),
            )
            conn.commit()
            cur.close()
            return None
        resolver_method = "provider_identity" if existing_id else "provider_metadata"
        ambiguous = False
        identity_conflict = False
        if existing_id and is_episode:
            cur.execute(
                "SELECT title, content_type FROM movies WHERE id = %s",
                (existing_id,),
            )
            provider_parent = cur.fetchone()
            series_types = {"web series", "tv series", "tv show", "series", "anime"}
            identity_conflict = bool(
                not provider_parent
                or provider_parent[0].casefold() != movie_name.casefold()
                or str(provider_parent[1] or "").casefold() not in series_types
            )
        if not existing_id:
            canonical_id, ambiguous = _find_movie_by_canonical_identity(
                cur,
                movie_name,
                movie_year,
                require_series=is_episode
                or content_type.casefold() in {"web series", "anime"},
                content_type=(
                    None
                    if is_episode or content_type.casefold() in {"web series", "anime"}
                    else content_type
                ),
            )
            if ambiguous and not (imdb_id or tmdb_id):
                _record_ingestion_evidence(
                    conn,
                    file_unique_id=file_unique_id,
                    raw_caption=raw_caption or raw_text,
                    raw_filename=raw_filename,
                    evidence=ai_data,
                    parsed_identity=asdict(parsed_identity),
                    provider_ids={"imdb_id": imdb_id, "tmdb_id": tmdb_id},
                    movie_id=None,
                    movie_file_id=None,
                    resolver_method="ambiguous_canonical_match",
                    confidence=parsed_identity.confidence,
                    status="pending_review",
                    warnings=parsed_identity.parse_warnings + ("Multiple exact canonical matches",),
                )
                conn.commit()
                cur.close()
                return None

            if canonical_id:
                cur.execute(
                    "SELECT imdb_id, tmdb_id FROM movies WHERE id = %s",
                    (canonical_id,),
                )
                stored_imdb_id, stored_tmdb_id = cur.fetchone()
                identity_conflict = bool(
                    (imdb_id and stored_imdb_id and imdb_id != stored_imdb_id)
                    or (tmdb_id and stored_tmdb_id and int(tmdb_id) != int(stored_tmdb_id))
                )
                if not identity_conflict:
                    existing_id = canonical_id
                    resolver_method = "exact_canonical_series_match"

            if not can_create_canonical_content(
                parsed_identity,
                provider_identity=bool(imdb_id or tmdb_id),
                existing_parent=bool(existing_id),
                ambiguous=ambiguous,
                provider_conflict=identity_conflict,
            ):
                _record_ingestion_evidence(
                    conn,
                    file_unique_id=file_unique_id,
                    raw_caption=raw_caption or raw_text,
                    raw_filename=raw_filename,
                    evidence=ai_data,
                    parsed_identity=asdict(parsed_identity),
                    provider_ids={"imdb_id": imdb_id, "tmdb_id": tmdb_id},
                    movie_id=None,
                    movie_file_id=None,
                    resolver_method=(
                        "conflicting_provider_identity"
                        if identity_conflict
                        else "episode_without_provider_or_existing_identity"
                    ),
                    confidence=parsed_identity.confidence,
                    status="pending_review",
                    warnings=parsed_identity.parse_warnings
                    + (
                        "Existing canonical row has conflicting provider IDs"
                        if identity_conflict
                        else "No provider identity or existing canonical content was found",
                    ),
                )
                conn.commit()
                cur.close()
                logger.warning(
                    "Ingestion held for review: unverified canonical content %r", movie_name
                )
                return None
        if existing_id and resolver_method == "exact_canonical_series_match":
            cur.execute(
                "UPDATE movies SET imdb_id = COALESCE(imdb_id, %s), "
                "tmdb_id = COALESCE(tmdb_id, %s) WHERE id = %s",
                (imdb_id, tmdb_id, existing_id),
            )
            cur.execute("SELECT title, year FROM movies WHERE id = %s", (existing_id,))
            existing_title, existing_year = cur.fetchone()
            _record_ingestion_evidence(
                conn,
                file_unique_id=file_unique_id,
                raw_caption=raw_caption or raw_text,
                raw_filename=raw_filename,
                evidence=ai_data,
                parsed_identity=asdict(parsed_identity),
                provider_ids={"imdb_id": imdb_id, "tmdb_id": tmdb_id},
                movie_id=existing_id,
                movie_file_id=None,
                resolver_method=resolver_method,
                confidence=parsed_identity.confidence,
                status="resolved",
                warnings=parsed_identity.parse_warnings,
            )
            conn.commit()
            cur.close()
            return {
                "movie_id": existing_id,
                "title": existing_title,
                "year": existing_year or year,
                "genre": genre,
                "rating": rating,
                "plot": plot,
                "category": category,
                "movie_lang": movie_lang,
                "poster_url": poster_url,
                "backdrop_poster_url": backdrop_poster_url if "backdrop_poster_url" in locals() else None,
                "imdb_id": imdb_id,
                "tmdb_id": tmdb_id,
                "cast_str": "",
                "parsed_identity": asdict(parsed_identity),
                "raw_caption": raw_caption or raw_text,
                "raw_filename": raw_filename,
                "resolver_method": resolver_method,
                "confidence": parsed_identity.confidence,
            }

        cast_str = ""
        if imdb_id:
            cast_str = await run_async(fetch_cast_from_imdb, imdb_id, 5)
        trailer_key = await run_async(
            resolve_trailer_key,
            title,
            year,
            imdb_id,
            category,
            movie_name,
        )
        artwork_poster_url, backdrop_poster_url = await run_async(
            fetch_tmdb_artwork,
            title,
            year,
            imdb_id,
            category,
        )
        if artwork_poster_url and not poster_url:
            poster_url = artwork_poster_url

        movie_values = (
            title, imdb_id, tmdb_id, poster_url, backdrop_poster_url, year,
            genre, rating, plot, category, content_type, movie_lang,
            extra_info, cast_str, trailer_key,
        )
        if existing_id:
            cur.execute(
                """
                UPDATE movies
                SET title = %s, imdb_id = COALESCE(%s, movies.imdb_id),
                    tmdb_id = COALESCE(%s, movies.tmdb_id),
                    poster_url = COALESCE(%s, poster_url),
                    backdrop_poster_url = COALESCE(%s, backdrop_poster_url),
                    year = CASE WHEN movies.year = 0 THEN %s ELSE movies.year END,
                    genre = COALESCE(%s, genre), rating = COALESCE(%s, rating),
                    description = COALESCE(%s, description),
                    category = COALESCE(%s, category),
                    content_type = COALESCE(%s, content_type),
                    language = CASE WHEN %s <> '' THEN %s ELSE movies.language END,
                    extra_info = CASE WHEN %s <> '' THEN %s ELSE movies.extra_info END,
                    "cast" = COALESCE(%s, "cast"),
                    trailer_key = COALESCE(%s, trailer_key)
                WHERE id = %s
                RETURNING id
                """,
                (
                    title, imdb_id, tmdb_id, poster_url, backdrop_poster_url,
                    year, genre, rating, plot, category, content_type,
                    movie_lang, movie_lang, extra_info, extra_info, cast_str,
                    trailer_key, existing_id,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO movies
                    (title, url, imdb_id, tmdb_id, poster_url, backdrop_poster_url,
                     year, genre, rating, description, category, content_type,
                     language, extra_info, "cast", trailer_key)
                VALUES (%s, '', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                movie_values,
            )
        movie_id = cur.fetchone()[0]
        _record_ingestion_evidence(
            conn,
            file_unique_id=file_unique_id,
            raw_caption=raw_caption or raw_text,
            raw_filename=raw_filename,
            evidence=ai_data,
            parsed_identity=asdict(parsed_identity),
            provider_ids={"imdb_id": imdb_id, "tmdb_id": tmdb_id},
            movie_id=movie_id,
            movie_file_id=None,
            resolver_method=resolver_method,
            confidence=parsed_identity.confidence,
            status="resolved",
            warnings=parsed_identity.parse_warnings,
        )
        conn.commit()
        cur.close()

        return {
            'movie_id':   movie_id,
            'title':      title,
            'year':       year,
            'genre':      genre,
            'rating':     rating,
            'plot':       plot,
            'category':   category,
            'movie_lang': movie_lang,
            'poster_url': poster_url,
            'backdrop_poster_url': backdrop_poster_url,
            'imdb_id':    imdb_id,
            'tmdb_id':    tmdb_id,
            'cast_str':   cast_str,
            'parsed_identity': asdict(parsed_identity),
            'raw_caption': raw_caption or raw_text,
            'raw_filename': raw_filename,
            'resolver_method': resolver_method,
            'confidence': parsed_identity.confidence,
        }
    except Exception as e:
        logger.error(f"_core_movie_processor DB Error: {e}")
        if conn: conn.rollback()
        return None
    finally:
        close_db_connection(conn)


# ==============================================================================
# 📤 _pm_save_file — pm_file_listener ka Phase 2 (ek jagah, sab use karein)
# superbatch_done bhi isko call karta hai — alag/duplicate code nahi
# ==============================================================================
def _parse_file_identity_for_parent(raw_caption, raw_filename, parent_title):
    """Keep episode metadata from the file only when it belongs to this parent."""
    _, parsed, _ = resolve_raw_content_identity(raw_caption, raw_filename)
    if not parsed or not (
        parsed.has_episode_marker or parsed.season_number is not None
    ):
        return {}
    if parsed.canonical_title.casefold() != str(parent_title or "").casefold():
        return {}
    result = asdict(parsed)
    result["canonical_title"] = parent_title
    return result


def _resolve_episode_file_parent(cur, requested_movie_id, raw_caption, raw_filename, year=None):
    """Resolve episode files to a unique existing series before any file is saved."""
    evidence = [
        parse_content_identity(value)
        for value in (raw_caption, raw_filename)
        if value
    ]
    if not any(
        parsed.has_episode_marker or parsed.season_number is not None
        for parsed in evidence
    ):
        return requested_movie_id, None

    _, parsed, error = resolve_raw_content_identity(raw_caption, raw_filename)
    if not parsed or error:
        return None, error or "Episode identity is ambiguous"
    resolved_year = int(year) if str(year or "").isdigit() else None
    canonical_id, ambiguous = _find_movie_by_canonical_identity(
        cur,
        parsed.canonical_title,
        resolved_year,
        require_series=True,
    )
    if ambiguous:
        return None, "Multiple canonical series rows match this episode"
    if not canonical_id:
        return None, "No existing canonical series row matches this episode"
    return canonical_id, parsed


async def _pm_save_file(message, context) -> str | None:
    """
    Unified Phase 2 saver. Gemini bilkul use nahi hota.
    Caption aur raw filename separately parse hote hain, field-wise merge hota hai,
    downgrade upload se PEHLE block hota hai, phir storage copy + DB upsert hota hai.
    """
    media = message.document or message.video
    if not media:
        logger.error("_pm_save_file: Unsupported media type")
        return None

    movie_id = BATCH_SESSION.get('movie_id')
    if not movie_id:
        logger.error("_pm_save_file: BATCH_SESSION movie_id missing")
        return None

    file_name = getattr(media, 'file_name', None) or "File"
    file_size = getattr(media, 'file_size', 0) or 0
    file_unique_id = getattr(media, 'file_unique_id', None)
    file_size_str = get_readable_file_size(file_size)
    current_lang = BATCH_SESSION.get('language', '')
    raw_caption = message.caption or message.text or ""

    # Independent local extraction; no Gemini in Phase 2.
    cap_data = await fallback_extraction(raw_caption) if raw_caption else {}
    fn_data = await fallback_extraction(file_name) if file_name else {}
    cap_clean = strip_caption_junk(raw_caption) if raw_caption else ""

    cap_label = generate_quality_label(cap_clean, file_size_str, current_lang) if cap_clean else ""
    fn_label = generate_quality_label(file_name, file_size_str, current_lang) if file_name else ""
    label = _merge_quality_labels(cap_label, fn_label)
    f_lang = _merge_csv_values(cap_data.get('language'), fn_data.get('language'))
    f_extra = _merge_extra_info(cap_data.get('extra_info'), fn_data.get('extra_info'))
    identity_title = (
        (BATCH_SESSION.get('parsed_identity') or {}).get('canonical_title')
        or BATCH_SESSION.get('movie_title')
    )
    file_identity = _parse_file_identity_for_parent(
        raw_caption, file_name, identity_title
    )

    # Downgrade check BEFORE copying to channels, taaki rejected orphan uploads na banein.
    precheck_conn = get_db_connection()
    if not precheck_conn:
        return None
    try:
        cur = precheck_conn.cursor()
        resolved_movie_id, episode_identity = _resolve_episode_file_parent(
            cur,
            movie_id,
            raw_caption,
            file_name,
            BATCH_SESSION.get("year"),
        )
        if not resolved_movie_id:
            _record_ingestion_evidence(
                precheck_conn,
                file_unique_id=file_unique_id,
                raw_caption=raw_caption,
                raw_filename=file_name,
                evidence={"caption": cap_data, "filename": fn_data},
                parsed_identity=file_identity,
                provider_ids={
                    "imdb_id": BATCH_SESSION.get("imdb_id"),
                    "tmdb_id": BATCH_SESSION.get("tmdb_id"),
                },
                movie_id=None,
                movie_file_id=None,
                resolver_method="episode_parent_unresolved",
                confidence=file_identity.get("confidence", 0),
                status="pending_review",
                warnings=(episode_identity,),
            )
            precheck_conn.commit()
            logger.warning("Episode file held before upload: %s", episode_identity)
            return None
        if resolved_movie_id != movie_id or episode_identity is not None:
            cur.execute("SELECT title FROM movies WHERE id = %s", (resolved_movie_id,))
            resolved_title = cur.fetchone()[0]
            movie_id = resolved_movie_id
            BATCH_SESSION.update(
                movie_id=movie_id,
                movie_title=resolved_title,
            )
            file_identity = _parse_file_identity_for_parent(
                raw_caption, file_name, resolved_title
            )
        cur.close()

        if file_unique_id:
            cur = precheck_conn.cursor()
            cur.execute(
                "SELECT id, movie_id FROM movie_files WHERE file_unique_id = %s",
                (file_unique_id,),
            )
            prior_file = cur.fetchone()
            cur.close()
            if prior_file:
                if prior_file[1] != movie_id:
                    logger.error(
                        "Telegram file %s is already attached to movie %s, not %s",
                        file_unique_id, prior_file[1], movie_id,
                    )
                    return None
                _link_movie_file_episodes(
                    precheck_conn, prior_file[0], movie_id, file_identity
                )
                _record_ingestion_evidence(
                    precheck_conn,
                    file_unique_id=file_unique_id,
                    raw_caption=raw_caption,
                    raw_filename=file_name,
                    evidence={"caption": cap_data, "filename": fn_data},
                    parsed_identity=file_identity,
                    provider_ids={
                        "imdb_id": BATCH_SESSION.get("imdb_id"),
                        "tmdb_id": BATCH_SESSION.get("tmdb_id"),
                    },
                    movie_id=movie_id,
                    movie_file_id=prior_file[0],
                    resolver_method=BATCH_SESSION.get("resolver_method", "batch_session"),
                    confidence=file_identity.get(
                        "confidence", BATCH_SESSION.get("confidence", 0)
                    ),
                    status="resolved",
                )
                precheck_conn.commit()
                return label
        rejected, existing = is_downgrade(movie_id, label, f_extra, precheck_conn)
    except Exception as exc:
        logger.error("_pm_save_file pre-upload downgrade check failed: %s", exc)
        return None
    finally:
        close_db_connection(precheck_conn)

    if rejected:
        logger.info("🛡️ _pm_save_file: REJECTED '%s' — DB already has better '%s'", label, existing)
        return None

    channels = get_storage_channels()
    if not channels:
        logger.error("_pm_save_file: No STORAGE_CHANNELS found")
        return None

    backup_map = {}
    for chat_id in channels:
        try:
            sent = await message.copy(chat_id=chat_id)
            backup_map[str(chat_id)] = sent.message_id
            await asyncio.sleep(0.3)
        except Exception as exc:
            logger.error("_pm_save_file upload failed for %s: %s", chat_id, exc)

    if not backup_map:
        logger.error("_pm_save_file: All uploads failed")
        return None

    main_channel_id = next((cid for cid in channels if str(cid) in backup_map), None)
    main_message_id = backup_map.get(str(main_channel_id)) if main_channel_id is not None else None
    main_url = f"https://t.me/c/{str(main_channel_id).replace('-100', '')}/{main_message_id}"

    thumb = getattr(media, 'thumbnail', None) or getattr(media, 'thumb', None)
    if thumb:
        BATCH_SESSION['extracted_thumb'] = getattr(thumb, 'file_id', None)

    conn = get_db_connection()
    if not conn:
        for chat_id, message_id in backup_map.items():
            try:
                await context.bot.delete_message(chat_id=int(chat_id), message_id=message_id)
            except Exception:
                pass
        return None

    try:
        movie_file_id = upsert_movie_file(
            conn, movie_id, label, file_size_str, main_url,
            json.dumps(backup_map), f_lang, f_extra, file_unique_id,
        )
        _link_movie_file_episodes(conn, movie_file_id, movie_id, file_identity)
        _record_ingestion_evidence(
            conn,
            file_unique_id=file_unique_id,
            raw_caption=raw_caption,
            raw_filename=file_name,
            evidence={"caption": cap_data, "filename": fn_data},
            parsed_identity=file_identity,
            provider_ids={
                "imdb_id": BATCH_SESSION.get("imdb_id"),
                "tmdb_id": BATCH_SESSION.get("tmdb_id"),
            },
            movie_id=movie_id,
            movie_file_id=movie_file_id,
            resolver_method=BATCH_SESSION.get("resolver_method", "batch_session"),
            confidence=file_identity.get(
                "confidence", BATCH_SESSION.get("confidence", 0)
            ),
            status="resolved",
        )
        BATCH_SESSION['file_count'] = BATCH_SESSION.get('file_count', 0) + 1
        logger.info(
            "_pm_save_file saved: %s — %s [%s]",
            BATCH_SESSION.get('movie_title'), label, file_size_str,
        )

        try:
            deleted, deleted_labels = auto_upgrade_delete(movie_id, label, f_extra, conn)
            if deleted > 0:
                BATCH_SESSION['file_count'] = max(0, BATCH_SESSION.get('file_count', 0) - deleted)
                logger.info("🔄 _pm_save_file: %s old print(s) deleted: %s", deleted, deleted_labels)
        except Exception as exc:
            logger.error("Auto-Upgrade error in _pm_save_file: %s", exc)

        return label

    except Exception as exc:
        logger.error("_pm_save_file DB error: %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass
        for chat_id, message_id in backup_map.items():
            try:
                await context.bot.delete_message(chat_id=int(chat_id), message_id=message_id)
            except Exception:
                pass
        return None
    finally:
        close_db_connection(conn)


async def pm_file_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 🛑 18+ Batch active hai toh yahan kuch nahi karna
    if BATCH_18_SESSION.get('active'):
        return

    # ==========================================
    # 🚀 SUPERBATCH: PM FILE LISTENER HI MUH HAI
    # Jab Superbatch active ho, files yahan se hi andar jayengi
    # Alag superbatch_listener ki zaroorat nahi — ek hi entry point
    # ==========================================
    if SUPER_BATCH_SESSION.get('active'):
        if not (update.effective_user and update.effective_user.id == SUPER_BATCH_SESSION.get('admin_id')):
            return

        record = await _collect_superbatch_file(update.effective_message)
        if record and _looks_like_song_file(update.effective_message, record):
            logger.info(f"Superbatch: skipping likely song file: {record.get('file_name')}")
            await update.effective_message.reply_text(
                f"⏭️ **Skip kiya (Song lag rahi hai):** `{record.get('file_name')}`\n"
                f"Agar ye galat hai (genuine movie/episode thi), ise `/batch` se manually add kar dena.",
                parse_mode='Markdown',
            )
            return

        if record:
            SUPER_BATCH_SESSION['files'].append(record)
            count = len(SUPER_BATCH_SESSION['files'])
            if count % 10 == 0:
                await update.effective_message.reply_text(
                    f"📥 **{count} files mil gayi hain!**\nJab sab bhej do, `/superdone` karo.",
                    parse_mode='Markdown',
                )
        return

    # 1. VIP Payment Check (Safe for channels)
    if context.user_data and context.user_data.get('payment_step') == 'screenshot' and update.message and update.message.photo:
        await payment_photo_handler(update, context)
        return

    message = update.effective_message
    if not message:
        return

    # 2. Security: Yeh function sirf PM mein chalega, aur sirf ADMIN ke liye
    # (Handler filter already ChatType.PRIVATE hai, yeh double-check hai)
    if not update.effective_user or not is_admin(update.effective_user.id):
        return

    # ==========================================
    # 🖼️ CUSTOM POSTER UPLOAD LOGIC (Photo & URL Both Supported)
    # Sirf PM se poster update hoga
    # ==========================================
    if BATCH_SESSION.get('active'):
        is_poster_update = False
        public_url = None
        
        # 1. Agar Admin ne Photo bheji hai
        if message.photo:
            is_poster_update = True
            status_msg = await message.reply_text("🖼️ Image received! Uploading poster to cloud...")
            photo_file_id = message.photo[-1].file_id
            public_url = await upload_image_to_telegraph(context.bot, photo_file_id)
            
        # 2. Agar Admin ne direct Image URL bheja hai (http/https se shuru hone wala)
        elif message.text and message.text.strip().startswith("http"):
            is_poster_update = True
            status_msg = await message.reply_text("🔗 Image URL received! Linking poster directly...")
            public_url = message.text.strip()

        # Agar dono mein se koi bhi step trigger hua hai (Photo ya URL)
        if is_poster_update:
            if public_url:
                movie_id = BATCH_SESSION['movie_id']
                conn = get_db_connection()
                if conn:
                    try:
                        cur = conn.cursor()
                        cur.execute("UPDATE movies SET poster_url = %s WHERE id = %s", (public_url, movie_id))
                        conn.commit()
                        cur.close()
                    except Exception as e:
                        logger.error(f"Poster Update Error: {e}")
                    finally:
                        close_db_connection(conn)

                await status_msg.edit_text("✅ **Poster Successfully Updated!**\nAb aap files bhej sakte hain ya `/done` kar sakte hain.", parse_mode='Markdown')
            else:
                await status_msg.edit_text("❌ Poster upload fail ho gaya. Kripya image ya URL dobara bhejein.")
            
            return # Yahan ruk jao taaki image/url aage file ki tarah save na ho
        
    # --- ISKE NEECHE TUMHARA PURANA PHASE 2 WALA CODE AAYEGA JO FILES SAVE KARTA HAI ---
    if not (message.document or message.video): return
    
    # ... (purana file save aur forward logic) ...
    message = update.effective_message
    if not (message.document or message.video or message.photo): return

    caption = message.caption or ""
    if caption.startswith('/post_query'):
        return

    # 🚀 THE MAIN FIX: Agar sirf Photo aayi hai (bina caption ke) aur Batch OFF hai,
    # toh isko Poster maan lo aur koi Error message mat do (Takrav khatam).
    if message.photo and not caption and not BATCH_SESSION.get('active'):
        return 

    async with auto_batch_lock:
        
        # ==========================================
        # 🤖 PHASE 1: START BATCH — _core_movie_processor se power lelo
        # ==========================================
        if not BATCH_SESSION.get('active'):

            raw_caption = message.caption or message.text or ""
            raw_filename = _get_message_filename(message)
            if not raw_caption and not raw_filename:
                await message.reply_text(
                    "❌ **Batch Off!**\nCaption ya Telegram filename mein movie identity nahi mili.",
                    parse_mode='Markdown',
                )
                return

            status_msg = await message.reply_text(
                "🧠 Caption + Filename Evidence → Gemini → TMDB/IMDb pipeline chal raha hai...",
                quote=True,
            )

            # Thumbnail extract karo BATCH_SESSION ke liye (poster backup)
            image_bytes = None
            try:
                thumb_file_id = None
                if message.photo:
                    thumb_file_id = message.photo[-1].file_id
                elif message.video and message.video.thumbnail:
                    thumb_file_id = message.video.thumbnail.file_id
                elif message.document and message.document.thumbnail:
                    thumb_file_id = message.document.thumbnail.file_id
                if thumb_file_id:
                    BATCH_SESSION['extracted_thumb'] = thumb_file_id
                    # image_bytes = bytes(await (await context.bot.get_file(thumb_file_id)).download_as_bytearray())
                    image_bytes = None  # TEMPORARY BYPASS
            except Exception as e:
                logger.error(f"Thumbnail extract error: {e}")

            # Same file ke caption aur raw Telegram filename ko local parser se
            # separately extract karke Gemini ek baar reconcile karega.
            reconciled_data = await process_file_with_evidence_engine(message)
            result = await _core_movie_processor(
                raw_caption or raw_filename,
                image_bytes,
                reconciled_data=reconciled_data,
                raw_caption=raw_caption,
                raw_filename=raw_filename,
                file_unique_id=getattr(message.document or message.video, 'file_unique_id', None),
            )

            if not result:
                await status_msg.edit_text("❌ Movie naam extract nahi ho paya.\n\n`/batch Movie Name` use karein.")
                return

            movie_id   = result['movie_id']
            title      = result['title']
            year       = result['year']
            category   = result['category']
            movie_lang = result['movie_lang']

            # File count check (existing files)
            file_count = 0
            conn = get_db_connection()
            if conn:
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM movie_files WHERE movie_id = %s", (movie_id,))
                    file_count = cur.fetchone()[0]
                    cur.close()
                except Exception:
                    pass
                finally:
                    close_db_connection(conn)

            BATCH_SESSION.update({
                'active': True, 'movie_id': movie_id, 'movie_title': title,
                'file_count': file_count, 'admin_id': ADMIN_USER_ID,
                'year': str(year) if year else "", 'category': category, 'language': movie_lang,
                'parsed_identity': result.get('parsed_identity', {}),
                'resolver_method': result.get('resolver_method', ''),
                'confidence': result.get('confidence', 0),
                'imdb_id': result.get('imdb_id'),
                'tmdb_id': result.get('tmdb_id'),
            })

            first_file_label = await _pm_save_file(message, context)
            first_file_status = (
                "✅ First file saved.\n"
                if first_file_label
                else "⚠️ First file was not saved; please retry it.\n"
            )
            keyboard = []
            if file_count > 0:
                keyboard.append([InlineKeyboardButton("🗑️ Delete OLD Files", callback_data=f"clearfiles_{movie_id}")])
            keyboard.append([InlineKeyboardButton("❌ Cancel Batch", callback_data="cancel_batch")])

            await status_msg.edit_text(
                f"✅ **Batch Started!**\n\n🎬 Movie: **{title}**\n📅 Year: {year if year else 'N/A'}\n🏷️ Category: {category}\n"
                f"{first_file_status}\n"
                f"🚀 **Ab apni files bhejna shuru karo!**\nJab ho jaye: `/done`",
                parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return

        # ==========================================
        # 📤 PHASE 2: SAVE FILES (Jab Batch ON ho)
        # ==========================================
        upload_status = await message.reply_text("⏳ Uploading file...", quote=True)
        # ... (Baaki ka Phase 2 ka code aapka same rahega)

        # 📤 PHASE 2: UNIFIED FILE SAVING
        label = await _pm_save_file(message, context)
        
        if label:
            file_size = message.document.file_size if message.document else (message.video.file_size if message.video else 0)
            file_size_str = get_readable_file_size(file_size)
            movie_title = BATCH_SESSION.get('movie_title', 'Movie')
            await upload_status.edit_text(
                f"✅ **Saved:** `{movie_title} {label}` [{file_size_str}]\n🔢 Total Files: {BATCH_SESSION.get('file_count', 0)}", 
                parse_mode='Markdown'
            )
        else:
            await upload_status.edit_text(
                "❌ **Save Failed or Blocked!**\nYa toh error aaya, ya DB mein pehle se better print (downgrade) hai. Logs check karein.", 
                parse_mode='Markdown'
            )
    
async def batch_done_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not BATCH_SESSION.get('active'): 
        await update.message.reply_text("❌ Koi batch active nahi hai!")
        return
    
    status_msg = await update.message.reply_text("🔄 **Batch complete kar raha hoon...**", parse_mode='Markdown')

    try:
        movie_id = BATCH_SESSION.get('movie_id')
        movie_title = BATCH_SESSION.get('movie_title', 'Unknown')
        movie_year = BATCH_SESSION.get('year', '')
        movie_category = BATCH_SESSION.get('category', '')
        
        # DB से क्वालिटी और डेटा निकालें
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT genre, language, \"cast\", poster_url, rating FROM movies WHERE id = %s", (movie_id,))
        minfo = cur.fetchone()
        cur.execute("SELECT quality FROM movie_files WHERE movie_id = %s", (movie_id,))
        qrows = cur.fetchall()
        cur.close()
        close_db_connection(conn)

        db_genre = minfo[0] if minfo and minfo[0] else "Unknown"
        db_lang = minfo[1] if minfo and minfo[1] else "Hindi (LiNE) + HC-ESubs"
        m_poster = minfo[3] if minfo else None
        m_rating = minfo[4] if minfo else "N/A"

        # क्वालिटी अलाइनमेंट
        res_list = sorted(list(set(re.search(r'(\d{3,4}p)', r[0]).group(1) for r in qrows if re.search(r'(\d{3,4}p)', r[0]))), key=lambda x: int(x.replace('p','')), reverse=True)
        dynamic_res = " | ".join(res_list) if res_list else "1080p | 720p | 480p"

        # 🎯 आपका पसंदीदा क्लीन फॉर्मेट
        caption = (
            f"🎬 <b>{movie_title}</b>\n"
            f"✨ Genre: {db_genre}\n"
            f"Language: {db_lang}\n"
            f"Quality: V2 HQ-HDTC {dynamic_res}\n"
            f"━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━\n"
            f"<b>Update Channel:</b> <a href='https://t.me/FlimfyBoxBackUp'>Join BackUp</a>\n"
            f"━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━\n"
            f"👇 <b>Download Below</b> 👇"
        )
        
        # 🚫 AI Alias Generation OFF — Flask Web App mein Google Suggest + pg_trgm already handles typos
        # generate_aliases_gemini() hata diya — Gemini API keys bachegi + DB clean rahega
        aliases = []
        alias_count = 0

        # 🚀 POST TO FORUM
        forum_post_status = "⏳ Posting to Forum..."
        

        # --- SECURE LINK FOR SUPERBATCH POST ---
        secure_url = f"https://flimfybox-bot-yht0.onrender.com/watch/{movie_id}"

        post_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Get Now", url=secure_url)],
            [InlineKeyboardButton("Join Channel", url=FILMFYBOX_CHANNEL_URL)]
        ])
        
        photo_to_send = m_poster if (m_poster and m_poster != 'N/A' and m_poster.startswith('http')) else None
        if not photo_to_send:
            thumb_file_id = context.bot_data.get(f"auto_thumb_{movie_id}")
            if thumb_file_id:
                photo_to_send = thumb_file_id
        if not photo_to_send: photo_to_send = DEFAULT_POSTER



        report = (
            f"🎉 **Batch Completed!**\n\n"
            f"🎬 **Movie:** `{movie_title}`\n"
            f"📅 **Year:** {movie_year if movie_year else 'N/A'}\n"
            f"🏷️ **Category:** {movie_category}\n"
            f"📂 **Files Saved:** {BATCH_SESSION.get('file_count', 0)}\n\n"
        )

        extracted_thumb = BATCH_SESSION.get('extracted_thumb')
        if extracted_thumb: context.bot_data[f"auto_thumb_{movie_id}"] = extracted_thumb

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🤖 Auto Post (HD TMDB Poster)", callback_data=f"autopost_{movie_id}")],
            [InlineKeyboardButton("📢 Manual Post (Send Poster)", callback_data=f"askposter_{movie_id}")]
        ])

        await status_msg.edit_text(report, parse_mode='Markdown', reply_markup=keyboard)

    except Exception as e:
        logger.error(f"Error in batch_done_command: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Error during /done: {e}")

    finally:
        BATCH_SESSION.update({
            'active': False, 'movie_id': None, 'movie_title': None, 
            'file_count': 0, 'admin_id': None, 'year': '', 'category': '', 
            'extracted_thumb': None
        })

                
async def handle_admin_poster(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin se photo lekar clean caption ke sath channel me post karega"""
    user_id = update.effective_user.id
    if not is_admin(user_id): 
        return

    # Check karo ki bot photo ka wait kar raha tha ya nahi
    movie_id = context.user_data.get('waiting_for_poster')
    if not movie_id: 
        return # Agar wait nahi kar raha tha, to ignore karo

    if not update.message.photo:
        await update.message.reply_text("❌ Please send a valid PHOTO.")
        return

    # Sabse acchi quality ki photo nikalo
    file_id = update.message.photo[-1].file_id
    status_msg = await update.message.reply_text("⏳ Publishing to channels...")

    # 1. Database se sirf Title nikalo
    conn = get_db_connection()
    if not conn: return
    cur = conn.cursor()
    cur.execute("SELECT title, category FROM movies WHERE id = %s", (movie_id,))
    res = cur.fetchone()
    cur.close()
    close_db_connection(conn)

    if not res:
        await status_msg.edit_text("❌ Movie not found in DB.")
        context.user_data.pop('waiting_for_poster', None)
        return
    
    m_title = res[0]
    m_category = res[1]

    # 🎯 FIX: This block must be indented to match the rest of the function
    channel_caption = (
        f"🎬 <b>{m_title}</b>\n"
        f"✨ Genre: {m_genre}\n"
        f"Language: {m_lang}\n"
        f"Quality: V2 HQ-HDTC {dynamic_res}\n"
        f"━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━\n"
        f"<b>Update Channel:</b> <a href='https://t.me/FlimfyBoxBackUp'>Join BackUp</a>\n"
        f"━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━\n"
        f"👇 <b>Download Below</b> 👇"
    )

    # 3. Download Buttons Banao
    secure_url = f"https://flimfybox-bot-yht0.onrender.com/watch/{movie_id}"

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Get Now", url=secure_url)],
        [InlineKeyboardButton("Join Channel", url=FILMFYBOX_CHANNEL_URL)]
    ])

    # 4. Channels me Post karo
    cat_lower = str(m_category).lower()
    if "anime" in cat_lower or "cartoon" in cat_lower or "animation" in cat_lower:
        target_channels = [ANIME_CHANNEL_ID]
    else:
        channels_str = os.environ.get('BROADCAST_CHANNELS', '')
        target_channels = [ch.strip() for ch in channels_str.split(',') if ch.strip()]

    if not target_channels:
        await status_msg.edit_text("❌ Error: No BROADCAST_CHANNELS found in .env")
        context.user_data.pop('waiting_for_poster', None)
        return

    # 👇 GLOBAL DUPLICATE CHECK — 7 din me kahi bhi post hui ho to skip
    if is_movie_posted_recently(movie_id, days=7):
        await status_msg.edit_text(f"⏭️ <b>{m_title}</b> pehle se 7 din ke andar post ho chuki hai. Skipping.", parse_mode='HTML')
        context.user_data.pop('waiting_for_poster', None)
        return

    sent_count = 0
    for chat_id_str in target_channels:
        try:
            chat_id = int(chat_id_str)
            sent_msg = await context.bot.send_photo(
                chat_id=chat_id,
                photo=file_id,
                caption=channel_caption,
                parse_mode='HTML',
                reply_markup=keyboard
            )
            
            # Restore Feature ke liye DB me save karo
            if sent_msg:
                save_post_to_db(
                    movie_id, chat_id, sent_msg.message_id, "FlimfyBoxBot",  # ✅ NAYA: bot3 hat gaya!
                    channel_caption, file_id, "photo", keyboard.to_dict(), None, "movies"
                )
                sent_count += 1
        except Exception as e:
            logger.error(f"Auto-post failed for {chat_id_str}: {e}")

    # 5. Finish and Clear State
    await status_msg.edit_text(f"✅ <b>Posted successfully to {sent_count} channels!</b>", parse_mode='HTML')
    context.user_data.pop('waiting_for_poster', None)

POST_QUERY_MEDIA_GROUPS = defaultdict(list)
POST_QUERY_TASKS = {}
GLOBAL_ALBUM_CACHE = {}

async def global_album_cacher(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.media_group_id:
        return
    mg_id = msg.media_group_id
    if mg_id not in GLOBAL_ALBUM_CACHE:
        if len(GLOBAL_ALBUM_CACHE) > 500:
            GLOBAL_ALBUM_CACHE.pop(next(iter(GLOBAL_ALBUM_CACHE)))
        GLOBAL_ALBUM_CACHE[mg_id] = []
    GLOBAL_ALBUM_CACHE[mg_id].append(msg)


async def collect_post_query_album(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        return

    message = update.message
    if not message or not message.media_group_id:
        return
        
    mg_id = message.media_group_id
    POST_QUERY_MEDIA_GROUPS[mg_id].append(message)
    
    if mg_id not in POST_QUERY_TASKS:
        POST_QUERY_TASKS[mg_id] = asyncio.create_task(process_post_query_album(mg_id, update, context))

def create_image_collage(image_bytes_list):
    from PIL import Image, ImageOps
    import math
    images = []
    for img_bytes in image_bytes_list:
        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            images.append(img)
        except Exception as e:
            logger.error(f"Failed to open image for collage: {e}")
            
    if not images:
        return None
        
    num_images = len(images)
    if num_images > 4:
        images = images[:4]
        num_images = 4
        
    cols = 2 if num_images >= 2 else 1
    rows = math.ceil(num_images / cols)
    
    cell_size = 600
    
    collage_w = cols * cell_size
    collage_h = rows * cell_size
    
    collage = Image.new("RGB", (collage_w, collage_h), color=(255, 255, 255))
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        
        fitted_img = ImageOps.fit(img, (cell_size, cell_size), method=Image.Resampling.LANCZOS)
        collage.paste(fitted_img, (col * cell_size, row * cell_size))
        
    output = BytesIO()
    output.name = "collage.jpg"
    collage.save(output, format='JPEG', quality=90)
    output.seek(0)
    return output

async def process_post_query_album(mg_id: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    await asyncio.sleep(2.5)  # Wait for all album parts to arrive
    
    messages = POST_QUERY_MEDIA_GROUPS.pop(mg_id, [])
    POST_QUERY_TASKS.pop(mg_id, None)
    
    if not messages:
        return
        
    # Find the message with the caption
    caption_msg = None
    for msg in messages:
        if msg.caption and msg.caption.startswith('/post_query'):
            caption_msg = msg
            break
            
    if not caption_msg:
        # Not a post query album, just ignore
        return

    # Now we process the album
    caption_text = caption_msg.caption
    raw_input = caption_text.replace('/post_query', '').strip()
    
    if ',' in raw_input:
        parts = raw_input.split(',', 1)
        query_text = parts[0].strip()
        custom_msg = parts[1].strip()
    else:
        query_text = raw_input
        custom_msg = ""

    if not query_text:
        await caption_msg.reply_text("❌ Movie name missing")
        return

    # Find Movie in DB
    movie_id = None
    movie_category = ""
    conn = get_db_connection()

    if conn:
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, category FROM movies WHERE title ILIKE %s LIMIT 1",
                (f"%{query_text}%",)
            )
            row = cur.fetchone()
            if row:
                movie_id = row[0]
                movie_category = row[1] or ""
            cur.close()
        except Exception as e:
            logger.error(f"DB Error: {e}")
        finally:
            close_db_connection(conn)

    # Generate Secure Links
    bot1 = "FlimfyBox_SearchBot"
    bot2 = "urmoviebot"
    bot3 = "FlimfyBoxBot"
    
    if movie_id:
        secure_link = f"https://flimfybox-bot-yht0.onrender.com/watch/{movie_id}"
        link1 = secure_link
        link2 = secure_link
        link3 = secure_link
    else:
        import re
        safe_query = re.sub(r'[^a-zA-Z0-9_-]', '', query_text.replace(' ', '_'))
        link_param = f"q_{safe_query}"[:64]
        
        link1 = f"https://t.me/{bot1}?start={link_param}"
        link2 = f"https://t.me/{bot2}?start={link_param}"
        link3 = f"https://t.me/{bot3}?start={link_param}"

    # Build Keyboard
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Get Now", url=link1)],
        [InlineKeyboardButton("Join Channel", url=FILMFYBOX_CHANNEL_URL)]
    ])

    # Build Caption
    channel_caption = f"🎬 <b>{query_text}</b>\n"
    if custom_msg:
        channel_caption += f"✨ <b>{custom_msg}</b>\n\n"
    else:
        channel_caption += "\n"
    
    channel_caption += (
        "➖➖➖➖➖➖➖\n"
        f"<b>Support:</b> <a href='https://t.me/+dxaCr_cMmGpkYTFl'>Join Chat</a>\n"
        "➖➖➖➖➖➖➖\n"
        "<b>👇 Download Below</b>"
    )

    # Send to Channels
    cat_lower = str(movie_category).lower()
    if "anime" in cat_lower or "cartoon" in cat_lower or "animation" in cat_lower:
        target_channels = [ANIME_CHANNEL_ID]
    else:
        channels_str = os.environ.get('BROADCAST_CHANNELS', '')
        target_channels = [ch.strip() for ch in channels_str.split(',') if ch.strip()]

    if not target_channels:
        await caption_msg.reply_text("❌ No BROADCAST_CHANNELS configured in .env")
        return

    if movie_id and is_movie_posted_recently(movie_id, days=7):
        await caption_msg.reply_text(f"⏭️ <b>{query_text}</b> pehle se 7 din ke andar post ho chuki hai. Skipping.", parse_mode='HTML')
        return

    # Build the collage
    messages.sort(key=lambda x: x.message_id)
    
    if any(m.video for m in messages):
        await caption_msg.reply_text("❌ Videos wale album supported nahi hain. Sirf images ka album ya single video bhejein.")
        return

    photo_msgs = [m for m in messages if m.photo]
    if not photo_msgs:
        return

    status_msg = await caption_msg.reply_text("⏳ Generating image collage, please wait...")

    image_bytes_list = []
    for m in photo_msgs:
        try:
            file = await context.bot.get_file(m.photo[-1].file_id)
            img_bytes = await file.download_as_bytearray()
            image_bytes_list.append(img_bytes)
        except Exception as e:
            logger.error(f"Error downloading image for collage: {e}")

    collage_bytesio = await run_async(create_image_collage, image_bytes_list)
    
    if not collage_bytesio:
        await status_msg.edit_text("❌ Failed to create image collage.")
        return

    sent_count = 0
    failed_list = []

    for chat_id_str in target_channels:
        try:
            chat_id = int(chat_id_str)
            logger.info(f"📤 Sending Collage to {chat_id}...")
            
            # Reset the pointer of the BytesIO object for each upload
            collage_bytesio.seek(0)
            
            # Send single photo with caption and keyboard
            sent_msg = await context.bot.send_photo(
                chat_id=chat_id,
                photo=collage_bytesio,
                caption=channel_caption,
                reply_markup=keyboard,
                parse_mode='HTML'
            )

            # Save post to db for restore
            if sent_msg and movie_id:
                try:
                    # Save the collage photo to db
                    save_post_to_db(
                        movie_id, chat_id, sent_msg.message_id, "FlimfyBoxBot",
                        channel_caption, sent_msg.photo[-1].file_id, "photo", keyboard.to_dict(), None, "movies"
                    )
                except Exception as save_err:
                    logger.warning(f"Album DB save failed (non-critical): {save_err}")

            if sent_msg:
                sent_count += 1

        except Exception as e:
            failed_list.append(f"{chat_id_str}: {str(e)[:30]}")
            logger.error(f"Error sending collage to {chat_id_str}: {e}")

    await status_msg.delete()

    # Final Report
    report = f"✅ <b>Post Processed (Collage: {len(photo_msgs)} images)</b>\n\n"
    report += f"📤 <b>Sent:</b> {sent_count}/{len(target_channels)}\n"
    report += f"❌ <b>Failed:</b> {len(failed_list)}\n\n"
    report += f"🎬 <b>Movie:</b> {query_text}\n"
    report += f"📝 <b>Extra:</b> {custom_msg or 'None'}"

    if failed_list:
        report += "\n\n<b>Errors:</b>\n"
        for err in failed_list[:3]:
            report += f"• {err}\n"

    await caption_msg.reply_text(report, parse_mode='HTML')


async def admin_post_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    ✅ FIXED: Smart Post Generator with proper error handling
    """
    try:
        user_id = update.effective_user.id
        if not is_admin(user_id):
            return

        message = update.message
        
        # 1. Check Media
        if not (message.photo or message.video):
            await message.reply_text("❌ Photo ya Video bhejo caption ke sath")
            return

        if message.media_group_id:
            # Media groups are handled by collect_post_query_album instead
            return

        caption_text = message.caption or ""
        if not caption_text.startswith('/post_query'):
            return

        # 2. Extract Media
        file_id = None
        media_type = 'photo'

        if message.photo:
            file_id = message.photo[-1].file_id
            media_type = 'photo'
        elif message.video:
            file_id = message.video.file_id
            media_type = 'video'

        # 3. Parse Query
        raw_input = caption_text.replace('/post_query', '').strip()
        
        if ',' in raw_input:
            parts = raw_input.split(',', 1)
            query_text = parts[0].strip()
            custom_msg = parts[1].strip()
        else:
            query_text = raw_input
            custom_msg = ""

        if not query_text:
            await message.reply_text("❌ Movie name missing")
            return

        # 4. Find Movie in DB
        movie_id = None
        movie_category = ""
        conn = get_db_connection()

        if conn:
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, category FROM movies WHERE title ILIKE %s LIMIT 1",
                    (f"%{query_text}%",)
                )
                row = cur.fetchone()
                if row:
                    movie_id = row[0]
                    movie_category = row[1] or ""
                cur.close()
            except Exception as e:
                logger.error(f"DB Error: {e}")
            finally:
                close_db_connection(conn)

        # 5. Generate Secure Links (Anti-Bot)
        bot1 = "FlimfyBox_SearchBot"
        bot2 = "urmoviebot"
        bot3 = "FlimfyBoxBot"
        
        if movie_id:
            # ✅ FIXED: Web App Secure Link (Exactly like /superdone)
            secure_link = f"https://flimfybox-bot-yht0.onrender.com/watch/{movie_id}"
            link1 = secure_link
            link2 = secure_link
            link3 = secure_link
        else:
            import re
            # ⚠️ Agar movie DB me nahi hai (Sirf search query hai), to purana link chalega
            # Clean text to contain only alphanumeric, underscores, and hyphens, and limit to 64 bytes
            safe_query = re.sub(r'[^a-zA-Z0-9_-]', '', query_text.replace(' ', '_'))
            link_param = f"q_{safe_query}"[:64]
            
            link1 = f"https://t.me/{bot1}?start={link_param}"
            link2 = f"https://t.me/{bot2}?start={link_param}"
            link3 = f"https://t.me/{bot3}?start={link_param}"

        # 6. Build Keyboard
        if movie_id:
            # ✅ Yahan se web_app= hata diya hai, ab direct tumhara /watch/ wala link khulega
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("Get Now", url=link1)],
                [InlineKeyboardButton("Join Channel", url=FILMFYBOX_CHANNEL_URL)]
            ])
        else:
            # Agar fallback tg:// link hai, toh normal URL rehne do
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("Get Now", url=link1)],
                [InlineKeyboardButton("Join Channel", url=FILMFYBOX_CHANNEL_URL)]
            ])
        # 7. Build Caption
        channel_caption = f"🎬 <b>{query_text}</b>\n"
        if custom_msg:
            channel_caption += f"✨ <b>{custom_msg}</b>\n\n"
        else:
            channel_caption += "\n"
        
        channel_caption += (
            "➖➖➖➖➖➖➖\n"
            f"<b>Support:</b> <a href='https://t.me/+dxaCr_cMmGpkYTFl'>Join Chat</a>\n"
            "➖➖➖➖➖➖➖\n"
            "<b>👇 Download Below</b>"
        )

        # 8. Send to Channels (Anime → Anime Channel, Baaki → BROADCAST_CHANNELS)
        cat_lower = str(movie_category).lower()
        if "anime" in cat_lower or "cartoon" in cat_lower or "animation" in cat_lower:
            target_channels = [ANIME_CHANNEL_ID]
        else:
            channels_str = os.environ.get('BROADCAST_CHANNELS', '')
            target_channels = [ch.strip() for ch in channels_str.split(',') if ch.strip()]

        if not target_channels:
            await message.reply_text("❌ No BROADCAST_CHANNELS configured in .env")
            return

        # 👇 GLOBAL DUPLICATE CHECK — 7 din me kahi bhi post hui ho to skip
        if movie_id and is_movie_posted_recently(movie_id, days=7):
            await message.reply_text(f"⏭️ <b>{query_text}</b> pehle se 7 din ke andar post ho chuki hai. Skipping.", parse_mode='HTML')
            return

        sent_count = 0
        failed_list = []

        for chat_id_str in target_channels:
            try:
                # ✅ FIXED: Parse channel ID properly
                try:
                    chat_id = int(chat_id_str)
                except ValueError:
                    failed_list.append(f"Invalid ID: {chat_id_str}")
                    continue

                logger.info(f"📤 Sending to {chat_id}...")

                sent_msg = None

                if media_type == 'video':
                    sent_msg = await context.bot.send_video(
                        chat_id=chat_id,
                        video=file_id,
                        caption=channel_caption,
                        reply_markup=keyboard,
                        parse_mode='HTML'
                    )
                else:
                    sent_msg = await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=file_id,
                        caption=channel_caption,
                        reply_markup=keyboard,
                        parse_mode='HTML'
                    )

                if sent_msg:
                    logger.info(f"✅ Sent to {chat_id}, Message ID: {sent_msg.message_id}")
                    sent_count += 1

            except telegram.error.BadRequest as e:
                error = str(e)
                if "group is deactivated" in error or "not found" in error:
                    failed_list.append(f"{chat_id_str}: Channel inactive/deleted")
                else:
                    failed_list.append(f"{chat_id_str}: {error}")
                logger.error(f"BadRequest for {chat_id_str}: {e}")
                
            except telegram.error.Forbidden as e:
                failed_list.append(f"{chat_id_str}: Bot blocked/no access")
                logger.error(f"Forbidden for {chat_id_str}: {e}")
                
            except Exception as e:
                failed_list.append(f"{chat_id_str}: {str(e)[:30]}")
                logger.error(f"Error sending to {chat_id_str}: {e}")

        # 9. Final Report
        report = f"""✅ <b>Post Processed ({media_type.capitalize()})</b>

📤 <b>Sent:</b> {sent_count}/{len(target_channels)}
❌ <b>Failed:</b> {len(failed_list)}

🎬 <b>Movie:</b> {query_text}
📝 <b>Extra:</b> {custom_msg or 'None'}"""

        if failed_list:
            report += "\n\n<b>Errors:</b>\n"
            for err in failed_list[:3]:  # Show first 3 errors
                report += f"• {err}\n"

        await message.reply_text(report, parse_mode='HTML')

    except Exception as e:
        logger.error(f"Critical error in post_query: {e}", exc_info=True)


async def admin_post_query_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Reply-based /post_query mode.
    Command format: /post_query Custom Movie Name (as a reply to a media message)
    """
    try:
        user_id = update.effective_user.id
        if not is_admin(user_id):
            return

        message = update.message
        # Check if it's a reply
        if not message.reply_to_message:
            return

        replied_msg = message.reply_to_message
        if not (replied_msg.photo or replied_msg.video):
            # Ignore if not replying to media
            return

        command_text = message.text or ""
        if not command_text.startswith('/post_query'):
            return

        query_text = command_text.replace('/post_query', '', 1).strip()
        if not query_text:
            await message.reply_text("❌ Movie name missing. Use: /post_query Custom Movie Name")
            return

        # 1. Database Lookup
        movie_id = None
        movie_category = ""
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, category FROM movies WHERE title ILIKE %s LIMIT 1",
                    (f"%{query_text}%",)
                )
                row = cur.fetchone()
                if row:
                    movie_id = row[0]
                    movie_category = row[1] or ""
                cur.close()
            except Exception as e:
                logger.error(f"DB Error: {e}")
            finally:
                close_db_connection(conn)

        # 2. Generate Secure Links
        bot1 = "FlimfyBox_SearchBot"
        bot2 = "urmoviebot"
        bot3 = "FlimfyBoxBot"
        
        if movie_id:
            secure_link = f"https://flimfybox-bot-yht0.onrender.com/watch/{movie_id}"
            link1 = secure_link
            link2 = secure_link
            link3 = secure_link
        else:
            import re
            safe_query = re.sub(r'[^a-zA-Z0-9_-]', '', query_text.replace(' ', '_'))
            link_param = f"q_{safe_query}"[:64]
            link1 = f"https://t.me/{bot1}?start={link_param}"
            link2 = f"https://t.me/{bot2}?start={link_param}"
            link3 = f"https://t.me/{bot3}?start={link_param}"

        # 3. Build Keyboard
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Get Now", url=link1)],
            [InlineKeyboardButton("Join Channel", url=FILMFYBOX_CHANNEL_URL)]
        ])

        # 4. Target Channels
        cat_lower = str(movie_category).lower()
        if "anime" in cat_lower or "cartoon" in cat_lower or "animation" in cat_lower:
            target_channels = [ANIME_CHANNEL_ID]
        else:
            channels_str = os.environ.get('BROADCAST_CHANNELS', '')
            target_channels = [ch.strip() for ch in channels_str.split(',') if ch.strip()]

        if not target_channels:
            await message.reply_text("❌ No BROADCAST_CHANNELS configured in .env")
            return

        # 5. Global Duplicate Check
        if movie_id and is_movie_posted_recently(movie_id, days=7):
            await message.reply_text(f"⏭️ <b>{query_text}</b> pehle se 7 din ke andar post ho chuki hai. Skipping.", parse_mode='HTML')
            return

        # 6. Copy Media Logic
        is_album = bool(replied_msg.media_group_id)
        sent_count = 0
        failed_list = []

        if is_album:
            mg_id = replied_msg.media_group_id
            if mg_id not in GLOBAL_ALBUM_CACHE:
                await message.reply_text("❌ Album not found in cache. Please forward the album to the bot again and then reply.")
                return
                
            album_messages = GLOBAL_ALBUM_CACHE[mg_id]
            album_messages.sort(key=lambda x: x.message_id)
            ordered_message_ids = [m.message_id for m in album_messages]
            
            # Find caption index in original album
            caption_index = -1
            for i, m in enumerate(album_messages):
                if m.caption:
                    caption_index = i
                    break
            
            if caption_index == -1:
                caption_index = 0

            for chat_id_str in target_channels:
                try:
                    chat_id = int(chat_id_str)
                    copied = await context.bot.copy_messages(
                        chat_id=chat_id,
                        from_chat_id=replied_msg.chat_id,
                        message_ids=ordered_message_ids
                    )
                    
                    if copied and len(copied) > caption_index:
                        await context.bot.edit_message_reply_markup(
                            chat_id=chat_id,
                            message_id=copied[caption_index].message_id,
                            reply_markup=keyboard
                        )
                    sent_count += 1
                except Exception as e:
                    failed_list.append(f"{chat_id_str}: {str(e)[:30]}")
                    logger.error(f"Error copying album to {chat_id_str}: {e}")
                    
        else:
            # Single photo/video
            for chat_id_str in target_channels:
                try:
                    chat_id = int(chat_id_str)
                    await context.bot.copy_message(
                        chat_id=chat_id,
                        from_chat_id=replied_msg.chat_id,
                        message_id=replied_msg.message_id,
                        reply_markup=keyboard
                    )
                    sent_count += 1
                except Exception as e:
                    failed_list.append(f"{chat_id_str}: {str(e)[:30]}")
                    logger.error(f"Error copying media to {chat_id_str}: {e}")

        # Final Report
        media_type = 'album' if is_album else ('video' if replied_msg.video else 'photo')
        report = f"✅ <b>Post Processed ({media_type.capitalize()}) [Reply Mode]</b>\n\n"
        report += f"📤 <b>Sent:</b> {sent_count}/{len(target_channels)}\n"
        report += f"❌ <b>Failed:</b> {len(failed_list)}\n\n"
        report += f"🎬 <b>Movie:</b> {query_text}"

        if failed_list:
            report += "\n\n<b>Errors:</b>\n"
            for err in failed_list[:3]:
                report += f"• {err}\n"

        await message.reply_text(report, parse_mode='HTML')
        
        # Delete the command message
        try:
            await message.delete()
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Critical error in admin_post_query_text: {e}", exc_info=True)
        await message.reply_text(f"❌ Error: {str(e)[:100]}")

# ==========================================
# 🚀 AUTO MASS-FORWARD & LINK SHORTENER
# ==========================================

async def shorten_link(long_url):
    """GPLinks API se link chota karke Earning link banata hai."""
    api_key = os.environ.get('GPLINKS_API_KEY')
    if not api_key:
        return long_url # Agar API key nahi hai, toh purana link hi chalne do
        
    api_url = f"https://gplinks.in/api?api={api_key}&url={long_url}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                data = await resp.json()
                if data.get("status") == "success":
                    return data.get("shortenedUrl")
    except Exception as e:
        print(f"Shortener Error: {e}")
    return long_url


# ==========================================
# 🚀 18+ MASS-FORWARD BATCH SYSTEM (SAFE)
# ==========================================

async def shorten_link(long_url):
    """GPLinks API se link chota karke Earning link banata hai."""
    api_key = os.environ.get('GPLINKS_API_KEY')
    if not api_key:
        return long_url
        
    api_url = f"https://gplinks.in/api?api={api_key}&url={long_url}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                data = await resp.json()
                if data.get("status") == "success":
                    return data.get("shortenedUrl")
    except Exception as e:
        logger.error(f"Shortener Error: {e}")
    return long_url

# ==================== 18+ BATCH SYSTEM (SAME AS NORMAL BATCH) ====================

BATCH_18_SESSION = {'active': False, 'movie_id': None, 'movie_title': None, 'file_count': 0, 'admin_id': None}

async def batch18_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """18+ बैच शुरू करें - बिल्कुल /batch की तरह काम करेगा"""
    if update.effective_user.id not in ADMIN_IDS:
        return

    if BATCH_SESSION.get('active'):  # अगर नॉर्मल बैच चल रहा है तो 18+ नहीं चलेगा
        await update.message.reply_text("❌ पहले से नॉर्मल बैच चल रहा है। कृपया उसे /done करें या /cancel करें।")
        return

    BATCH_18_SESSION.update({
        'active': True,
        'admin_id': update.effective_user.id,
        'movie_id': None,
        'movie_title': None,
        'file_count': 0
    })

    await update.message.reply_text(
        "🔞 **18+ बैच मोड चालू!**\n\n"
        "👉 अब आप जिस 18+ मूवी/सीरीज़ की फ़ाइलें भेजना चाहते हैं, उसकी **पहली फ़ाइल** कैप्शन के साथ भेजें।\n"
        "👉 बॉट उसका टाइटल, साल, भाषा आदि निकालकर आपको दिखाएगा।\n"
        "👉 इसके बाद आप उसी मूवी की बाकी सभी फ़ाइलें (कोई भी क्वालिटी/एपिसोड) एक-एक करके भेज सकते हैं।\n"
        "👉 सब भेजने के बाद `/done18` लिखें।",
        parse_mode='Markdown'
    )

# ============================================================================
# 🔞 18+ BATCH LISTENER (Fully Optimized & Fixed)
# ============================================================================

# ============================================================================
# 🔞 BATCH18 MULTI-SOURCE EVIDENCE HELPERS (ISOLATED)
# ============================================================================

def batch18_parse_filename_evidence(raw_text: str) -> dict:
    """Deterministically parse an adult-series filename/caption without inventing facts."""
    raw = str(raw_text or '').strip()
    base = re.sub(r'\.(?:mkv|mp4|avi|mov|webm|ts)$', '', raw, flags=re.I)
    year_match = re.search(r'\b((?:19|20)\d{2})\b', base)
    year = year_match.group(1) if year_match else ''
    compact = re.search(r'\bS(?:EASON)?\s*0*(\d{1,2})\s*P(?:ART)?\s*0*(\d{1,3})\b', base, re.I)
    season = re.search(r'\bS(?:EASON)?\s*0*(\d{1,2})(?=\b|P)', base, re.I)
    part = re.search(r'(?:\b|(?<=\d))P(?:ART)?\s*0*(\d{1,3})\b', base, re.I)
    if compact:
        season = compact
        part = re.search(r'P(?:ART)?\s*0*(\d{1,3})\b', compact.group(0), re.I)
    extras = []
    if season:
        extras.append(f'S{int(season.group(1)):02d}')
    if part:
        extras.append(f'P{int(part.group(1)):02d}')
    for tag in ('UNRATED', 'UNCUT', 'EXTENDED', 'COMBINED', 'COMPLETE'):
        if re.search(rf'\b{tag}\b', base, re.I):
            extras.append(tag)
    technical = re.compile(
        r'\b(?:19|20)\d{2}\b|\bS(?:EASON)?\s*\d{1,2}\s*P(?:ART)?\s*\d{1,3}\b|\bS(?:EASON)?\s*\d{1,2}(?=\b|P)|(?:\b|(?<=\d))P(?:ART)?\s*\d{1,3}\b|'
        r'\b(?:UNRATED|UNCUT|EXTENDED|COMBINED|COMPLETE|SERIES|WEB\s*SERIES|HOT|ADULT|18\+|ULLU|WOOW|ATRANGII|VOOVI|KOOKU|PRIMESHOTS|PRIME\s*SHOTS|NEONX|ALTT)\b|'
        r'\b(?:\d{3,4}p|2160p|4K|HEVC|H\.?265|H\.?264|HDRIP|WEB[- ]?DL|HDTV|x26[45]|AAC|DDP?\d*|DTS|MULTI|DUAL|HINDI|ENGLISH)\b',
        re.I,
    )
    title = technical.sub(' ', base)
    title = re.sub(r'[_\.\[\]\(\){}]+', ' ', title)
    title = re.sub(r'[-]+', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip(' -')
    # Remove common uploader prefixes/suffixes only after technical cleanup.
    title = re.sub(r'^(?:www\.)?[^ ]+\s+(?=[A-Z][a-z])', '', title, flags=re.I) if title.lower().startswith(('www.', 'www ')) else title
    title = re.sub(r'\b(?:x265|x264|aac|mkv|mp4)\b', '', title, flags=re.I)
    title = re.sub(r'\s+', ' ', title).strip(' -')
    return {
        'title': title or 'UNKNOWN',
        'year': year,
        'extra_info': ' '.join(extras),
        'category': 'Adult',
        'raw': raw,
    }


def _batch18_merge_source(source_names, source_name):
    if source_name and source_name not in source_names:
        source_names.append(source_name)


def _batch18_relevant_candidate(item: dict, title: str, year: str = '') -> bool:
    """Reject generic same-word search hits before they become metadata evidence."""
    blob = re.sub(r'[^a-z0-9]+', ' ', f"{item.get('title', '')} {item.get('snippet', '')}".lower()).strip()
    target = re.sub(r'[^a-z0-9]+', ' ', str(title or '').lower()).strip()
    if not target or not blob:
        return False
    if target in blob:
        return True
    tokens = [token for token in target.split() if len(token) > 2]
    if len(tokens) < 2:
        return False
    overlap = sum(1 for token in set(tokens) if re.search(rf'\b{re.escape(token)}\b', blob))
    required = max(2, int(round(len(set(tokens)) * 0.75)))
    if overlap < required:
        return False
    return not year or str(year) in blob or overlap == len(set(tokens))


async def _batch18_cse_search(query: str, image: bool = False) -> list:
    """Use configured Google CSE only as a search transport; never treat one hit as truth."""
    api_key = os.environ.get('GOOGLE_API_KEY')
    cx_id = os.environ.get('GOOGLE_CX_ID')
    if not api_key or not cx_id:
        logger.info('Batch18 source unavailable: Google CSE credentials missing')
        return []
    try:
        params = {'key': api_key, 'cx': cx_id, 'q': query, 'num': 10}
        if image:
            params['searchType'] = 'image'
        response = await run_async(requests.get, 'https://www.googleapis.com/customsearch/v1', params=params, timeout=12)
        data = response.json()
        return data.get('items', []) or []
    except Exception as exc:
        logger.warning('Batch18 CSE query failed (%s): %s', query, exc)
        return []


async def _batch18_html_search(query: str) -> list:
    """No-key fallback: parse public search-result HTML, not search snippets as facts."""
    encoded = quote(query)
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36'}
    results = []
    for engine, url in (
        ('Bing HTML', f'https://www.bing.com/search?q={encoded}'),
        ('DDG HTML', f'https://html.duckduckgo.com/html/?q={encoded}'),
    ):
        try:
            response = await run_async(requests.get, url, headers=headers, timeout=12)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            selectors = ('li.b_algo', '.result')
            nodes = []
            for selector in selectors:
                nodes.extend(soup.select(selector))
            for node in nodes[:10]:
                anchor = node.select_one('h2 a, .result__a, a.result__url')
                if not anchor:
                    continue
                href = anchor.get('href', '')
                label = anchor.get_text(' ', strip=True)
                snippet_node = node.select_one('.b_caption p, .result__snippet')
                snippet = snippet_node.get_text(' ', strip=True) if snippet_node else node.get_text(' ', strip=True)
                if href and label:
                    results.append({'title': label, 'snippet': snippet[:600], 'link': href, 'pagemap': {}, 'engine': engine})
            if results:
                return results
        except Exception as exc:
            logger.info('Batch18 %s fallback failed: %s', engine, exc)
    return results


async def _batch18_youtube_search(title: str, year: str) -> list:
    """Prefer YouTube Data API when configured, otherwise search indexed YouTube pages."""
    key = os.environ.get('YOUTUBE_API_KEY')
    if key:
        try:
            response = await run_async(requests.get, 'https://www.googleapis.com/youtube/v3/search', params={
                'key': key, 'part': 'snippet', 'q': f'{title} {year} official trailer',
                'type': 'video', 'maxResults': 10,
            }, timeout=12)
            items = response.json().get('items', []) or []
            return [{'title': x.get('snippet', {}).get('title', ''),
                     'snippet': x.get('snippet', {}).get('description', ''),
                     'link': f"https://www.youtube.com/watch?v={x.get('id', {}).get('videoId', '')}",
                     'pagemap': {}} for x in items]
        except Exception as exc:
            logger.warning('Batch18 YouTube API failed: %s', exc)
    items = await _batch18_cse_search(f'site:youtube.com "{title}" {year} (trailer OR teaser OR "official")')
    return items or await _batch18_html_search(f'site:youtube.com "{title}" {year} (trailer OR teaser OR "official")')


async def _batch18_ocr_image(url: str) -> str:
    """Best-effort OCR from a public poster/thumbnail; optional dependency, never fatal."""
    if not url:
        return ''
    try:
        import pytesseract
        response = await run_async(requests.get, url, timeout=12, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code != 200 or not response.content:
            return ''
        image = Image.open(BytesIO(response.content))
        return (await run_async(pytesseract.image_to_string, image))[:1000].strip()
    except Exception as exc:
        logger.info('Batch18 poster OCR unavailable/failed: %s', exc)
        return ''


async def _batch18_internal_history(title: str, year: str) -> list:
    """Look for prior locally stored series records; failure must not block the batch."""
    hits = []
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return hits
        cur = conn.cursor()
        cur.execute('SELECT id, title, year, poster_url, extra_info FROM movies WHERE title ILIKE %s LIMIT 10', (f'%{title}%',))
        for row in cur.fetchall() or []:
            hits.append({'id': row[0], 'title': row[1], 'year': row[2], 'poster': row[3], 'extra_info': row[4]})
        cur.close()
    except Exception as exc:
        logger.info('Batch18 internal history unavailable: %s', exc)
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass
    return hits


async def _batch18_public_evidence(title: str, year: str) -> dict:
    """Search promotional ecosystems and return evidence, not fabricated metadata."""
    sources, candidates, snippets, posters, ocr_text = [], [], [], [], []
    queries = [
        ('YouTube', f'site:youtube.com "{title}" {year} official trailer'),
        ('Ullu social', f'site:facebook.com/Ulluappnow "{title}"'),
        ('Atrangii social', f'site:youtube.com "{title}" "Atrangii Originals"'),
        ('OTT social', f'("{title}" OR "{title.replace(" ", "-")}") {year} (Ullu OR WOOW OR Atrangii OR Kooku OR PrimeShots) trailer'),
        ('Instagram promo', f'site:instagram.com "{title}" {year}'),
        ('Archive trace', f'site:web.archive.org "{title}"'),
        ('News promo', f'"{title}" {year} cast release trailer web series'),
    ]
    for source_name, query in queries:
        items = await _batch18_cse_search(query, image=False)
        if not items:
            items = await _batch18_html_search(query)
            if items:
                source_name = f'{source_name} via web search'
        accepted_from_source = False
        for item in items[:10]:
            title_text = item.get('title', '')
            snippet = item.get('snippet', '')
            link = item.get('link', '')
            item = {'source': source_name, 'title': title_text, 'snippet': snippet, 'link': link}
            if not _batch18_relevant_candidate(item, title, year):
                continue
            candidates.append(item)
            accepted_from_source = True
            if snippet:
                snippets.append(snippet)
        if accepted_from_source:
            _batch18_merge_source(sources, source_name)
    # Direct archive lookup works even when the original OTT page is gone.
    try:
        archive_response = await run_async(requests.get, 'https://web.archive.org/cdx/search/cdx', params={
            'url': f'*{title.replace(" ", "*")}*', 'output': 'json', 'filter': 'statuscode:200',
            'fl': 'timestamp,original,statuscode,mimetype', 'collapse': 'urlkey', 'limit': 20,
        }, timeout=12)
        archive_rows = archive_response.json() if archive_response.status_code == 200 else []
        if isinstance(archive_rows, list) and len(archive_rows) > 1:
            _batch18_merge_source(sources, 'Internet Archive')
            for row in archive_rows[1:]:
                if len(row) >= 2:
                    candidates.append({'source': 'Internet Archive', 'title': title, 'snippet': f'Archived URL: {row[1]}', 'link': f'https://web.archive.org/web/{row[0]}/{row[1]}'})
    except Exception as exc:
        logger.info('Batch18 Internet Archive lookup unavailable: %s', exc)

    image_items = await _batch18_cse_search(f'"{title}" {year} poster thumbnail official', image=True)
    if image_items:
        _batch18_merge_source(sources, 'Poster/Image search')
    for item in image_items[:10]:
        image = item.get('link') or item.get('pagemap', {}).get('cse_image', [{}])[0].get('src')
        if image:
            posters.append(image)
    for poster in posters[:3]:
        text = await _batch18_ocr_image(poster)
        if text:
            ocr_text.append(text)
            candidates.append({'source': 'Poster OCR', 'title': title, 'snippet': text, 'link': poster})
            _batch18_merge_source(sources, 'Poster OCR')
    return {'sources': sources, 'candidates': candidates, 'snippets': snippets, 'posters': posters, 'ocr': ocr_text}


async def _batch18_source_pipeline(title: str, year: str) -> dict:
    evidence = await _batch18_public_evidence(title, year)
    yt = await _batch18_youtube_search(title, year)
    accepted_youtube = [x for x in yt[:10] if _batch18_relevant_candidate(x, title, year)]
    if accepted_youtube:
        _batch18_merge_source(evidence['sources'], 'YouTube')
        evidence['candidates'].extend(
            {'source': 'YouTube', 'title': x.get('title', ''), 'snippet': x.get('snippet', ''), 'link': x.get('link', '')}
            for x in accepted_youtube
        )
    history = await _batch18_internal_history(title, year)
    if history:
        _batch18_merge_source(evidence['sources'], 'Internal history')
    evidence['history'] = history
    evidence['source_count'] = len(evidence['sources'])
    evidence['candidate_count'] = len(evidence['candidates'])
    logger.info('Batch18 evidence sources=%s candidates=%s posters=%s ocr=%s history=%s', evidence['sources'] or ['none'], evidence['candidate_count'], len(evidence['posters']), len(evidence['ocr']), len(history))
    return evidence


# ============================================================================
# 🔞 ADULT METADATA COMBO ENGINE - 5 Sources Pipeline
# ============================================================================
async def fetch_adult_metadata_combo(
    movie_name: str,
    movie_year: str = "",
    movie_lang: str = "Hindi",
    raw_caption: str = "",
    raw_filename: str = ""
) -> dict:
    """
    5-source combo pipeline for adult/OTT content:
    1. TMDB (adult_mode=True)
    2. Wikipedia API (free, no key)
    3. DuckDuckGo Instant Answer (free, no key)
    4. Google Custom Search (text + image)
    5. Gemini AI (generate from knowledge - most reliable for Ullu/AltBalaji)

    Returns best combined result from all available sources.
    """
    result = {
        "title": movie_name,
        "year": int(movie_year) if str(movie_year).isdigit() else 0,
        "poster_url": None,
        "genre": "Adult, Romance, Drama",
        "imdb_id": None,
        "rating": "18+",
        "plot": None,
        "cast": "",
        "category": "Adult",
        "source": "Filename/Caption",
        "evidence_sources": [],
        "identity_status": "Filename-confirmed"
    }

    logger.info(f"🔍 Batch18 multi-source search: '{movie_name}' ({movie_year})")
    evidence = await _batch18_source_pipeline(movie_name, movie_year)
    result["evidence_sources"] = evidence.get("sources", [])
    if evidence.get("history"):
        result["identity_status"] = "Internally corroborated"
    elif evidence.get("sources"):
        result["identity_status"] = "Public-trace corroborated"
    else:
        result["identity_status"] = "Filename-confirmed; public trace not found"
    # Score independent clues; do not accept a random same-name result.
    scored = []
    normalized_title = re.sub(r'[^a-z0-9]+', ' ', movie_name.lower()).strip()
    title_tokens = {token for token in normalized_title.split() if len(token) > 2}
    for item in evidence.get("candidates", []):
        blob = re.sub(r'[^a-z0-9]+', ' ', f"{item.get('title', '')} {item.get('snippet', '')}".lower())
        overlap = len(title_tokens.intersection(blob.split()))
        score = overlap * 10
        if normalized_title and normalized_title in blob:
            score += 30
        if str(movie_year) and str(movie_year) in blob:
            score += 10
        if any(tag in blob for tag in ('official trailer', 'ullu originals', 'atrangii originals', 'woow', 'kooku', 'primeshots')):
            score += 15
        if item.get('source') in ('YouTube', 'Ullu social', 'Atrangii social', 'Internet Archive', 'Poster OCR'):
            score += 8
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if scored and scored[0][0] >= 35:
        best_score, best_item = scored[0]
        if best_item.get('snippet') and not result["plot"]:
            result["plot"] = best_item['snippet'][:500]
        result["source"] = best_item.get("source", "Public trace")
        result["identity_status"] = f"Evidence score {best_score}"
        result["evidence_sources"] = list(dict.fromkeys(result["evidence_sources"] + [best_item.get('source', 'Public trace')]))
    if not result["poster_url"] and evidence.get("posters"):
        result["poster_url"] = evidence["posters"][0]
    _batch18_merge_source(result["evidence_sources"], "Filename/Caption")

    # ─────────────────────────────────────────────
    # SOURCE 1: TMDB with include_adult=true
    # ─────────────────────────────────────────────
    # ⚠️ STRICT YEAR CHECK: Agar caption mein year diya hai aur TMDB ne
    # 3 saal se zyada purani cheez pakdi, toh wo result REJECT karo.
    # Example: Caption="2026", TMDB returned "1972" → REJECT
    try:
        tmdb_data = await run_async(fetch_movie_metadata, movie_name, movie_year, movie_lang, adult_mode=True)
        if tmdb_data:
            t_title, t_year, t_poster, t_genre, t_imdb, t_rating, t_plot, t_cat = tmdb_data

            # Year mismatch check
            year_ok = True
            if movie_year and str(movie_year).isdigit() and t_year and t_year > 0:
                caption_year = int(movie_year)
                if abs(caption_year - t_year) > 3:
                    year_ok = False
                    logger.warning(
                        f"⛔ TMDB year mismatch REJECTED: caption={caption_year}, TMDB={t_year} "
                        f"for '{movie_name}' — ye galat movie hai!"
                    )

            if year_ok:
                if t_title and t_title != movie_name: result["title"] = t_title
                if t_year and t_year > 0: result["year"] = t_year
                if t_genre and t_genre not in ("Romance, Drama", ""): result["genre"] = t_genre
                if t_imdb: result["imdb_id"] = t_imdb
                if t_rating and t_rating != "N/A": result["rating"] = t_rating
                if t_plot and len(t_plot) > 20: result["plot"] = t_plot
                result["source"] = "TMDB"
                logger.info(f"✅ TMDB accepted: {result['title']} ({t_year})")

                # Poster: agar TMDB ne direct nahi diya, TMDB search se poster dhundho
                if t_poster:
                    result["poster_url"] = t_poster
                elif t_imdb:
                    # IMDB ID se TMDB poster
                    try:
                        tmdb_api_key = "9fa44f5e9fbd41415df930ce5b81c4d7"
                        find_url = f"https://api.themoviedb.org/3/find/{t_imdb}?api_key={tmdb_api_key}&external_source=imdb_id"
                        find_resp = await run_async(requests.get, find_url, timeout=8)
                        find_data = find_resp.json()
                        all_results = find_data.get("movie_results", []) + find_data.get("tv_results", [])
                        for r in all_results:
                            if r.get("poster_path"):
                                result["poster_url"] = f"https://image.tmdb.org/t/p/original{r['poster_path']}"
                                logger.info(f"✅ TMDB poster found via IMDB ID")
                                break
                    except Exception as pe:
                        logger.warning(f"⚠️ TMDB poster fetch failed: {pe}")
                else:
                    # Title se TMDB poster search
                    try:
                        tmdb_api_key = "9fa44f5e9fbd41415df930ce5b81c4d7"
                        search_url = f"https://api.themoviedb.org/3/search/multi?api_key={tmdb_api_key}&query={quote(result['title'])}&include_adult=true"
                        search_resp = await run_async(requests.get, search_url, timeout=8)
                        search_data = search_resp.json()
                        for item in search_data.get("results", []):
                            if item.get("poster_path"):
                                result["poster_url"] = f"https://image.tmdb.org/t/p/original{item['poster_path']}"
                                logger.info(f"✅ TMDB poster found via title search")
                                break
                    except Exception as pe:
                        logger.warning(f"⚠️ TMDB title poster search failed: {pe}")

            else:
                logger.info(f"⏭️ TMDB skipped — wrong year match, trying other sources...")
    except Exception as e:
        logger.warning(f"⚠️ TMDB failed: {e}")

    # ─────────────────────────────────────────────
    # SOURCE 2: Wikipedia API (no key needed)
    # ─────────────────────────────────────────────
    try:
        wiki_query = quote(f"{movie_name} web series")
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_query}"
        wiki_resp = await run_async(requests.get, wiki_url, timeout=8)
        if wiki_resp.status_code == 200:
            wiki_data = wiki_resp.json()
            wiki_plot = wiki_data.get("extract", "")
            wiki_thumb = wiki_data.get("thumbnail", {}).get("source")
            if wiki_plot and len(wiki_plot) > 30 and not result["plot"]:
                result["plot"] = wiki_plot[:400]
                result["source"] = result["source"] + "+Wiki" if result["source"] != "Default" else "Wikipedia"
            if wiki_thumb and not result["poster_url"]:
                result["poster_url"] = wiki_thumb
            logger.info(f"✅ Wikipedia data found for: {movie_name}")
    except Exception as e:
        logger.warning(f"⚠️ Wikipedia failed: {e}")

    # Hindi Wikipedia fallback
    if not result["plot"]:
        try:
            wiki_hi_url = f"https://hi.wikipedia.org/api/rest_v1/page/summary/{quote(movie_name)}"
            wiki_resp = await run_async(requests.get, wiki_hi_url, timeout=8)
            if wiki_resp.status_code == 200:
                wiki_data = wiki_resp.json()
                wiki_plot = wiki_data.get("extract", "")
                if wiki_plot and len(wiki_plot) > 20:
                    result["plot"] = wiki_plot[:400]
                    result["source"] = result["source"] + "+HiWiki" if result["source"] != "Default" else "HiWiki"
        except Exception:
            pass

    # ─────────────────────────────────────────────
    # SOURCE 3: DuckDuckGo Instant Answer (no key)
    # ─────────────────────────────────────────────
    if not result["plot"] or not result["poster_url"]:
        try:
            ddg_query = quote(f"{movie_name} {movie_year} ullu altbalaji web series")
            ddg_url = f"https://api.duckduckgo.com/?q={ddg_query}&format=json&no_html=1&skip_disambig=1"
            ddg_resp = await run_async(requests.get, ddg_url, timeout=8,
                                       headers={"User-Agent": "Mozilla/5.0"})
            if ddg_resp.status_code == 200:
                ddg_data = ddg_resp.json()
                ddg_abstract = ddg_data.get("Abstract", "")
                ddg_image = ddg_data.get("Image", "")
                if ddg_abstract and len(ddg_abstract) > 30 and not result["plot"]:
                    result["plot"] = ddg_abstract[:400]
                    result["source"] = result["source"] + "+DDG" if result["source"] != "Default" else "DuckDuckGo"
                if ddg_image and not result["poster_url"]:
                    img_src = ddg_image if ddg_image.startswith("http") else f"https://duckduckgo.com{ddg_image}"
                    result["poster_url"] = img_src
                # Related topics se bhi kuch nikalte hain
                if not result["plot"]:
                    for topic in ddg_data.get("RelatedTopics", [])[:3]:
                        text = topic.get("Text", "")
                        if text and len(text) > 30:
                            result["plot"] = text[:400]
                            break
                logger.info(f"✅ DuckDuckGo data processed for: {movie_name}")
        except Exception as e:
            logger.warning(f"⚠️ DuckDuckGo failed: {e}")

    # ─────────────────────────────────────────────
    # SOURCE 4: Google Custom Search (text + image)
    # ─────────────────────────────────────────────
    try:
        google_data = await fetch_metadata_from_google(
            f"{movie_name} {movie_year} web series cast plot",
            movie_year
        )
        if google_data:
            if not result["poster_url"] and google_data.get("poster"):
                result["poster_url"] = google_data["poster"]
            if not result["plot"] or len(result["plot"]) < 50:
                g_plot = google_data.get("plot", "")
                if g_plot and len(g_plot) > 30:
                    result["plot"] = g_plot
            result["source"] = result["source"] + "+Google" if result["source"] != "Default" else "Google"
            logger.info(f"✅ Google data found for: {movie_name}")
    except Exception as e:
        logger.warning(f"⚠️ Google search failed: {e}")

    # Google Image search specifically for poster (separate query)
    if not result["poster_url"]:
        try:
            poster_data = await fetch_metadata_from_google(
                f"{movie_name} poster official ullu altbalaji",
                movie_year
            )
            if poster_data and poster_data.get("poster"):
                result["poster_url"] = poster_data["poster"]
        except Exception:
            pass

    # ─────────────────────────────────────────────
    # SOURCE 5: Gemini AI (optional evidence extraction only; never invent)
    # Training data mein Ullu/AltBalaji content hai
    # ─────────────────────────────────────────────
    # Sirf tab call karo jab plot ya cast khaali ho
    gemini_needed = not result["plot"] or len(result.get("plot","")) < 50 or not result["cast"]
    if gemini_needed:
        # ─────────────────────────────────────────────
        # SOURCE 5: Gemini AI with full key rotation
        # ─────────────────────────────────────────────
        import google.generativeai as genai

        api_keys = []
        std_key = os.environ.get("GEMINI_API_KEY")
        if std_key: api_keys.append(std_key)
        for i in range(1, 10):
            k = os.environ.get(f"GEMINI_API_KEY_{i}")
            if k: api_keys.append(k)

        # Platform detect
        raw_caption_lower = movie_name.lower()
        platform_hint = "Indian OTT (Ullu / AltBalaji / Akkuott / PrimeShots / Kooku)"
        for plat in ["ullu", "altbalaji", "akkuott", "primeshots", "kooku", "neonx", "hotx", "voovi", "bigmoviezoo"]:
            if plat in raw_caption_lower:
                platform_hint = plat.capitalize()
                break

        prompt = f"""You are an expert database of Indian adult OTT web series (Ullu, AltBalaji, Akkuott, PrimeShots, Kooku, NeonX, etc.).

IMPORTANT: I need metadata for a RECENT web series, NOT old Bollywood movies.

Title: "{movie_name}"
Release Year: {movie_year or "2024-2026"} <- THIS IS THE YEAR, use it exactly
Platform: {platform_hint}
Content Type: Adult / 18+ Web Series (NOT classic cinema)

STRICT RULES:
- "year" MUST be {movie_year or "2024"} or close to it — do NOT return years like 1972, 1990 etc.
- This is a web series, NOT an old film
- If you do not have verified information, return empty plot and cast. Never generate realistic or guessed facts.

Respond ONLY in this exact JSON format (no markdown, no backticks):
{{
  "title": "{movie_name}",
  "year": {movie_year or 2024},
  "genre": "Adult, Romance, Drama",
  "rating": "18+",
  "plot": "2-3 line story summary in Hindi or English",
  "cast": "Actor1, Actor2, Actress1",
  "category": "Web Series"
}}"""

        safety = {
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }

        gemini_success = False
        for key_idx, api_key in enumerate(api_keys):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-flash-latest')
                gemini_resp = await run_async(model.generate_content, prompt, safety_settings=safety)

                raw = gemini_resp.text.strip()
                raw = re.sub(r'```json\s*', '', raw)
                raw = re.sub(r'```\s*', '', raw)
                raw = re.sub(r'\s*```', '', raw).strip()

                g_data = json.loads(raw)

                if not result["plot"] or len(result["plot"]) < 50:
                    gem_plot = g_data.get("plot", "")
                    if gem_plot and len(gem_plot) > 20:
                        result["plot"] = gem_plot
                if not result["cast"]:
                    result["cast"] = g_data.get("cast", "")
                if not result["genre"] or result["genre"] == "Adult, Romance, Drama":
                    gem_genre = g_data.get("genre", "")
                    if gem_genre: result["genre"] = gem_genre
                if result["year"] == 0:
                    gem_year = g_data.get("year", 0)
                    try:
                        if int(str(gem_year)) > 2000:
                            result["year"] = int(str(gem_year))
                    except: pass

                result["source"] = result["source"] + "+Gemini" if result["source"] != "Default" else "Gemini AI"
                logger.info(f"✅ Gemini success with key #{key_idx + 1} for: {movie_name}")
                gemini_success = True
                break  # Success — baaki keys try mat karo

            except json.JSONDecodeError as je:
                logger.warning(f"⚠️ Gemini key #{key_idx+1} JSON parse failed: {je}")
                break  # JSON error = response aaya, parse fail — retry se fayda nahi
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                    logger.warning(f"⚠️ Gemini key #{key_idx+1} quota exceeded, trying next key...")
                    continue  # Next key try karo
                else:
                    logger.warning(f"⚠️ Gemini key #{key_idx+1} failed: {e}")
                    break  # Unknown error — stop

        if not gemini_success:
            logger.warning(f"⚠️ All {len(api_keys)} Gemini keys exhausted for: {movie_name}")

    # ─────────────────────────────────────────────
    # FINAL: Default fallback values fill karo
    # ─────────────────────────────────────────────
    if not result["plot"]:
        result["plot"] = ""
    if not result["genre"]:
        result["genre"] = "Adult"
    if not result["rating"]:
        result["rating"] = "18+"

    logger.info(
        f"📊 Adult Combo Result for '{movie_name}': "
        f"source={result['source']}, poster={'✅' if result['poster_url'] else '❌'}, "
        f"plot={'✅' if result['plot'] else '❌'}, cast={'✅' if result['cast'] else '❌'}"
    )

    return result


async def batch18_listener(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    🔞 18+ BATCH LISTENER: Auto-extracts metadata and saves files.
    Fixed: Clean logs, better adult detection, proper error handling.
    """
    # === GUARD CLAUSES ===
    if not BATCH_18_SESSION.get('active'):
        return
    
    if update.effective_user.id != BATCH_18_SESSION.get('admin_id'):
        return

    # Takrav se bachne ke liye checks
    if BATCH_SESSION.get('active') or SUPER_BATCH_SESSION.get('active'):
        return

    message = update.effective_message
    if not message or not (message.document or message.video):
        return

    # === PHASE 1: FIRST FILE = METADATA & MOVIE CREATION ===
    if BATCH_18_SESSION.get('movie_id') is None:
        raw_caption = (message.caption or message.text or "").strip()
        media = message.document or message.video
        raw_filename = (getattr(media, 'file_name', None) or "").strip() if media else ""
        if not raw_caption and not raw_filename:
            await message.reply_text(
                "❌ **18+ बैच:** पहली फाइल के साथ caption या filename mein series ka naam zaroor dein.",
                parse_mode='Markdown'
            )
            return
        
        # 🎯 Adult content auto-detection from both caption and actual filename
        raw_lower = f"{raw_caption} {raw_filename}".lower()
        force_adult = any(tag in raw_lower for tag in ['unrated', '18+', 'adult', 'hot', 'bhabhi', 'mastani'])
        
        status_msg = await message.reply_text(
            "🔞 **Analyzing 18+ content...**" + (" (Forced Adult Mode)" if force_adult else ""),
            quote=True
        )

        # === BATCH18 EVIDENCE EXTRACTION ===
        # Read the exact same file's caption and Telegram filename together.
        # This stays inside batch18; other batch flows are not changed here.
        try:
            ai_data = await process_file_with_evidence_engine(message)
            movie_name = ai_data.get("title", "UNKNOWN")
            movie_year = ai_data.get("year", "")
            movie_lang = ai_data.get("language", "Hindi") or "Hindi"
            movie_extra = ai_data.get("extra_info", "")
            evidence_category = ai_data.get("category", "Web Series") or "Web Series"
            logger.info(
                "Batch18 identity evidence: title=%s year=%s filename=%s extra=%s",
                movie_name, movie_year, raw_filename or "<none>", movie_extra or "<none>",
            )

            # Batch18 always remains Adult; the reconciler supplies only identity
            # fields such as clean title/year/language/season/part evidence.
            gemini_category = "Adult"
            
        except Exception as e:
            logger.error(f"Batch18 evidence extraction failed: {e}")
            fallback_text = raw_caption or raw_filename
            fallback_data = await fallback_extraction(fallback_text)
            movie_name = fallback_data.get("title", "UNKNOWN")
            movie_year = fallback_data.get("year", "")
            movie_lang = fallback_data.get("language", "Hindi") or "Hindi"
            movie_extra = fallback_data.get("extra_info", "")
            gemini_category = "Adult" if force_adult else "Web Series"

        identity_source, parsed_identity, identity_error = resolve_raw_content_identity(
            raw_caption, raw_filename
        )
        if not parsed_identity:
            _record_pending_identity(
                getattr(message.document or message.video, "file_unique_id", None),
                raw_caption,
                raw_filename,
                {"identity_source": "batch18"},
                asdict(parse_content_identity(raw_caption or raw_filename)),
                identity_error or "No plausible canonical content title was found",
            )
            await status_msg.edit_text(
                "❌ Name identify nahi ho paya. Sahi naam ke sath dobara bhejein."
            )
            return
        movie_name = parsed_identity.canonical_title
        movie_year = movie_year or parsed_identity.year or ""
        if not str(movie_year).isdigit():
            movie_year = parsed_identity.year or ""
        if parsed_identity.has_episode_marker or parsed_identity.season_number is not None:
            movie_extra = ""
        if not is_safe_canonical_title(movie_name):
            _record_pending_identity(
                getattr(message.document or message.video, "file_unique_id", None),
                raw_caption,
                raw_filename,
                {"identity_source": "batch18"},
                asdict(parsed_identity),
                "No safe canonical title remained after parsing",
            )
            await status_msg.edit_text("❌ Identity unclear; content was not added.")
            return
        
        await status_msg.edit_text(
            f"✅ **Extracted:** 🎬 `{movie_name}` ({movie_year or 'N/A'})\n"
            f"⏳ Fetching adult metadata and public evidence...",
            parse_mode='Markdown'
        )

        # === 🚀 COMBO METADATA ENGINE (5 Sources) ===
        await status_msg.edit_text(
            f"🔍 **Searching:** `{movie_name}`\n"
            f"⏳ Trying available public traces; filename/caption evidence remains primary...",
            parse_mode='Markdown'
        )

        combo = await fetch_adult_metadata_combo(movie_name, movie_year, movie_lang, raw_caption, raw_filename)

        title     = combo["title"]
        if not is_safe_canonical_title(title):
            title = movie_name
        is_episode = parsed_identity.has_episode_marker or parsed_identity.season_number is not None
        if is_episode:
            provider_title = parse_content_identity(title).canonical_title
            if provider_title.casefold() != movie_name.casefold():
                _record_pending_identity(
                    getattr(message.document or message.video, "file_unique_id", None),
                    raw_caption,
                    raw_filename,
                    ai_data if "ai_data" in locals() else {},
                    asdict(parsed_identity),
                    "Provider title does not confirm the raw episode's canonical series",
                )
                await status_msg.edit_text(
                    "⚠️ Series identity could not be verified. No movie row or file was created."
                )
                return
            title = movie_name
        year      = combo["year"]
        poster_url = combo["poster_url"]
        genre     = combo["genre"]
        imdb_id   = combo["imdb_id"]
        rating    = combo["rating"]
        plot      = combo["plot"]
        cast_str  = combo["cast"]
        category  = gemini_category  # Always keep Adult category
        data_source = combo["source"]
        evidence_sources = combo.get("evidence_sources", [])
        identity_status = combo.get("identity_status", "Unverified")
        trailer_key = await run_async(
            resolve_trailer_key, title, year, imdb_id, category, movie_name
        )

        # IMDB cast fetch (extra — agar imdb_id mila ho)
        if imdb_id and not cast_str:
            try:
                cast_str = await run_async(fetch_cast_from_imdb, imdb_id, 5)
            except Exception:
                pass

        # === DATABASE INSERTION ===
        conn = get_db_connection()
        if not conn:
            await status_msg.edit_text("❌ Database Connection Failed.")
            return

        try:
            cur = conn.cursor()
            
            tmdb_id = await run_async(
                resolve_tmdb_id_from_imdb, imdb_id, category
            )
            existing = None
            try:
                existing_id = _find_movie_by_provider_identity(cur, imdb_id, tmdb_id)
            except ValueError as exc:
                media = message.document or message.video
                _record_ingestion_evidence(
                    conn,
                    file_unique_id=getattr(media, "file_unique_id", None),
                    raw_caption=raw_caption,
                    raw_filename=raw_filename,
                    evidence=ai_data if "ai_data" in locals() else {},
                    parsed_identity=asdict(parsed_identity),
                    provider_ids={"imdb_id": imdb_id, "tmdb_id": tmdb_id},
                    movie_id=None,
                    movie_file_id=None,
                    resolver_method="conflicting_provider_identity",
                    confidence=parsed_identity.confidence,
                    status="pending_review",
                    warnings=parsed_identity.parse_warnings + (str(exc),),
                )
                conn.commit()
                cur.close()
                await status_msg.edit_text(
                    "⚠️ Provider identities conflict. No movie row or file was created."
                )
                return
            resolver_method = "provider_identity" if existing_id else "provider_metadata"
            identity_conflict = False
            ambiguous = False
            if existing_id and is_episode:
                cur.execute(
                    "SELECT title, content_type FROM movies WHERE id = %s",
                    (existing_id,),
                )
                provider_parent = cur.fetchone()
                series_types = {"web series", "tv series", "tv show", "series", "anime"}
                identity_conflict = bool(
                    not provider_parent
                    or provider_parent[0].casefold() != movie_name.casefold()
                    or str(provider_parent[1] or "").casefold() not in series_types
                )
            if not existing_id:
                series_identity = (
                    is_episode
                    or "series" in str(evidence_category).casefold()
                )
                canonical_id, ambiguous = _find_movie_by_canonical_identity(
                    cur,
                    movie_name,
                    movie_year,
                    require_series=series_identity,
                    content_type=None if series_identity else "Movie",
                )
                if canonical_id:
                    cur.execute(
                        "SELECT imdb_id, tmdb_id FROM movies WHERE id = %s",
                        (canonical_id,),
                    )
                    stored_imdb_id, stored_tmdb_id = cur.fetchone()
                    identity_conflict = bool(
                        (imdb_id and stored_imdb_id and imdb_id != stored_imdb_id)
                        or (
                            tmdb_id and stored_tmdb_id
                            and int(tmdb_id) != int(stored_tmdb_id)
                        )
                    )
                    if not identity_conflict:
                        existing_id = canonical_id
                        resolver_method = "exact_canonical_match"

            if not can_create_canonical_content(
                parsed_identity,
                provider_identity=bool(imdb_id or tmdb_id),
                existing_parent=bool(existing_id),
                ambiguous=ambiguous,
                provider_conflict=identity_conflict,
            ):
                reason = (
                    "Multiple exact canonical matches"
                    if ambiguous
                    else "Existing canonical row has conflicting provider IDs"
                    if identity_conflict
                    else "No provider identity or existing canonical content was found"
                )
                media = message.document or message.video
                _record_ingestion_evidence(
                    conn,
                    file_unique_id=getattr(media, "file_unique_id", None),
                    raw_caption=raw_caption,
                    raw_filename=raw_filename,
                    evidence=ai_data if "ai_data" in locals() else {},
                    parsed_identity=asdict(parsed_identity),
                    provider_ids={"imdb_id": imdb_id, "tmdb_id": tmdb_id},
                    movie_id=None,
                    movie_file_id=None,
                    resolver_method="unverified_content_identity",
                    confidence=parsed_identity.confidence,
                    status="pending_review",
                    warnings=parsed_identity.parse_warnings + (reason,),
                )
                conn.commit()
                cur.close()
                await status_msg.edit_text(
                    "⚠️ Content identity needs review. No movie row or file was created."
                )
                return
            if existing_id:
                cur.execute(
                    "SELECT id, poster_url, year FROM movies WHERE id = %s",
                    (existing_id,),
                )
                existing = cur.fetchone()

            if existing:
                # Update existing with better data if available
                existing_id, existing_poster, existing_year = existing
                final_poster = poster_url if poster_url else existing_poster
                final_year = year if (year and year > 0) else existing_year
                normalized_category, content_type = normalize_catalog_labels(
                    category=category,
                    language=movie_lang,
                    extra_info=movie_extra,
                    genre=genre,
                    title=title,
                )
                if parsed_identity.has_episode_marker or parsed_identity.season_number is not None:
                    content_type = "Web Series"
                
                cur.execute("""
                    UPDATE movies 
                    SET imdb_id = COALESCE(%s, imdb_id),
                        tmdb_id = COALESCE(%s, tmdb_id),
                        poster_url = COALESCE(%s, poster_url),
                        year = CASE WHEN %s > 0 THEN %s ELSE year END,
                        genre = COALESCE(%s, genre),
                        rating = COALESCE(%s, rating),
                        description = COALESCE(%s, description),
                        category = %s,
                        content_type = %s,
                        language = COALESCE(NULLIF(%s, ''), language),
                        extra_info = COALESCE(NULLIF(%s, ''), extra_info),
                        "cast" = COALESCE(%s, "cast"),
                        trailer_key = COALESCE(%s, trailer_key)
                    WHERE id = %s
                    RETURNING id
                """, (imdb_id, tmdb_id, final_poster, final_year, final_year, genre,
                      rating, plot, normalized_category, content_type, movie_lang,
                      movie_extra, cast_str, trailer_key, existing_id))
                movie_id = cur.fetchone()[0]
                logger.info(f"🔄 Updated existing movie: {title} (ID: {movie_id})")
                
            else:
                # Insert new movie
                category, content_type = normalize_catalog_labels(
                    category=category,
                    language=movie_lang,
                    extra_info=movie_extra,
                    genre=genre,
                    title=title,
                )
                if parsed_identity.has_episode_marker or parsed_identity.season_number is not None:
                    content_type = "Web Series"
                cur.execute("""
                    INSERT INTO movies 
                    (title, url, imdb_id, tmdb_id, poster_url, year, genre, rating, 
                     description, category, content_type, language, extra_info, "cast", trailer_key)
                    VALUES (%s, '', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (title, imdb_id, tmdb_id, poster_url, year, genre, rating, 
                      plot, category, content_type, movie_lang, movie_extra, cast_str, trailer_key))
                movie_id = cur.fetchone()[0]
                logger.info(f"✅ Created new movie: {title} (ID: {movie_id})")

            conn.commit()

            # Check for old files
            cur.execute("SELECT COUNT(*) FROM movie_files WHERE movie_id = %s", (movie_id,))
            file_count_old = cur.fetchone()[0]
            cur.close()

            # Update session
            BATCH_18_SESSION.update({
                'movie_id': movie_id,
                'movie_title': title,
                'file_count': 0,
                'year': str(year) if year else movie_year,
                'category': category,
                'language': movie_lang,
                'parsed_identity': asdict(parsed_identity),
                'resolver_method': resolver_method,
                'confidence': parsed_identity.confidence,
                'imdb_id': imdb_id,
                'tmdb_id': tmdb_id,
                'raw_caption': raw_caption,
                'raw_filename': raw_filename,
            })
            _record_ingestion_evidence(
                conn,
                file_unique_id=getattr(message.document or message.video, "file_unique_id", None),
                raw_caption=raw_caption,
                raw_filename=raw_filename,
                evidence=ai_data if "ai_data" in locals() else {},
                parsed_identity=asdict(parsed_identity),
                provider_ids={"imdb_id": imdb_id, "tmdb_id": tmdb_id},
                movie_id=movie_id,
                movie_file_id=None,
                resolver_method=resolver_method,
                confidence=parsed_identity.confidence,
                status="resolved",
                warnings=parsed_identity.parse_warnings,
            )
            conn.commit()

            # Build success message
            cast_display = f"\n👥 **Cast:** {cast_str}" if cast_str else ""
            poster_display = "✅ Found" if poster_url else "❌ Not Found"
            
            success_msg = (
                f"✅ **18+ Metadata Ready**\n"
                f"📡 **Sources:** {', '.join(evidence_sources) if evidence_sources else data_source}\n"
                f"🧾 **Identity:** {identity_status}\n\n"
                f"🎬 **Title:** `{title}`\n"
                f"📅 **Year:** {year if year else 'N/A'}\n"
                f"🎭 **Genre:** {genre}\n"
                f"⭐️ **Rating:** {rating}\n"
                f"🖼️ **Poster:** {poster_display}\n"
                f"🏷️ **Category:** {category}\n"
                f"{cast_display}\n"
                f"🚀 **Ab files bhejein, phir `/done18` likhein.**"
            )

            # Build keyboard
            keyboard = []
            if file_count_old > 0:
                keyboard.append([InlineKeyboardButton(
                    "🗑️ Delete OLD Files", 
                    callback_data=f"clearfiles_{movie_id}"
                )])
            keyboard.append([InlineKeyboardButton(
                "❌ Cancel Batch", 
                callback_data="cancel_batch18"
            )])

            await status_msg.edit_text(
                success_msg, 
                parse_mode='Markdown', 
                reply_markup=InlineKeyboardMarkup(keyboard)
            )

        except Exception as e:
            logger.error(f"❌ 18+ DB Error: {e}")
            if conn: conn.rollback()
            await status_msg.edit_text(f"❌ Database Error: {e}")
            return
        finally:
            close_db_connection(conn)

    # === PHASE 2: SUBSEQUENT FILES ===
    upload_status = await message.reply_text(
        "⏳ Saving 18+ file...", 
        quote=True
    )

    media = message.document or message.video
    file_unique_id = getattr(media, "file_unique_id", None)
    file_name = getattr(media, "file_name", None) or "File"
    raw_caption = message.caption or message.text or ""
    file_identity = _parse_file_identity_for_parent(
        raw_caption,
        file_name,
        (BATCH_18_SESSION.get("parsed_identity") or {}).get("canonical_title")
        or BATCH_18_SESSION.get("movie_title"),
    )
    parent_conn = get_db_connection()
    if not parent_conn:
        await upload_status.edit_text("❌ Database Connection Failed.")
        return
    try:
        cur = parent_conn.cursor()
        resolved_movie_id, episode_identity = _resolve_episode_file_parent(
            cur,
            BATCH_18_SESSION.get("movie_id"),
            raw_caption,
            file_name,
            BATCH_18_SESSION.get("year"),
        )
        if not resolved_movie_id:
            _record_ingestion_evidence(
                parent_conn,
                file_unique_id=file_unique_id,
                raw_caption=raw_caption,
                raw_filename=file_name,
                evidence={"source": "batch18"},
                parsed_identity=file_identity,
                provider_ids={
                    "imdb_id": BATCH_18_SESSION.get("imdb_id"),
                    "tmdb_id": BATCH_18_SESSION.get("tmdb_id"),
                },
                movie_id=None,
                movie_file_id=None,
                resolver_method="episode_parent_unresolved",
                confidence=file_identity.get("confidence", 0),
                status="pending_review",
                warnings=(episode_identity,),
            )
            parent_conn.commit()
            await upload_status.edit_text(
                "⚠️ Episode parent could not be uniquely verified; no file was uploaded."
            )
            return
        if resolved_movie_id != BATCH_18_SESSION.get("movie_id") or episode_identity is not None:
            cur.execute("SELECT title FROM movies WHERE id = %s", (resolved_movie_id,))
            resolved_title = cur.fetchone()[0]
            BATCH_18_SESSION.update(
                movie_id=resolved_movie_id,
                movie_title=resolved_title,
            )
            file_identity = _parse_file_identity_for_parent(
                raw_caption, file_name, resolved_title
            )
        cur.close()
    except Exception as exc:
        logger.exception("Batch18 episode parent resolution failed")
        await upload_status.edit_text("❌ Could not verify the episode's series parent.")
        return
    finally:
        close_db_connection(parent_conn)

    if file_unique_id:
        duplicate_conn = get_db_connection()
        if not duplicate_conn:
            await upload_status.edit_text("❌ Database Connection Failed.")
            return
        try:
            cur = duplicate_conn.cursor()
            cur.execute(
                "SELECT id, movie_id FROM movie_files WHERE file_unique_id = %s",
                (file_unique_id,),
            )
            prior_file = cur.fetchone()
            cur.close()
            if prior_file:
                if prior_file[1] != BATCH_18_SESSION.get("movie_id"):
                    await upload_status.edit_text(
                        "❌ This Telegram file is already attached to a different title."
                    )
                    return
                _link_movie_file_episodes(
                    duplicate_conn, prior_file[0], prior_file[1], file_identity
                )
                _record_ingestion_evidence(
                    duplicate_conn,
                    file_unique_id=file_unique_id,
                    raw_caption=raw_caption,
                    raw_filename=file_name,
                    evidence={"source": "batch18_retry"},
                    parsed_identity=file_identity,
                    provider_ids={
                        "imdb_id": BATCH_18_SESSION.get("imdb_id"),
                        "tmdb_id": BATCH_18_SESSION.get("tmdb_id"),
                    },
                    movie_id=prior_file[1],
                    movie_file_id=prior_file[0],
                    resolver_method=BATCH_18_SESSION.get("resolver_method", "batch18_session"),
                    confidence=file_identity.get(
                        "confidence", BATCH_18_SESSION.get("confidence", 0)
                    ),
                    status="resolved",
                )
                duplicate_conn.commit()
                await upload_status.edit_text("✅ Duplicate upload detected; existing file kept.")
                return
        except Exception as exc:
            logger.error("Batch18 duplicate check failed: %s", exc)
            await upload_status.edit_text("❌ Could not verify duplicate file identity.")
            return
        finally:
            close_db_connection(duplicate_conn)

    # Get storage channels
    channels = get_storage_channels()
    backup_map = {}
    
    if channels:
        for chat_id in channels:
            try:
                sent = await message.copy(chat_id=chat_id)
                backup_map[str(chat_id)] = sent.message_id
            except Exception as e:
                logger.error(f"18+ Backup failed for {chat_id}: {e}")

    # Extract file info
    file_size = (message.document.file_size if message.document 
                 else (message.video.file_size if message.video else 0))
    file_size_str = get_readable_file_size(file_size)

    # 🧹 Caption Clean: Links, @usernames, promotions hatao before quality detection
    text_for_detection = strip_caption_junk(message.caption) if message.caption else file_name
    current_lang = BATCH_18_SESSION.get('language', 'Hindi')
    label = generate_quality_label(text_for_detection, file_size_str, current_lang)

    # 🚀 FIXED: Batch ki baaki files ke liye sirf Fallback (Regex) use karein (API Key bachegi)
    try:
        ai_data_f = await fallback_extraction(text_for_detection)
        f_lang = ai_data_f.get('language', '')
        f_extra = ai_data_f.get('extra_info', '')
    except:
        f_lang = ''
        f_extra = ''
    # Build main URL
    main_url = ""
    if channels and backup_map:
        main_channel = channels[0]
        main_url = f"https://t.me/c/{str(main_channel).replace('-100', '')}/{backup_map.get(str(main_channel))}"

    # Save to database
    conn = get_db_connection()
    if conn:
        try:
            # 🛡️ Anti-Downgrade Shield: Pehle check karo ki DB mein better file toh nahi hai
            rejected, existing = is_downgrade(BATCH_18_SESSION['movie_id'], label, f_extra, conn)
            if rejected:
                logger.info(f"🛡️ Batch18: REJECTED '{label}' — DB already has better '{existing}'")
                await upload_status.edit_text(
                    f"🛡️ **Downgrade Blocked!**\n"
                    f"❌ `{label}` save nahi hua\n"
                    f"✅ DB mein pehle se better print hai: `{existing}`",
                    parse_mode='Markdown'
                )
                close_db_connection(conn)
                return

            movie_file_id = upsert_movie_file(
                conn, BATCH_18_SESSION['movie_id'], label, file_size_str, main_url,
                json.dumps(backup_map), f_lang, f_extra, file_unique_id,
            )
            _link_movie_file_episodes(
                conn, movie_file_id, BATCH_18_SESSION["movie_id"], file_identity
            )
            _record_ingestion_evidence(
                conn,
                file_unique_id=file_unique_id,
                raw_caption=raw_caption,
                raw_filename=file_name,
                evidence={"source": "batch18_file", "language": f_lang, "extra_info": f_extra},
                parsed_identity=file_identity,
                provider_ids={
                    "imdb_id": BATCH_18_SESSION.get("imdb_id"),
                    "tmdb_id": BATCH_18_SESSION.get("tmdb_id"),
                },
                movie_id=BATCH_18_SESSION["movie_id"],
                movie_file_id=movie_file_id,
                resolver_method=BATCH_18_SESSION.get("resolver_method", "batch18_session"),
                confidence=file_identity.get(
                    "confidence", BATCH_18_SESSION.get("confidence", 0)
                ),
                status="resolved",
            )
            
            BATCH_18_SESSION['file_count'] += 1

            # 🔄 Auto-Upgrade: पुरानी घटिया prints delete करो
            upgrade_msg = ""
            try:
                deleted, deleted_labels = auto_upgrade_delete(BATCH_18_SESSION['movie_id'], label, f_extra, conn)
                if deleted > 0:
                    BATCH_18_SESSION['file_count'] = max(0, BATCH_18_SESSION['file_count'] - deleted)
                    upgrade_msg = f"\n🔄 Upgraded! {deleted} पुरानी print(s) auto-deleted"
                    logger.info(f"🔄 Batch18: {deleted} पुरानी print(s) auto-deleted: {deleted_labels}")
            except Exception as ue:
                logger.error(f"Auto-Upgrade error in Batch18: {ue}")
            
            await upload_status.edit_text(
                f"✅ **Saved:** `{BATCH_18_SESSION['movie_title']} {label}` [{file_size_str}]\n"
                f"📦 Total Files: {BATCH_18_SESSION['file_count']}{upgrade_msg}",
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"18+ File Save Error: {e}")
            if conn: conn.rollback()
            await upload_status.edit_text(f"❌ Save Error: {e}")
        finally:
            close_db_connection(conn)
    else:
        await upload_status.edit_text("❌ Database connection failed")


# ============================================================================
# 🔞 18+ BATCH DONE (Optimized)
# ============================================================================

async def batch18_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Complete 18+ batch and post to adult channel"""
    
    # Validation
    if not BATCH_18_SESSION.get('active'):
        await update.message.reply_text("❌ कोई सक्रिय 18+ बैच नहीं है।")
        return

    if update.effective_user.id != BATCH_18_SESSION.get('admin_id'):
        return

    movie_id = BATCH_18_SESSION.get('movie_id')
    movie_title = BATCH_18_SESSION.get('movie_title', 'Unknown')
    file_count = BATCH_18_SESSION.get('file_count', 0)

    if not movie_id or file_count == 0:
        await update.message.reply_text(
            "❌ कोई फ़ाइल सेव नहीं की गई। बैच रद्द किया जा रहा है।"
        )
        BATCH_18_SESSION.update({
            'active': False, 'movie_id': None, 'movie_title': None,
            'file_count': 0, 'admin_id': None
        })
        return

    # Get adult channel
    adult_channel_id_str = os.environ.get('ADULT_CHANNEL_ID')
    if not adult_channel_id_str:
        await update.message.reply_text("❌ .env में ADULT_CHANNEL_ID सेट नहीं है।")
        return
    
    try:
        ADULT_CHANNEL_ID = int(adult_channel_id_str)
    except ValueError:
        await update.message.reply_text("❌ ADULT_CHANNEL_ID invalid है।")
        return

    status_msg = await update.message.reply_text(
        f"🔄 **{movie_title}** का 18+ पोस्ट बन रहा है..."
    )

    # Fetch movie data
    conn = get_db_connection()
    if not conn:
        await status_msg.edit_text("❌ Database error.")
        return

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT poster_url, year, genre, rating, language, description 
            FROM movies WHERE id = %s
        """, (movie_id,))
        m_data = cur.fetchone()
        
        if not m_data:
            await status_msg.edit_text("❌ Movie DB में नहीं मिली।")
            return
            
        poster_url, year, genre, rating, language, description = m_data

        # Get qualities
        cur.execute(
            "SELECT quality FROM movie_files WHERE movie_id = %s", 
            (movie_id,)
        )
        qrows = cur.fetchall()
        cur.close()
        
    except Exception as e:
        await status_msg.edit_text(f"❌ DB Error: {e}")
        return
    finally:
        close_db_connection(conn)

    # Build quality string
    res_list = set()
    for r in qrows:
        match = re.search(r'(\d{3,4}p)', r[0])
        if match:
            res_list.add(match.group(1))
    
    res_list = sorted(list(res_list), key=lambda x: int(x.replace('p', '')), reverse=True)
    dynamic_res = " | ".join(res_list) if res_list else "1080p | 720p | 480p"

    # Process poster
    raw_photo = poster_url if (poster_url and poster_url != 'N/A' and poster_url.startswith('http')) else None
    if raw_photo:
        photo_to_send = await make_landscape_poster(raw_photo)
    else:
        photo_to_send = DEFAULT_POSTER

    # Build caption
    safe_title = movie_title.replace('<', '').replace('>', '')
    unicode_title = get_safe_font(safe_title)
    
    style_choice = random.choice([1, 2])
    
    if style_choice == 1:
        caption = (
            f"🔞 <b>{safe_title}</b>\n"
            f"➖➖➖➖➖➖➖➖➖➖\n"
            f"✨ <b>Genre:</b> {genre or 'Romance, Drama'}\n"
            f"🔊 <b>Language:</b> {language or 'Hindi'}\n"
            f"💿 <b>Quality:</b> V2 HQ-HDTC {dynamic_res}\n"
            f"➖➖➖➖➖➖➖➖➖➖\n"
            f"<b>Update Channel:</b> <a href='https://t.me/FlimfyBoxBackUp'>Join BackUp</a>\n"
            f"👇 <b>Download Below</b> 👇"
        )
    else:
        caption = (
            f"🔥 <b>{unicode_title}</b>\n"
            f" ├ ✨ Genre: {genre or 'Romance, Drama'}\n"
            f" ├ 🔊 Language: {language or 'Hindi'}\n"
            f" └ 💿 Quality: V2 HQ-HDTC {dynamic_res}\n"
            f"━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━\n"
            f"<b>Update Channel:</b> <a href='https://t.me/FlimfyBoxBackUp'>Join BackUp</a>\n"
            f"👇 <b>Download Below</b> 👇"
        )

    # Build keyboard
    secure_url = f"https://flimfybox-bot-yht0.onrender.com/watch/{movie_id}"
    post_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Get Now", url=secure_url)],
        [InlineKeyboardButton("Join Channel", url=FILMFYBOX_CHANNEL_URL)]
    ])

    # Send to adult channel
    try:
        if hasattr(photo_to_send, 'read'):
            photo_to_send.seek(0)
            
        sent = await context.bot.send_photo(
            chat_id=ADULT_CHANNEL_ID,
            photo=photo_to_send,
            caption=caption,
            parse_mode='HTML',
            reply_markup=post_keyboard
        )

        # Save to DB for restore feature
        if sent:
            save_post_to_db(
                movie_id=movie_id,
                channel_id=ADULT_CHANNEL_ID,
                message_id=sent.message_id,
                bot_username="FlimfyBoxBot",
                caption=caption,
                media_file_id=sent.photo[-1].file_id if sent.photo else None,
                media_type="photo",
                keyboard_data=post_keyboard.to_dict(),
                topic_id=None,
                content_type="adult"
            )

        await status_msg.edit_text(
            f"✅ **18+ बैच पूर्ण!**\n\n"
            f"🎬 {movie_title}\n"
            f"📦 कुल फ़ाइलें: {file_count}\n"
            f"📢 एडल्ट चैनल में पोस्ट भेज दी गई।"
        )

    except Exception as e:
        logger.error(f"18+ Post Error: {e}")
        await status_msg.edit_text(f"❌ पोस्ट भेजने में एरर: {e}")

    # Clear session
    BATCH_18_SESSION.update({
        'active': False,
        'movie_id': None,
        'movie_title': None,
        'file_count': 0,
        'admin_id': None,
        'year': '',
        'category': '',
        'language': ''
    })


# ============================================================================
# 🔞 18+ BATCH CANCEL
# ============================================================================

async def batch18_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel active 18+ batch"""
    if update.effective_user.id == BATCH_18_SESSION.get('admin_id'):
        BATCH_18_SESSION.update({
            'active': False,
            'movie_id': None,
            'movie_title': None,
            'file_count': 0,
            'admin_id': None
        })
        await update.message.reply_text("🛑 18+ बैच रद्द कर दिया गया।")

async def admin_post_18(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Premium 18+ Post - Single Item (Fixed Crash)"""
    try:
        user_id = update.effective_user.id
        if not is_admin(user_id):
            return

        message = update.message
        replied_msg = message.reply_to_message

        media_msg    = None
        command_text = ""
        embed_link   = "" 

        if message.text and message.text.startswith('/post18'):
            command_text = message.text
            if replied_msg and (replied_msg.photo or replied_msg.video or replied_msg.document):
                media_msg = replied_msg
        elif message.caption and message.caption.startswith('/post18'):
            media_msg    = message
            command_text = message.caption

        if not command_text.startswith('/post18'): return

        status_msg = await message.reply_text("⏳ <b>Processing Premium Post...</b>", parse_mode='HTML')

        if "|" in command_text:
            parts        = command_text.split('|', 1)
            command_text = parts[0].strip()
            embed_link   = parts[1].strip()

        user_photo_id, user_video_id = None, None
        if media_msg:
            if media_msg.photo: user_photo_id = media_msg.photo[-1].file_id
            elif media_msg.video: user_video_id = media_msg.video.file_id
            elif media_msg.document:
                mime = getattr(media_msg.document, 'mime_type', '') or ''
                if "image" in mime: user_photo_id = media_msg.document.file_id
                else: user_video_id = media_msg.document.file_id

        raw_input = command_text.replace('/post18', '').strip()
        if ',' in raw_input:
            parts = raw_input.split(',', 1)
            query_text, custom_msg = parts[0].strip(), parts[1].strip()
        else:
            query_text, custom_msg = raw_input, ""

        if not query_text:
            await status_msg.edit_text("❌ Movie name missing!")
            return

        metadata = await run_async(fetch_movie_metadata, query_text)

        display_title = f"<b>{get_safe_font(query_text)}</b>"
        year_str, rating_str, genre_str = "", "", "Romance, Drama"
        plot_str = custom_msg or "Exclusive Full HD Episode."
        imdb_poster = None

        if metadata:
            m_title, m_year, m_poster, m_genre, m_imdb, m_rating, m_plot, m_cat = metadata
            if m_title and m_title != "N/A": display_title = f"<b>{get_safe_font(m_title)}</b>"
            if m_year and str(m_year) != "0": year_str = str(m_year)
            if m_genre and m_genre != "N/A": genre_str = m_genre
            if not custom_msg and m_plot and m_plot != "N/A": plot_str = m_plot[:220] + "..."
            if m_poster and m_poster != "N/A": imdb_poster = m_poster

        link_section = ""
        if embed_link:
            short_link = await shorten_link(embed_link) # Naya GPLink integration
            link_section = (
                f"\n┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n\n"
                f'📺 <b>Watch Online & Download:</b>\n👉 {short_link}'
            )

        year_display = f" ({year_str})" if year_str else ""
        channel_caption = (
            f"╔═══════════════════════╗\n"
            f"      🔥 {display_title} 🔥\n"
            f"      ━━━{year_display}━━━\n"
            f"╚═══════════════════════╝\n"
            f"\n"
            f"🔞 18+  |  💎 <b>Premium Quality</b>\n"
            f"🚨 <i>Only For Adults (18+)</i>"
            f"{link_section}\n\n"
            f"🔞 <b>Join BackUp:</b> https://t.me/FlimfyBoxBackUp" 
        )

        target_channel = os.environ.get('ADULT_CHANNEL_ID')
        if not target_channel:
            await status_msg.edit_text("❌ ADULT_CHANNEL_ID missing!")
            return

        poster_final = user_photo_id or imdb_poster or DEFAULT_POSTER
        if user_photo_id:
            try:
                telegram_file = await context.bot.get_file(user_photo_id)
                poster_final = await make_landscape_poster(
                    bytes(await telegram_file.download_as_bytearray())
                )
            except Exception as exc:
                logger.warning("Uploaded poster processing failed: %s", exc)
        elif imdb_poster:
            poster_final = await make_landscape_poster(poster_final)
        sent_post = None

        try:
            if user_video_id:
                sent_post = await context.bot.send_video(chat_id=int(target_channel), video=user_video_id, caption=channel_caption, parse_mode='HTML')
            else:
                sent_post = await context.bot.send_photo(chat_id=int(target_channel), photo=poster_final, caption=channel_caption, parse_mode='HTML')
        except Exception as post_err:
            await status_msg.edit_text(f"❌ Post failed:\n<code>{post_err}</code>", parse_mode='HTML')
            return

        await status_msg.edit_text(f"✅ <b>Premium Post Done!</b>\n🎬 Movie: <b>{query_text}</b>", parse_mode='HTML')

    except Exception as e:
        logger.error(f"Post18 Critical Error: {e}")
        try: await message.reply_text(f"❌ Error: {e}")
        except: pass
# ==================== ADMIN COMMANDS ====================
async def add_movie(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to add a movie manually (Supports Unreleased)"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Sorry Darling, sirf 𝑶𝒘𝒏𝒆𝒓 hi is command ka istemal kar sakte hain.")
        return

    conn = None
    try:
        parts = context.args
        if len(parts) < 2:
            await update.message.reply_text("Galat Format! Aise use karein:\n/addmovie MovieName Link/FileID/unreleased")
            return

        value = parts[-1]  # Last part is link/id/unreleased
        title = " ".join(parts[:-1]) # Rest is title
        if not is_safe_canonical_title(title):
            await update.message.reply_text(
                "❌ Use a canonical movie/series title without episode or release metadata."
            )
            return

        logger.info(f"Adding movie: {title} with value: {value}")

        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()

        # CASE 1: UNRELEASED MOVIE
        if value.strip().lower() == "unreleased":
            # is_unreleased = TRUE set karenge
            cur.execute(
                """
                INSERT INTO movies (title, url, file_id, is_unreleased) 
                VALUES (%s, %s, %s, %s) 
                """,
                (title.strip(), "", None, True)
            )
            message = f"✅ '{title}' ko successfully **Unreleased** mark kar diya gaya hai. (Cute message activate ho gaya ✨)"

        # CASE 2: TELEGRAM FILE ID
        elif any(value.startswith(prefix) for prefix in ["BQAC", "BAAC", "CAAC", "AQAC"]):
            cur.execute(
                """
                INSERT INTO movies (title, url, file_id, is_unreleased) 
                VALUES (%s, %s, %s, %s) 
                """,
                (title.strip(), "", value.strip(), False)
            )
            message = f"✅ '{title}' ko File ID ke sath add kar diya gaya hai."

        # CASE 3: URL LINK
        elif "http" in value or "." in value:
            normalized_url = value.strip()
            if not value.startswith(('http://', 'https://')):
                await update.message.reply_text("❌ Invalid URL format. URL must start with http:// or https://")
                return

            cur.execute(
                """
                INSERT INTO movies (title, url, file_id, is_unreleased) 
                VALUES (%s, %s, %s, %s) 
                """,
                (title.strip(), normalized_url, None, False)
            )
            message = f"✅ '{title}' ko URL ke sath add kar diya gaya hai."

        else:
            await update.message.reply_text("❌ Invalid format. Please provide valid File ID, URL, or type 'unreleased'.")
            return

        conn.commit()
        await update.message.reply_text(message)

        # Notify Users logic (Agar movie sach mein release hui hai to hi notify karein)
        if value.strip().lower() != "unreleased":
            cur.execute("SELECT id, title, url, file_id FROM movies WHERE title = %s", (title.strip(),))
            movie_found = cur.fetchone()

            if movie_found:
                movie_id, title, url, file_id = movie_found
                value_to_send = file_id if file_id else url

                num_notified = await notify_users_for_movie(context, title, value_to_send)
                # Group notification optional
                # await notify_in_group(context, title)
                await update.message.reply_text(f"📢 Notification: {num_notified} users notified.")

    except Exception as e:
        logger.error(f"Error in add_movie command: {e}")
        await update.message.reply_text(f"Ek error aaya: {e}")
    finally:
        if conn:
            close_db_connection(conn)

ASK_MOVIE, ASK_USER = range(20, 22) # Naye states

async def notify_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Step 1: Admin types /notify"""
    if update.effective_user.id not in ADMIN_IDS: return ConversationHandler.END
    
    await update.message.reply_text("🎬 <b>Smart Notify Started!</b>\n\n👉 सबसे पहले मुझे <b>Movie / Series</b> का नाम बताइए:", parse_mode='HTML')
    return ASK_MOVIE

async def notify_ask_movie(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Step 2: Admin gives Movie Name"""
    # Cancel command check
    if update.message.text == '/cancel':
        await update.message.reply_text("❌ Notify Cancelled.")
        return ConversationHandler.END
        
    context.user_data['notify_movie'] = update.message.text
    await update.message.reply_text("👤 <b>अब User का Username या User ID बताइए:</b>\n(जैसे @username या 123456789)", parse_mode='HTML')
    return ASK_USER

async def notify_ask_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Step 3: Admin gives Username/ID -> Bot sends Template using Multi-Bot"""
    if update.message.text == '/cancel':
        await update.message.reply_text("❌ Notify Cancelled.")
        return ConversationHandler.END

    user_input = update.message.text.replace('@', '').strip()
    movie_name = context.user_data.get('notify_movie', 'Movie')

    # Find user ID from DB
    conn = get_db_connection()
    if not conn:
        await update.message.reply_text("❌ DB Error!")
        return ConversationHandler.END

    try:
        cur = conn.cursor()
        if user_input.isdigit(): # ID di hai
            cur.execute("SELECT first_name, username FROM user_requests WHERE user_id = %s LIMIT 1", (int(user_input),))
            target_user_id = int(user_input)
            res = cur.fetchone()
            if res:
                first_name = res[0] or "User"
                username = res[1]
            else:
                first_name = "User"
                username = None
        else: # Username diya hai
            cur.execute("SELECT user_id, first_name, username FROM user_requests WHERE username ILIKE %s LIMIT 1", (user_input,))
            res = cur.fetchone()
            if not res:
                await update.message.reply_text(f"❌ '{user_input}' database me nahi mila. ID try karein.")
                return ConversationHandler.END
            target_user_id, first_name, username = res

        # 🎨 Beautiful Premium Template with Mention
        if username:
            user_mention_link = f"<a href='https://t.me/{username}'>{first_name}</a>"
        else:
            user_mention_link = f"<a href='tg://user?id={target_user_id}'>{first_name}</a>"
        msg = (
            f"<b>━━━━━ 🎉 𝗡𝗲𝘄 𝗨𝗽𝗱𝗮𝘁𝗲 𝗙𝗼𝗿 𝗨𝗼𝘂! ━━━━━</b>\n\n"
            f"✦ Hey {user_mention_link}!\n\n"
            f"◈ आपकी Requested File अब उपलब्ध है।\n\n"
            f"🎬 File: <b>{movie_name}</b>\n\n"
            f"इसे पाने के लिए अभी बॉट में मूवी का नाम टाइप करें और एन्जॉय करें! 😊\n\n"
            f"<b>━━━━━━━━━━━━━━━━━━━</b>\n"
            f"◈ Regards, <b>@{ADMIN_USERNAME}</b>"
        )

        # Multi-bot send function call karo
        success = await send_multi_bot_message(target_user_id, msg)

        if success:
            await update.message.reply_text(f"✅ <b>Perfect!</b> Notification successfully {first_name} ko bhej di gayi hai.", parse_mode='HTML')
        else:
            await update.message.reply_text("❌ <b>Fail!</b> User ne teeno bots ko block kar diya hai.", parse_mode='HTML')

    finally:
        close_db_connection(conn)

    return ConversationHandler.END

async def update_buttons_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return

    if len(context.args) < 2:
        await update.message.reply_text("Usage: /fixbuttons <old_bot_username> <new_bot_username>")
        return

    old_bot = context.args[0].lstrip("@")
    new_bot = context.args[1].lstrip("@")

    status_msg = await update.message.reply_text(
        "🚀 **Safe Update Mode On...**\nStarting to fix buttons slowly to avoid ban.",
        parse_mode='Markdown'
    )

    conn = get_db_connection()
    if not conn:
        await status_msg.edit_text("❌ DB connection failed.")
        return

    cur = conn.cursor()
    cur.execute(
        "SELECT movie_id, channel_id, message_id FROM channel_posts WHERE bot_username = %s",
        (old_bot,)
    )
    posts = cur.fetchall()

    total = len(posts)
    success = 0

    for (m_id, ch_id, msg_id) in posts:
        try:
            # --- SECURE LINK FOR OLD POSTS UPDATE ---
            secure_url = f"https://flimfybox-bot-yht0.onrender.com/watch/{m_id}"

            new_keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("📥 Download Server 1", url=secure_url)],
                [InlineKeyboardButton("📢 Join Channel", url=FILMFYBOX_CHANNEL_URL)]
            ])
            await context.bot.edit_message_reply_markup(
                chat_id=ch_id,
                message_id=msg_id,
                reply_markup=new_keyboard
            )

            success += 1
            await asyncio.sleep(3)
            if success % 50 == 0:
                await asyncio.sleep(10)
                await status_msg.edit_text(f"☕ Break...\nUpdated: {success}/{total}")

        except RetryAfter as e:
            await asyncio.sleep(e.retry_after + 5)
            continue
        except TelegramError as e:
            if "Message to edit not found" in str(e):
                cur.execute("DELETE FROM channel_posts WHERE channel_id = %s AND message_id = %s", (ch_id, msg_id))
                conn.commit()
            logger.error(f"Error editing {msg_id}: {e}")

    cur.close()
    close_db_connection(conn)
    await status_msg.edit_text(f"✅ Updated {success}/{total} posts safely.", parse_mode='Markdown')

async def bulk_add_movies(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Add multiple movies at once"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Sorry Darling, सिर्फ एडमिन ही इस कमांड का इस्तेमाल कर सकते हैं।")
        return

    try:
        full_text = update.message.text
        lines = full_text.split('\n')

        if len(lines) <= 1 and not context.args:
            await update.message.reply_text("""
गलत फॉर्मेट! ऐसे इस्तेमाल करें:

/bulkadd
Movie1 https://link1.com
Movie2 https://link2.com
Movie3 file_id_here
""")
            return

        success_count = 0
        failed_count = 0
        results = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('/bulkadd'):
                continue

            parts = line.split()
            if len(parts) < 2:
                failed_count += 1
                results.append(f"❌ Invalid line format: {line}")
                continue

            url_or_id = parts[-1]
            title = ' '.join(parts[:-1])
            if not is_safe_canonical_title(title):
                failed_count += 1
                results.append(f"❌ {title} - title contains episode or release metadata")
                continue

            try:
                conn = get_db_connection()
                if not conn:
                    failed_count += 1
                    results.append(f"❌ {title} - Database connection failed")
                    continue

                cur = conn.cursor()

                if any(url_or_id.startswith(prefix) for prefix in ["BQAC", "BAAC", "CAAC", "AQAC"]):
                    cur.execute(
                        "INSERT INTO movies (title, url, file_id) VALUES (%s, %s, %s)",
                        (title.strip(), "", url_or_id.strip())
                    )
                else:
                    normalized_url = normalize_url(url_or_id)
                    cur.execute(
                        "INSERT INTO movies (title, url, file_id) VALUES (%s, %s, NULL)",
                        (title.strip(), normalized_url.strip())
                    )

                conn.commit()
                close_db_connection(conn)

                success_count += 1
                results.append(f"✅ {title}")
            except Exception as e:
                failed_count += 1
                results.append(f"❌ {title} - Error: {str(e)}")

        result_message = f"""
📊 Bulk Add Results:

Successfully added: {success_count}
Failed: {failed_count}

Details:
""" + "\n".join(results[:10])

        if len(results) > 10:
            result_message += f"\n\n... और {len(results) - 10} more items"

        await update.message.reply_text(result_message)

    except Exception as e:
        logger.error(f"Error in bulk_add_movies: {e}")
        await update.message.reply_text(f"Bulk add में error: {e}")

async def add_alias(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Add an alias for an existing movie"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Sorry Darling, सिर्फ एडमिन ही इस कमांड का इस्तेमाल कर सकते हैं।")
        return

    conn = None
    try:
        if not context.args or len(context.args) < 2:
            await update.message.reply_text("गलत फॉर्मेट! ऐसे इस्तेमाल करें:\n/addalias मूवी_का_असली_नाम alias_name")
            return

        parts = context.args
        alias = parts[-1]
        movie_title = " ".join(parts[:-1])

        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()

        cur.execute("SELECT id FROM movies WHERE title = %s", (movie_title,))
        movie = cur.fetchone()

        if not movie:
            await update.message.reply_text(f"❌ '{movie_title}' डेटाबेस में नहीं मिली। पहले मूवी को add करें।")
            return

        movie_id = movie

        cur.execute(
            "INSERT INTO movie_aliases (movie_id, alias) VALUES (%s, %s) ON CONFLICT (movie_id, alias) DO NOTHING",
            (movie_id, alias.lower())
        )

        conn.commit()
        await update.message.reply_text(f"✅ Alias '{alias}' successfully added for '{movie_title}'")

    except Exception as e:
        logger.error(f"Error adding alias: {e}")
        await update.message.reply_text(f"Error: {e}")
    finally:
        if conn:
            close_db_connection(conn)

async def list_aliases(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all aliases for a movie"""
    conn = None
    try:
        if not context.args:
            await update.message.reply_text("कृपया मूवी का नाम दें:\n/aliases मूवी_का_नाम")
            return

        movie_title = " ".join(context.args)

        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()

        cur.execute("""
            SELECT m.title, COALESCE(array_agg(ma.alias), '{}'::text[])
            FROM movies m
            LEFT JOIN movie_aliases ma ON m.id = ma.movie_id
            WHERE m.title = %s
            GROUP BY m.title
        """, (movie_title,))

        result = cur.fetchone()

        if not result:
            await update.message.reply_text(f"'{movie_title}' डेटाबेस में नहीं मिली।")
            return

        title, aliases = result
        aliases_list = "\n".join(f"- {alias}" for alias in aliases) if aliases else "कोई aliases नहीं हैं"

        await update.message.reply_text(f"🎬 **{title}**\n\n**Aliases:**\n{aliases_list}", parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Error listing aliases: {e}")
        await update.message.reply_text(f"Error: {e}")
    finally:
        if conn:
            close_db_connection(conn)
async def bulk_add_aliases(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Add multiple aliases at once"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Sorry Darling, सिर्फ एडमिन ही इस कमांड का इस्तेमाल कर सकते हैं।")
        return

    conn = None
    try:
        full_text = update.message.text
        lines = full_text.split('\n')

        if len(lines) <= 1 and not context.args:
            await update.message.reply_text("""
गलत फॉर्मेट! ऐसे इस्तेमाल करें:

/aliasbulk
Movie1: alias1, alias2, alias3
Movie2: alias4, alias5
""")
            return

        success_count = 0
        failed_count = 0

        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('/aliasbulk'):
                continue

            if ':' not in line:
                continue

            movie_title, aliases_str = line.split(':', 1)
            movie_title = movie_title.strip()
            aliases = [alias.strip() for alias in aliases_str.split(',') if alias.strip()]

            cur.execute("SELECT id FROM movies WHERE title = %s", (movie_title,))
            movie = cur.fetchone()

            if not movie:
                failed_count += len(aliases)
                continue

            movie_id = movie

            for alias in aliases:
                try:
                    cur.execute(
                        "INSERT INTO movie_aliases (movie_id, alias) VALUES (%s, %s) ON CONFLICT (movie_id, alias) DO NOTHING",
                        (movie_id, alias.lower())
                    )
                    success_count += 1
                except:
                    failed_count += 1

        conn.commit()

        await update.message.reply_text(f"""
📊 Alias Bulk Add Results:

Successfully added: {success_count}
Failed: {failed_count}
""")

    except Exception as e:
        logger.error(f"Error in bulk alias add: {e}")
        await update.message.reply_text(f"Error: {e}")
    finally:
        if conn:
            close_db_connection(conn)

async def notify_manually(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manually notify users about a movie"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Sorry Darling, सिर्फ एडमिन ही इस कमांड का इस्तेमाल कर सकते हैं।")
        return

    try:
        if not context.args:
            await update.message.reply_text("Usage: /notify <movie_title>")
            return

        movie_title = " ".join(context.args)

        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()
        cur.execute("SELECT id, title, url, file_id FROM movies WHERE title ILIKE %s LIMIT 1", (f'%{movie_title}%',))
        movie_found = cur.fetchone()
        cur.close()
        close_db_connection(conn)

        if movie_found:
            movie_id, title, url, file_id = movie_found
            value_to_send = file_id if file_id else url
            num_notified = await notify_users_for_movie(context, title, value_to_send)
            await notify_in_group(context, title)
            await update.message.reply_text(f"{num_notified} users को '{title}' के लिए notify किया गया है।")
        else:
            await update.message.reply_text(f"'{movie_title}' डेटाबेस में नहीं मिली।")
    except Exception as e:
        logger.error(f"Error in notify_manually: {e}")
        await update.message.reply_text(f"एक एरर आया: {e}")

async def notify_user_by_username(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send text notification to specific user"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("⛔ Admin only command.")
        return

    try:
        if not context.args or len(context.args) < 2:
            await update.message.reply_text("Usage: /notifyuser @username Your message here")
            return

        target_username = context.args[0].replace('@', '')
        message_text = ' '.join(context.args[1:])

        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT user_id, first_name FROM user_requests WHERE username ILIKE %s LIMIT 1",
            (target_username,)
        )
        user = cur.fetchone()

        if not user:
            await update.message.reply_text(f"❌ User `@{target_username}` not found in database.", parse_mode='Markdown')
            cur.close()
            close_db_connection(conn)
            return

        user_id, first_name = user

        await context.bot.send_message(
            chat_id=user_id,
            text=message_text
        )

        await update.message.reply_text(f"✅ Message sent to `@{target_username}` ({first_name})", parse_mode='Markdown')

        cur.close()
        close_db_connection(conn)

    except telegram.error.Forbidden:
        await update.message.reply_text(f"❌ User blocked the bot.")
    except Exception as e:
        logger.error(f"Error in notify_user_by_username: {e}")
        await update.message.reply_text(f"❌ Error: {e}")

async def broadcast_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Broadcast HTML message to all users with formatting support"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("⛔ Admin only command.")
        return

    try:
        # Command ke baad wala pura text (Formatting ke sath)
        if not context.args:
            await update.message.reply_text("Usage: /broadcast <b>Message Title</b>\n\nYour formatted text here...")
            return

        # Pure message ko extract karein
        message_text = update.message.text.replace('/broadcast', '').strip()

        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()
        cur.execute("SELECT DISTINCT user_id FROM user_requests")
        all_users = cur.fetchall()

        if not all_users:
            await update.message.reply_text("No users found in database.")
            cur.close()
            close_db_connection(conn)
            return

        status_msg = await update.message.reply_text(f"📤 Broadcasting to {len(all_users)} users...\n⏳ Please wait...")

        success_count = 0
        failed_count = 0

        for user_id_tuple in all_users:
            user_id = user_id_tuple[0]
            try:
                # 📢 YAHAN PAR 'HTML' USE HOGA
                await context.bot.send_message(
                    chat_id=user_id,
                    text=message_text,
                    parse_mode='HTML',  # Isse Enter aur Bold kaam karega
                    disable_web_page_preview=True
                )
                success_count += 1
                await asyncio.sleep(0.05) # Flood protection
            except telegram.error.Forbidden:
                failed_count += 1
            except Exception as e:
                failed_count += 1

        await status_msg.edit_text(
            f"📊 <b>Broadcast Complete</b>\n\n"
            f"✅ Sent: {success_count}\n"
            f"❌ Failed: {failed_count}",
            parse_mode='HTML'
        )

        cur.close()
        close_db_connection(conn)

    except Exception as e:
        logger.error(f"Error in broadcast_message: {e}")
        await update.message.reply_text(f"❌ Error: {e}")

async def schedule_notification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Schedule a notification for later"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("⛔ Admin only command.")
        return

    try:
        if not context.args or len(context.args) < 3:
            await update.message.reply_text(
                "Usage: /schedulenotify <minutes> <@username> <message>\n"
                "Example: /schedulenotify 30 @john New movie arriving soon!"
            )
            return

        delay_minutes = int(context.args[0])
        target_username = context.args[1].replace('@', '')
        message_text = ' '.join(context.args[2:])

        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT user_id, first_name FROM user_requests WHERE username ILIKE %s LIMIT 1",
            (target_username,)
        )
        user = cur.fetchone()

        if not user:
            await update.message.reply_text(f"❌ User `@{target_username}` not found.", parse_mode='Markdown')
            cur.close()
            close_db_connection(conn)
            return

        user_id, first_name = user

        async def send_scheduled_notification():
            await asyncio.sleep(delay_minutes * 60)
            try:
                await context.bot.send_message(
                    chat_id=user_id,
                    text=message_text
                )
                logger.info(f"Scheduled notification sent to {user_id}")
            except Exception as e:
                logger.error(f"Failed to send scheduled notification to {user_id}: {e}")

        asyncio.create_task(send_scheduled_notification())

        await update.message.reply_text(
            f"⏰ Notification scheduled!\n\n"
            f"To: `@{target_username}` ({first_name})\n"
            f"Delay: {delay_minutes} minutes\n"
            f"Message: {message_text[:50]}...",
            parse_mode='Markdown'
        )

        cur.close()
        close_db_connection(conn)

    except ValueError:
        await update.message.reply_text("❌ Invalid delay. Please provide number of minutes.")
    except Exception as e:
        logger.error(f"Error in schedule_notification: {e}")
        await update.message.reply_text(f"❌ Error: {e}")

async def notify_user_with_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Notify user with media by replying to a message"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("⛔ Admin only command.")
        return

    try:
        if not update.message.reply_to_message:
            await update.message.reply_text(
                "❌ Please reply to a message (file/video/audio/photo) with:\n"
                "/notifyuserwithmedia @username Optional message"
            )
            return

        if not context.args:
            await update.message.reply_text(
                "Usage: /notifyuserwithmedia @username [optional message]\n"
                "Example: /notifyuserwithmedia @amit002 Here's your requested movie!"
            )
            return

        target_username = context.args[0].replace('@', '')
        optional_message = ' '.join(context.args[1:]) if len(context.args) > 1 else None

        replied_message = update.message.reply_to_message

        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT user_id, first_name FROM user_requests WHERE username ILIKE %s LIMIT 1",
            (target_username,)
        )
        user = cur.fetchone()

        if not user:
            await update.message.reply_text(f"❌ User `@{target_username}` not found in database.", parse_mode='Markdown')
            cur.close()
            close_db_connection(conn)
            return

        user_id, first_name = user

        notification_header = ""
        if optional_message:
            notification_header = optional_message

        warning_msg = await context.bot.send_message(
            chat_id=user_id,
            text="ᯓ➤This file automatically❕️deletes after 2 minutes❕️so please forward it to another chat જ⁀➴",
            parse_mode='Markdown'
        )

        sent_msg = None
        media_type = "unknown"
        join_keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("➡️ Join Channel", url="https://t.me/FlimfyBoxx")]])

        if replied_message.document:
            media_type = "file"
            sent_msg = await context.bot.send_document(
                chat_id=user_id,
                document=replied_message.document.file_id,
                caption=notification_header if notification_header else None,
                reply_markup=join_keyboard
            )
        elif replied_message.video:
            media_type = "video"
            sent_msg = await context.bot.send_video(
                chat_id=user_id,
                video=replied_message.video.file_id,
                caption=notification_header if notification_header else None,
                reply_markup=join_keyboard
            )
        elif replied_message.audio:
            media_type = "audio"
            sent_msg = await context.bot.send_audio(
                chat_id=user_id,
                audio=replied_message.audio.file_id,
                caption=notification_header if notification_header else None,
                reply_markup=join_keyboard
            )
        elif replied_message.photo:
            media_type = "photo"
            photo = replied_message.photo[-1]
            sent_msg = await context.bot.send_photo(
                chat_id=user_id,
                photo=photo.file_id,
                caption=notification_header if notification_header else None,
                reply_markup=join_keyboard
            )
        if sent_msg:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                # Hum save kar rahe hain ki is movie ka post is channel me is ID par hai
                cur.execute(
                    "INSERT INTO channel_posts (movie_id, channel_id, message_id, bot_username) VALUES (%s, %s, %s, %s)",
                    (movie_id, chat_id, sent_msg.message_id, "FlimfyBoxBot") # Current Main Bot Username
                )
                conn.commit()
                cur.close()
                close_db_connection(conn)
            except Exception as e:
                logger.error(f"Failed to save post ID: {e}")
        
        elif replied_message.text:
            media_type = "text"
            text_to_send = replied_message.text
            if optional_message:
                text_to_send = f"{optional_message}\n\n{text_to_send}"
            sent_msg = await context.bot.send_message(
                chat_id=user_id,
                text=text_to_send
            )
        else:
            await update.message.reply_text("❌ Unsupported media type.")
            cur.close()
            close_db_connection(conn)
            return

        if sent_msg and media_type != "text":
            asyncio.create_task(
                delete_messages_after_delay(
                    context,
                    user_id,
                    [sent_msg.message_id, warning_msg.message_id],
                    USER_FILE_DELETE_SECONDS
                )
            )

        confirmation = f"✅ **Notification Sent!**\n\n"
        confirmation += f"To: `@{target_username}` ({first_name})\n"
        confirmation += f"Media Type: {media_type.capitalize()}"

        await update.message.reply_text(confirmation, parse_mode='Markdown')

        cur.close()
        close_db_connection(conn)

    except telegram.error.Forbidden:
        await update.message.reply_text(f"❌ User blocked the bot.")
    except Exception as e:
        logger.error(f"Error in notify_user_with_media: {e}")
        await update.message.reply_text(f"❌ Error: {e}")

async def broadcast_with_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Broadcast media to all users"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("⛔ Admin only command.")
        return

    replied_message = update.message.reply_to_message
    if not replied_message:
        await update.message.reply_text("❌ Please reply to a media message to broadcast it.")
        return

    try:
        optional_message = ' '.join(context.args) if context.args else None

        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()
        cur.execute("SELECT DISTINCT user_id, first_name, username FROM user_requests")
        all_users = cur.fetchall()

        if not all_users:
            await update.message.reply_text("No users found in database.")
            cur.close()
            close_db_connection(conn)
            return

        status_msg = await update.message.reply_text(
            f"📤 Broadcasting media to {len(all_users)} users...\n⏳ Please wait..."
        )

        success_count = 0
        failed_count = 0
        join_keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("➡️ Join Channel", url="https://t.me/FlimfyBoxx")]])

        for user_id, first_name, username in all_users:
            try:
                sent_msg = None
                text_msg = None
                if optional_message:
                    text_msg = await context.bot.send_message(
                        chat_id=user_id,
                        text=optional_message
                    )

                if replied_message.document:
                    sent_msg = await context.bot.send_document(
                        chat_id=user_id,
                        document=replied_message.document.file_id,
                        reply_markup=join_keyboard
                    )
                elif replied_message.video:
                    sent_msg = await context.bot.send_video(
                        chat_id=user_id,
                        video=replied_message.video.file_id,
                        reply_markup=join_keyboard
                    )
                elif replied_message.audio:
                    sent_msg = await context.bot.send_audio(
                        chat_id=user_id,
                        audio=replied_message.audio.file_id,
                        reply_markup=join_keyboard
                    )
                elif replied_message.photo:
                    photo = replied_message.photo[-1]
                    sent_msg = await context.bot.send_photo(
                        chat_id=user_id,
                        photo=photo.file_id,
                        reply_markup=join_keyboard
                    )

                if sent_msg:
                    # copy_message returns MessageId, not a full Message object.
                    is_file_message = any(
                        getattr(sent_msg, media_type, None)
                        for media_type in ('document', 'video', 'audio', 'photo')
                    )
                    track_user_message_for_deletion(
                        context, user_id, sent_msg, is_file=is_file_message
                    )
                if text_msg:
                    track_user_message_for_deletion(context, user_id, text_msg)

                success_count += 1
                await asyncio.sleep(0.1)

            except telegram.error.Forbidden:
                failed_count += 1
            except Exception as e:
                failed_count += 1
                logger.error(f"Failed broadcast to {user_id}: {e}")

        await status_msg.edit_text(
            f"📊 **Broadcast Complete**\n\n"
            f"✅ Sent: {success_count}\n"
            f"❌ Failed: {failed_count}\n"
            f"📝 Total: {len(all_users)}"
        )

        cur.close()
        close_db_connection(conn)

    except Exception as e:
        logger.error(f"Error in broadcast_with_media: {e}")
        await update.message.reply_text(f"❌ Error: {e}")

async def quick_notify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick notify - sends media to specific requesters"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("⛔ Admin only command.")
        return

    replied_message = update.message.reply_to_message
    if not replied_message:
        await update.message.reply_text("❌ Reply to a media message first!")
        return

    if not context.args:
        await update.message.reply_text("Usage: /qnotify <@username | MovieTitle>")
        return

    try:
        query = ' '.join(context.args)

        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()

        target_users = []

        if query.startswith('@'):
            username = query.replace('@', '')
            cur.execute(
                "SELECT DISTINCT user_id, first_name, username FROM user_requests WHERE username ILIKE %s",
                (username,)
            )
            target_users = cur.fetchall()
        else:
            cur.execute(
                "SELECT DISTINCT user_id, first_name, username FROM user_requests WHERE movie_title ILIKE %s AND notified = FALSE",
                (f'%{query}%',)
            )
            target_users = cur.fetchall()

        if not target_users:
            await update.message.reply_text(f"❌ No users found for '{query}'")
            cur.close()
            close_db_connection(conn)
            return

        success_count = 0
        failed_count = 0
        join_keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("➡️ Join Channel", url="https://t.me/FlimfyBoxx")]])

        for user_id, first_name, username in target_users:
            try:
                sent_msg = None
                caption = f"🎬 {query}" if not query.startswith('@') else None
                if replied_message.document:
                    sent_msg = await context.bot.send_document(
                        chat_id=user_id,
                        document=replied_message.document.file_id,
                        caption=caption,
                        reply_markup=join_keyboard
                    )
                elif replied_message.video:
                    sent_msg = await context.bot.send_video(
                        chat_id=user_id,
                        video=replied_message.video.file_id,
                        caption=caption,
                        reply_markup=join_keyboard
                    )

                # 🛡️ AUTO-DELETE: Copyright Protection — 2 minutes baad file delete
                if sent_msg:
                    track_message_for_deletion(
                        context, user_id, sent_msg.message_id, USER_FILE_DELETE_SECONDS
                    )

                success_count += 1

                if not query.startswith('@'):
                    cur.execute(
                        "UPDATE user_requests SET notified = TRUE WHERE user_id = %s AND movie_title ILIKE %s",
                        (user_id, f'%{query}%')
                    )
                    conn.commit()

                await asyncio.sleep(0.1)

            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to send to {user_id}: {e}")

        await update.message.reply_text(
            f"✅ Sent to {success_count} user(s)\n"
            f"❌ Failed for {failed_count} user(s)\n"
            f"Query: {query}"
        )

        cur.close()
        close_db_connection(conn)

    except Exception as e:
        logger.error(f"Error in quick_notify: {e}")
        await update.message.reply_text(f"❌ Error: {e}")

async def forward_to_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Forward message from channel to user"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("⛔ Admin only command.")
        return

    replied_message = update.message.reply_to_message
    if not replied_message:
        await update.message.reply_text("❌ Reply to a message first!")
        return

    if not context.args:
        await update.message.reply_text("Usage: /forwardto @username_or_userid")
        return

    try:
        target_username = context.args[0].replace('@', '')

        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT user_id, first_name FROM user_requests WHERE username ILIKE %s LIMIT 1",
            (target_username,)
        )
        user = cur.fetchone()

        if not user:
            await update.message.reply_text(f"❌ User `@{target_username}` not found.", parse_mode='Markdown')
            cur.close()
            close_db_connection(conn)
            return

        user_id, first_name = user

        await replied_message.forward(chat_id=user_id)

        await update.message.reply_text(f"✅ Forwarded to `@{target_username}` ({first_name})", parse_mode='Markdown')

        cur.close()
        close_db_connection(conn)

    except Exception as e:
        logger.error(f"Error in forward_to_user: {e}")
        await update.message.reply_text(f"❌ Error: {e}")

async def get_user_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get user information"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("⛔ Admin only command.")
        return

    if not context.args:
        await update.message.reply_text("Usage: /userinfo @username")
        return

    try:
        target_username = context.args[0].replace('@', '')

        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()

        cur.execute("""
            SELECT
                user_id,
                username,
                first_name,
                COUNT(*) as total_requests,
                SUM(CASE WHEN notified = TRUE THEN 1 ELSE 0 END) as fulfilled,
                MAX(requested_at) as last_request
            FROM user_requests
            WHERE username ILIKE %s
            GROUP BY user_id, username, first_name
        """, (target_username,))

        user_info = cur.fetchone()

        if not user_info:
            await update.message.reply_text(f"❌ No data found for `@{target_username}`", parse_mode='Markdown')
            cur.close()
            close_db_connection(conn)
            return

        user_id, username, first_name, total, fulfilled, last_request = user_info
        fulfilled = fulfilled or 0

        cur.execute("""
            SELECT movie_title, requested_at, notified
            FROM user_requests
            WHERE user_id = %s
            ORDER BY requested_at DESC
            LIMIT 5
        """, (user_id,))
        recent_requests = cur.fetchall()

        username_str = f"`@{username}`" if username else "N/A"

        info_text = f"""
👤 **User Information**

**Basic Info:**
• Name: {first_name}
• Username: {username_str}
• User ID: `{user_id}`

**Statistics:**
• Total Requests: {total}
• Fulfilled: {fulfilled}
• Pending: {total - fulfilled}
• Last Request: {last_request.strftime('%Y-%m-%d %H:%M') if last_request else 'N/A'}

**Recent Requests:**
"""

        if recent_requests:
            for movie, req_time, notified in recent_requests:
                status = "✅" if notified else "⏳"
                info_text += f"{status} {movie} - {req_time.strftime('%m/%d %H:%M')}\n"
        else:
            info_text += "No recent requests."

        await update.message.reply_text(info_text, parse_mode='Markdown')

        cur.close()
        close_db_connection(conn)

    except Exception as e:
        logger.error(f"Error in get_user_info: {e}")
        await update.message.reply_text(f"❌ Error: {e}")

async def list_all_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all bot users with Accurate Count from Activity Log"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("⛔ Admin only command.")
        return

    try:
        page = 1
        if context.args and context.args[0].isdigit():
            page = int(context.args[0])

        per_page = 10
        offset = (page - 1) * per_page

        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()

        # 1. ✅ REAL TOTAL COUNT (From user_activity table)
        # Ye un sabhi unique users ko ginega jinhone kabhi bhi bot use kiya hai
        cur.execute("SELECT COUNT(DISTINCT user_id) FROM user_activity")
        result = cur.fetchone()
        total_users = result[0] if result else 0

        # 2. GET LIST (From user_requests table because it has Names)
        # Note: List mein shayad kam log dikhein (sirf wo jinhone request kiya hai), 
        # lekin uppar Total Count sahi dikhega.
        cur.execute("""
            SELECT 
                user_id, 
                username, 
                first_name, 
                COUNT(*) as requests, 
                MAX(requested_at) as last_seen
            FROM user_requests 
            GROUP BY user_id, username, first_name 
            ORDER BY MAX(requested_at) DESC 
            LIMIT %s OFFSET %s
        """, (per_page, offset))

        users = cur.fetchall()

        # Calculate pages based on the list available (user_requests)
        cur.execute("SELECT COUNT(DISTINCT user_id) FROM user_requests")
        listable_users = cur.fetchone()[0]
        total_pages = (listable_users + per_page - 1) // per_page if listable_users > 0 else 1

        users_text = f"👥 **Bot Users** (Page {page}/{total_pages})\n"
        users_text += f"📊 **Total Unique Users: {total_users}**\n\n"

        if not users:
            users_text += "No active requesters found on this page."
        else:
            for idx, (user_id, username, first_name, req_count, last_seen) in enumerate(users, start=offset+1):
                username_str = f"`@{username}`" if username else "N/A"
                safe_name = (first_name or "Unknown").replace("<", "&lt;").replace(">", "&gt;")
                
                users_text += f"{idx}. <b>{safe_name}</b> ({username_str})\n"
                users_text += f"   🆔 `{user_id}` | 📥 Reqs: {req_count}\n"
                users_text += f"   🕒 {last_seen.strftime('%Y-%m-%d %H:%M')}\n\n"

        if total_users > listable_users:
            users_text += f"\n⚠️ *Note:* {total_users - listable_users} users ne bot use kiya hai par koi Request nahi bheji (isliye list me naam nahi hai)."

        await update.message.reply_text(users_text, parse_mode='HTML')

        cur.close()
        close_db_connection(conn)

    except Exception as e:
        logger.error(f"Error in list_all_users: {e}")
        await update.message.reply_text(f"❌ Error: {e}")

async def get_bot_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get comprehensive bot statistics"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("⛔ Admin only command.")
        return

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("❌ Database connection failed.")
            return

        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM movies")
        total_movies = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(DISTINCT user_id) FROM user_requests")
        total_users = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM user_requests")
        total_requests = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM user_requests WHERE notified = TRUE")
        fulfilled = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM user_requests WHERE DATE(requested_at) = CURRENT_DATE")
        today_requests = cur.fetchone()[0]

        cur.execute("""
            SELECT first_name, username, COUNT(*) as req_count
            FROM user_requests
            GROUP BY user_id, first_name, username
            ORDER BY req_count DESC
            LIMIT 5
        """)
        top_users = cur.fetchall()

        fulfillment_rate = (fulfilled / total_requests * 100) if total_requests > 0 else 0

        stats_text = f"""
📊 **Bot Statistics**

**Database:**
• Movies: {total_movies}
• Users: {total_users}
• Total Requests: {total_requests}
• Fulfilled: {fulfilled}
• Pending: {total_requests - fulfilled}

**Activity:**
• Today's Requests: {today_requests}
• Fulfillment Rate: {fulfillment_rate:.1f}%

**Top Requesters:**
"""
        if top_users:
            for name, username, count in top_users:
                username_str = f"`@{username}`" if username else "N/A"
                stats_text += f"• {name} ({username_str}): {count} requests\n"
        else:
            stats_text += "No user data available."
            
        await update.message.reply_text(stats_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in get_bot_stats: {e}")
        await update.message.reply_text(f"❌ Error while fetching stats: {e}")
        
    finally:
        if cur: cur.close()
        if conn: close_db_connection(conn)

async def fix_missing_metadata(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Magic Command: Finds movies with missing info and fixes them - UPDATED
    """
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("⛔ सिर्फ एडमिन के लिए!")
        return

    status_msg = await update.message.reply_text("⏳ **Scanning Database for incomplete movies...**", parse_mode='Markdown')

    conn = get_db_connection()
    if not conn:
        await status_msg.edit_text("❌ Database connection failed.")
        return

    try:
        cur = conn.cursor()
        # Find movies where ANY key info is missing (Genre, Poster, or Year)
        cur.execute("SELECT title FROM movies WHERE genre IS NULL OR poster_url IS NULL OR year IS NULL")
        movies_to_fix = cur.fetchall()
        
        if not movies_to_fix:
            await status_msg.edit_text("✅ **All Good!** Database mein sabhi movies ka metadata complete hai.")
            return

        total = len(movies_to_fix)
        await status_msg.edit_text(f"🧐 Found **{total}** movies to fix. Starting update process... (This may take time)")

        success_count = 0
        failed_count = 0

        for index, (title,) in enumerate(movies_to_fix):
            try:
                # Progress update every 10 movies
                if index % 10 == 0:
                    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

                # ✅ FETCH CORRECT METADATA (6 Values)
                metadata = fetch_movie_metadata(title)
                if metadata:
                    new_title, year, poster_url, genre, imdb_id, rating, plot, category, seasons_data = metadata

                    # Only update if we found something useful
                    if genre or poster_url or year > 0:
                        # ✅ CORRECT SQL UPDATE QUERY (Order Matters!)
                        cur.execute("""
                            UPDATE movies 
                            SET genre = %s, 
                                poster_url = %s, 
                                year = %s, 
                                imdb_id = %s, 
                                rating = %s
                            WHERE title = %s
                        """, (genre, poster_url, year, imdb_id, rating, title))
                        
                        conn.commit()
                        success_count += 1
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
                
                # Sleep slightly to respect API limits
                await asyncio.sleep(0.5) 

            except Exception as e:
                # 🛑 ROLLBACK IS CRITICAL HERE
                if conn:
                    conn.rollback() 
                logger.error(f"Failed to fix {title}: {e}")
                failed_count += 1

        # Final Report
        await status_msg.edit_text(
            f"🎉 **Repair Complete!**\n\n"
            f"✅ Fixed: {success_count}\n"
            f"❌ Failed: {failed_count}\n"
            f"📊 Total Processed: {total}\n\n"
            f"Database updated successfully! 🚀",
            parse_mode='Markdown'
        )

    except Exception as e:
        logger.error(f"Error in fix_metadata: {e}")
        await status_msg.edit_text(f"❌ Error: {e}")
    finally:
        if cur: cur.close()
        if conn: close_db_connection(conn)

async def restore_posts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /restore <new_channel_id> <content_type> [delay]

    Examples:
    /restore -100111111111 movies       -> Sirf movies restore
    /restore -100222222222 adult        -> Sirf 18+ restore
    /restore -100333333333 series 5     -> Series, 5 sec delay
    /restore -100444444444 anime 3      -> Anime, 3 sec delay
    /restore -100111111111 all 3        -> Sab kuch (careful!)
    """
    if update.effective_user.id not in ADMIN_IDS:
        return

    # --- Argument Check ---
    if len(context.args) < 2:
        await update.message.reply_text(
            "📋 <b>Restore Command Guide:</b>\n\n"
            "<code>/restore &lt;channel_id&gt; &lt;type&gt; [delay]</code>\n\n"
            "<b>Types Available:</b>\n"
            "🎬 <code>movies</code>  - Normal movies\n"
            "🔞 <code>adult</code>   - 18+ content\n"
            "📺 <code>series</code>  - Web series\n"
            "🎌 <code>anime</code>   - Anime\n"
            "📦 <code>all</code>     - Everything\n\n"
            "<b>Examples:</b>\n"
            "<code>/restore -100123456789 movies</code>\n"
            "<code>/restore -100987654321 adult 5</code>",
            parse_mode='HTML'
        )
        return

    # --- Parse Arguments ---
    try:
        new_channel_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text(
            "❌ Channel ID galat hai!\n"
            "Sahi format: <code>-100XXXXXXXXXX</code>",
            parse_mode='HTML'
        )
        return

    content_type = context.args[1].lower().strip()

    # Valid types check
    valid_types = ['movies', 'adult', 'series', 'anime', 'all']
    if content_type not in valid_types:
        await update.message.reply_text(
            f"❌ Type galat hai: <code>{content_type}</code>\n\n"
            f"✅ Valid types: <code>{', '.join(valid_types)}</code>",
            parse_mode='HTML'
        )
        return

    # Delay (default 3 sec)
    delay = 3
    if len(context.args) > 2:
        try:
            delay = int(context.args[2])
            delay = max(2, min(delay, 30))  # 2 se 30 ke beech
        except ValueError:
            pass

    # --- Database Se Posts Nikalo ---
    conn = get_db_connection()
    if not conn:
        await update.message.reply_text("❌ Database error.")
        return

    cur = conn.cursor()

    if content_type == "all":
        cur.execute("""
            SELECT id, movie_id, caption, media_file_id,
                   media_type, keyboard_data, topic_id, content_type
            FROM channel_posts
            WHERE is_restored = FALSE OR is_restored IS NULL
            ORDER BY posted_at ASC
        """)
    else:
        cur.execute("""
            SELECT id, movie_id, caption, media_file_id,
                   media_type, keyboard_data, topic_id, content_type
            FROM channel_posts
            WHERE (is_restored = FALSE OR is_restored IS NULL)
              AND content_type = %s
            ORDER BY posted_at ASC
        """, (content_type,))

    posts = cur.fetchall()
    cur.close()
    close_db_connection(conn)

    if not posts:
        type_emoji = {
            'movies': '🎬', 'adult': '🔞',
            'series': '📺', 'anime': '🎌', 'all': '📦'
        }
        await update.message.reply_text(
            f"{type_emoji.get(content_type, '📦')} "
            f"<b>{content_type.upper()}</b> type ki koi bhi "
            f"post restore ke liye nahi mili.",
            parse_mode='HTML'
        )
        return

    total = len(posts)
    est_minutes = (total * delay) // 60

    status_msg = await update.message.reply_text(
        f"🔄 <b>Restore Starting...</b>\n\n"
        f"📦 Type: <code>{content_type.upper()}</code>\n"
        f"📊 Total Posts: <code>{total}</code>\n"
        f"⏱ Delay: <code>{delay}</code> seconds\n"
        f"⌛ Est. Time: ~<code>{est_minutes}</code> min\n\n"
        f"<i>Please wait, do not stop the bot...</i>",
        parse_mode='HTML'
    )

    success = 0
    failed  = 0
    skipped = 0

    bot_info = await context.bot.get_me()
    new_bot  = bot_info.username

    for idx, (post_id, movie_id, caption, media_file_id,
              media_type, keyboard_data_raw, topic_id, c_type) in enumerate(posts, 1):
        try:
            # 1. Keyboard Rebuild (Naye bot ke links ke saath)
            new_keyboard = None
            if keyboard_data_raw:
                try:
                    kd = (keyboard_data_raw
                          if isinstance(keyboard_data_raw, dict)
                          else json.loads(keyboard_data_raw))

                    rebuilt_rows = []
                    for row in kd.get("inline_keyboard", []):
                        new_row = []
                        for btn in row:
                            new_url = btn.get("url", "")
                            # Purane bot names replace karo
                            for old_b in [
                                "FlimfyBox_SearchBot",
                                "urmoviebot",
                                "FlimfyBoxBot"
                            ]:
                                if old_b in new_url:
                                    new_url = new_url.replace(old_b, new_bot)
                            new_row.append(
                                InlineKeyboardButton(btn["text"], url=new_url)
                            )
                        rebuilt_rows.append(new_row)

                    if rebuilt_rows:
                        new_keyboard = InlineKeyboardMarkup(rebuilt_rows)
                except Exception as kb_err:
                    logger.warning(f"Keyboard error post {post_id}: {kb_err}")

            # 2. Post Bhejo
            sent = None
            extra = {}
            if topic_id and topic_id != 100:
                extra['message_thread_id'] = topic_id

            if media_type == "photo" and media_file_id:
                sent = await safe_send(context.bot.send_photo(
                    chat_id      = new_channel_id,
                    photo        = media_file_id,
                    caption      = caption or "",
                    parse_mode   = 'Markdown',
                    reply_markup = new_keyboard,
                    **extra
                ))
            elif media_type == "video" and media_file_id:
                sent = await safe_send(context.bot.send_video(
                    chat_id      = new_channel_id,
                    video        = media_file_id,
                    caption      = caption or "",
                    parse_mode   = 'Markdown',
                    reply_markup = new_keyboard,
                    **extra
                ))
            elif caption:
                sent = await safe_send(context.bot.send_message(
                    chat_id      = new_channel_id,
                    text         = caption,
                    parse_mode   = 'Markdown',
                    reply_markup = new_keyboard,
                    **extra
                ))
            else:
                skipped += 1
                continue

            # 3. DB Update
            if sent:
                conn2 = get_db_connection()
                if conn2:
                    try:
                        cur2 = conn2.cursor()
                        cur2.execute("""
                            UPDATE channel_posts
                            SET is_restored  = TRUE,
                                restored_at  = NOW(),
                                channel_id   = %s,
                                message_id   = %s,
                                bot_username = %s
                            WHERE id = %s
                        """, (new_channel_id, sent.message_id, new_bot, post_id))
                        conn2.commit()
                        cur2.close()
                    except Exception as db_e:
                        logger.error(f"DB update error: {db_e}")
                    finally:
                        close_db_connection(conn2)
                success += 1

            # 4. Progress (Har 10 posts pe update)
            if idx % 10 == 0 or idx == total:
                try:
                    await status_msg.edit_text(
                        f"🔄 <b>Restoring {content_type.upper()}...</b>\n\n"
                        f"📊 Progress: <code>{idx}/{total}</code>\n"
                        f"✅ Success:  <code>{success}</code>\n"
                        f"❌ Failed:   <code>{failed}</code>\n"
                        f"⏭ Skipped:  <code>{skipped}</code>",
                        parse_mode='HTML'
                    )
                except Exception:
                    pass

            # 5. Delay (Telegram Flood se bachao)
            await asyncio.sleep(delay)

        except RetryAfter as e:
            wait = e.retry_after + 5
            logger.warning(f"Rate limited! Waiting {wait}s")
            try:
                await status_msg.edit_text(
                    f"⏸ <b>Telegram ne slow kiya!</b>\n"
                    f"Waiting <code>{wait}</code> seconds...\n"
                    f"Progress: <code>{idx}/{total}</code>",
                    parse_mode='HTML'
                )
            except Exception:
                pass
            await asyncio.sleep(wait)

        except telegram.error.Forbidden:
            await status_msg.edit_text(
                f"❌ <b>Bot ko channel mein admin access nahi!</b>\n\n"
                f"Steps:\n"
                f"1. Channel open karo\n"
                f"2. Bot ko Admin banao\n"
                f"3. Dobara /restore karo"
            )
            return

        except Exception as e:
            failed += 1
            logger.error(f"Restore failed post {post_id}: {e}")
            await asyncio.sleep(1)

    # Final Report
    type_emoji = {
        'movies': '🎬', 'adult': '🔞',
        'series': '📺', 'anime': '🎌', 'all': '📦'
    }
    await status_msg.edit_text(
        f"🎉 <b>Restore Complete!</b>\n\n"
        f"{type_emoji.get(content_type,'📦')} Type: "
        f"<code>{content_type.upper()}</code>\n"
        f"📦 Total:   <code>{total}</code>\n"
        f"✅ Success: <code>{success}</code>\n"
        f"❌ Failed:  <code>{failed}</code>\n"
        f"⏭ Skipped: <code>{skipped}</code>\n\n"
        f"📢 New Channel: <code>{new_channel_id}</code>\n"
        f"🤖 Bot: @{new_bot}",
        parse_mode='HTML'
    )

async def admin_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show admin commands help"""
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("⛔ Admin only command.")
        return

    help_text = """
👑 **Admin Commands Guide**

**Media Notifications:**
• `/notifyuserwithmedia @user [msg]` - Reply to media + send to user
• `/qnotify <@user|MovieTitle>` - Quick notify (reply to media)
• `/forwardto @user` - Forward channel message (reply to msg)
• `/broadcastmedia [msg]` - Broadcast media to all (reply to media)

**Text Notifications:**
• `/notifyuser @user <msg>` - Send text message
• `/broadcast <msg>` - Text broadcast to all
• `/schedulenotify <min> @user <msg>` - Schedule notification

**User Management:**
• `/userinfo @username` - Get user stats
• `/listusers [page]` - List all users

**Movie Management:**
• `/addmovie <Title> <URL|FileID>` - Add movie
• `/bulkadd` - Bulk add movies (multi-line)
• `/addalias <Title> <alias>` - Add alias
• `/aliasbulk` - Bulk add aliases (multi-line)
• `/aliases <MovieTitle>` - List aliases
• `/notify <MovieTitle>` - Auto-notify requesters

**Stats & Help:**
• `/stats` - Bot statistics
• `/adminhelp` - This help message
"""

    await update.message.reply_text(help_text, parse_mode='Markdown')

# ==================== ERROR HANDLER ====================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Log errors and handle them gracefully"""
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

    if isinstance(update, Update) and update.effective_message:
        try:
            if update.callback_query:
                await update.callback_query.answer(
                    "Something went wrong while opening this option. Please try again.",
                    show_alert=True
                )
                return

            # ✅ IMPROVED: Only send ReplyKeyboardMarkup in Private Chats to prevent Channel crashes
            is_private = update.effective_chat and update.effective_chat.type == "private"
            keyboard_markup = get_main_keyboard() if is_private else None

            error_msg = str(context.error)
            if "too many values to unpack" in error_msg:
                await update.effective_message.reply_text(
                    "❌ Error: Data format issue. Please try again.",
                    reply_markup=keyboard_markup
                )
            elif "unpacking" in error_msg:
                await update.effective_message.reply_text(
                    "❌ Error: Could not process your request. Please try again.",
                    reply_markup=keyboard_markup
                )
            else:
                error_message = await update.effective_message.reply_text(
                    "Sorry, something went wrong. Please try again later.",
                    reply_markup=keyboard_markup
                )
                if is_private:
                    track_user_message_for_deletion(
                        context,
                        update.effective_chat.id,
                        error_message
                    )
        except Exception as e:
            logger.error(f"Failed to send error message to user: {e}")

# ==================== FLASK APP (Premium Edition) ====================

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import logging
import json
import psycopg2
from datetime import datetime
import requests
from urllib.parse import quote
import random
import re
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
flask_app = Flask(__name__)
_web_app_origin = urlunparse(urlparse(WEB_APP_URL)._replace(path='', params='', query='', fragment=''))
_cors_origins = [
    origin.strip().rstrip('/')
    for origin in os.environ.get(
        'CORS_ALLOWED_ORIGINS',
        f'{_web_app_origin},http://localhost:3000,http://127.0.0.1:3000'
    ).split(',')
    if origin.strip()
]
CORS(flask_app, resources={r"/*": {"origins": _cors_origins}})

# ==================== DATABASE HELPERS (use existing functions) ====================
# Make sure these functions are already defined in your main code:
# get_db_connection(), close_db_connection(), store_user_request()
# We'll assume they are available.

from webapp_routes import register_webapp_routes

register_webapp_routes(
    flask_app,
    api_movies_cache=api_movies_cache,
    search_cache=search_cache,
    get_db_connection=get_db_connection,
    close_db_connection=close_db_connection,
    store_user_request=store_user_request,
    TMDB_API_KEY=TMDB_API_KEY,
    logger=logger
)

# Guaranteed dependency-free health endpoint. Register it only if the imported
# webapp route module has not already provided one.
def _miniapp_healthz():
    return jsonify({'status': 'ok', 'service': 'flimfybox-mini-app'}), 200


def _miniapp_readyz():
    if _shutdown_requested.is_set() or not _startup_complete.is_set():
        return jsonify({'status': 'not_ready', 'service': 'flimfybox-mini-app'}), 503
    return jsonify({'status': 'ready', 'service': 'flimfybox-mini-app'}), 200

if not any(rule.rule == '/healthz' for rule in flask_app.url_map.iter_rules()):
    flask_app.add_url_rule(
        '/healthz', 'miniapp_healthz', _miniapp_healthz, methods=['GET', 'HEAD']
    )
if not any(rule.rule == '/readyz' for rule in flask_app.url_map.iter_rules()):
    flask_app.add_url_rule(
        '/readyz', 'miniapp_readyz', _miniapp_readyz, methods=['GET', 'HEAD']
    )

# ==================== RUN FLASK ====================

def run_flask():
    """Run and supervise the Mini App HTTP server without stopping Telegram polling."""
    port = int(os.environ.get('PORT', '10000'))
    restart_delay = max(3, int(os.environ.get('MINIAPP_RESTART_DELAY', '5')))

    while not _shutdown_requested.is_set():
        server_started = False
        try:
            try:
                from waitress import serve
                logger.info("🌐 Mini App HTTP server listening via Waitress on 0.0.0.0:%s", port)
                server_started = True
                serve(flask_app, host='0.0.0.0', port=port, threads=8)
            except ImportError:
                # Keep the service reachable if a deployment omitted waitress.
                logger.warning("⚠️ waitress unavailable; using Flask threaded fallback")
                server_started = True
                flask_app.run(
                    host='0.0.0.0',
                    port=port,
                    debug=False,
                    threaded=True,
                    use_reloader=False
                )

            # A serving function returning means the web server stopped without
            # taking down the Telegram bot. Restart it after a short backoff.
            logger.error("❌ Mini App HTTP server returned unexpectedly")
        except Exception:
            logger.exception("❌ Mini App HTTP server failed")

        if _shutdown_requested.is_set():
            break
        state = 'after start' if server_started else 'before start'
        logger.warning(
            "🔁 Mini App server supervisor restarting %s in %s seconds",
            state,
            restart_delay
        )
        time.sleep(restart_delay)
    logger.info("🛑 Mini App HTTP server supervisor stopped")


# Uncomment the following lines only if you want to run Flask standalone (not recommended inside main)
# if __name__ == '__main__':
#     run_flask()

# ==================== BATCH UPLOAD HANDLERS (OLD - TO BE REMOVED) ====================

# Note: Purane batch functions ko replace kar diya gaya hai naye multi-channel batch functions se
# Isliye ye functions delete kar diye gaye hain aur unki jagah naye functions upar add kiye gaye hain.

# ==================== NEW REQUEST SYSTEM (CONFIRMATION FLOW) ====================

async def start_request_flow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Step 1: User clicks 'Request This Movie' -> Show Short & Stylish Guidelines"""
    query = update.callback_query
    try:
        await query.answer()
    except Exception:
        logger.exception("Could not acknowledge request callback")

    # Failed search results can open the request confirmation directly in chat.
    # The regular `request_` entry point remains unchanged and still asks for a name.
    if query.data.startswith("request_prefill_"):
        movie_title = unquote(query.data[len("request_prefill_"):]).strip()
        if not movie_title:
            await _replace_search_result_message(
                query,
                context,
                "⚠️ The requested title was empty. Please search again.",
            )
            return ConversationHandler.END
        context.user_data['temp_request_name'] = movie_title
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Yes, Confirm", callback_data="confirm_yes"),
            InlineKeyboardButton("❌ No, Cancel", callback_data="confirm_no")
        ]])
        try:
            await _replace_search_result_message(
                query,
                context,
                (
                    "🔔 <b>Confirmation Required</b>\n\n"
                    f"Do you want to request <b>'{html_escape(movie_title)}'</b>?"
                ),
                reply_markup=keyboard,
            )
            return CONFIRMATION
        except Exception:
            logger.exception("Could not open prefilled request confirmation")
            context.user_data.pop('temp_request_name', None)
            return ConversationHandler.END
    
    # --- NEW STYLISH & SHORT TEXT ---
    request_instruction_text = (
        "📝 𝗥𝗲𝗾𝘂𝗲𝘀𝘁 𝗥𝘂𝗹𝗲𝘀..!!\n\n"
        "बस मूवी/सीरीज़ का <b>असली नाम</b> लिखें।✔️\n\n"
        "फ़ालतू शब्द (Download, HD, Please) न लिखें।♻️\n\n"
        "<b><a href='https://www.google.com/'>𝗚𝗼𝗼𝗴𝗹𝗲</a></b> से सही स्पेलिंग चेक कर लें। ☜\n\n"
        "✐ᝰ𝗘𝘅𝗮𝗺𝗽𝗹𝗲\n\n"
        "सही है.!‼️    \n"
        "─────────────────────\n"
        "Animal ✔️ | Animal Movie Download ❌\n"
        "─────────────────────\n"
        "Mirzapur S03 ✔️ | Mirzapur New Season ❌\n"
        "─────────────────────\n\n"
        "👇 <b>अब नीचे मूवी का नाम भेजें:</b>"
    )
    
    # Message Edit karein
    try:
        request_message = await _replace_search_result_message(
            query,
            context,
            request_instruction_text,
            disable_web_page_preview=True,
        )
    except Exception:
        logger.exception("Could not open request instructions")
        return ConversationHandler.END
    
    # Is instruction message ko bhi delete list me daal dein (2 min baad)
    track_message_for_deletion(
        context,
        update.effective_chat.id,
        getattr(request_message, "message_id", query.message.message_id),
        USER_TEXT_DELETE_SECONDS,
    )
    
    # State change -> Ab Bot sirf Name ka wait karega
    return WAITING_FOR_NAME

async def handle_request_name_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Step 2: User sends name -> Bot asks for Confirmation (Not saved yet)"""
    user_name_input = update.message.text.strip()
    chat_id = update.effective_chat.id
    
    # User ka message delete karne ke liye (Clean Chat)
    track_message_for_deletion(context, chat_id, update.message.message_id, USER_TEXT_DELETE_SECONDS)

    # ✅ FIXED: Safety Check - Agar user ne koi Menu Button daba diya
    MENU_BUTTONS = ['🔍 Search Movies', '📂 Browse by Genre', '🙋 Request Movie', '📊 My Stats', '❓ Help']

    if user_name_input.startswith('/') or user_name_input in MENU_BUTTONS:
        msg = await update.message.reply_text("❌ **Request Process Cancelled.**")
        track_message_for_deletion(context, chat_id, msg.message_id, USER_TEXT_DELETE_SECONDS)
        # Us button ka original function chala do
        await main_menu_or_search(update, context)
        return ConversationHandler.END

    # Name ko temporary memory me rakho
    context.user_data['temp_request_name'] = user_name_input
    
    # Confirmation Keyboard (Yes/No)
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Yes, Confirm", callback_data="confirm_yes"),
            InlineKeyboardButton("❌ No, Cancel", callback_data="confirm_no")
        ]
    ])
    
    msg = await update.message.reply_text(
        f"🔔 <b>Confirmation Required</b>\n\n"
        f"क्या आप <b>'{user_name_input}'</b> को रिक्वेस्ट करना चाहते हैं?\n\n"
        f"नाम सही है तो <b>Yes</b> दबाएं, नहीं तो <b>No</b> दबाकर दोबारा कोशिश करें।",
        reply_markup=keyboard,
        parse_mode='HTML'
    )
    
    # ⚡ Ye Confirmation message 60 seconds me delete ho jayega
    track_message_for_deletion(context, chat_id, msg.message_id, USER_TEXT_DELETE_SECONDS)
    
    return CONFIRMATION

async def handle_confirmation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Step 3: Handle Yes/No buttons"""
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    
    choice = query.data
    user = query.from_user
    
    if choice == "confirm_no":
        await _replace_search_result_message(
            query,
            context,
            "❌ Request Cancelled. आप दोबारा सर्च या रिक्वेस्ट कर सकते हैं।",
        )
        # Cancel message auto delete in 10 seconds
        track_message_for_deletion(context, chat_id, query.message.message_id, USER_TEXT_DELETE_SECONDS)
        context.user_data.pop('temp_request_name', None)
        return ConversationHandler.END
        
    elif choice == "confirm_yes":
        movie_title = context.user_data.get('temp_request_name')
        
        # --- FINAL SAVE TO DATABASE ---
        try:
            stored = await run_async(
                store_user_request,
                user.id,
                user.username,
                user.first_name,
                movie_title,
                query.message.chat.id if query.message.chat.type != "private" else None,
                query.message.message_id,
            )
        except Exception:
            logger.exception("Could not save request from confirmation callback")
            await _replace_search_result_message(
                query,
                context,
                "⚠️ The request could not be submitted right now. Please try again.",
            )
            context.user_data.pop('temp_request_name', None)
            return ConversationHandler.END
        
        if stored:
            # Notify Admin
            group_info = query.message.chat.title if query.message.chat.type != "private" else None
            await send_admin_notification(context, user, movie_title, group_info)
            
            success_text = f"""
✅ <b>Request Sent to Admin!</b>

🎬 Movie: <b>{html_escape(str(movie_title or ''))}</b>

📝 आपकी रिक्वेस्ट 𝑶𝒘𝒏𝒆𝒓 <b>@Ownermahi</b> / <b>@Ownermahi</b> को मिली गई है।
⏳ जैसे ही मूवी उपलब्ध होगी, वो खुद आपको यहाँ सूचित (Notify) कर देंगे।

<i>हमसे जुड़े रहने के लिए धन्यवाद! 🙏</i>
            """
            await _replace_search_result_message(
                query, context, success_text, parse_mode='HTML'
            )
        else:
            await _replace_search_result_message(
                query,
                context,
                "❌ Error: Request save नहीं हो पाई। शायद यह पहले से पेंडिंग है।",
            )
            
        # ⚡ Success Message Auto Delete (60 Seconds)
        track_message_for_deletion(context, chat_id, query.message.message_id, USER_TEXT_DELETE_SECONDS)
            
        context.user_data.pop('temp_request_name', None)
        return ConversationHandler.END

async def timeout_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """2 Minute Timeout Handler"""
    if update.effective_message:
        msg = await update.effective_message.reply_text("⏳ <b>Session Expired:</b> रिक्वेस्ट का समय समाप्त हो गया।", parse_mode='HTML')
        track_message_for_deletion(context, update.effective_chat.id, msg.message_id, USER_TEXT_DELETE_SECONDS)
    return ConversationHandler.END

async def main_menu_or_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 👇 SABSE PEHLE SAFEGUARD LAGAYEIN: Ignore channel posts or anonymous updates
    if not update.effective_user:
        return

    if SEARCH_TIMING_ENABLED:
        logger.info("Search timing stage=update_received")
        
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    pre_search_query = (
        update.message.text.strip()
        if update.message and update.message.text
        else ""
    )
    is_menu_query = pre_search_query in {
        '🔍 Search Movies', '📊 My Stats', '❓ Help', '🙋 Request Movie'
    }
    progress_message = None
    if pre_search_query and not is_menu_query:
        progress_message = await send_search_progress(
            update, context, pre_search_query
        )
        if progress_message:
            context.user_data['_search_progress_message'] = progress_message

    # 👇 VIP Payment UTR Check 👇 (Ab yeh safe hai kyunki channel filter ho chuka hai)
    if context.user_data and context.user_data.get('payment_step') == 'utr':
        await payment_utr_handler(update, context)
        return
        
    # === 1. FSub Check (Only in Private Chat) ===
    if update.effective_chat.type == "private":
        fsub_start = time.perf_counter() if SEARCH_TIMING_ENABLED else None
        check = await is_user_member(context, user_id)
        if fsub_start is not None:
            logger.info(
                "Search timing stage=fsub_check duration_ms=%d",
                int((time.perf_counter() - fsub_start) * 1000),
            )
        if not check['is_member']:
            if update.message and update.message.text:
                context.user_data['pending_search_query'] = update.message.text.strip()

            msg = await update.message.reply_text(
                get_join_message(check['channel'], check['group']),
                reply_markup=get_join_keyboard(),
                parse_mode='Markdown'
            )
            track_message_for_deletion(context, chat_id, msg.message_id, USER_TEXT_DELETE_SECONDS)
            await remove_search_progress(progress_message)
            context.user_data.pop('_search_progress_message', None)
            return
    # ============================================

    if not update.message or not update.message.text:
        return

    query_text = update.message.text.strip()
    
    # === 2. Menu Button Logic ===
    if query_text == '🔍 Search Movies':
        msg = await update.message.reply_text("Great! Just type the name of the movie you want to search for.")
        track_message_for_deletion(context, chat_id, msg.message_id, USER_TEXT_DELETE_SECONDS)
        return

    elif query_text == '🙋 Request Movie':
        web_app_url = WEB_APP_URL
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 Open Request Portal", web_app=WebAppInfo(url=web_app_url))]
        ])
        msg = await update.message.reply_text(
            "👇 **स्मार्ट रिक्वेस्ट पोर्टल:**\n\nयहाँ मूवी का नाम सर्च करें। अगर स्पेलिंग गलत हुई, तो हमारा AI उसे सही कर देगा और आप सीधा रिक्वेस्ट भेज पाएंगे!", 
            reply_markup=keyboard, 
            parse_mode='Markdown'
        )
        track_message_for_deletion(context, chat_id, msg.message_id, USER_TEXT_DELETE_SECONDS)
        return

    elif query_text == '📊 My Stats':
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM user_requests WHERE user_id = %s", (user_id,))
                req = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM user_requests WHERE user_id = %s AND notified = TRUE", (user_id,))
                ful = cur.fetchone()[0]
                
                stats_msg = await update.message.reply_text(
                    f"📊 **Your Stats**\n\n📝 Total Requests: {req}\n✅ Fulfilled: {ful}",
                    parse_mode='Markdown'
                )
                track_message_for_deletion(context, chat_id, stats_msg.message_id, USER_TEXT_DELETE_SECONDS)
            except Exception as e:
                logger.error(f"Stats Error: {e}")
            finally:
                close_db_connection(conn)
        return

    elif query_text == '❓ Help':
        help_text = (
            "🤖 **How to use:**\n\n"
            "1. **Search:** Just type any movie name (e.g., 'Avengers').\n"
            "2. **Request:** If not found, use the Request button.\n"
            "3. **Download:** Click the buttons provided."
        )
        msg = await update.message.reply_text(help_text, parse_mode='Markdown')
        track_message_for_deletion(context, chat_id, msg.message_id, USER_TEXT_DELETE_SECONDS)
        return

    # === 3. If no button matched, Search for the Movie ===
    await search_movies(update, context)

# 👇👇👇 IS FUNCTION KO REPLACE KARO (Line ~1665) 👇👇👇

async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Search a group message and report results, absence, or failure."""
    handler_entry = time.perf_counter()
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    if text.startswith('/'):
        return
    if len(text) < 2:
        return

    requester_id = update.effective_user.id if update.effective_user else None
    search_started = handler_entry
    if SEARCH_TIMING_ENABLED:
        logger.info("Search timing stage=group_handler_entry query=%r", text[:80])
    progress_message = None
    progress_start = time.perf_counter()
    try:
        progress_message = await update.message.reply_text(
            f'🔎 Search for "{html_escape(text[:100])}"...'
        )
    except Exception:
        logger.exception("Could not send group-search progress")
    if SEARCH_TIMING_ENABLED:
        logger.info(
            "Search timing stage=group_progress query=%r telegram_ms=%.2f",
            text[:80],
            (time.perf_counter() - progress_start) * 1000,
        )
    try:
        movies = await run_async(get_movies_fast_sql, text, limit=5)
    except Exception:
        logger.exception("Group movie search failed for query %r", text[:200])
        try:
            response_started = time.perf_counter()
            await _update_search_progress(
                progress_message,
                update.message,
                "⚠️ Search is temporarily unavailable. Please try again.",
            )
            if SEARCH_TIMING_ENABLED:
                logger.info(
                    "Group search timing query=%r total_ms=%.2f response_ms=%.2f "
                    "fallback_ms=0 result=error",
                    text[:80],
                    (time.perf_counter() - search_started) * 1000,
                    (time.perf_counter() - response_started) * 1000,
                )
        except Exception:
            logger.exception("Could not send group-search error response")
        return

    try:
        schedule_recommendation_event(
            user_id=requester_id,
            event_type='group_search',
            source='group',
            metadata={'query': text[:200], 'chat_id': update.effective_chat.id},
        )
    except Exception:
        logger.exception("Could not record group-search event")

    if not movies:
        result_format_start = time.perf_counter()
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "🔎 Check spelling",
                url="https://www.google.com/search?{}".format(
                    urlencode({"q": text[:200]})
                ),
            )
        ]])
        if SEARCH_TIMING_ENABLED:
            logger.info(
                "Search timing stage=group_result_format query=%r format_ms=%.2f",
                text[:80],
                (time.perf_counter() - result_format_start) * 1000,
            )
        try:
            response_started = time.perf_counter()
            await _update_search_progress(
                progress_message,
                update.message,
                (
                    "❌ No matching movie found for:\n"
                    f"<b>{html_escape(text[:200])}</b>\n\n"
                    "Check the spelling, add the year, or search privately with /start."
                ),
                reply_markup=keyboard,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            if SEARCH_TIMING_ENABLED:
                logger.info(
                    "Group search timing query=%r total_ms=%.2f response_ms=%.2f "
                    "fallback_ms=0 result=not_found",
                    text[:80],
                    (time.perf_counter() - search_started) * 1000,
                    (time.perf_counter() - response_started) * 1000,
                )
        except Exception:
            logger.exception("Could not send group no-results response")
        return

    selection_start = time.perf_counter()
    try:
        chosen_movie = _select_single_search_result(text, movies)
    except Exception:
        logger.exception("Could not select a group-search result")
        try:
            await _update_search_progress(
                progress_message,
                update.message,
                "⚠️ Search results could not be displayed. Please try again."
            )
        except Exception:
            logger.exception("Could not send group result error response")
        return
    if SEARCH_TIMING_ENABLED:
        logger.info(
            "Search timing stage=group_result_selection query=%r python_ms=%.2f",
            text[:80],
            (time.perf_counter() - selection_start) * 1000,
        )

    if chosen_movie:
        movie_id, title, url, file_id = chosen_movie[:4]
        try:
            schedule_recommendation_event(
                user_id=requester_id,
                event_type='group_selection',
                source='group',
                movie_id=movie_id,
                metadata={
                    'query': text[:200],
                    'title': title,
                    'chat_id': update.effective_chat.id,
                },
            )
        except Exception:
            logger.exception("Could not record group result selection")
        try:
            response_started = time.perf_counter()
            await process_movie_exact_match(
                update,
                context,
                movie_id,
                title,
                status_message=progress_message,
            )
            if SEARCH_TIMING_ENABLED:
                logger.info(
                    "Group search timing query=%r total_ms=%.2f response_ms=%.2f "
                    "fallback_ms=0 result=exact",
                    text[:80],
                    (time.perf_counter() - search_started) * 1000,
                    (time.perf_counter() - response_started) * 1000,
                )
        except Exception:
            logger.exception("Could not display exact group-search result")
            try:
                await _update_search_progress(
                    progress_message,
                    update.message,
                    "⚠️ Search results could not be displayed. Please try again.",
                )
            except Exception:
                logger.exception("Could not send group result error response")
        return

    result_format_start = time.perf_counter()
    try:
        if requester_id is not None:
            context.user_data['search_results'] = movies
            context.user_data['search_query'] = text
            keyboard = create_movie_selection_keyboard(
                movies, page=0, requester_id=requester_id
            )
        else:
            keyboard = None
        result_text = (
            f"<b>🎬 Search results</b>\n\n"
            f"Found <b>{len(movies)}</b> results for "
            f"'<b>{html_escape(text[:200])}</b>'. Select the correct title:"
        )
        if SEARCH_TIMING_ENABLED:
            logger.info(
                "Search timing stage=group_result_format query=%r format_ms=%.2f",
                text[:80],
                (time.perf_counter() - result_format_start) * 1000,
            )
        response_started = time.perf_counter()
        msg = await _update_search_progress(
            progress_message,
            update.message,
            result_text,
            reply_markup=keyboard,
            parse_mode='HTML',
        )
        if SEARCH_TIMING_ENABLED:
            logger.info(
                "Group search timing query=%r total_ms=%.2f response_ms=%.2f "
                "fallback_ms=0 result=multiple",
                text[:80],
                (time.perf_counter() - search_started) * 1000,
                (time.perf_counter() - response_started) * 1000,
            )
    except Exception:
        logger.exception("Could not send group movie search results")
        try:
            await update.message.reply_text(
                "⚠️ Search results could not be displayed. Please try again."
            )
        except Exception:
            logger.exception("Could not send group result error response")
        return
    try:
        track_message_for_deletion(
            context,
            update.effective_chat.id,
            msg.message_id,
            USER_TEXT_DELETE_SECONDS,
        )
    except Exception:
        logger.exception("Could not schedule group result cleanup")

async def group_member_welcome(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome new human members with a clickable display-name mention."""
    if not update.message or not update.message.new_chat_members:
        return
    if update.effective_chat.type not in ("group", "supergroup"):
        return

    for member in update.message.new_chat_members:
        if member.is_bot:
            continue

        first_name = html_escape((member.first_name or "there").strip())
        identity = f"<a href='tg://user?id={member.id}'>{first_name}</a>"
        group_name = html_escape(
            (update.effective_chat.title or "this group").strip()
        )
        app_button = InlineKeyboardMarkup([
            # Telegram Web App buttons are not valid in group chats. A normal
            # HTTPS URL opens the same Mini App safely from a group welcome.
            [InlineKeyboardButton("🎬 Open FlimfyBox", url=WEB_APP_URL)]
        ])
        welcome_text = (
            f"<b>Hey ♥️ {identity}, Welcome to {group_name}...</b>"
        )
        msg = await update.message.reply_text(
            welcome_text,
            parse_mode="HTML",
            reply_markup=app_button
        )
        track_user_message_for_deletion(context, update.effective_chat.id, msg)

async def web_app_data_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mini App se aane wali movie ID ko receive karega aur movie bhejega"""
    if update.effective_message.web_app_data:
        received_data = update.effective_message.web_app_data.data
        chat_id = update.effective_chat.id
        
        if received_data.startswith("movie_"):
            movie_id = int(received_data.split("_")[1])
            
            # Loading message dikhayein
            status_msg = await context.bot.send_message(chat_id=chat_id, text="⏳ <b>Fetching your movie from Web App...</b>", parse_mode='HTML')
            
            # Movie bhejne wala purana function call karein
            await deliver_movie_on_start(update, context, movie_id)
            
            try:
                await status_msg.delete()
            except:
                pass

async def upcoming_reminder_worker(app: Application):
    """Process release and local-availability notifications independently."""
    try:
        bot_info = await app.bot.get_me()
        logger.info(f"📣 Upcoming reminder worker started for @{bot_info.username}")
    except Exception as exc:
        logger.error(f"Upcoming reminder worker startup failed: {exc}")
        return

    while True:
        conn = None
        try:
            conn = get_db_connection()
            if conn is None:
                await asyncio.sleep(60)
                continue

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, tmdb_id, movie_title, release_date,
                           release_notification_requested,
                           availability_notification_requested
                    FROM upcoming_notifications
                    WHERE (release_notification_requested AND release_notified_at IS NULL
                           AND release_date <= CURRENT_DATE)
                       OR (availability_notification_requested
                           AND availability_notified_at IS NULL)
                    ORDER BY created_at ASC
                    LIMIT 50
                    """
                )
                rows = cur.fetchall()

            if not rows:
                await asyncio.sleep(60)
                continue

            for notification_id, user_id, tmdb_id, title, release_date, release_requested, availability_requested in rows:
                try:
                    normalized_tmdb_id = str(tmdb_id).replace('tmdb_', '').strip()
                    if not normalized_tmdb_id or not normalized_tmdb_id.isdigit():
                        raise ValueError(f'Invalid tmdb_id for reminder: {tmdb_id!r}')

                    cur = conn.cursor()
                    cur.execute("""
                        SELECT EXISTS(
                            SELECT 1
                            FROM movies m
                            LEFT JOIN movie_files mf ON mf.movie_id = m.id
                            WHERE (
                                m.tmdb_id = %s
                                OR (
                                    m.tmdb_id IS NULL
                                    AND LOWER(REGEXP_REPLACE(m.title, '[^a-z0-9]', '', 'g'))
                                        = LOWER(REGEXP_REPLACE(%s, '[^a-z0-9]', '', 'g'))
                                )
                            )
                            AND (
                                NULLIF(m.url, '') IS NOT NULL
                                OR NULLIF(m.file_id, '') IS NOT NULL
                                OR NULLIF(mf.url, '') IS NOT NULL
                                OR NULLIF(mf.file_id, '') IS NOT NULL
                            )
                        )
                    """, (int(normalized_tmdb_id), title))
                    local_available = bool(cur.fetchone()[0])
                    cur.close()
                    today = datetime.utcnow().date()
                    if release_requested and release_date <= today:
                        await safe_send(app.bot.send_message(
                            chat_id=int(user_id),
                            text=(
                                f"🔔 <b>{html_escape(title or 'This title')}</b> is now released!\n\n"
                                f"You can now rate this title.\n"
                                f"Download will be available separately."
                            ),
                            parse_mode='HTML'
                        ))
                        with conn.cursor() as cur:
                            cur.execute("""
                                UPDATE upcoming_notifications
                                SET release_notified_at = CURRENT_TIMESTAMP,
                                    notified_at = COALESCE(notified_at, CURRENT_TIMESTAMP)
                                WHERE id = %s AND release_notified_at IS NULL
                            """, (notification_id,))
                        conn.commit()
                    if availability_requested and local_available:
                        await safe_send(app.bot.send_message(
                            chat_id=int(user_id),
                            text=(
                                f"📥 <b>{html_escape(title or 'This title')}</b> is now available for download!\n\n"
                                f"The download is now ready on FlimfyBox."
                            ),
                            parse_mode='HTML'
                        ))
                        with conn.cursor() as cur:
                            cur.execute("""
                                UPDATE upcoming_notifications
                                SET availability_notified_at = CURRENT_TIMESTAMP
                                WHERE id = %s AND availability_notified_at IS NULL
                            """, (notification_id,))
                        conn.commit()
                except Exception as exc:
                    logger.warning('Upcoming reminder processing failed for user_id=%s: %s', user_id, exc)
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception('Upcoming reminder worker error: %s', exc)
        finally:
            if conn:
                close_db_connection(conn)
        await asyncio.sleep(45)


async def auto_delete_worker(app: Application):
    """
    Background worker jo har 5 second me DB check karega, 
    messages delete karega aur fir DB se bhi entry uda dega (Self-Cleaning).
    """
    try:
        bot_info = await app.bot.get_me()
        bot_username = bot_info.username
    except Exception as e:
        logger.error(f"Worker bot info error: {e}")
        return

    logger.info(f"🧹 Auto-Delete Worker Started for @{bot_username}")

    while True:
        conn = None
        try:
            conn = get_db_connection()
            if conn:
                cur = conn.cursor()
                # 1. Wo messages dhoondo jinka time pura ho chuka hai
                cur.execute(
                    "SELECT id, chat_id, message_id FROM auto_delete_queue WHERE bot_username = %s AND delete_at <= NOW() LIMIT 50",
                    (bot_username,)
                )
                rows = cur.fetchall()
                
                for row in rows:
                    row_id, chat_id, msg_id = row
                    
                    # 2. Telegram se file delete karo
                    try:
                        await app.bot.delete_message(chat_id=chat_id, message_id=msg_id)
                    except telegram.error.BadRequest as exc:
                        # Telegram returns BadRequest for an already deleted
                        # message. Remove that queue row; retry other failures.
                        if "not found" in str(exc).lower() or "message to delete not found" in str(exc).lower():
                            cur.execute("DELETE FROM auto_delete_queue WHERE id = %s", (row_id,))
                            conn.commit()
                        else:
                            logger.warning(
                                "Auto-delete retry needed for chat=%s message=%s: %s",
                                chat_id, msg_id, exc
                            )
                    except Exception as exc:
                        logger.warning(
                            "Auto-delete retry needed for chat=%s message=%s: %s",
                            chat_id, msg_id, exc
                        )
                    else:
                        cur.execute("DELETE FROM auto_delete_queue WHERE id = %s", (row_id,))
                        conn.commit()
                    
                cur.close()
        except Exception as e:
            logger.error(f"Auto-delete worker error: {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
        finally:
            if conn:
                close_db_connection(conn)
            
        # Har 5 second me database check karega
        await asyncio.sleep(5)


async def keep_miniapp_alive_worker():
    """Check the local server and, when configured, the public health URL."""
    port = int(os.environ.get('PORT', '10000'))
    local_url = f'http://127.0.0.1:{port}/healthz'
    configured_url = (os.environ.get('PUBLIC_URL') or WEB_APP_URL or '').strip()
    parsed_url = urlparse(configured_url)
    public_url = (
        f'{parsed_url.scheme}://{parsed_url.netloc}/healthz'
        if parsed_url.scheme and parsed_url.netloc else ''
    )
    targets = list(dict.fromkeys([local_url] + ([public_url] if public_url else [])))
    logger.info('💓 Mini App keep-alive worker started: %s', ', '.join(targets))

    timeout = aiohttp.ClientTimeout(total=15)
    headers = {'User-Agent': 'FlimfyBox-MiniApp-HealthCheck/1.0'}
    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            while True:
                for health_url in targets:
                    try:
                        async with session.get(health_url, allow_redirects=False) as response:
                            if response.status != 200:
                                logger.warning('⚠️ Mini App health check %s returned HTTP %s', health_url, response.status)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning('⚠️ Mini App health check failed for %s: %s', health_url, exc)
                await asyncio.sleep(300)
    except asyncio.CancelledError:
        logger.info('🛑 Mini App keep-alive worker stopped')
        raise

# 👇 YAHAN SE COPY KARO AUR EXACTLY 'def register_handlers' KE THEEK UPAR PASTE KARO 👇

async def payment_photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Agar user screenshot stage par hai
    if context.user_data.get('payment_step') == 'screenshot':
        context.user_data['screenshot_id'] = update.message.photo[-1].file_id
        context.user_data['payment_step'] = 'utr'
        await update.message.reply_text(
            "✅ <b>Screenshot Received!</b>\n\n🔢 Ab <b>UTR ya Reference Number</b> type karke bhejein.", 
            parse_mode='HTML'
        )
        return True # Matlab photo handle ho gayi
    return False

async def payment_utr_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Agar user UTR stage par hai
    if context.user_data.get('payment_step') == 'utr':
        utr_number = update.message.text.strip()
        user = update.effective_user
        screenshot_id = context.user_data.get('screenshot_id')
        
        # Admin ko alert bhejna
        admin_id = int(os.environ.get('ADMIN_USER_ID', '123456789')) 
        admin_text = (
            f"🔔 <b>NEW PAYMENT PENDING</b>\n\n"
            f"👤 Name: {user.first_name}\n"
            f"🆔 ID: <code>{user.id}</code>\n"
            f"🔢 UTR: <code>{utr_number}</code>"
        )
        try:
            await context.bot.send_photo(chat_id=admin_id, photo=screenshot_id, caption=admin_text, parse_mode='HTML')
        except Exception as e:
            pass
            
        await update.message.reply_text(
            "⏳ <b>Verification Pending!</b>\n\n✅ Payment details admin ko bhej di gayi hai. Thodi der me VIP access mil jayega.", 
            parse_mode='HTML'
        )
        # Process complete, ab reset kar do
        context.user_data.pop('payment_step', None)
        context.user_data.pop('screenshot_id', None)
        return True
    return False


# ==================== MULTI-BOT SETUP (REPLACES OLD MAIN) ====================

def register_handlers(application: Application):
    """
    यह फंक्शन हर बॉट पर लॉजिक (Handlers) सेट करेगा।
    ताकि तीनों बॉट्स सेम काम करें।
    """
    # -----------------------------------------------------------
    # 1. NEW REQUEST SYSTEM HANDLER (With 2 Min Timeout)
    # -----------------------------------------------------------
    # नोट: ConversationHandler को हर बार नया बनाना जरूरी है
    request_conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(start_request_flow, pattern="^request_")],
        states={
            WAITING_FOR_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_request_name_input)
            ],
            CONFIRMATION: [
                CallbackQueryHandler(handle_confirmation_callback, pattern="^confirm_")
            ]
        },
        fallbacks=[
            CommandHandler('cancel', cancel),
            CommandHandler('start', start)
        ],
        conversation_timeout=120,
    )
    application.add_handler(request_conv_handler)

    notify_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("notify", notify_start)],
        states={
            ASK_MOVIE: [MessageHandler(filters.TEXT & ~filters.COMMAND, notify_ask_movie)],
            ASK_USER: [MessageHandler(filters.TEXT & ~filters.COMMAND, notify_ask_user)]
        },
        fallbacks=[CommandHandler('cancel', notify_ask_user)], # Dummy fallback to catch /cancel
        conversation_timeout=120
    )
    application.add_handler(notify_conv_handler)
    
    # -----------------------------------------------------------
    # 2. GLOBAL HANDLERS
    # -----------------------------------------------------------

    # 👇 YAHAN PAR 'application' LIKHNA HAI 'app' KI JAGAH 👇
    
    
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE, main_menu_or_search))
    
    # Button Callback
    application.add_handler(CallbackQueryHandler(button_callback))

    # -----------------------------------------------------------
    # 3. ADMIN & BATCH COMMANDS
    # -----------------------------------------------------------
    application.add_handler(CommandHandler("addmovie", add_movie))
    application.add_handler(CommandHandler("bulkadd", bulk_add_movies))
    application.add_handler(CommandHandler("addalias", add_alias))
    application.add_handler(CommandHandler("aliases", list_aliases))
    application.add_handler(CommandHandler("aliasbulk", bulk_add_aliases))
    # Add handler to collect media groups in private chats for post_query albums
    application.add_handler(MessageHandler((filters.PHOTO | filters.VIDEO) & filters.ChatType.PRIVATE, global_album_cacher), group=-2)
    application.add_handler(MessageHandler((filters.PHOTO | filters.VIDEO) & filters.ChatType.PRIVATE, collect_post_query_album), group=-1)
    application.add_handler(MessageHandler((filters.PHOTO | filters.VIDEO) & filters.CaptionRegex(r'^/post_query'), admin_post_query))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex(r'^/post_query'), admin_post_query_text))
    application.add_handler(MessageHandler(filters.Regex(r'^/post18'), admin_post_18))
    application.add_handler(CommandHandler("fixbuttons", update_buttons_command))
    application.add_handler(CommandHandler("restore", restore_posts_command))

    # 🚀 NEW: Add this line to catch the poster image
    application.add_handler(MessageHandler(filters.PHOTO & filters.ChatType.PRIVATE, handle_admin_poster), group=0)

    # 🚀 SUPER BATCH COMMANDS
    # ✅ superbatch_listener HATA DIYA — ab pm_file_listener hi "muh" hai
    # Jab SUPER_BATCH_SESSION active ho, pm_file_listener (group=2) khud files collect karta hai
    application.add_handler(CommandHandler("superbatch", superbatch_start))
    application.add_handler(CommandHandler("superdone", superbatch_done))
    
    
    # ==========================================
    # 🔞 18+ BATCH SYSTEM HANDLERS
    # ==========================================
    application.add_handler(CommandHandler("batch18", batch18_start))
    application.add_handler(CommandHandler("done18", batch18_done))
    application.add_handler(CommandHandler("cancel18", batch18_cancel))

    
    # ✅ FIX: group=1 जोड़ा गया ताकि यह दूसरे फाइल्स को ब्लॉक न करे
    application.add_handler(MessageHandler(filters.ChatType.PRIVATE & filters.FORWARDED, batch18_listener), group=1)
    
    # Batch Commands
    application.add_handler(CommandHandler("batch", batch_add_command))
    application.add_handler(CommandHandler("done", batch_done_command))
    application.add_handler(CommandHandler("batchid", batch_id_command))
    application.add_handler(CommandHandler("fixdata", fix_missing_metadata))
    application.add_handler(CommandHandler("post", post_to_topic_command))
    
    # ✅ FIX: group=2 — Sirf PM (Private Chat) mein hi pm_file_listener chalega
    # Channel ya group se koi bhi message yahan nahi aayega
    application.add_handler(MessageHandler(
        filters.ChatType.PRIVATE &
        (filters.Document.ALL | filters.VIDEO | filters.PHOTO | (filters.TEXT & ~filters.COMMAND)),
        pm_file_listener
    ), group=2)

    # -----------------------------------------------------------
    # 4. GENRE & GROUP HANDLERS
    # -----------------------------------------------------------
    application.add_handler(CommandHandler("genres", show_genre_selection))
    application.add_handler(
        MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, group_member_welcome)
    )
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS, handle_group_message))

    application.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, web_app_data_handler))
      
    # -----------------------------------------------------------
    # 5. NOTIFICATION & STATS
    # -----------------------------------------------------------
    application.add_handler(CommandHandler("notifyuser", notify_user_by_username))
    application.add_handler(CommandHandler("broadcast", broadcast_message))
    application.add_handler(CommandHandler("schedulenotify", schedule_notification))
    application.add_handler(CommandHandler("notifyuserwithmedia", notify_user_with_media))
    application.add_handler(CommandHandler("qnotify", quick_notify))
    application.add_handler(CommandHandler("forwardto", forward_to_user))
    application.add_handler(CommandHandler("broadcastmedia", broadcast_with_media))

    application.add_handler(CommandHandler("userinfo", get_user_info))
    application.add_handler(CommandHandler("listusers", list_all_users))
    application.add_handler(CommandHandler("adminhelp", admin_help))
    application.add_handler(CommandHandler("stats", get_bot_stats))

    # Error Handler
    application.add_error_handler(error_handler)


async def main():
    """Main function to run MULTIPLE bots concurrently"""
    logger.info("🚀 Starting Multi-Bot System...")
    stop_signal = asyncio.Event()

    def request_shutdown(signum, _frame):
        logger.info("🛑 Shutdown signal received: %s", signal.Signals(signum).name)
        _shutdown_requested.set()
        stop_signal.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(signum, request_shutdown)
        except (OSError, ValueError):
            logger.warning("Signal handler unavailable for %s", signum)

    # =================================================================
    # 1. Flask Server FIRST (Render timeout se bachao)
    # =================================================================
    flask_thread        = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    logger.info("🌐 Flask server started.")

    # =================================================================
    # 2. Database Setup
    # =================================================================
    try:
        run_migrations(DATABASE_URL)
    except Exception as e:
        logger.exception("❌ Database migrations failed; startup aborted: %s", e)
        _shutdown_requested.set()
        close_db_pool()
        return

    # 3. Get Tokens from ENV
    # =================================================================
    tokens = [
        os.environ.get("TELEGRAM_BOT_TOKEN"),  # Bot 1
        os.environ.get("BOT_TOKEN_2"),          # Bot 2
        os.environ.get("BOT_TOKEN_3")           # Bot 3
    ]

    # Khali tokens filter karo aur duplicate hatao
    tokens = list(set([t for t in tokens if t]))

    if not tokens:
        logger.error("❌ No tokens found! Check Environment Variables.")
        _shutdown_requested.set()
        close_db_pool()
        return

    # =================================================================
    # 4. Initialize & Start All Bots
    # =================================================================
    apps = []
    logger.info(f"🤖 Found {len(tokens)} tokens. Initializing bots...")

    worker_tasks = []
    for i, token in enumerate(tokens):
        if _shutdown_requested.is_set():
            break
        try:
            logger.info(f"🔹 Initializing Bot {i+1}...")

            app = (
                Application.builder()
                .token(token)
                # A slow file/metadata operation must not hold every other
                # user's update in the polling queue.
                .concurrent_updates(32)
                .read_timeout(30)
                .write_timeout(30)
                .build()
            )

            register_handlers(app)

            await app.initialize()
            await app.start()
            await app.updater.start_polling(drop_pending_updates=True)
            worker_tasks.append(asyncio.create_task(auto_delete_worker(app)))
            apps.append(app)

            bot_info = await app.bot.get_me()
            logger.info(f"✅ Bot {i+1} Started: @{bot_info.username}")

        except Exception as e:
            logger.error(f"❌ Failed to start Bot {i+1}: {e}")

    if not apps:
        logger.error("❌ No bots could be started.")
        _shutdown_requested.set()
        close_db_pool()
        return

    # Keep the Mini App route warm independently of Telegram updates. This is
    # intentionally one task for the process, not one per bot token.
    if apps:
        worker_tasks.append(asyncio.create_task(upcoming_reminder_worker(apps[0])))
    worker_tasks.append(asyncio.create_task(keep_miniapp_alive_worker()))
    _startup_complete.set()
    logger.info("✅ Startup complete; service is ready.")

    # =================================================================
    # 5. Keep Script Alive
    # =================================================================
    await stop_signal.wait()

    _startup_complete.clear()
    _shutdown_requested.set()
    logger.info("🛑 Graceful shutdown started.")
    for task in worker_tasks:
        task.cancel()
    if worker_tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*worker_tasks, return_exceptions=True),
                timeout=10
            )
        except asyncio.TimeoutError:
            logger.error("❌ Background workers did not stop within 10 seconds.")

    for app in apps:
        try:
            await asyncio.wait_for(app.updater.stop(), timeout=10)
            await asyncio.wait_for(app.stop(), timeout=10)
            await asyncio.wait_for(app.shutdown(), timeout=10)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    close_db_pool()
    logger.info("✅ Graceful shutdown complete.")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Critical Error: {e}")
