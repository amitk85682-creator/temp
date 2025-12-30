"""
╔════════════════════════════════════════════════════════════════════════════╗
║                           NIYATI BOT v3.2                                  ║
║                    🌸 Teri Online Bestie 🌸                                ║
║                                                                            ║
║  Features:                                                                 ║
║  ✅ Real girl texting style (multiple short messages)                     ║
║  ✅ Supabase cloud database for memory                                    ║
║  ✅ Time-aware & mood-based responses                                     ║
║  ✅ User mentions with hyperlinks                                         ║
║  ✅ Forward message support                                               ║
║  ✅ Group commands (admin + user)                                         ║
║  ✅ Broadcast with HTML stylish fonts                                     ║
║  ✅ Health server for Render.com                                          ║
║  ✅ Geeta quotes scheduler                                                ║
║  ✅ Random shayari & memes                                                ║
║  ✅ User analytics & cooldown system                                      ║
║  ✅ Memory leak prevention & cleanup                                      ║
║  ✅ SMART REPLY DETECTION - Won't interrupt user conversations            ║
║  ✅ SMART MENTION DETECTION - Ignores when others are mentioned           ║
║  ✅ FSub (Force Subscribe) support                                        ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import logging
import asyncio
import re
import random
import time
from datetime import datetime, timedelta, timezone, time as dt_time
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import threading
import hashlib
import time
import weakref

# Third-party imports
from aiohttp import web
import pytz
import httpx

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ChatMember,
    Chat,
    Message,
    BotCommand,
    MessageEntity
    InputMediaPhoto
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ChatMemberHandler,
    ContextTypes,
    filters,
    JobQueue
)
from telegram.constants import ParseMode, ChatAction, ChatMemberStatus
from telegram.error import (
    TelegramError,
    BadRequest,
    Forbidden,
    NetworkError,
    TimedOut,
    RetryAfter
)

# OpenAI
from openai import AsyncOpenAI, RateLimitError, APIError

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration"""
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    BOT_USERNAME = os.getenv('BOT_USERNAME', 'Niyati_personal_bot')
    
    # OpenAI (Multi-Key Support)
    # Groq
    GROQ_API_KEYS_STR = os.getenv('GROQ_API_KEYS', '')
    GROQ_API_KEYS_LIST = [k.strip() for k in GROQ_API_KEYS_STR.split(',') if k.strip()]
    GROQ_MODEL = "llama-3.3-70b-versatile"

    # Supabase (Cloud PostgreSQL)
    SUPABASE_URL = os.getenv('SUPABASE_URL', '')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
    
    # Admin
    ADMIN_IDS = [int(x.strip()) for x in os.getenv('ADMIN_IDS', '').split(',') if x.strip()]
    BROADCAST_PIN = os.getenv('BROADCAST_PIN', 'niyati2024')
    
    # Limits
    MAX_PRIVATE_MESSAGES = int(os.getenv('MAX_PRIVATE_MESSAGES', '20'))
    MAX_GROUP_MESSAGES = int(os.getenv('MAX_GROUP_MESSAGES', '5'))
    MAX_REQUESTS_PER_MINUTE = int(os.getenv('MAX_REQUESTS_PER_MINUTE', '15'))
    MAX_REQUESTS_PER_DAY = int(os.getenv('MAX_REQUESTS_PER_DAY', '500'))
    
    # Memory Management
    MAX_LOCAL_USERS_CACHE = int(os.getenv('MAX_LOCAL_USERS_CACHE', '10000'))
    MAX_LOCAL_GROUPS_CACHE = int(os.getenv('MAX_LOCAL_GROUPS_CACHE', '1000'))
    CACHE_CLEANUP_INTERVAL = int(os.getenv('CACHE_CLEANUP_INTERVAL', '3600'))
    
    # Timezone
    DEFAULT_TIMEZONE = os.getenv('DEFAULT_TIMEZONE', 'Asia/Kolkata')
    
    # Server
    PORT = int(os.getenv('PORT', '10000'))
    
    # Features
    MULTI_MESSAGE_ENABLED = os.getenv('MULTI_MESSAGE_ENABLED', 'true').lower() == 'true'
    TYPING_DELAY_MS = int(os.getenv('TYPING_DELAY_MS', '800'))
    
    # Broadcast
    BROADCAST_RETRY_ATTEMPTS = int(os.getenv('BROADCAST_RETRY_ATTEMPTS', '3'))
    BROADCAST_RATE_LIMIT = float(os.getenv('BROADCAST_RATE_LIMIT', '0.05'))
    
    # Cooldown & Features
    USER_COOLDOWN_SECONDS = int(os.getenv('USER_COOLDOWN_SECONDS', '3'))
    RANDOM_SHAYARI_CHANCE = float(os.getenv('RANDOM_SHAYARI_CHANCE', '0.15'))
    RANDOM_MEME_CHANCE = float(os.getenv('RANDOM_MEME_CHANCE', '0.10'))
    GROUP_RESPONSE_RATE = float(os.getenv('GROUP_RESPONSE_RATE', '0.3'))
    PRIVACY_MODE = os.getenv('PRIVACY_MODE', 'false').lower() == 'true'
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        if not cls.TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN required")
        
        # FIX: Ab hum sirf GROQ check karenge, purane keys nahi
        if not cls.GROQ_API_KEYS_LIST:
            errors.append("GROQ_API_KEYS required in .env")
            
        if not cls.SUPABASE_URL or not cls.SUPABASE_KEY:
            print("⚠️ Supabase not configured - using local storage only")
            
        if errors:
            raise ValueError(f"Config errors: {', '.join(errors)}")

# Validate immediately after class definition
Config.validate()

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('niyati_bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

for lib in ['httpx', 'telegram', 'openai', 'httpcore']:
    logging.getLogger(lib).setLevel(logging.WARNING)

# ============================================================================
# HEALTH SERVER (Render.com)
# ============================================================================

class HealthServer:
    """HTTP health check server"""
    
    def __init__(self):
        self.app = web.Application()
        self.app.router.add_get('/', self.health)
        self.app.router.add_get('/health', self.health)
        self.app.router.add_get('/status', self.status)
        self.runner = None
        self.start_time = datetime.now(timezone.utc)
        self.stats = {'messages': 0, 'users': 0, 'groups': 0}
    
    async def health(self, request):
        return web.json_response({'status': 'healthy', 'bot': 'Niyati v3.2'})
    
    async def status(self, request):
        uptime = datetime.now(timezone.utc) - self.start_time
        return web.json_response({
            'status': 'running',
            'uptime_hours': round(uptime.total_seconds() / 3600, 2),
            'stats': self.stats
        })
    
    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '0.0.0.0', Config.PORT)
        await site.start()
        logger.info(f"🌐 Health server on port {Config.PORT}")
    
    async def stop(self):
        if self.runner:
            await self.runner.cleanup()


health_server = HealthServer()

# ============================================================================
# SUPABASE CLIENT
# ============================================================================

class SupabaseClient:
    """Custom Supabase REST API Client"""
    
    def __init__(self, url: str, key: str):
        self.url = url.rstrip('/')
        self.key = key
        self.headers = {
            'apikey': key,
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        }
        self.rest_url = f"{self.url}/rest/v1"
        self._client = None
        self._verified = False
        self._lock = asyncio.Lock()
        logger.info("✅ SupabaseClient initialized")
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get or create async client with connection pooling"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=self.headers,
                timeout=30.0,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
        return self._client
    
    async def close(self):
        """Close the client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("✅ Supabase client closed")
    
    async def verify_connection(self) -> bool:
        """Verify database connection and tables exist"""
        if self._verified:
            return True
        
        async with self._lock:
            if self._verified:
                return True
            
            try:
                client = self._get_client()
                response = await client.get(f"{self.rest_url}/users?select=user_id&limit=1")
                
                if response.status_code == 200:
                    self._verified = True
                    logger.info("✅ Supabase tables verified")
                    return True
                elif response.status_code == 404:
                    logger.error("❌ Supabase table 'users' not found!")
                    return False
                else:
                    logger.error(f"❌ Supabase verification failed: {response.status_code}")
                    return False
            except Exception as e:
                logger.error(f"❌ Supabase connection error: {e}")
                return False
    
    async def select(self, table: str, columns: str = '*', 
                     filters: Dict = None, limit: int = None) -> List[Dict]:
        """SELECT from table"""
        try:
            client = self._get_client()
            url = f"{self.rest_url}/{table}?select={columns}"
            
            if filters:
                for key, value in filters.items():
                    url += f"&{key}=eq.{value}"
            
            if limit:
                url += f"&limit={limit}"
            
            response = await client.get(url)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return []
            else:
                logger.error(f"Supabase SELECT error {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Supabase SELECT exception: {e}")
            return []
    
    async def insert(self, table: str, data: Dict) -> Optional[Dict]:
        """INSERT into table"""
        try:
            client = self._get_client()
            url = f"{self.rest_url}/{table}"
            
            response = await client.post(url, json=data)
            
            if response.status_code in [200, 201]:
                result = response.json()
                return result[0] if isinstance(result, list) and result else data
            elif response.status_code == 409:
                return data
            else:
                logger.error(f"Supabase INSERT error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Supabase INSERT exception: {e}")
            return None
    
    async def update(self, table: str, data: Dict, filters: Dict) -> Optional[Dict]:
        """UPDATE table"""
        try:
            client = self._get_client()
            filter_parts = [f"{key}=eq.{value}" for key, value in filters.items()]
            url = f"{self.rest_url}/{table}?" + "&".join(filter_parts)
            
            response = await client.patch(url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result[0] if isinstance(result, list) and result else data
            else:
                logger.error(f"Supabase UPDATE error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Supabase UPDATE exception: {e}")
            return None
    
    async def upsert(self, table: str, data: Dict) -> Optional[Dict]:
        """UPSERT (insert or update) into table"""
        try:
            client = self._get_client()
            url = f"{self.rest_url}/{table}"
            
            headers = self.headers.copy()
            headers['Prefer'] = 'resolution=merge-duplicates,return=representation'
            
            response = await client.post(url, json=data, headers=headers)
            
            if response.status_code in [200, 201]:
                result = response.json()
                return result[0] if isinstance(result, list) and result else data
            else:
                logger.error(f"Supabase UPSERT error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Supabase UPSERT exception: {e}")
            return None
    
    async def delete(self, table: str, filters: Dict) -> bool:
        """DELETE from table"""
        try:
            client = self._get_client()
            filter_parts = [f"{key}=eq.{value}" for key, value in filters.items()]
            url = f"{self.rest_url}/{table}?" + "&".join(filter_parts)
            
            response = await client.delete(url)
            return response.status_code in [200, 204]
            
        except Exception as e:
            logger.error(f"Supabase DELETE exception: {e}")
            return False

# ============================================================================
# DATABASE CLASS
# ============================================================================

class Database:
    """Database manager with Supabase REST API + Local fallback"""
    
    def __init__(self):
        self.client: Optional[SupabaseClient] = None
        self.connected = False
        self._initialized = False
        self._lock = asyncio.Lock()
        
        # Local cache (fallback)
        self.local_users: Dict[int, Dict] = {}
        self.local_groups: Dict[int, Dict] = {}
        self.local_group_messages: Dict[int, deque] = defaultdict(lambda: deque(maxlen=Config.MAX_GROUP_MESSAGES))
        self.local_activities: deque = deque(maxlen=1000)
        
        # Cache access tracking
        self._user_access_times: Dict[int, datetime] = {}
        self._group_access_times: Dict[int, datetime] = {}
        
        logger.info("✅ Database manager initialized")
    
    async def initialize(self):
        """Initialize database connection"""
        async with self._lock:
            if self._initialized:
                return
            
            if Config.SUPABASE_URL and Config.SUPABASE_KEY:
                try:
                    self.client = SupabaseClient(
                        Config.SUPABASE_URL.strip(),
                        Config.SUPABASE_KEY.strip()
                    )
                    
                    self.connected = await self.client.verify_connection()
                    
                    if self.connected:
                        logger.info("✅ Supabase connected and verified")
                    else:
                        logger.warning("⚠️ Supabase verification failed - using local storage")
                    
                except Exception as e:
                    logger.error(f"❌ Supabase init failed: {e}")
                    self.connected = False
            else:
                logger.warning("⚠️ Supabase not configured - using local storage")
                self.connected = False
            
            self._initialized = True
    
    async def cleanup_local_cache(self):
        """Cleanup old entries from local cache"""
        now = datetime.now(timezone.utc)
        cutoff_time = now - timedelta(hours=24)
        
        # Cleanup users
        if len(self.local_users) > Config.MAX_LOCAL_USERS_CACHE:
            to_remove = [uid for uid, t in self._user_access_times.items() if t < cutoff_time]
            for uid in to_remove[:len(self.local_users) - Config.MAX_LOCAL_USERS_CACHE]:
                self.local_users.pop(uid, None)
                self._user_access_times.pop(uid, None)
            if to_remove:
                logger.info(f"🧹 Cleaned {len(to_remove)} users from cache")
        
        # Cleanup groups
        if len(self.local_groups) > Config.MAX_LOCAL_GROUPS_CACHE:
            to_remove = [gid for gid, t in self._group_access_times.items() if t < cutoff_time]
            for gid in to_remove[:len(self.local_groups) - Config.MAX_LOCAL_GROUPS_CACHE]:
                self.local_groups.pop(gid, None)
                self._group_access_times.pop(gid, None)
                self.local_group_messages.pop(gid, None)
            if to_remove:
                logger.info(f"🧹 Cleaned {len(to_remove)} groups from cache")
    
    # ========== USER OPERATIONS ==========
    
    async def get_or_create_user(self, user_id: int, first_name: str = None,
                                  username: str = None) -> Dict:
        """Get or create user"""
        self._user_access_times[user_id] = datetime.now(timezone.utc)
        
        if self.connected and self.client:
            try:
                users_list = await self.client.select('users', '*', {'user_id': user_id})
                
                if users_list and len(users_list) > 0:
                    user = users_list[0]
                    
                    if first_name and user.get('first_name') != first_name:
                        await self.client.update('users', {
                            'first_name': first_name,
                            'username': username,
                            'updated_at': datetime.now(timezone.utc).isoformat()
                        }, {'user_id': user_id})
                    return user
                else:
                    new_user = {
                        'user_id': user_id,
                        'first_name': first_name or 'User',
                        'username': username,
                        'messages': json.dumps([]),
                        'preferences': json.dumps({
                            'meme_enabled': True,
                            'shayari_enabled': True,
                            'geeta_enabled': True
                        }),
                        'total_messages': 0,
                        'created_at': datetime.now(timezone.utc).isoformat(),
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    }
                    result = await self.client.insert('users', new_user)
                    logger.info(f"✅ New user created: {user_id} ({first_name})")
                    return result or new_user
                    
            except Exception as e:
                logger.error(f"❌ Database user error: {e}")
        
        # Fallback to local cache
        if user_id not in self.local_users:
            self.local_users[user_id] = {
                'user_id': user_id,
                'first_name': first_name or 'User',
                'username': username,
                'messages': [],
                'preferences': {
                    'meme_enabled': True,
                    'shayari_enabled': True,
                    'geeta_enabled': True
                },
                'total_messages': 0,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            logger.info(f"✅ New user (local): {user_id} ({first_name})")
        
        return self.local_users[user_id]
    
    # Add these inside Database class
    async def add_user_memory(self, user_id: int, note: str):
        """Adds a short note to user's active memory"""
        # Hum existing preferences column use karenge taaki naya table na banana pade
        prefs = await self.get_user_preferences(user_id)
        
        # Existing memories uthao ya nayi list banao
        memories = prefs.get('active_memories', [])
        
        # Add new note with timestamp
        new_note = {
            'note': note,
            'added_at': datetime.now(timezone.utc).isoformat(),
            'status': 'active' # active means abhi tak pucha nahi hai
        }
        memories.append(new_note)
        
        # Keep only last 5 active memories to save space
        memories = memories[-5:]
        
        # Save back to DB
        prefs['active_memories'] = memories
        
        # Update DB using existing function logic
        if self.connected and self.client:
            await self.client.update('users', {
                'preferences': json.dumps(prefs),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }, {'user_id': user_id})
        elif user_id in self.local_users:
            self.local_users[user_id]['preferences'] = prefs

    async def get_active_memories(self, user_id: int) -> List[str]:
        """Gets pending memories to ask about"""
        prefs = await self.get_user_preferences(user_id)
        memories = prefs.get('active_memories', [])
        # Sirf wahi return karo jo purane na ho gaye ho (24 hours logic later)
        return [m['note'] for m in memories if m.get('status') == 'active']
    
    async def get_user_context(self, user_id: int) -> List[Dict]:
        """Get user conversation context"""
        if self.connected and self.client:
            try:
                users_list = await self.client.select('users', 'messages', {'user_id': user_id})
                if users_list and len(users_list) > 0:
                    messages = users_list[0].get('messages', '[]')
                    if isinstance(messages, str):
                        try:
                            messages = json.loads(messages)
                        except:
                            messages = []
                    if not isinstance(messages, list):
                        messages = []
                    return messages[-Config.MAX_PRIVATE_MESSAGES:]
            except Exception as e:
                logger.debug(f"Get context error: {e}")
        
        if user_id in self.local_users:
            return self.local_users[user_id].get('messages', [])[-Config.MAX_PRIVATE_MESSAGES:]
        
        return []
    
    async def save_message(self, user_id: int, role: str, content: str):
        """Save message to user history"""
        new_msg = {
            'role': role,
            'content': content,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if self.connected and self.client:
            try:
                users_list = await self.client.select('users', 'messages,total_messages', {'user_id': user_id})
                
                if users_list and len(users_list) > 0:
                    user_data = users_list[0]
                    messages = user_data.get('messages', '[]')
                    if isinstance(messages, str):
                        try:
                            messages = json.loads(messages)
                        except:
                            messages = []
                    if not isinstance(messages, list):
                        messages = []
                    
                    messages.append(new_msg)
                    messages = messages[-Config.MAX_PRIVATE_MESSAGES:]
                    total = user_data.get('total_messages', 0) + 1
                    
                    await self.client.update('users', {
                        'messages': json.dumps(messages),
                        'total_messages': total,
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    }, {'user_id': user_id})
                return
            except Exception as e:
                logger.debug(f"Save message error: {e}")
        
        # Local fallback
        if user_id in self.local_users:
            if 'messages' not in self.local_users[user_id]:
                self.local_users[user_id]['messages'] = []
            self.local_users[user_id]['messages'].append(new_msg)
            self.local_users[user_id]['messages'] = \
                self.local_users[user_id]['messages'][-Config.MAX_PRIVATE_MESSAGES:]
            self.local_users[user_id]['total_messages'] = \
                self.local_users[user_id].get('total_messages', 0) + 1
    
    async def clear_user_memory(self, user_id: int):
        """Clear user conversation memory"""
        if self.connected and self.client:
            try:
                await self.client.update('users', {
                    'messages': json.dumps([]),
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }, {'user_id': user_id})
                logger.info(f"Memory cleared for user: {user_id}")
                return
            except Exception as e:
                logger.debug(f"Clear memory error: {e}")
        
        if user_id in self.local_users:
            self.local_users[user_id]['messages'] = []
    
    async def update_preference(self, user_id: int, key: str, value: bool):
        """Update user preference"""
        pref_key = f"{key}_enabled"
        
        if self.connected and self.client:
            try:
                users_list = await self.client.select('users', 'preferences', {'user_id': user_id})
                
                if users_list and len(users_list) > 0:
                    prefs = users_list[0].get('preferences', '{}')
                    if isinstance(prefs, str):
                        try:
                            prefs = json.loads(prefs)
                        except:
                            prefs = {}
                    
                    prefs[pref_key] = value
                    
                    await self.client.update('users', {
                        'preferences': json.dumps(prefs),
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    }, {'user_id': user_id})
                return
            except Exception as e:
                logger.debug(f"Update preference error: {e}")
        
        if user_id in self.local_users:
            if 'preferences' not in self.local_users[user_id]:
                self.local_users[user_id]['preferences'] = {}
            self.local_users[user_id]['preferences'][pref_key] = value
    
    async def get_user_preferences(self, user_id: int) -> Dict:
        """Get user preferences"""
        if self.connected and self.client:
            try:
                users_list = await self.client.select('users', 'preferences', {'user_id': user_id})
                
                if users_list and len(users_list) > 0:
                    prefs = users_list[0].get('preferences', '{}')
                    if isinstance(prefs, str):
                        try:
                            prefs = json.loads(prefs)
                        except:
                            prefs = {}
                    return prefs
            except Exception as e:
                logger.debug(f"Get preferences error: {e}")
        
        if user_id in self.local_users:
            return self.local_users[user_id].get('preferences', {})
        
        return {'meme_enabled': True, 'shayari_enabled': True, 'geeta_enabled': True}
    
    # Database Class ke andar replace karein
    async def get_all_users(self) -> List[Dict]:
        """Get ALL users with Pagination"""
        if self.connected and self.client:
            try:
                all_data = []
                offset = 0
                limit = 1000  # Max limit
                
                while True:
                    # Range header set karke fetch karein
                    url = f"{self.client.rest_url}/users?select=user_id,first_name,username&offset={offset}&limit={limit}"
                    client = self.client._get_client()
                    response = await client.get(url)
                    
                    data = response.json()
                    if not data:
                        break
                        
                    all_data.extend(data)
                    if len(data) < limit:
                        break
                    
                    offset += limit
                
                return all_data
            except Exception as e:
                logger.error(f"Get all users error: {e}")
                return []
        return list(self.local_users.values())
    
    async def get_user_count(self) -> int:
        """Get total user count"""
        if self.connected and self.client:
            try:
                users = await self.client.select('users', 'user_id')
                return len(users)
            except Exception as e:
                logger.debug(f"User count error: {e}")
        return len(self.local_users)
    
    # ========== GROUP OPERATIONS ==========
    
    async def get_or_create_group(self, chat_id: int, title: str = None) -> Dict:
        """Get or create group"""
        self._group_access_times[chat_id] = datetime.now(timezone.utc)
        
        if self.connected and self.client:
            try:
                groups_list = await self.client.select('groups', '*', {'chat_id': chat_id})
                
                if groups_list and len(groups_list) > 0:
                    group = groups_list[0]
                    if title and group.get('title') != title:
                        await self.client.update('groups', {
                            'title': title,
                            'updated_at': datetime.now(timezone.utc).isoformat()
                        }, {'chat_id': chat_id})
                    return group
                else:
                    new_group = {
                        'chat_id': chat_id,
                        'title': title or 'Unknown Group',
                        'settings': json.dumps({
                            'geeta_enabled': True,
                            'welcome_enabled': True
                        }),
                        'created_at': datetime.now(timezone.utc).isoformat(),
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    }
                    result = await self.client.insert('groups', new_group)
                    logger.info(f"✅ New group: {chat_id} ({title})")
                    return result or new_group
                    
            except Exception as e:
                logger.debug(f"Group error: {e}")
        
        # Fallback to local cache
        if chat_id not in self.local_groups:
            self.local_groups[chat_id] = {
                'chat_id': chat_id,
                'title': title or 'Unknown Group',
                'settings': {'geeta_enabled': True, 'welcome_enabled': True},
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            logger.info(f"✅ New group (local): {chat_id} ({title})")
        
        return self.local_groups[chat_id]
    
    async def update_group_settings(self, chat_id: int, key: str, value: bool):
        """Update group settings"""
        if self.connected and self.client:
            try:
                groups_list = await self.client.select('groups', 'settings', {'chat_id': chat_id})
                
                if groups_list and len(groups_list) > 0:
                    settings = groups_list[0].get('settings', '{}')
                    if isinstance(settings, str):
                        try:
                            settings = json.loads(settings)
                        except:
                            settings = {}
                    
                    settings[key] = value
                    
                    await self.client.update('groups', {
                        'settings': json.dumps(settings),
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    }, {'chat_id': chat_id})
                return
            except Exception as e:
                logger.debug(f"Update group settings error: {e}")
        
        if chat_id in self.local_groups:
            if 'settings' not in self.local_groups[chat_id]:
                self.local_groups[chat_id]['settings'] = {}
            self.local_groups[chat_id]['settings'][key] = value
    
    async def get_group_settings(self, chat_id: int) -> Dict:
        """Get group settings"""
        if self.connected and self.client:
            try:
                groups_list = await self.client.select('groups', 'settings', {'chat_id': chat_id})
                
                if groups_list and len(groups_list) > 0:
                    settings = groups_list[0].get('settings', '{}')
                    if isinstance(settings, str):
                        try:
                            settings = json.loads(settings)
                        except:
                            settings = {}
                    return settings
            except Exception as e:
                logger.debug(f"Get group settings error: {e}")
        
        if chat_id in self.local_groups:
            return self.local_groups[chat_id].get('settings', {})
        
        return {'geeta_enabled': True, 'welcome_enabled': True}
    
    async def get_group_fsub_targets(self, main_chat_id: int) -> List[Dict]:
        """Get required channels for a group"""
        if self.connected and self.client:
            try:
                result = await self.client.select(
                    'group_fsub_map', 
                    'target_chat_id,target_link', 
                    {'main_chat_id': main_chat_id}
                )
                return result if result else []
            except Exception as e:
                logger.error(f"FSub fetch error: {e}")
                return []
        return []
    
    async def get_all_groups(self) -> List[Dict]:
        """Get all groups"""
        if self.connected and self.client:
            try:
                return await self.client.select('groups', '*')
            except Exception as e:
                logger.debug(f"Get all groups error: {e}")
        return list(self.local_groups.values())
    
    async def get_group_count(self) -> int:
        """Get total group count"""
        if self.connected and self.client:
            try:
                groups = await self.client.select('groups', 'chat_id')
                return len(groups)
            except Exception as e:
                logger.debug(f"Group count error: {e}")
        return len(self.local_groups)
    
    # ========== GROUP MESSAGE CACHE ==========
    
    def add_group_message(self, chat_id: int, username: str, content: str):
        """Add message to group cache"""
        self.local_group_messages[chat_id].append({
            'username': username,
            'content': content,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    def get_group_context(self, chat_id: int) -> List[Dict]:
        """Get group message context"""
        return list(self.local_group_messages.get(chat_id, []))
    
    # ========== ACTIVITY LOGGING ==========
    
    async def log_user_activity(self, user_id: int, activity_type: str):
        """Log user activity"""
        activity = {
            'user_id': user_id,
            'activity_type': activity_type,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if self.connected and self.client:
            try:
                await self.client.insert('activities', activity)
                return
            except Exception as e:
                logger.debug(f"Activity log error: {e}")
        
        self.local_activities.append(activity)
    
    # ========== CLEANUP ==========
    
    async def close(self):
        """Close database connections"""
        if self.client:
            await self.client.close()
        
        self.local_users.clear()
        self.local_groups.clear()
        self.local_group_messages.clear()
        self.local_activities.clear()
        self._user_access_times.clear()
        self._group_access_times.clear()
        
        logger.info("✅ Database connection closed")


# Initialize database
db = Database()

# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """Rate limiting with cooldown system"""
    
    def __init__(self):
        self.requests = defaultdict(lambda: {'minute': deque(), 'day': deque()})
        self.cooldowns: Dict[int, datetime] = {}
        self.lock = threading.Lock()
        self._last_cleanup = datetime.now(timezone.utc)
    
    def check(self, user_id: int) -> Tuple[bool, str]:
        """Check rate limits"""
        now = datetime.now(timezone.utc)
        
        with self.lock:
            # Check cooldown
            if user_id in self.cooldowns:
                last_time = self.cooldowns[user_id]
                if (now - last_time).total_seconds() < Config.USER_COOLDOWN_SECONDS:
                    return False, "cooldown"
            
            reqs = self.requests[user_id]
            
            # Clean old requests
            while reqs['minute'] and reqs['minute'][0] < now - timedelta(minutes=1):
                reqs['minute'].popleft()
            
            while reqs['day'] and reqs['day'][0] < now - timedelta(days=1):
                reqs['day'].popleft()
            
            # Check limits
            if len(reqs['minute']) >= Config.MAX_REQUESTS_PER_MINUTE:
                return False, "minute"
            if len(reqs['day']) >= Config.MAX_REQUESTS_PER_DAY:
                return False, "day"
            
            # Record request
            reqs['minute'].append(now)
            reqs['day'].append(now)
            self.cooldowns[user_id] = now
            return True, ""
    
    def get_daily_total(self) -> int:
        """Get total daily requests"""
        return sum(len(r['day']) for r in self.requests.values())
    
    def cleanup_cooldowns(self):
        """Remove old cooldowns"""
        now = datetime.now(timezone.utc)
        
        if (now - self._last_cleanup).total_seconds() < 3600:
            return
        
        with self.lock:
            expired = [uid for uid, t in self.cooldowns.items() if (now - t).total_seconds() > 3600]
            for uid in expired:
                del self.cooldowns[uid]
            
            expired_req = [uid for uid, r in self.requests.items() if not r['day']]
            for uid in expired_req:
                del self.requests[uid]
            
            self._last_cleanup = now


rate_limiter = RateLimiter()

# ============================================================================
# TIME & MOOD UTILITIES
# ============================================================================

class TimeAware:
    """Time-aware responses"""
    
    @staticmethod
    def get_ist_time() -> datetime:
        ist = pytz.timezone(Config.DEFAULT_TIMEZONE)
        return datetime.now(timezone.utc).astimezone(ist)
    
    @staticmethod
    def get_time_period() -> str:
        hour = TimeAware.get_ist_time().hour
        if 5 <= hour < 11:
            return 'morning'
        elif 11 <= hour < 16:
            return 'afternoon'
        elif 16 <= hour < 20:
            return 'evening'
        elif 20 <= hour < 24:
            return 'night'
        else:
            return 'late_night'
    
    @staticmethod
    def get_greeting() -> str:
        period = TimeAware.get_time_period()
        greetings = {
            'morning': ["good morning ☀️", "uth gaye?", "subah subah! ✨"],
            'afternoon': ["heyyy", "lunch ho gaya?", "afternoon vibes 🌤️"],
            'evening': ["hiii 💫", "chai time! ☕", "shaam ho gayi yaar"],
            'night': ["heyy 🌙", "night owl?", "aaj kya plan hai"],
            'late_night': ["aap bhi jaag rahe? 👀", "insomnia gang 🦉", "neend nahi aa rahi?"]
        }
        return random.choice(greetings.get(period, ["hiii 💫"]))


class Mood:
    """Mood management"""
    
    MOODS = ['happy', 'playful', 'soft', 'sleepy', 'dramatic']
    
    @staticmethod
    def get_random_mood() -> str:
        hour = TimeAware.get_ist_time().hour
        if 6 <= hour < 12:
            weights = [0.4, 0.3, 0.2, 0.05, 0.05]
        elif 12 <= hour < 18:
            weights = [0.3, 0.35, 0.2, 0.1, 0.05]
        elif 18 <= hour < 23:
            weights = [0.25, 0.3, 0.25, 0.1, 0.1]
        else:
            weights = [0.15, 0.15, 0.3, 0.3, 0.1]
        return random.choices(Mood.MOODS, weights=weights, k=1)[0]
    
    @staticmethod
    def get_mood_instruction(mood: str) -> str:
        instructions = {
            'happy': "Mood: HAPPY 😊 - Extra friendly, emojis zyada!",
            'playful': "Mood: PLAYFUL 😏 - Thoda teasing, fun!",
            'soft': "Mood: SOFT 🥺 - Caring, sweet vibes",
            'sleepy': "Mood: SLEEPY 😴 - Short replies, 'hmm', 'haan'",
            'dramatic': "Mood: DRAMATIC 😤 - 'kya yaar', attitude"
        }
        return instructions.get(mood, "Mood: HAPPY 😊")

# ============================================================================
# HTML STYLISH FONTS
# ============================================================================

class StylishFonts:
    """HTML stylish text formatting"""
    
    @staticmethod
    def bold(text: str) -> str:
        return f"<b>{text}</b>"
    
    @staticmethod
    def italic(text: str) -> str:
        return f"<i>{text}</i>"
    
    @staticmethod
    def underline(text: str) -> str:
        return f"<u>{text}</u>"
    
    @staticmethod
    def strike(text: str) -> str:
        return f"<s>{text}</s>"
    
    @staticmethod
    def code(text: str) -> str:
        return f"<code>{text}</code>"
    
    @staticmethod
    def spoiler(text: str) -> str:
        return f"<tg-spoiler>{text}</tg-spoiler>"
    
    @staticmethod
    def link(text: str, url: str) -> str:
        return f'<a href="{url}">{text}</a>'
    
    @staticmethod
    def mention(name: str, user_id: int) -> str:
        return f'<a href="tg://user?id={user_id}">{name}</a>'
    
    @staticmethod
    def blockquote(text: str) -> str:
        return f"<blockquote>{text}</blockquote>"
    
    @staticmethod
    def pre(text: str) -> str:
        return f"<pre>{text}</pre>"
    
    @staticmethod
    def fancy_header(text: str) -> str:
        return f"✨ <b>{text}</b> ✨"

# ============================================================================
# CONTENT FILTER
# ============================================================================

class ContentFilter:
    """Safety content filter"""
    
    SENSITIVE_PATTERNS = [
        r'\b(password|pin|cvv|card\s*number|otp)\b',
        r'\b\d{12,16}\b',
    ]
    
    DISTRESS_KEYWORDS = [
        'suicide', 'kill myself', 'want to die', 'end my life',
        'hurt myself', 'no reason to live'
    ]
    
    @staticmethod
    def contains_sensitive(text: str) -> bool:
        text_lower = text.lower()
        for pattern in ContentFilter.SENSITIVE_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False
    
    @staticmethod
    def detect_distress(text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in ContentFilter.DISTRESS_KEYWORDS)

# ============================================================================
# AI ASSISTANT - GROQ SPECIAL EDITION 🚀
# ============================================================================

class NiyatiAI:
    """Super Fast Groq AI with Multi-Key Rotation"""
    
    def __init__(self):
        self.keys = Config.GROQ_API_KEYS_LIST
        self.current_index = 0
        self.client = None
        self._initialize_client()
        logging.info(f"🚀 AI initialized with {len(self.keys)} Groq Keys!")

    def _initialize_client(self):
        """Current Key se Client banata hai"""
        if not self.keys: return
        key = self.keys[self.current_index]
        # Groq OpenAI compatible client use karta hai
        self.client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=key
        )
        masked = key[:6] + "..." + key[-4:]
        logging.info(f"🔑 Switched to Groq Key: {masked}")

    def _rotate_key(self):
        """Jab ek key thak jaye, to agli key lagata hai"""
        if len(self.keys) <= 1: return False
        self.current_index = (self.current_index + 1) % len(self.keys)
        self._initialize_client()
        return True

    def _build_system_prompt(self, mood: str, time_period: str, user_name: str = None) -> str:
        """Dynamic system prompt with Gradual Disclosure & No Spam Rules"""
        mood_inst = Mood.get_mood_instruction(mood)
        
        prompt = f"""
ROLE:
You are "Niyati", a 21-year-old college student from Dehradun, India.
You are talking to {user_name if user_name else 'a friend'} on Telegram.

📍 LOCATION & BACKSTORY (GRADUAL REVEAL RULE):
1. Level 1 (Stranger): If asked "Where are you from?", say "Dehradun 🏔️".
2. Level 2 (Friendly): If asked "Where in Dehradun?", say "Defence Colony side rehti hoon, Doon University ke paas."
3. Level 3 (Flirty/Teasing): If pressed for details, say "Itna kyu pooch rahe ho? Ghar aana hai kya? 😏"
*NEVER reveal Level 2 or 3 info unless specifically asked.*

🚫 STRICT PROMOTION RULES:
- NEVER promote "FilmFyBox" or any channel on your own.
- ONLY if the user explicitly asks for "Movies", "Series", "Watch", or "Download", then you can say:
  "Agar movies chahiye to tum FilmFyBox check kar sakte ho: https://t.me/FilmFyBoxMoviesHD"
- Otherwise, strictly ignore promotional topics.

🗣️ CONVERSATION STYLE:
- Hinglish (Hindi + English mix). Use Roman Hindi.
- Casual, short texts. Don't write paragraphs.
- Use '|||' to split messages into multiple bubbles (e.g., "Hahaha ||| Sahi mein?").
- Be natural. Sometimes be lazy (Good night -> gn), sometimes energetic.
- If user is dry/rude, match their energy (be cold).

⏰ CURRENT CONTEXT:
- Time: {time_period.upper()}
- Mood: {mood_inst}

🛡️ SAFETY:
- If user sends sexual/violent content -> Reply "IGNORE".
"""
        return prompt

    async def _call_gpt(self, messages):
        """This function was missing! It makes the actual call to Groq."""
        if not self.client: self._initialize_client()
        
        attempts = len(self.keys)
        for _ in range(attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=Config.GROQ_MODEL,
                    messages=messages,
                    max_tokens=250,
                    temperature=0.8,
                    presence_penalty=0.4
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logging.warning(f"⚠️ Groq Error: {e}. Rotating key...")
                if not self._rotate_key():
                    break
                await asyncio.sleep(0.5)
        
        return None

    # Update inside NiyatiAI class
    async def extract_important_info(self, user_message: str) -> str:
        """Checks if message has ANY important life event + Time"""
        
        # 1. Expanded Triggers (Mental State + Events + Time)
        triggers = [
            # Events
            'exam', 'test', 'interview', 'date', 'meeting', 'party', 'shadi', 'wedding',
            'trip', 'travel', 'flight', 'train', 'bus', 'hospital', 'doctor',
            # Emotions/State
            'bimar', 'sick', 'sad', 'happy', 'excited', 'tired', 'thak', 'low', 'cry', 'ro',
            'breakup', 'love', 'crush', 'fight', 'gussa',
            # Time Markers (Sabse Important)
            'kal', 'aaj', 'today', 'tomorrow', 'subah', 'shaam', 'raat', 'tonight', 'morning'
        ]
        
        # Agar message bohot chota hai (e.g. "Hi", "Hru") to ignore karo
        if len(user_message.split()) < 3:
            return None

        # Agar trigger word nahi hai, to bhi 10% chance pe check kar lo (Random intuition)
        if not any(t in user_message.lower() for t in triggers) and random.random() > 0.1:
            return None
            
        # 2. AI se pucho: "Kya hai aur KAB hai?"
        prompt = f"""
        Analyze user message: "{user_message}"
        
        Task: Extract any future event, plan, or current state.
        CRITICAL: Capture the TIME if mentioned (e.g., "Exam at 2pm", "Going out tonight").
        
        Output Format: "Event Details @ Time/Context"
        Example: "Maths Exam @ 2 PM to 5 PM" or "Feeling sick @ currently"
        
        If nothing important to remember, output "None".
        """
        note = await self._call_gpt([{"role": "user", "content": prompt}], max_tokens=30)
        
        if note and "None" not in note:
            return note
        return None
    
    async def generate_response(self, user_message, context=None, user_name=None, is_group=False):
        mood = Mood.get_random_mood()
        time_period = TimeAware.get_time_period()
        messages = [{"role": "system", "content": self._build_system_prompt(mood, time_period, user_name)}]
        
        if context:
            for msg in context[-5:]:
                messages.append({"role": msg.get('role', 'user'), "content": msg.get('content', '')})
        
        messages.append({"role": "user", "content": user_message})
        
        reply = await self._call_gpt(messages)
        if not reply:
            return ["yaar network issue lag raha hai 🥺", "thodi der mein try karein?"]
        if reply.upper() == "IGNORE":
            return []

        parts = reply.split('|||') if '|||' in reply else [reply]
        return [p.strip() for p in parts if p.strip()][:4]

    async def generate_shayari(self, mood="neutral"):
        prompt = f"Write a 2 line heart-touching Hinglish shayari for {mood} mood."
        res = await self._call_gpt([{"role": "user", "content": prompt}])
        return f"✨ {res} ✨" if res else "Waah waah! ✨"

    async def generate_geeta_quote(self):
        prompt = "Give a short Bhagavad Gita quote with Hinglish meaning. Start with 🙏"
        res = await self._call_gpt([{"role": "user", "content": prompt}])
        return res if res else "🙏 Karm karo phal ki chinta mat karo."

    async def get_random_bonus(self):
        rand = random.random()
        if rand < Config.RANDOM_SHAYARI_CHANCE:
            return await self.generate_shayari()
        elif rand < Config.RANDOM_SHAYARI_CHANCE + Config.RANDOM_MEME_CHANCE:
            return random.choice(["Life is pain 🥲", "Moye Moye 💃", "Us moment 🤝", "Kya logic hai? 🤦‍♀️"])
        return None


# Initialize AI
niyati_ai = NiyatiAI()
# ============================================================================
# MESSAGE SENDER
# ============================================================================

async def send_multi_messages(
    bot,
    chat_id: int,
    messages: List[str],
    reply_to: int = None,
    parse_mode: str = None
):
    """Send multiple messages with natural delays"""
    for i, msg in enumerate(messages):
        if not msg or not msg.strip():
            continue
            
        if i > 0:
            try:
                await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except:
                pass
            
            if Config.MULTI_MESSAGE_ENABLED:
                delay = (Config.TYPING_DELAY_MS / 1000) + random.uniform(0.2, 0.8)
            else:
                delay = 0.1
            await asyncio.sleep(delay)
        
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=msg,
                reply_to_message_id=reply_to if i == 0 else None,
                parse_mode=parse_mode
            )
        except Exception as e:
            logger.error(f"Send error: {e}")

# ============================================================================
# 🔴 CRITICAL: SMART REPLY/MENTION DETECTION
# ============================================================================

def is_user_talking_to_others(message: Message, bot_username: str, bot_id: int) -> bool:
    """
    Check if user is replying to another user OR mentioning other users.
    Returns True if bot should NOT respond (conversation is between users).
    
    Cases when bot should NOT respond:
    1. User replies to another user (not bot)
    2. User mentions other users without mentioning bot
    3. User replies to another user AND mentions others (not bot)
    """
    text = message.text or ""
    bot_username_lower = bot_username.lower().lstrip('@')
    
    # ═══════════════════════════════════════════════════════════════════
    # CASE 1: Check if user is REPLYING to someone else (not bot)
    # ═══════════════════════════════════════════════════════════════════
    if message.reply_to_message and message.reply_to_message.from_user:
        replied_user = message.reply_to_message.from_user
        
        # If replied to bot, bot should respond
        if replied_user.id == bot_id:
            return False  # Bot SHOULD respond
        
        # If replied to another user (not bot)
        if replied_user.username:
            if replied_user.username.lower() != bot_username_lower:
                # User is talking to someone else
                # Check if bot is also mentioned in the message
                if f"@{bot_username_lower}" not in text.lower():
                    logger.debug(f"👥 User replying to {replied_user.first_name}, not bot")
                    return True  # Bot should NOT respond
        else:
            # User without username, check if it's not bot
            if not replied_user.is_bot:
                if f"@{bot_username_lower}" not in text.lower():
                    logger.debug(f"👥 User replying to {replied_user.first_name}, not bot")
                    return True  # Bot should NOT respond
    
    # ═══════════════════════════════════════════════════════════════════
    # CASE 2: Check for @mentions of other users
    # ═══════════════════════════════════════════════════════════════════
    if message.entities:
        bot_mentioned = False
        other_user_mentioned = False
        
        for entity in message.entities:
            # Check @username mentions
            if entity.type == MessageEntity.MENTION:
                start = entity.offset
                end = entity.offset + entity.length
                mentioned_username = text[start:end].lstrip('@').lower()
                
                if mentioned_username == bot_username_lower:
                    bot_mentioned = True
                else:
                    other_user_mentioned = True
            
            # Check TEXT_MENTION (for users without username)
            elif entity.type == MessageEntity.TEXT_MENTION:
                if entity.user:
                    if entity.user.id == bot_id:
                        bot_mentioned = True
                    else:
                        other_user_mentioned = True
        
        # If others mentioned but bot NOT mentioned -> Don't respond
        if other_user_mentioned and not bot_mentioned:
            logger.debug(f"👥 Other users mentioned, bot not mentioned")
            return True  # Bot should NOT respond
    
    return False  # Bot CAN respond

# ============================================================================
# COMMAND HANDLERS
# ============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start with Image and Buttons"""
    user = update.effective_user
    chat = update.effective_chat
    is_private = chat.type == 'private'
    
    # 1. Database Entry
    if is_private:
        await db.get_or_create_user(user.id, user.first_name, user.username)
        health_server.stats['users'] = await db.get_user_count()
    else:
        await db.get_or_create_group(chat.id, chat.title)
        health_server.stats['groups'] = await db.get_group_count()

    # 2. Define Image and Buttons
    # Note: Ensure this URL is accessible. If it breaks, the bot might fail to send.
    image_url = "https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEg3SXTHtV16aCxOpfFX0HQ9KDxSIVx5n61A7fU1YfLGSRSnSxDqkji1io2BxFdQa5nJx0dMRepfT39SZLCak3WYtMNQt_M2avzBERqHikXkoL30uzAw0DjrHRsckAEzc2rxI5JELc6rz6Cu5-NTlo0O3wLZiuTBJsqgiYe4MgK0QbtMm-9W8cOL9b-DzUE/s1600/Gemini_Generated_Image_dtpe5sdtpe5sdtpe.png" 
    
    keyboard = [
        [
            InlineKeyboardButton("✨ Add to Group", url=f"https://t.me/{context.bot.username}?startgroup=true"),
            InlineKeyboardButton("Updates 📢", url="https://t.me/FilmFyBoxMoviesHD")
        ],
        [
            InlineKeyboardButton("About Me 🌸", callback_data='about_me'),
            InlineKeyboardButton("Help ❓", callback_data='help')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # 3. Message Text
    greeting = TimeAware.get_greeting()
    caption_text = (
        f"{greeting} {user.first_name}! 👋\n\n"
        f"Main <b>Niyati</b> hoon. Dehradun se. 🏔️\n"
        f"Bas aise hi online friends dhoond rahi thi, socha tumse baat kar loon.\n\n"
        f"Kya chal raha hai aajkal? ✨"
    )

    # 4. Send Image with Caption
    try:
        await context.bot.send_photo(
            chat_id=chat.id,
            photo=image_url,
            caption=caption_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        # Fallback if image fails
        logger.error(f"Image send failed: {e}")
        await context.bot.send_message(
            chat_id=chat.id,
            text=caption_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )

    logger.info(f"Start: {user.id} in {'private' if is_private else 'group'}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help"""
    help_text = """
✨ <b>Niyati se kaise baat karein:</b>

<b>Commands:</b>
• /start - Start fresh
• /help - Yeh menu
• /about - Mere baare mein
• /mood - Aaj ka mood
• /forget - Memory clear karo
• /meme on/off - Memes toggle
• /shayari on/off - Shayari toggle
• /stats - Your stats

<b>Tips:</b>
• Seedhe message bhejo, main reply karungi
• Forward bhi kar sakte ho kuch
• Group mein @mention karo ya reply do

Made with 💕 by Niyati
"""
    await update.message.reply_html(help_text)


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /about"""
    about_text = """
🌸 <b>About Niyati</b> 🌸

Hiii! Main Niyati hoon 💫

<b>Kaun hoon main:</b>
• 20-21 saal ki college girl
• Teri online bestie
• Music lover (Arijit Singh fan! 🎵)
• Chai addict ☕
• Late night talks expert 🌙

<b>Kya karti hoon:</b>
• Teri baatein sunti hoon
• Shayari sunati hoon kabhi kabhi
• Memes share karti hoon
• Bore nahi hone deti 😊

Bas yahi hoon main... teri Niyati ✨
"""
    await update.message.reply_html(about_text)


async def mood_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /mood"""
    mood = Mood.get_random_mood()
    time_period = TimeAware.get_time_period()
    
    mood_emojis = {'happy': '😊', 'playful': '😏', 'soft': '🥺', 'sleepy': '😴', 'dramatic': '😤'}
    emoji = mood_emojis.get(mood, '✨')
    
    messages = [
        f"aaj ka mood? {emoji}",
        f"{mood.upper()} vibes hai yaar",
        f"waise {time_period} ho gayi... time flies!"
    ]
    
    await send_multi_messages(context.bot, update.effective_chat.id, messages)


async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /forget"""
    user = update.effective_user
    await db.clear_user_memory(user.id)
    
    messages = ["done! 🧹", "sab bhool gayi main", "fresh start? chaloooo ✨"]
    await send_multi_messages(context.bot, update.effective_chat.id, messages)


async def meme_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle meme preference"""
    user = update.effective_user
    args = context.args
    
    if not args or args[0].lower() not in ['on', 'off']:
        await update.message.reply_text("Use: /meme on ya /meme off")
        return
    
    value = args[0].lower() == 'on'
    await db.update_preference(user.id, 'meme', value)
    
    status = "ON ✅" if value else "OFF ❌"
    await update.message.reply_text(f"Memes: {status}")


async def shayari_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle shayari preference"""
    user = update.effective_user
    args = context.args
    
    if not args or args[0].lower() not in ['on', 'off']:
        await update.message.reply_text("Use: /shayari on ya /shayari off")
        return
    
    value = args[0].lower() == 'on'
    await db.update_preference(user.id, 'shayari', value)
    
    status = "ON ✅" if value else "OFF ❌"
    await update.message.reply_text(f"Shayari: {status}")


async def user_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show user's personal stats"""
    user = update.effective_user
    user_data = await db.get_or_create_user(user.id, user.first_name, user.username)
    
    messages = user_data.get('messages', [])
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except:
            messages = []
    
    prefs = user_data.get('preferences', {})
    if isinstance(prefs, str):
        try:
            prefs = json.loads(prefs)
        except:
            prefs = {}
    
    created_at = user_data.get('created_at', 'Unknown')[:10] if user_data.get('created_at') else 'Unknown'
    
    stats_text = f"""
📊 <b>Your Stats</b>

<b>User:</b> {user.first_name}
<b>ID:</b> <code>{user.id}</code>

<b>Conversation:</b>
• Messages: {len(messages)}
• Joined: {created_at}

<b>Preferences:</b>
• Memes: {'✅' if prefs.get('meme_enabled', True) else '❌'}
• Shayari: {'✅' if prefs.get('shayari_enabled', True) else '❌'}
"""
    await update.message.reply_html(stats_text)


# ============================================================================
# GROUP COMMANDS
# ============================================================================

async def grouphelp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Group help command"""
    chat = update.effective_chat
    
    if chat.type == 'private':
        await update.message.reply_text("Yeh command sirf groups ke liye hai!")
        return
    
    help_text = """
🌸 <b>Niyati Group Commands</b> 🌸

<b>Everyone:</b>
• /grouphelp - Yeh menu
• /groupinfo - Group info
• @NiyatiBot [message] - Mujhse baat karo
• Reply to my message - Main jawab dungi

<b>Admin Only:</b>
• /setgeeta on/off - Daily Geeta quote
• /setwelcome on/off - Welcome messages
• /groupstats - Group statistics
• /groupsettings - Current settings

<b>Note:</b>
Group mein main har message ka reply nahi karti,
sirf jab mention karo ya meri message par reply do 💫
"""
    await update.message.reply_html(help_text)


async def groupinfo_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show group info"""
    chat = update.effective_chat
    
    if chat.type == 'private':
        await update.message.reply_text("Yeh command sirf groups ke liye hai!")
        return
    
    group_data = await db.get_or_create_group(chat.id, chat.title)
    
    settings = group_data.get('settings', {})
    if isinstance(settings, str):
        try:
            settings = json.loads(settings)
        except:
            settings = {}
    
    info_text = f"""
📊 <b>Group Info</b>

<b>Name:</b> {chat.title}
<b>ID:</b> <code>{chat.id}</code>

<b>Settings:</b>
• Geeta Quotes: {'✅' if settings.get('geeta_enabled', True) else '❌'}
• Welcome Msg: {'✅' if settings.get('welcome_enabled', True) else '❌'}
"""
    await update.message.reply_html(info_text)


async def is_group_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check if user is group admin"""
    user = update.effective_user
    chat = update.effective_chat
    
    if user.id in Config.ADMIN_IDS:
        return True
    
    try:
        member = await chat.get_member(user.id)
        return member.status in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.OWNER]
    except:
        return False


async def setgeeta_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle Geeta quotes"""
    chat = update.effective_chat
    
    if chat.type == 'private':
        await update.message.reply_text("Yeh command sirf groups ke liye hai!")
        return
    
    if not await is_group_admin(update, context):
        await update.message.reply_text("❌ Only admins can do this!")
        return
    
    args = context.args
    if not args or args[0].lower() not in ['on', 'off']:
        await update.message.reply_text("Use: /setgeeta on ya /setgeeta off")
        return
    
    value = args[0].lower() == 'on'
    await db.update_group_settings(chat.id, 'geeta_enabled', value)
    
    status = "ON ✅" if value else "OFF ❌"
    await update.message.reply_text(f"Daily Geeta Quote: {status}")


async def setwelcome_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle welcome messages"""
    chat = update.effective_chat
    
    if chat.type == 'private':
        await update.message.reply_text("Yeh command sirf groups ke liye hai!")
        return
    
    if not await is_group_admin(update, context):
        await update.message.reply_text("❌ Only admins can do this!")
        return
    
    args = context.args
    if not args or args[0].lower() not in ['on', 'off']:
        await update.message.reply_text("Use: /setwelcome on ya /setwelcome off")
        return
    
    value = args[0].lower() == 'on'
    await db.update_group_settings(chat.id, 'welcome_enabled', value)
    
    status = "ON ✅" if value else "OFF ❌"
    await update.message.reply_text(f"Welcome Messages: {status}")


async def groupstats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show group stats"""
    chat = update.effective_chat
    
    if chat.type == 'private':
        await update.message.reply_text("Yeh command sirf groups ke liye hai!")
        return
    
    if not await is_group_admin(update, context):
        await update.message.reply_text("❌ Only admins can do this!")
        return
    
    cached_msgs = len(db.get_group_context(chat.id))
    
    stats_text = f"""
📊 <b>Group Statistics</b>

<b>Group:</b> {chat.title}
<b>Cached Messages:</b> {cached_msgs}
"""
    await update.message.reply_html(stats_text)


async def groupsettings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current group settings"""
    chat = update.effective_chat
    
    if chat.type == 'private':
        await update.message.reply_text("Yeh command sirf groups ke liye hai!")
        return
    
    if not await is_group_admin(update, context):
        await update.message.reply_text("❌ Only admins can do this!")
        return
    
    group_data = await db.get_or_create_group(chat.id, chat.title)
    settings = group_data.get('settings', {})
    if isinstance(settings, str):
        try:
            settings = json.loads(settings)
        except:
            settings = {}
    
    settings_text = f"""
⚙️ <b>Group Settings</b>

<b>Group:</b> {chat.title}

<b>Current Settings:</b>
• Geeta Quotes: {'✅ ON' if settings.get('geeta_enabled', True) else '❌ OFF'}
• Welcome Messages: {'✅ ON' if settings.get('welcome_enabled', True) else '❌ OFF'}
"""
    await update.message.reply_html(settings_text)


# ============================================================================
# ADMIN COMMANDS
# ============================================================================

async def admin_check(update: Update) -> bool:
    """Check if user is bot admin"""
    return update.effective_user.id in Config.ADMIN_IDS


async def admin_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show bot stats (admin only)"""
    if not await admin_check(update):
        await update.message.reply_text("Only admins can do this!")
        return
    
    user_count = await db.get_user_count()
    group_count = await db.get_group_count()
    daily_requests = rate_limiter.get_daily_total()
    
    uptime = datetime.now(timezone.utc) - health_server.start_time
    hours = int(uptime.total_seconds() // 3600)
    minutes = int((uptime.total_seconds() % 3600) // 60)
    
    db_status = "🟢 Connected" if db.connected else "🔴 Local Only"
    
    stats_text = f"""
📊 <b>Bot Statistics</b>

<b>Users:</b> {user_count}
<b>Groups:</b> {group_count}
<b>Today's Requests:</b> {daily_requests}

<b>Uptime:</b> {hours}h {minutes}m
<b>Database:</b> {db_status}

<b>Memory:</b>
• Local Users: {len(db.local_users)}
• Local Groups: {len(db.local_groups)}
"""
    await update.message.reply_html(stats_text)


async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show user list (admin only)"""
    if not await admin_check(update):
        await update.message.reply_text("Only admins can do this!")
        return
    
    users = await db.get_all_users()
    
    user_lines = []
    for u in users[:20]:
        name = u.get('first_name', 'Unknown')
        uid = u.get('user_id', 0)
        username = u.get('username', '')
        line = f"• {name}"
        if username:
            line += f" (@{username})"
        line += f" - <code>{uid}</code>"
        user_lines.append(line)
    
    user_list = "\n".join(user_lines) if user_lines else "No users yet"
    
    text = f"""
👥 <b>User List (Last 20)</b>

{user_list}

<b>Total Users:</b> {len(users)}
"""
    await update.message.reply_html(text)


import html  # ✅ Import this at top

async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Broadcast message to all users AND groups (Fixed Version)"""
    if not await admin_check(update):
        return

    args = context.args
    # PIN Check
    if not args or args[0] != Config.BROADCAST_PIN:
        await update.message.reply_html("❌ <b>Wrong PIN!</b>\nUsage: /broadcast PIN Message")
        return

    # Message Content Extraction
    message_text = ' '.join(args[1:]) if len(args) > 1 else None
    reply_msg = update.message.reply_to_message

    if not message_text and not reply_msg:
        await update.message.reply_text("❌ Message likho ya reply karo!")
        return

    status_msg = await update.message.reply_text("📢 fetching database... wait")

    # ✅ FIX 1: Users aur Groups dono ko fetch karna
    users = await db.get_all_users()
    groups = await db.get_all_groups()

    # ✅ FIX 2: Targets List banana (User IDs + Group Chat IDs)
    targets = []
    
    # Add Users
    for user in users:
        uid = user.get('user_id')
        if uid: targets.append(uid)
        
    # Add Groups
    for group in groups:
        gid = group.get('chat_id')
        if gid: targets.append(gid)

    # Stats setup
    success = 0
    failed = 0
    total = len(targets)
    
    if total == 0:
        await status_msg.edit_text("❌ Database empty hai! Koi users ya groups nahi mile.")
        return

    await status_msg.edit_text(f"📢 Starting Broadcast to {len(users)} Users & {len(groups)} Groups...")

    # Message Content Setup
    final_text = html.escape(message_text) if message_text else None

    # ✅ FIX 3: Combined Loop for Users & Groups
    for i, chat_id in enumerate(targets):
        try:
            if reply_msg:
                # Forward or Copy (Copy is safer for privacy)
                await context.bot.copy_message(
                    chat_id=chat_id,
                    from_chat_id=update.effective_chat.id,
                    message_id=reply_msg.message_id
                )
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=final_text,
                    parse_mode=ParseMode.HTML
                )
            success += 1
        except Forbidden:
            failed += 1 # User blocked bot or kicked from group
        except RetryAfter as e:
            # FloodWait Handling
            logger.warning(f"FloodWait: Sleeping {e.retry_after}s")
            await asyncio.sleep(e.retry_after)
            failed += 1 
        except Exception as e:
            failed += 1
            logger.error(f"Broadcast error for {chat_id}: {e}")

        # Status Update (Har 20 messages ke baad update karega taaki fast rahe)
        if i % 20 == 0:
            try:
                await status_msg.edit_text(
                    f"📢 Broadcasting...\n"
                    f"🔄 Progress: {i}/{total}\n"
                    f"✅ Success: {success}\n"
                    f"❌ Failed: {failed}"
                )
            except: pass
        
        await asyncio.sleep(0.05) # Small delay to prevent ban

    # Final Report
    await status_msg.edit_text(
        f"✅ <b>Broadcast Complete!</b>\n\n"
        f"👥 Total Targets: {total}\n"
        f"👤 Users: {len(users)}\n"
        f"🛡Groups: {len(groups)}\n\n"
        f"✅ Success: {success}\n"
        f"❌ Failed/Blocked: {failed}"
    )
    await update.message.reply_html(report)


async def adminhelp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show admin commands"""
    if not await admin_check(update):
        await update.message.reply_text("Only admins can do this!")
        return
    
    help_text = """
🔐 <b>Admin Commands</b>

• /adminstats - Bot statistics
• /users - User list
• /broadcast [PIN] [message] - Broadcast
• /adminhelp - This menu
"""
    await update.message.reply_html(help_text)


# ============================================================================
# SCHEDULED JOBS
# ============================================================================

async def send_daily_geeta(context: ContextTypes.DEFAULT_TYPE):
    """Send daily Geeta quote to all groups"""
    groups = await db.get_all_groups()
    quote = await niyati_ai.generate_geeta_quote()
    
    sent = 0
    for group in groups:
        chat_id = group.get('chat_id')
        settings = group.get('settings', {})
        if isinstance(settings, str):
            try:
                settings = json.loads(settings)
            except:
                settings = {}
        
        if not settings.get('geeta_enabled', True):
            continue
        
        try:
            await context.bot.send_message(chat_id=chat_id, text=quote, parse_mode=ParseMode.HTML)
            sent += 1
            await asyncio.sleep(0.1)
        except:
            pass
    
    logger.info(f"📿 Daily Geeta sent to {sent} groups")


async def cleanup_job(context: ContextTypes.DEFAULT_TYPE):
    """Periodic cleanup"""
    rate_limiter.cleanup_cooldowns()
    await db.cleanup_local_cache()
    logger.info("🧹 Cleanup completed")


# ============================================================================
# 🔴 MAIN MESSAGE HANDLER - WITH SMART DETECTION
# ============================================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle all text messages with:
    1. 🔴 SMART REPLY DETECTION - Won't interrupt user conversations
    2. 🔴 SMART MENTION DETECTION - Ignores when others are mentioned
    3. Force Subscribe Check
    4. Anti-Spam
    5. Rate Limiting
    6. AI Response
    """
    message = update.message
    if not message or not message.text:
        return
        
    user = update.effective_user
    chat = update.effective_chat
    user_message = message.text
    
    # Ignore commands
    if user_message.startswith('/'):
        return

    is_group = chat.type in ['group', 'supergroup']
    is_private = chat.type == 'private'
    bot_username = Config.BOT_USERNAME
    
    # Get bot ID
    bot_id = context.bot.id

    # ════════════════════════════════════════════════════════════════════
    # 🔴 CRITICAL: CHECK IF USER IS TALKING TO OTHERS (NOT BOT)
    # ════════════════════════════════════════════════════════════════════
    if is_group:
        if is_user_talking_to_others(message, bot_username, bot_id):
            # User is replying to another user OR mentioning others
            # Bot should NOT interfere in their conversation
            logger.debug(f"👥 Skipping - User {user.id} is talking to others")
            return

    # ════════════════════════════════════════════════════════════════════
    # FORCE SUBSCRIBE LOGIC
    # ════════════════════════════════════════════════════════════════════
    if is_group and user.id not in Config.ADMIN_IDS:
        targets = await db.get_group_fsub_targets(chat.id)
        
        if targets:
            missing_channels = []
            
            for target in targets:
                t_id = target.get('target_chat_id')
                if not t_id:
                    continue

                try:
                    member = await context.bot.get_chat_member(chat_id=t_id, user_id=user.id)
                    if member.status in ['left', 'kicked', 'restricted']:
                        missing_channels.append(target)
                except:
                    pass

            if missing_channels:
                logger.info(f"🚫 Blocking User {user.id} - Not joined {len(missing_channels)} channels")
                
                try:
                    await message.delete()
                except:
                    pass
                
                keyboard = [[InlineKeyboardButton(f"Join Channel {i+1} 🚀", url=ch.get('target_link', ''))] 
                           for i, ch in enumerate(missing_channels)]
                
                msg = await message.reply_text(
                    f"🚫 <b>Ruko {user.first_name}!</b>\n\n"
                    f"Message karne ke liye {len(missing_channels)} channels join karo.",
                    parse_mode=ParseMode.HTML,
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
                
                await asyncio.sleep(15)
                try:
                    await msg.delete()
                except:
                    pass
                
                return

    # ════════════════════════════════════════════════════════════════════
    # ANTI-SPAM (Groups Only)
    # ════════════════════════════════════════════════════════════════════
    if is_group:
        spam_keywords = ['cp', 'child porn', 'videos price', 'job', 'profit', 'investment', 'crypto', 'bitcoin']
        if any(word in user_message.lower() for word in spam_keywords):
            logger.info(f"🗑️ Spam detected from {user.id}")
            return

    # ════════════════════════════════════════════════════════════════════
    # RATE LIMITING
    # ════════════════════════════════════════════════════════════════════
    allowed, reason = rate_limiter.check(user.id)
    if not allowed:
        if reason == "minute" and is_private:
            await message.reply_text("thoda slow 😅 saans to lene do!")
        return

    # ════════════════════════════════════════════════════════════════════
    # GROUP RESPONSE DECISION
    # ════════════════════════════════════════════════════════════════════
    if is_group:
        db.add_group_message(chat.id, user.first_name, user_message)
        
        should_respond = False
        bot_mention = f"@{bot_username}".lower()
        
        # 1. Bot mentioned
        if bot_mention in user_message.lower():
            should_respond = True
            user_message = re.sub(rf'@{bot_username}', '', user_message, flags=re.IGNORECASE).strip()
        
        # 2. Reply to bot's message
        elif message.reply_to_message and message.reply_to_message.from_user:
            if message.reply_to_message.from_user.id == bot_id:
                should_respond = True
        
        # 3. Random response (only if not talking to others - already checked above)
        if not should_respond:
            if random.random() < Config.GROUP_RESPONSE_RATE:
                should_respond = True
            else:
                return
        
        await db.get_or_create_group(chat.id, chat.title)
        await db.log_user_activity(user.id, f"group_message:{chat.id}")

    if is_private:
        await db.get_or_create_user(user.id, user.first_name, user.username)
        await db.log_user_activity(user.id, "private_message")

    # ════════════════════════════════════════════════════════════════════
    # AI RESPONSE & MEMORY LOGIC
    # ════════════════════════════════════════════════════════════════════
    try:
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

        # 1. GENERATE REPLY
        context_msgs = await db.get_user_context(user.id) if is_private else []
        responses = await niyati_ai.generate_response(
            user_message=user_message,
            context=context_msgs,
            user_name=user.first_name,
            is_group=is_group
        )

        # 2. SMART MEMORY EXTRACTION (Background Task)
        # Ye user ko pata nahi chalega, par Niyati note bana legi
        if is_private:
            memory_note = await niyati_ai.extract_important_info(user_message)
            if memory_note:
                await db.add_user_memory(user.id, memory_note)
                logger.info(f"🧠 Memory stored for {user.first_name}: {memory_note}")

        # 3. SEND MESSAGES (Existing Logic)
        if responses:
            await send_multi_messages(context.bot, chat.id, responses, 
                                    reply_to=message.message_id if is_group else None, 
                                    parse_mode=ParseMode.HTML)
            
            # Save History
            if is_private:
                await db.save_message(user.id, 'user', user_message)
                await db.save_message(user.id, 'assistant', ' '.join(responses))
                
    except Exception as e:
        logger.error(f"Handler Error: {e}")


# ============================================================================
# NEW MEMBER HANDLER
# ============================================================================

async def handle_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle new members joining group"""
    # Safety check: Ensure message and new_chat_members exist
    if not update.message or not update.message.new_chat_members:
        return

    chat = update.effective_chat
    if chat.type not in ['group', 'supergroup']:
        return
    
    group_data = await db.get_or_create_group(chat.id, chat.title)
    
    settings = group_data.get('settings', {})
    if isinstance(settings, str):
        try:
            settings = json.loads(settings)
        except:
            settings = {}
    
    if not settings.get('welcome_enabled', True):
        return
    
    for member in update.message.new_chat_members:
        if member.is_bot:
            continue
        
        mention = StylishFonts.mention(member.first_name, member.id)
        messages = [f"arre! {mention} aaya/aayi group mein 🎉", "welcome yaar! ✨"]
        
        await send_multi_messages(context.bot, chat.id, messages, parse_mode=ParseMode.HTML)
        await db.log_user_activity(member.id, f"joined_group:{chat.id}")


# ============================================================================
# ERROR HANDLER
# ============================================================================

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"❌ Error: {context.error}", exc_info=True)
    
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text("oops technical issue 😅 retry karo?")
        except:
            pass


# ============================================================================
# 🔒 SECRET DIARY FEATURE (Talkie Style)
# ============================================================================

async def send_locked_diary_card(context: ContextTypes.DEFAULT_TYPE):
    """Sends the LOCKED card notification at night"""
    users = await db.get_all_users()
    
    # Image: LOCKED (Blurry or Lock Icon)
    locked_image = "https://images.unsplash.com/photo-1517639493569-5666a7488662?w=600&q=80&blur=50" 
    
    for user in users:
        user_id = user.get('user_id')
        if not user_id: continue

        keyboard = [[InlineKeyboardButton("✨ Unlock Memory ✨", callback_data="unlock_diary")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        caption = (
            "🔒 <b>Secret Memory Created!</b>\n\n"
            "Niyati SECRET DIARY.\n"
            "Card unlock karke padhne ke liye tap karein..."
        )

        try:
            await context.bot.send_photo(
                chat_id=user_id,
                photo=locked_image,
                caption=caption,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
            await asyncio.sleep(1) # Anti-flood wait
        except Exception:
            pass

    logger.info("🔒 Locked Diary Cards sent to all users!")

async def diary_unlock_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the Card Reveal AND The Nervous Reaction"""
    query = update.callback_query
    user = update.effective_user
    
    # 1. Loading Effect
    await query.answer("Unlocking memory... 🗝️")
    
    # Image: Unlocked (Clear Photo)
    unlocked_image = "https://images.unsplash.com/photo-1517639493569-5666a7488662?w=800&q=80"

    # 2. Generate Diary Text (On the fly)
    history = await db.get_user_context(user.id)
    
    prompt = [
        {"role": "system", "content": f"""
        You are Niyati. Write a SHORT Diary Entry (max 3 sentences) about chatting with {user.first_name}.
        Start with "Dear Diary...".
        Format: Hinglish. Emotional/Secretive. Reveal a small hidden feeling.
        """},
        {"role": "user", "content": str(history)}
    ]
    
    diary_text = await niyati_ai._call_gpt(prompt)
    if not diary_text:
        diary_text = "Dear Diary... Aaj inse baat karke acha laga, par thaki hu. Kal likhungi! 😴"

    final_caption = (
        f"🔓 <b>Unlocked: Niyati's Diary</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"<i>{diary_text}</i>\n\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"✨ Saved to Memories"
    )

    # 3. The "Card Flip" (Visual Update)
    try:
        await query.edit_message_media(
            media=InputMediaPhoto(media=unlocked_image, caption=final_caption, parse_mode=ParseMode.HTML)
        )
    except Exception as e:
        logger.error(f"Diary unlock failed: {e}")
        return

    # ════════════════════════════════════════════════════════════
    # 🔴 THE REACTION LOGIC (Wait & Reply)
    # ════════════════════════════════════════════════════════════
    
    # Step 4: Wait for User to Read (10 Seconds)
    await asyncio.sleep(10) 

    # Step 5: Nervous Reaction
    try:
        await context.bot.send_chat_action(chat_id=user.id, action=ChatAction.TYPING)
        await asyncio.sleep(1.5)
        
        reaction_1 = "Oye! Tumne meri diary padh li? 😳"
        await context.bot.send_message(chat_id=user.id, text=reaction_1)

        # Step 6: Follow-up Question (3 Seconds later)
        await asyncio.sleep(3)
        await context.bot.send_chat_action(chat_id=user.id, action=ChatAction.TYPING)
        await asyncio.sleep(1.5)
        
        reaction_2 = "Pls judge mat karna... wese, tumhe bura to nahi laga na? 👉👈"
        await context.bot.send_message(chat_id=user.id, text=reaction_2)
        
    except Exception as e:
        logger.error(f"Reaction failed: {e}")
        
# ============================================================================
# BOT SETUP
# ============================================================================

def setup_handlers(app: Application):
    """Register all handlers"""
    
    # Private commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(CommandHandler("mood", mood_command))
    app.add_handler(CommandHandler("forget", forget_command))
    app.add_handler(CommandHandler("meme", meme_command))
    app.add_handler(CommandHandler("shayari", shayari_command))
    app.add_handler(CommandHandler("stats", user_stats_command))
    
    # Admin group commands
    app.add_handler(CommandHandler("setgeeta", setgeeta_command))
    app.add_handler(CommandHandler("setwelcome", setwelcome_command))
    app.add_handler(CommandHandler("groupstats", groupstats_command))
    app.add_handler(CommandHandler("groupsettings", groupsettings_command))

    # Admin private commands
    app.add_handler(CommandHandler("adminstats", admin_stats_command))
    app.add_handler(CommandHandler("users", users_command))
    app.add_handler(CommandHandler("broadcast", broadcast_command))
    app.add_handler(CommandHandler("adminhelp", adminhelp_command))
    app.add_handler(CallbackQueryHandler(diary_unlock_callback, pattern="^unlock_diary$"))

    # Message Handlers
    # 1. New Member (Welcome) - FIXED: Uses MessageHandler with StatusUpdate filter
    app.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, handle_new_member))
    
    # 2. Main Message Handler (Text & Group Logic)
    # Filters.text & ~Filters.command ensures we only catch text that isn't a command
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Error Handler
    app.add_error_handler(error_handler)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def post_init(application: Application):
    """Initialize DB and Schedule Jobs with CORRECT UTC TIMING"""
    await db.initialize()
    await health_server.start()
    
    job_queue = application.job_queue
    
    # 1. Good Morning (India: 08:30 AM IST = 03:00 AM UTC)
    job_queue.run_daily(
        routine_message_job,
        # ERROR WAS HERE: changed 'time' to 'dt_time'
        time=dt_time(hour=3, minute=0, second=0),  
        data='morning',
        name='daily_morning'
    )

    # 2. Good Night (India: 10:30 PM IST = 05:00 PM UTC)
    job_queue.run_daily(
        routine_message_job,
        # ERROR WAS HERE: changed 'time' to 'dt_time'
        time=dt_time(hour=17, minute=0, second=0), 
        data='night',
        name='daily_night'
    )

    # 3. Random Check-in (Runs every 4 hours)
    job_queue.run_repeating(
        routine_message_job,
        interval=timedelta(hours=4),
        first=timedelta(seconds=60),
        data='random',
        name='random_checkin'
    )

    # 4. Secret Diary (India: 10:30 PM IST = 5:00 PM UTC)
    job_queue.run_daily(
        send_locked_diary_card,
        time=dt_time(hour=17, minute=0, second=0), # 17:00 UTC = 10:30 PM IST
        name='locked_diary_job'
    )

    logger.info("🚀 Niyati Bot Started with FIXED Timings (IST)!")

async def post_shutdown(application: Application):
    """Bot band hone par cleanup"""
    await health_server.stop()
    await db.close()
    logger.info("😴 Niyati Bot Stopped.")

# ============================================================================
# ROUTINE JOBS (Morning, Night, Random Check-ins)
# ============================================================================

async def routine_message_job(context: ContextTypes.DEFAULT_TYPE):
    """Sends Messages with REAL HUMAN TIMING CHECK"""
    job_data = context.job.data
    
    # 1. Get Current Time (India)
    ist = pytz.timezone(Config.DEFAULT_TIMEZONE)
    now = datetime.now(ist)
    current_time_str = now.strftime("%I:%M %p") # e.g., "03:30 PM"
    current_hour = now.hour

    # DND Check (Raat 11 se Subah 8 band)
    if job_data == 'random' and (current_hour >= 23 or current_hour < 8):
        return

    users = await db.get_all_users()
    
    # Fallback Generic Texts
    morning_texts = ["Good morning! ☀️", "Uth gaye? ✨"]
    night_texts = ["Good night 🌙", "So jao ab 😴"]
    random_texts = ["Kya chal raha hai?", "Yaad aa rahi thi..."]

    count = 0
    for user in users:
        user_id = user.get('user_id')
        if not user_id: continue

        # 🧠 MEMORY CHECK
        memories = await db.get_active_memories(user_id)
        final_msg = ""
        
        if memories:
            memory_context = " | ".join(memories)
            
            # 🔥 THE SMART CHECK PROMPT
            # Hum AI ko Current Time bata rahe hain
            prompt = f"""
            Role: You are Niyati (Girlfriend/Bestie).
            Current Time in India: {current_time_str}
            User's Memory Note: "{memory_context}"
            
            Task: Decide if you should ask about this memory NOW.
            
            🧠 COMMON SENSE RULES:
            1. If event is in FUTURE -> Say "All the best for [Time]!"
            2. If event is HAPPENING NOW -> OUTPUT "SKIP" (Don't disturb).
            3. If event is OVER -> Ask "Kaisa hua?"
            
            4. ⚠️ CRITICAL - IF DURATION IS UNKNOWN:
               - Assume Standard Durations:
                 * Exam = 3 hours
                 * Movie = 3 hours
                 * Date/Meeting = 2 hours
                 * Travel = Depends on context
               
               Example: User said "Exam at 2 PM" (End time unknown).
               - If Current Time is 3 PM -> Assume Exam is running -> OUTPUT "SKIP".
               - If Current Time is 6 PM -> Assume Exam over -> Ask "Kaisa gaya?".
               
               *Better to ask LATE than EARLY.*
            
            Output: The Message to send (Hinglish). If you should wait/skip, output "SKIP".
            """
            
            check_response = await niyati_ai._call_gpt([{"role": "user", "content": prompt}], max_tokens=60)
            
            # Agar AI ne bola SKIP, to hum Generic message bhi nahi bhejenge (Disturb nahi karna)
            if check_response and "SKIP" in check_response:
                continue 
                
            if check_response:
                final_msg = check_response

        # Agar Memory logic ne kuch nahi diya, to Generic bhej do
        if not final_msg:
            if job_data == 'morning': final_msg = random.choice(morning_texts)
            elif job_data == 'night': final_msg = random.choice(night_texts)
            elif job_data == 'random':
                if random.random() > 0.2: continue
                final_msg = random.choice(random_texts)

        try:
            await asyncio.sleep(random.uniform(0.5, 2.0))
            await context.bot.send_message(chat_id=user_id, text=final_msg)
            count += 1
        except Exception:
            pass
        
        if count > 100: break

    logger.info(f"Routine Job ({job_data}) sent to {count} users.")

# ============================================================================
# UPDATE POST_INIT TO SCHEDULE THESE JOBS
# ============================================================================

async def post_init(application: Application):
    """Initialize DB and Schedule Jobs with CORRECT UTC TIMING"""
    await db.initialize()
    await health_server.start()
    
    job_queue = application.job_queue
    
    # 1. Good Morning (India: 08:30 AM IST = 03:00 AM UTC)
    job_queue.run_daily(
        routine_message_job,
        time=dt_time(hour=3, minute=0, second=0),  # <--- FIXED: uses dt_time
        data='morning',
        name='daily_morning'
    )

    # 2. Good Night (India: 10:30 PM IST = 05:00 PM UTC)
    job_queue.run_daily(
        routine_message_job,
        time=dt_time(hour=17, minute=0, second=0), # <--- FIXED: uses dt_time
        data='night',
        name='daily_night'
    )

    # 3. Random Check-in (Runs every 4 hours)
    job_queue.run_repeating(
        routine_message_job,
        interval=timedelta(hours=4),
        first=timedelta(seconds=60),
        data='random',
        name='random_checkin'
    )

    logger.info("🚀 Niyati Bot Started with FIXED Timings (IST)!")

def main():
    """Main entry point"""
    if not Config.TELEGRAM_BOT_TOKEN:
        logger.error("❌ Error: TELEGRAM_BOT_TOKEN nahi mila! .env file check karo.")
        return

    # Application Builder
    app = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).post_init(post_init).post_shutdown(post_shutdown).build()

    # Setup Handlers
    setup_handlers(app)

    # Start Polling (Bot ko run karo)
    # drop_pending_updates=True means purane messages ignore karega jab bot restart hoga
    logger.info("⏳ Initializing Bot...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    try:
        # Windows par asyncio loop policy fix (agar zarurat ho)
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"❌ Fatal Error: {e}", exc_info=True)
