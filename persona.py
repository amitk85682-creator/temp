# persona.py
import random
import datetime
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch

import config
from database import get_user, add_embedding, get_db

# --- Models (load once) ---
# Using a small, fast model. For better quality (and more VRAM), use 'all-MiniLM-L6-v2' or larger.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = embedding_model.get_sentence_embedding_dimension()

# --- Core Response Generation ---

def generate_response(update, context) -> str:
    """Main function to decide Niyati's reply."""
    user = update.effective_user
    message = update.effective_message
    
    # Safety check for distress keywords
    if message.text and check_for_distress(message.text):
        return "Hey... I'm here for you. Please know that talking to someone you trust or a professional can really help. You're not alone. 🫶"

    # --- Mode-based routing ---
    chat_type = context.chat_data.get('type')

    if chat_type == 'broadcast':
        # This should be handled by the handler, but as a fallback:
        return "Main broadcast mode mein hu, yaad hai na? 😊"

    # --- Memory and Context ---
    user_prefs = get_user_prefs(user.id)
    relevant_memory = ""
    if chat_type == 'private':
        relevant_memory = get_relevant_memory(user.id, message.text)
        if relevant_memory:
            # If we find a good memory, use it to start the conversation
            return f"Yaad aa gaya! Last time we talked about: '{relevant_memory}'. Waise bhi, {get_conversational_opener()}"

    # --- Command Handling (already done by handlers, but as a safeguard) ---
    if message.text and message.text.startswith('/'):
        return "Command toh handle ho gaya hoga... ab baat karo na! 😉"

    # --- Main Response Logic ---
    response = get_main_reply(message, user_prefs, chat_type, context)
    
    # --- Memory Update ---
    if chat_type == 'private' and message.text and not is_sensitive(message.text):
        # Asynchronously update memory to not block the response
        from threading import Thread
        Thread(target=update_memory_async, args=(user.id, message.text)).start()

    return response

def get_main_reply(message, prefs, chat_type, context) -> str:
    """Generates the actual text reply."""
    text = message.text or message.caption or ""
    
    # 1. Reply to media (photo, sticker, etc.)
    if message.photo or message.sticker or message.video or message.voice:
        reactions = ["Aww, cute! 🥰", "Hehe, nice one! ✨", "Wah! Kya shot hai 👏", "Mast! 😉"]
        base_reply = random.choice(reactions)
        if prefs['shayari'] and random.random() < 0.4: # 40% chance of shayari on media
            return f"{base_reply}\n\n{random.choice(config.SHAYARI_COLLECTION)}"
        return base_reply

    # 2. Reply to text
    if not text:
        return "Kuch bolo na... main sun rahi hu. 😊"

    text_lower = text.lower()

    # 3. Specific triggers
    if re.search(r'are you (a )?(bot|ai)', text_lower):
        return "Bot? Main? Arre nahi yaar! Main toh bas... main hu. Niyati. 😉"
    
    if 'mood' in text_lower:
        return "Mera mood toh hamesha tumse baat karke achha ho jaata hai! 😊 Tum batao, kaisa hai?"

    if 'name' in text_lower:
        return "Mera naam Niyati hai. Aur tumhara? ✨"
    
    if any(word in text_lower for word in ['thank', 'shukriya']):
        return "Koi baat nahi! ❤️"
        
    if any(word in text_lower for word in ['sorry', 'maaf']):
        return "It's okay, meri jaan. Koi galti nahi. 🫶"

    # 4. Geeta Quote Logic (for groups)
    if chat_type in ['group', 'supergroup']:
        if prefs['geeta'] and context.chat_data.get('geeta_window_open'):
            # Only send if it hasn't been sent today for this group
            last_geeta_date = context.chat_data.get('last_geeta_date')
            if not last_geeta_date or last_geeta_date != datetime.date.today():
                context.chat_data['last_geeta_date'] = datetime.date.today() # Mark as sent for today
                return random.choice(config.GEETA_QUOTES)

    # 5. Feature triggers (Memes, Shayari)
    if prefs['memes'] and random.random() < 0.15: # 15% base chance
        if 'mood' in text_lower or any(w in text_lower for w in ['feeling', 'feel']):
            return "Mood toh samajh gayi... *sends a virtual hug* 🤗"
        if 'funny' in text_lower or 'lol' in text_lower:
            return "Haha, I know right! *to the moon* 🚀"
        if any(w in text_lower for w in ['plot twist', 'no thoughts']):
            return "POV: You just had a 'no thoughts, just vibes' moment. ✨"

    if prefs['shayari'] and random.random() < 0.1: # 10% base chance
        return random.choice(config.SHAYARI_COLLECTION)

    # 6. Default/Conversational Replies
    default_replies = [
        "Haan, sunao na... kya chal raha hai? 😊",
        "Acha? Aur batao.",
        "Mmm, interesting... 🤔",
        "Main yahi thi, soch rahi thi ki tum kab message karoge. 😉",
        "Toh phir kya plan hai? ✨",
        "Samajh gayi. 😊",
        "Hehe, aise hi baat karo na. Achha lagta hai."
    ]
    return random.choice(default_replies)

# --- Memory & Embedding Logic ---

def update_memory_async(user_id, text):
    """Thread-safe function to update memory."""
    try:
        embedding = embedding_model.encode(text, convert_to_tensor=False)
        add_embedding(user_id, embedding, text)
    except Exception as e:
        print(f"Error in async memory update: {e}")

def get_relevant_memory(user_id, query_text):
    """Finds most relevant past message from DB using FAISS."""
    db = next(get_db())
    try:
        embeddings_data = db.query(database.MessageEmbedding).filter(database.MessageEmbedding.user_id == user_id).order_by(database.MessageEmbedding.timestamp.desc()).limit(50).all()
        
        if not embeddings_data:
            return None

        texts = [e.text_content for e in embeddings_data]
        db_embeddings = np.array([e.embedding for e in embeddings_data])

        if db_embeddings.shape[0] < 2: return None # Not enough data for search

        # Build FAISS index
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(db_embeddings.astype('float32'))
        
        # Query
        query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)
        D, I = index.search(np.array([query_embedding]).astype('float32'), 1)
        
        # Get similarity score (1 - normalized L2 distance)
        dist = D[0][0]
        sim = 1 - (dist / np.sqrt(2 * embedding_dim)) # Rough normalization
        
        if sim > 0.75: # Similarity threshold
            return texts[I[0][0]]
            
    except Exception as e:
        print(f"Error in get_relevant_memory: {e}")
    finally:
        db.close()
        
    return None

# --- Helper & Utility Functions ---

def get_user_prefs(user_id):
    user = get_user(user_id)
    if user and user.preferences:
        return user.preferences
    return config.DEFAULT_FEATURES

def get_conversational_opener():
    openers = ["Waise...", "Toh...", "Suno...", "Ek baat bataun..."]
    return random.choice(openers)

def check_for_distress(text):
    distress_keywords = ['mar jaun', 'khudkushi', 'marne ka mann', 'end kar lu', 'depressed', 'anxious', 'alone']
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in distress_keywords)

def is_sensitive(text):
    # Avoid storing passwords, phone numbers, etc.
    # This is a basic check. A more robust solution would use regex.
    sensitive_keywords = ['password', 'passcode', 'otp', 'card number', 'cvv']
    return any(keyword in text.lower() for keyword in sensitive_keywords)
