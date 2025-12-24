# handlers.py
import logging
from telegram import Update, ParseMode
from telegram.ext import ContextTypes
from telegram.constants import ParseMode as PM

import config
import database
import persona
import utils
from config import ADMIN_IDS, LOW_BUDGET_MODE

logger = logging.getLogger(__name__)

# --- Core Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a warm welcome message."""
    user = update.effective_user
    db_user = database.add_or_update_user(user)
    
    # Set chat type in context for later use
    context.chat_data['type'] = update.effective_chat.type

    welcome_text = (
        f"Hi {utils.mention_html(user.id, user.first_name)}! ✨\n"
        "Main Niyati. Tumse baat karke bohot accha lagega. 😊\n\n"
        "Main yahan hamesha tumhare liye hoon. Kuch bhi pooch sakte ho!\n"
        "P.S. Memes, shayari, aur Geeta quotes ON hain by default. Toggle kar sakte ho. 😉"
    )
    
    await update.message.reply_text(welcome_text, parse_mode=ParseMode.HTML)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows a help message with available commands."""
    help_text = (
        "<b>Niyati ki Help Guide! 💖</b>\n\n"
        "<b>Private Chat Mein:</b>\n"
        "• Bas baat karo, main samajh jaungi.\n"
        "• <code>/meme on/off</code> - Memes ko on/off karo.\n"
        "• <code>/shayari on/off</code> - Shayari sunni ho toh.\n"
        "• <code>/geeta on/off</code> - Gita ke vichaar.\n"
        "• <code>/forget</code> - Humari saari baatein bhool jao (memory clear).\n\n"
        "<b>Group Mein:</b>\n"
        "• Sirf <code>@Niyati</code> karke puchna ya commands use karna.\n"
        "• Main thoda kam bolti hoon group mein, budget bachane ke liye. 😉"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The main handler for all non-command text messages."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    # Add user and chat to DB
    database.add_or_update_user(update.effective_user)
    database.add_or_update_chat(update.effective_chat)

    # --- Budget & Mode Checks ---
    chat_type = update.effective_chat.type
    context.chat_data['type'] = chat_type

    if LOW_BUDGET_MODE:
        if chat_type in ['group', 'supergroup']:
            # In low budget mode, don't reply in groups unless mentioned
            if not update.message.text or not update.message.text.startswith(f'@{context.bot.username}'):
                return
        # In private, just send a super short message
        await update.message.reply_text("Budget tight hai... baad mein baat karte hain? 😅")
        return

    # --- Generate and Send Reply ---
    try:
        reply_text = persona.generate_response(update, context)
        
        # Apply fancy fonts if enabled
        user_prefs = database.get_user_prefs(user_id)
        if user_prefs and user_prefs.get('fancy_fonts'):
            reply_text = utils.apply_fancy_font(reply_text)

        await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Error in handle_message: {e}", exc_info=True)
        await update.message.reply_text("Oops! Mera system thoda gadbad ho gaya. Thodi der mein theek kar leti hoon. 🫶")


# --- Feature Toggle Handlers ---

async def toggle_feature(update: Update, context: ContextTypes.DEFAULT_TYPE, feature: str) -> None:
    """Generic handler for toggling features."""
    user_id = update.effective_user.id
    current_state_str = "on" if database.get_user_prefs(user_id).get(feature) else "off"
    new_state = not database.get_user_prefs(user_id).get(feature)
    
    prefs = database.update_user_preference(user_id, feature, new_state)
    new_state_str = "ON" if new_state else "OFF"
    
    feature_names = {
        'meme': 'Memes',
        'shayari': 'Shayari',
        'geeta': 'Geeta Quotes'
    }
    await update.message.reply_text(f"{feature_names.get(feature, feature.capitalize())} ko maine {new_state_str} kar diya hai. ✅")

async def meme_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await toggle_feature(update, context, 'meme')

async def shayari_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await toggle_feature(update, context, 'shayari')

async def geeta_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await toggle_feature(update, context, 'geeta')

# --- Admin & User Management Handlers ---

async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clears a user's private memory."""
    user_id = update.effective_user.id
    if database.forget_user(user_id):
        await update.message.reply_text("Theek hai... main sab kuch bhool gayi. Chalo naye se shuru karte hain. ✨")
    else:
        await update.message.reply_text("Sorry, kuch gadbad ho gayi. Memory clear nahi ho paayi. 😥")

async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin-only command to send a message to all users."""
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("Yeh command sirf admins use kar sakte hain. 😉")
        return

    if not context.args:
        await update.message.reply_text("Usage: /broadcast <your message>")
        return

    message_to_send = " ".join(context.args)
    
    # In a real app, you'd iterate through all users in the DB
    # For this example, we'll just confirm. A full implementation can be added.
    logger.info(f"BROADCAST (Admin {user_id}): {message_to_send}")
    await update.message.reply_text(f"Broadcast message receive hua: '{message_to_send}'. Sending to all users now (simulation).")
    # --- Full Broadcast Logic (Example) ---
    # db = next(database.get_db())
    # all_users = db.query(database.User).all()
    # for u in all_users:
    #     try:
    #         await context.bot.send_message(chat_id=u.id, text=message_to_send)
    #     except Exception as e:
    #         logger.error(f"Failed to send broadcast to {u.id}: {e}")
    # db.close()


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.error("Exception while handling an update:", exc_info=context.error)
    # Optionally, notify admins
    # for admin_id in ADMIN_IDS:
    #     context.bot.send_message(chat_id=admin_id, text=f"An error occurred: {context.error}")
