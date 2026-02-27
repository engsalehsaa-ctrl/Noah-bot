#!/usr/bin/env python3
"""
نوح (Noah) - Telegram AI Bot
Powered by Poe API - All AI models with one key
"""

import os
import re
import logging
from io import BytesIO
from collections import defaultdict

import httpx
import openai
from telegram import Update, ChatAction
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
POE_API_KEY = os.environ["POE_API_KEY"]

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Poe API client (OpenAI-compatible)
poe_client = openai.AsyncOpenAI(
    api_key=POE_API_KEY,
    base_url="https://api.poe.com/v1",
)

# Models for each task (same as Poe Noah bot!)
MODELS = {
    "chat": "GPT-5.2",
    "image": "Imagen-4-Ultra",
    "voice": "ElevenLabs-v2.5-Turbo",
    "music": "ElevenLabs-Music",
    "video": "Sora-2",
    "summarize": "Claude-Sonnet-4.5",
    "search": "Gemini-3-Flash",
    "code": "Claude-Sonnet-4.5",
    "translate": "GPT-5.2",
    "classify": "Grok-4.1-Fast-Non-Reasoning",
}

# Per-user conversation history
user_histories = defaultdict(list)
MAX_HISTORY = 20


# ═══════════════════════════════════════════════════════════
# Poe API Helpers
# ═══════════════════════════════════════════════════════════

async def call_text(model, messages, system=None, **params):
    """Call Poe API for text models (chat, code, translate, etc.)."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(messages)

    kwargs = {"model": model, "messages": msgs, "stream": False}
    if params:
        kwargs["extra_body"] = params

    response = await poe_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


async def call_media(model, prompt, **params):
    """Call Poe API for media models (image, voice, music, video)."""
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if params:
        kwargs["extra_body"] = params

    response = await poe_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def extract_urls(text):
    """Extract media URLs from bot response."""
    if not text:
        return []
    return re.findall(r'https?://[^\s\)\]>"\']+', text)


async def download_file(url):
    """Download file from URL."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))


async def translate_to_english(text):
    """Translate Arabic text to English for media generation."""
    result = await call_text(
        MODELS["classify"],
        [{"role": "user", "content": f"Translate to English as a short prompt. Only output the translation:\n{text}"}],
    )
    return result.strip()


async def send_long_message(update, text):
    """Split long messages for Telegram's 4096 char limit."""
    if not text:
        return
    MAX_LEN = 4000
    if len(text) <= MAX_LEN:
        await update.message.reply_text(text)
    else:
        for i in range(0, len(text), MAX_LEN):
            await update.message.reply_text(text[i:i + MAX_LEN])


# ═══════════════════════════════════════════════════════════
# Task Handlers
# ═══════════════════════════════════════════════════════════

async def do_chat(update, content, user_id):
    """💬 Chat with GPT-5.2 (with conversation memory)."""
    try:
        await update.message.chat.send_action(ChatAction.TYPING)
        history = user_histories[user_id]
        messages = history + [{"role": "user", "content": content}]

        response_text = await call_text(
            MODELS["chat"],
            messages,
            system=(
                "You are Noah (نوح), a helpful bilingual (Arabic/English) AI assistant. "
                "Respond in the same language the user uses. Be helpful, friendly, and concise."
            ),
        )

        history.append({"role": "user", "content": content})
        history.append({"role": "assistant", "content": response_text})
        user_histories[user_id] = history[-MAX_HISTORY:]

        await send_long_message(update, response_text)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        await update.message.reply_text("❌ حدث خطأ، حاول مرة أخرى\nError, try again.")


async def do_image(update, content):
    """🖼️ Generate images with Imagen-4-Ultra."""
    if not content:
        await update.message.reply_text("🖼️ الرجاء وصف الصورة\nPlease describe the image.")
        return
    try:
        await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
        prompt = await translate_to_english(content) if is_arabic(content) else content

        response_text = await call_media(MODELS["image"], prompt, aspect_ratio="1:1")

        urls = extract_urls(response_text)
        if urls:
            img_bytes = await download_file(urls[0])
            await update.message.reply_photo(photo=BytesIO(img_bytes), caption=f"🖼️ {content}")
        else:
            await update.message.reply_text(response_text or "❌ لم أتمكن من توليد الصورة")
    except Exception as e:
        logger.error(f"Image error: {e}")
        await update.message.reply_text("❌ خطأ في توليد الصورة\nImage generation error.")


async def do_voice(update, content):
    """🔊 Text-to-speech with ElevenLabs."""
    if not content:
        await update.message.reply_text("🔊 الرجاء كتابة النص\nPlease provide text.")
        return
    try:
        await update.message.chat.send_action(ChatAction.RECORD_VOICE)
        voice = "Sana" if is_arabic(content) else "Jessica"
        lang = "ar" if is_arabic(content) else "en"

        response_text = await call_media(
            MODELS["voice"], content, voice=voice, language=lang
        )

        urls = extract_urls(response_text)
        if urls:
            audio_bytes = await download_file(urls[0])
            await update.message.reply_voice(voice=BytesIO(audio_bytes))
        else:
            await update.message.reply_text("❌ خطأ في الصوت\nVoice error.")
    except Exception as e:
        logger.error(f"Voice error: {e}")
        await update.message.reply_text("❌ خطأ في الصوت\nVoice error.")


async def do_music(update, content):
    """🎵 Generate music with ElevenLabs-Music."""
    if not content:
        await update.message.reply_text("🎵 الرجاء وصف الموسيقى\nPlease describe the music.")
        return
    try:
        await update.message.reply_text("🎵 جارِ التوليد (قد يستغرق دقيقة)...\nGenerating music...")
        prompt = await translate_to_english(content) if is_arabic(content) else content

        response_text = await call_media(MODELS["music"], prompt, music_length_ms=30000)

        urls = extract_urls(response_text)
        if urls:
            audio_bytes = await download_file(urls[0])
            await update.message.reply_audio(audio=BytesIO(audio_bytes), title=f"🎵 {content[:30]}")
        else:
            await update.message.reply_text("❌ خطأ في الموسيقى\nMusic error.")
    except Exception as e:
        logger.error(f"Music error: {e}")
        await update.message.reply_text("❌ خطأ في الموسيقى\nMusic error.")


async def do_video(update, content):
    """🎬 Generate video with Sora-2."""
    if not content:
        await update.message.reply_text("🎬 الرجاء وصف الفيديو\nPlease describe the video.")
        return
    try:
        await update.message.reply_text("🎬 جارِ التوليد (قد يستغرق عدة دقائق)...\nGenerating video...")
        prompt = await translate_to_english(content) if is_arabic(content) else content

        response_text = await call_media(MODELS["video"], prompt, size="1280x720")

        urls = extract_urls(response_text)
        if urls:
            video_bytes = await download_file(urls[0])
            await update.message.reply_video(video=BytesIO(video_bytes), caption=f"🎬 {content}")
        else:
            await update.message.reply_text(response_text or "❌ خطأ في الفيديو")
    except Exception as e:
        logger.error(f"Video error: {e}")
        await update.message.reply_text("❌ خطأ في الفيديو\nVideo error.")


async def do_translate(update, content):
    """🌍 Translation with GPT-5.2."""
    if not content:
        await update.message.reply_text("🌍 الرجاء إرسال النص\nPlease provide text.")
        return
    try:
        await update.message.chat.send_action(ChatAction.TYPING)
        response_text = await call_text(
            MODELS["translate"],
            [{"role": "user", "content": content}],
            system=(
                "You are a professional translator. If Arabic→English, if English→Arabic, "
                "if other→both Arabic and English. Only output the translation."
            ),
        )
        await send_long_message(update, response_text)
    except Exception as e:
        logger.error(f"Translate error: {e}")
        await update.message.reply_text("❌ خطأ في الترجمة\nTranslation error.")


async def do_summarize(update, content):
    """📝 Summarize with Claude-Sonnet-4.5."""
    if not content:
        await update.message.reply_text("📝 الرجاء إرسال النص\nPlease provide text.")
        return
    try:
        await update.message.chat.send_action(ChatAction.TYPING)
        response_text = await call_text(
            MODELS["summarize"],
            [{"role": "user", "content": f"Summarize:\n\n{content}"}],
            system="Summarize concisely in the same language. Use bullet points.",
        )
        await send_long_message(update, response_text)
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        await update.message.reply_text("❌ خطأ في التلخيص\nSummarization error.")


async def do_search(update, content):
    """🌐 Web search with Gemini-3-Flash."""
    if not content:
        await update.message.reply_text("🌐 الرجاء كتابة ما تريد البحث عنه\nPlease provide query.")
        return
    try:
        await update.message.chat.send_action(ChatAction.TYPING)
        response_text = await call_text(
            MODELS["search"],
            [{"role": "user", "content": content}],
            system="Search and provide accurate, current info. Cite sources. Respond in the user's language.",
            web_search=True,
        )
        await send_long_message(update, response_text)
    except Exception as e:
        logger.error(f"Search error: {e}")
        await update.message.reply_text("❌ خطأ في البحث\nSearch error.")


async def do_code(update, content):
    """💻 Coding help with Claude-Sonnet-4.5."""
    if not content:
        await update.message.reply_text("💻 الرجاء وصف ما تحتاجه\nPlease describe what you need.")
        return
    try:
        await update.message.chat.send_action(ChatAction.TYPING)
        response_text = await call_text(
            MODELS["code"],
            [{"role": "user", "content": content}],
            system="You are an expert programmer. Help clearly with well-commented code. Respond in the user's language.",
        )
        await send_long_message(update, response_text)
    except Exception as e:
        logger.error(f"Code error: {e}")
        await update.message.reply_text("❌ خطأ في البرمجة\nCoding error.")


# ═══════════════════════════════════════════════════════════
# Intent Classification
# ═══════════════════════════════════════════════════════════

def classify_intent(text):
    """Keyword-based classification for speed (no API call needed)."""
    t = text.lower()

    keywords = {
        "image": ["ارسم", "صمم", "صورة", "draw", "generate image", "create image", "picture"],
        "voice": ["اقرأ لي", "تكلم", "حول لصوت", "حوله صوت", "read aloud", "speak", "say this"],
        "music": ["موسيقى", "أغنية", "لحن", "music", "song", "melody", "beat"],
        "video": ["فيديو", "مقطع", "video", "clip", "animate"],
        "translate": ["ترجم", "ترجمة", "translate"],
        "summarize": ["لخص", "تلخيص", "ملخص", "summarize", "summary"],
        "search": ["ابحث", "بحث عن", "search", "find info", "ما آخر", "أخبار"],
        "code": ["كود", "برمج", "اكتب كود", "code", "program", "function", "debug"],
    }

    for task, kws in keywords.items():
        if any(kw in t for kw in kws):
            return task

    return "chat"


# ═══════════════════════════════════════════════════════════
# Telegram Handlers
# ═══════════════════════════════════════════════════════════

WELCOME_MSG = """مرحباً! أنا نوح 🤖
Hello! I'm Noah 🤖

مدعوم بـ Poe API - كل النماذج بمفتاح واحد!
Powered by Poe API - All AI models!

💬 محادثة (GPT-5.2)
🖼️ صور (Imagen-4-Ultra)
🔊 صوت (ElevenLabs)
🎵 موسيقى (ElevenLabs-Music)
🎬 فيديو (Sora-2)
📝 تلخيص (Claude)
🌐 بحث (Gemini + Web)
💻 كود (Claude)
🌍 ترجمة (GPT-5.2)

الأوامر / Commands:
/image - صورة
/voice - صوت
/music - موسيقى
/video - فيديو
/translate - ترجمة
/summarize - تلخيص
/search - بحث
/code - برمجة
/clear - مسح المحادثة

أو أرسل رسالتك مباشرة! ✨"""


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME_MSG)


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_histories.pop(update.message.from_user.id, None)
    await update.message.reply_text("✅ تم مسح المحادثة\nConversation cleared")


async def route_task(update, task, content):
    """Route to the correct handler."""
    user_id = update.message.from_user.id
    handlers = {
        "chat": lambda: do_chat(update, content, user_id),
        "image": lambda: do_image(update, content),
        "voice": lambda: do_voice(update, content),
        "music": lambda: do_music(update, content),
        "video": lambda: do_video(update, content),
        "translate": lambda: do_translate(update, content),
        "summarize": lambda: do_summarize(update, content),
        "search": lambda: do_search(update, content),
        "code": lambda: do_code(update, content),
    }
    await handlers.get(task, handlers["chat"])()


async def cmd_task(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle English slash commands."""
    text = update.message.text
    cmd = text.split()[0][1:].lower()
    content = text.partition(" ")[2].strip()
    cmd_map = {
        "image": "image", "voice": "voice", "music": "music",
        "video": "video", "translate": "translate", "summarize": "summarize",
        "search": "search", "code": "code",
    }
    await route_task(update, cmd_map.get(cmd, "chat"), content)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular messages + Arabic commands."""
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()

    # Check Arabic slash commands
    arabic_cmds = {
        "/صورة": "image", "/صوت": "voice", "/موسيقى": "music",
        "/فيديو": "video", "/ترجمة": "translate", "/تلخيص": "summarize",
        "/بحث": "search", "/كود": "code",
    }
    for prefix, task in arabic_cmds.items():
        if text.startswith(prefix):
            content = text[len(prefix):].strip()
            await route_task(update, task, content)
            return

    # Auto-classify by keywords
    task = classify_intent(text)
    await route_task(update, task, text)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # System commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))

    # Task commands (English)
    for cmd in ["image", "voice", "music", "video", "translate", "summarize", "search", "code"]:
        app.add_handler(CommandHandler(cmd, cmd_task))

    # Regular messages + Arabic commands
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    logger.info("🤖 Noah bot is running with Poe API!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
