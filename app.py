"""
app.py
──────
Chainlit web-chat interface for the Scholarship Assistant.

Authentication: hardcoded haseeb / 123 via @cl.password_auth_callback.
This is a temporary placeholder — swap it for a real auth provider later.
"""

import asyncio
from datetime import datetime

import chainlit as cl

from pipeline import get_pipeline


# ── Authentication ─────────────────────────────────────────────────────────────
# Hardcoded credentials for local development only.
_HARDCODED_USERS = {
    "haseeb": "123",
}


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    username = username.strip().lower()
    password = password.strip()
    if _HARDCODED_USERS.get(username) == password:
        return cl.User(
            identifier=username,
            metadata={"role": "user", "provider": "credentials"},
        )
    print(f"[auth] Login failed for '{username}'")
    return None


# ── Chat handlers ──────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """
    Runs once when a new chat session opens.
    Initialises the pipeline singleton and sends a personalised welcome message.
    """
    pipeline = get_pipeline()

    # Give this session a unique ID so history is correctly scoped per session
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pipeline.set_session(session_id)
    cl.user_session.set("session_id", session_id)

    profile      = pipeline.db.get_profile()
    stored_name  = profile.get("name", "")
    display_name = stored_name if stored_name and stored_name != "Student" else None

    greeting_line = (
        f"Welcome back, **{display_name}**! 👋"
        if display_name
        else "👋 **Welcome to the Scholarship Assistant!**"
    )

    # Remind the user if there is a leftover pending profile update
    pending      = pipeline.db.get_pending_updates()
    pending_note = ""
    if pending:
        fields       = ", ".join(f"{k}={v}" for k, v in pending.items())
        pending_note = (
            f"\n\n> ⚠️ **Pending profile update from last session:** {fields}\n"
            "> Reply **yes** to apply it or **no** to discard it."
        )

    welcome_msg = (
        f"{greeting_line}\n\n"
        "I can help you:\n"
        "- 🔍 Find scholarships matching your background\n"
        "- 📅 Check deadlines and eligibility criteria\n"
        "- 📋 Understand application requirements\n\n"
        "**Try asking:**\n"
        "- *\"Scholarships for postgraduate students in Sindh\"*\n"
        "- *\"What is the HEC Need-Based Scholarship deadline?\"*\n"
        "- *\"My name is Haseeb, I study CS in Sindh with GPA 3.2\"*\n"
        "- *\"Which scholarship suits me?\"*"
        + pending_note
    )

    await cl.Message(content=welcome_msg, author="Scholarship Assistant").send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles every user message.
    Runs process_query in a thread executor so LLM inference does not block
    the async event loop (inference can take several seconds on CPU).
    """
    pipeline   = get_pipeline()
    session_id = cl.user_session.get("session_id")
    if session_id:
        pipeline.set_session(session_id)

    # Show a "thinking" indicator while the LLM works
    async with cl.Step(name="Thinking…", show_input=False):
        loop     = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, pipeline.process_query, message.content
        )

    await cl.Message(content=response, author="Scholarship Assistant").send()