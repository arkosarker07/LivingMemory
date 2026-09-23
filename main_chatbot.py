"""
main_chatbot.py
===============
Living Memory — Gradio Chatbot UI

Ties all middleware together:
  1. User sends message
  2. read_path.chat()        → retrieve memories → LLM reply
  3. write_path.process_memory() → score and save user message
  4. write_path.process_memory() → score and save assistant reply
  5. write_counter increments
  6. Every 50 saves → run_consolidator() fires in background
  7. Memory stats panel updates after every turn

Run with:
    python main_chatbot.py
"""

import uuid
import threading
import gradio as gr
from datetime import datetime

from middleware.read_path    import chat as read_chat
from middleware.write_path   import process_memory, get_write_path_stats
from middleware.consolidator import run_consolidator, CONSOLIDATION_TRIGGER

# ── Global state ──────────────────────────────────────────────────────────────
# write_counter tracks how many memories have been saved this session.
# When it hits CONSOLIDATION_TRIGGER the consolidator fires.
write_counter  = 0
consolidation_log = []   # stores last consolidation result for display
state_lock = threading.Lock()
consolidation_running = False


# ══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def maybe_consolidate():
    """
    Checks write_counter and fires consolidator if threshold is reached.
    Runs in a background thread so it never blocks the UI response.
    """
    global write_counter, consolidation_log, consolidation_running

    with state_lock:
        if write_counter < CONSOLIDATION_TRIGGER or consolidation_running:
            return
        write_counter = 0
        consolidation_running = True
        print(f"\n[Chatbot] Write counter hit {CONSOLIDATION_TRIGGER}. "
              f"Firing consolidator in background...")

        def _run():
            global consolidation_running
            try:
                stats = run_consolidator()
                with state_lock:
                    consolidation_log.append(stats)
                    # Keep only last 5 consolidation records
                    if len(consolidation_log) > 5:
                        consolidation_log.pop(0)
            finally:
                with state_lock:
                    consolidation_running = False

        threading.Thread(target=_run, daemon=True).start()


def respond(
    user_message: str,
    chat_history: list,
    session_id: str
) -> tuple[list, str, str]:
    """
    Full pipeline for one conversation turn.

    Args:
        user_message: what the user typed
        chat_history: Gradio chat history list
        session_id:   unique ID for this browser session

    Returns:
        (updated_chat_history, stats_text, memory_log_text)
    """
    global write_counter

    if not user_message.strip():
        return chat_history, get_stats_text(), get_memory_log()

    # ── Build short-term sliding window (last 6 turns = 12 messages) ─────────
    # This gives the LLM immediate context within the session
    # without relying on the vector DB for very recent messages
    # Strip extra Gradio-specific fields (like 'metadata') to avoid API errors
    session_history = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history[-12:]]

    # ── Step 1: Read path — retrieve memories and get LLM reply ──────────────
    try:
        reply, retrieved_memories = read_chat(
            user_message   = user_message,
            session_id     = session_id,
            session_history= session_history,
        )
    except Exception as e:
        reply = f"[Error getting reply: {e}]"
        retrieved_memories = []
        reply_failed = True
    else:
        reply_failed = False

    # ── Step 2: Write path — save user message ────────────────────────────────
    try:
        user_result = process_memory(user_message, "user", session_id)
        if user_result in ("saved", "updated"):
            with state_lock:
                write_counter += 1
    except Exception as e:
        print(f"[Chatbot] Warning: failed to save user message: {e}")

    # ── Step 3: Write path — save assistant reply ─────────────────────────────
    if not reply_failed:
        try:
            asst_result = process_memory(reply, "assistant", session_id)
            if asst_result in ("saved", "updated"):
                with state_lock:
                    write_counter += 1
        except Exception as e:
            print(f"[Chatbot] Warning: failed to save assistant reply: {e}")

    # ── Step 4: Maybe consolidate ─────────────────────────────────────────────
    maybe_consolidate()

    # ── Step 5: Update chat history ───────────────────────────────────────────
    chat_history.append({"role": "user",      "content": user_message})
    chat_history.append({"role": "assistant", "content": reply})

    return chat_history, get_stats_text(), get_memory_log()


# ══════════════════════════════════════════════════════════════════════════════
# STATS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_stats_text() -> str:
    """Returns formatted memory stats for the stats panel."""
    global write_counter

    stats = get_write_path_stats()

    lines = [
        "## 🧠 Memory Stats",
        "",
        f"**Total memories:**     {stats.get('total', 0)}",
        f"**Active:**             {stats.get('active', 0)}",
        f"**Archived:**           {stats.get('archived', 0)}",
        f"**Contradictions:**     {stats.get('contradictions', 0)}",
        "",
        "---",
        f"**Write counter:**      {write_counter} / {CONSOLIDATION_TRIGGER}",
        f"**Next consolidation:** {CONSOLIDATION_TRIGGER - write_counter} saves away",
        "",
        "---",
        f"*Updated: {datetime.now().strftime('%H:%M:%S')}*"
    ]

    # Add last consolidation result if available
    if consolidation_log:
        last = consolidation_log[-1]
        lines += [
            "",
            "## 🔀 Last Consolidation",
            f"**Clusters merged:**  {last.get('clusters_merged', 0)}",
            f"**Memories deleted:** {last.get('memories_deleted', 0)}",
            f"**Memories saved:**   {last.get('memories_saved', 0)}",
            f"**Net reduction:**    -{last.get('memories_deleted', 0) - last.get('memories_saved', 0)}",
        ]
        if last.get("skipped_reason"):
            lines.append(f"**Skipped:** {last['skipped_reason']}")

    return "\n".join(lines)


def get_memory_log() -> str:
    """Returns a description of what the memory system is doing."""
    stats = get_write_path_stats()
    total = stats.get("total", 0)

    if total == 0:
        return "No memories stored yet. Start chatting!"

    lines = [
        "## 📋 Memory Activity",
        "",
        "The system is actively:",
        f"- Storing **{stats.get('active', 0)}** retrievable memories",
        f"- Keeping **{stats.get('archived', 0)}** archived (contradicted) memories",
        f"- Tracking **{stats.get('contradictions', 0)}** contradiction events",
        "",
        "**Write Path:** Scoring and filtering every message before saving.",
        "**Read Path:** Applying decay scores to retrieve only relevant memories.",
        f"**Consolidator:** Fires every {CONSOLIDATION_TRIGGER} saves to merge duplicates.",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

def build_ui():
    with gr.Blocks(
        title="Living Memory — AI Chatbot",
    ) as app:

        # ── Session ID — unique per browser tab ───────────────────────────────
        # Use a fixed session ID so memories persist across restarts and browser refreshes
        session_id = gr.State(value="default_user")

        # ── Header ─────────────────────────────────────────────────────────────
        gr.Markdown("""
# 🧠 Living Memory Chatbot
**Thesis Prototype** — Dynamic Memory Management Middleware for LLM Agents

*Memories are scored, decayed, and consolidated automatically in the background.*
        """)

        # ── Main layout: chat on left, stats on right ──────────────────────────
        with gr.Row():

            # Left column — chat
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_label=True,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Your message",
                        scale=5,
                        lines=1,
                        autofocus=True,
                    )
                    send_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1,
                    )

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    refresh_btn = gr.Button("🔄 Refresh Stats", variant="secondary")

            # Right column — stats
            with gr.Column(scale=2):
                stats_panel = gr.Markdown(
                    value=get_stats_text(),
                    elem_classes=["stats-box"],
                    label="Memory Stats",
                )

                gr.Markdown("---")

                memory_log = gr.Markdown(
                    value=get_memory_log(),
                    label="Memory Activity",
                )

        # ── How it works accordion ─────────────────────────────────────────────
        with gr.Accordion("ℹ️ How Living Memory works", open=False):
            gr.Markdown("""
**Write Path** (every message):
1. Math filter — scores novelty and semantic density
2. Collision override — if very similar to existing memory, sends to LLM
3. Single LLM call — scores importance (0-10) and detects contradictions
4. Saves if score ≥ 4, archives old memory if contradiction detected

**Read Path** (every query):
1. Fetches top-20 candidate memories by cosine similarity
2. Scores each: `U = Similarity × Importance × e^(-λ×Δt) × (1 + log(1 + access_count))`
3. Filters below utility threshold (active forgetting)
4. Injects top-5 into LLM prompt

**Consolidator** (every 50 saves):
1. Fetches all active episodic memories
2. Clusters by semantic similarity (agglomerative, threshold 0.75)
3. Merges clusters ≥ 3 into one semantic summary via LLM
4. Hard-deletes originals — reduces DB size and retrieval noise
            """)

        # ── Event handlers ─────────────────────────────────────────────────────

        def on_send(user_message, chat_history, sid):
            """Handles send button and Enter key."""
            if not user_message.strip():
                return "", chat_history, get_stats_text(), get_memory_log()

            updated_history, stats, log = respond(user_message, chat_history, sid)
            return "", updated_history, stats, log

        def on_clear():
            """Clears chat history but keeps memories in DB."""
            return [], get_stats_text(), get_memory_log()

        def on_refresh():
            """Refreshes stats panels without sending a message."""
            return get_stats_text(), get_memory_log()

        # Send on button click
        send_btn.click(
            fn=on_send,
            inputs=[msg_input, chatbot, session_id],
            outputs=[msg_input, chatbot, stats_panel, memory_log],
        )

        # Send on Enter key
        msg_input.submit(
            fn=on_send,
            inputs=[msg_input, chatbot, session_id],
            outputs=[msg_input, chatbot, stats_panel, memory_log],
        )

        # Clear chat
        clear_btn.click(
            fn=on_clear,
            outputs=[chatbot, stats_panel, memory_log],
        )

        # Refresh stats
        refresh_btn.click(
            fn=on_refresh,
            outputs=[stats_panel, memory_log],
        )

    return app


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*55)
    print("  Living Memory Chatbot — Starting...")
    print("="*55)
    print(f"  Consolidation trigger: every {CONSOLIDATION_TRIGGER} saves")
    print(f"  Database: ./data/middleware_db")
    print(f"  Opening browser at: http://localhost:7860")
    print("="*55)

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,       # set True to get a public gradio.live link
        inbrowser=True,    # auto-opens browser
        theme=gr.themes.Soft(),
        css="""
        .stats-box { font-size: 0.9em; }
        footer { display: none !important; }
        """
    )