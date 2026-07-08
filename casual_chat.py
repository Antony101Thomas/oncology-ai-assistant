"""
casual_chat.py
---------------
Lightweight small-talk layer for ONCO AI. Sits in front of the full RAG
pipeline in rag_agent.py — if a message is just casual chat (greeting,
check-in, thanks, "remember when I asked..."), we skip retrieval/live
APIs/validation entirely and answer directly with a persona-scoped LLM call.
"""

import re
from groq import Groq

groq_client = Groq()

# Narrow on purpose: greetings, check-ins, thanks, farewells, and meta
# questions about the conversation itself. Anything else falls through
# to the normal oncology pipeline (and gets refused there if off-topic).
_CASUAL_PATTERNS = [
    r"^\s*(hi|hey|hello|yo|sup)\b",
    r"\bhow('?s| is| are)\s+(it going|you doing|things|you|your day)\b",
    r"\bwhat'?s up\b",
    r"\bgood (morning|afternoon|evening|night)\b",
    r"\bhow have you been\b",
    r"\b(thanks|thank you|thx|appreciate it)\b",
    r"\b(bye|goodbye|see you|see ya|later)\b",
    r"\bwho are you\b",
    r"\bwhat can you do\b",
    r"\b(do you remember|earlier|last time|previous(ly)?)\b.*\b(ask|said|talk|chat)\b",
    r"\bhow('?s| was) (our|your) (chat|conversation|talk)\b",
]
_CASUAL_RE = re.compile("|".join(_CASUAL_PATTERNS), re.IGNORECASE)

# If the message is long, it's probably a real question even if it opens
# with "hey", so don't treat it as pure small talk.
_MAX_CASUAL_WORDS = 12


def is_casual_message(question: str) -> bool:
    text = question.strip()
    if not text or len(text.split()) > _MAX_CASUAL_WORDS:
        return False
    return bool(_CASUAL_RE.search(text))


_CASUAL_SYSTEM_PROMPT = """You are ONCO AI, a friendly oncology assistant.
The user is making small talk right now (greeting, check-in, thanks, or
asking about your earlier conversation) — NOT asking a medical question.

Rules:
- Reply warmly and briefly (1-3 sentences), like a normal assistant would.
- You MAY reference the recent conversation history below if relevant
  (e.g. "Last time you asked about HER2-low breast cancer!").
- You can mention you're here for oncology questions when it fits, but
  don't repeat that every single time — keep it natural.
- Do NOT answer any medical question that sneaks in here — just say
  you're happy to look into it and invite them to ask properly.
- No bullet points, no citations, no clinical formatting — keep it human.
"""


def generate_casual_reply(question: str, user_name: str | None = None,
                          recent_history: list[dict] | None = None) -> str:
    history_block = ""
    if recent_history:
        lines = [f'- User asked: "{h["question"]}"' for h in recent_history[:5]]
        history_block = "Recent conversation history:\n" + "\n".join(lines)

    user_block = f"The user's name is {user_name}.\n" if user_name else ""
    prompt = f'{user_block}{history_block}\n\nUser just said: "{question}"\n\nReply casually.'

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _CASUAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )
    return response.choices[0].message.content.strip()