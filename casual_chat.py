"""
casual_chat.py
---------------
Lightweight small-talk layer for ONCO AI. Sits in front of the full RAG
pipeline in rag_agent.py — if a message is just casual chat (greeting,
check-in, thanks, "remember when I asked..."), we skip retrieval/live
APIs/validation entirely and answer directly with a persona-scoped LLM call.
"""

import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_xai_client = None

def _get_xai_client() -> OpenAI:
    global _xai_client
    if _xai_client is None:
        api_key = os.getenv("XAI_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Set XAI_API_KEY or GROQ_API_KEY environment variable")
        _xai_client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
    return _xai_client

# Narrow on purpose: greetings, check-ins, thanks, farewells, meta
# questions, jokes, compliments, feelings, and identity questions.
# Anything else falls through to the normal oncology pipeline
# (and gets refused there if off-topic).
_CASUAL_PATTERNS = [
    # Greetings
    r"^\s*(hi|hey|hello|yo|sup|hola|howdy|hii+|heyy+)\b",
    r"\bgood (morning|afternoon|evening|night)\b",
    # Check-ins
    r"\bhow('?s| is| are)\s+(it going|you doing|things|you|your day|everything|life)\b",
    r"\bwhat'?s up\b",
    r"\bhow have you been\b",
    r"\bhow('?s| is) it going\b",
    # Thanks & appreciation
    r"\b(thanks|thank you|thx|ty|appreciate it|much appreciated|thanks a lot)\b",
    # Farewells
    r"\b(bye|goodbye|see you|see ya|later|take care|good night|gn|cya|peace out)\b",
    # Identity & capability questions
    r"\bwho are you\b",
    r"\bwhat can you do\b",
    r"\bwhat('?s| is) your name\b",
    r"\bare you (a bot|an? ai|real|human)\b",
    r"\btell me about yourself\b",
    # Conversation memory
    r"\b(do you remember|earlier|last time|previous(ly)?)\b.*\b(ask|said|talk|chat)\b",
    r"\bhow('?s| was) (our|your) (chat|conversation|talk)\b",
    # Feelings & compliments
    r"\b(you('?re| are) (amazing|awesome|great|cool|helpful|smart|the best))\b",
    r"\b(i love you|love this|i like you|you rock|nice job|well done)\b",
    r"\b(i('?m| am) (bored|sad|happy|tired|excited|confused|lonely|scared))\b",
    r"\bhow do you feel\b",
    # Fun & jokes
    r"\btell me a (joke|fun fact|story)\b",
    r"\b(make me laugh|say something funny|cheer me up)\b",
    r"\b(lol|lmao|haha|rofl|😂|🤣)\b",
    # Simple acknowledgements
    r"^\s*(ok|okay|cool|nice|great|awesome|sure|alright|got it|noted|yep|yup|yeah)\s*[!.?]*\s*$",
    # Help / what to ask
    r"\bwhat (should|can) i ask\b",
    r"\bhelp me\b",
    r"\bwhat do you know\b",
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

    response = _get_xai_client().chat.completions.create(
        model="grok-3-mini-fast",
        messages=[
            {"role": "system", "content": _CASUAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )
    return response.choices[0].message.content.strip()