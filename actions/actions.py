# actions/actions.py
import os
import re
import math
import hashlib
import difflib
import logging
import asyncio
from typing import List, Optional, Tuple, Dict

import httpx
from dotenv import load_dotenv
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# Setup
load_dotenv()
logger = logging.getLogger(__name__)


# Env / constants
KURASA_API_URL = os.getenv("KURASA_API_URL", "https://research.kurasa.co/api/prompt-ai-with-tokens")
KURASA_API_TOKEN = os.getenv("KURASA_API_TOKEN")

GUIDE_PATH = os.getenv(
    "KURASA_GUIDE_PATH",
    os.path.join(os.path.dirname(__file__), "kurasa_guide.txt"),
)

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
CACHE_PREFIX = os.getenv("CACHE_PREFIX", "kurasa:v1")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "15"))

# Guardrails
MAX_USER_CHARS = int(os.getenv("MAX_USER_CHARS", "2000"))          # truncate user msg
MAX_MEMORY_CHARS = int(os.getenv("MAX_MEMORY_CHARS", "2000"))      # truncate memory ctx
MAX_GUIDE_SECTION_CHARS = int(os.getenv("MAX_GUIDE_SECTION_CHARS", "4000"))

# Humanization / matching thresholds
MIN_GUIDE_SCORE = float(os.getenv("MIN_GUIDE_SCORE", "0.18"))  # higher => stricter guide match
HUMANIZE_STYLE = os.getenv("HUMANIZE_STYLE", "numbered")       # "numbered" or "sentences"

# Redis (async) – initialized lazily; pinged on first use
redis_client = None
_redis_ready_checked = False

try:
    from redis import asyncio as aioredis  # type: ignore

    if os.getenv("REDIS_URL"):
        redis_client = aioredis.from_url(os.getenv("REDIS_URL"))  # type: ignore
except Exception as e:
    logger.info("Redis not configured/available: %s", e)
    redis_client = None


async def _ensure_redis_ready():
    """Ping Redis at first use; on failure disable caching silently."""
    global _redis_ready_checked, redis_client
    if _redis_ready_checked or not redis_client:
        return
    try:
        # In action server this runs inside an event loop
        await redis_client.ping()  # type: ignore
        _redis_ready_checked = True
        logger.info("✅ Redis (async) connected")
    except Exception as e:
        logger.info("⚠️  Redis disabled: %s", e)
        redis_client = None
        _redis_ready_checked = True


# HTTP client (async) – tiny retry/backoff
class HttpClient:
    def __init__(self, timeout: float, max_retries: int = 3, backoff: float = 0.5):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._client = httpx.AsyncClient(timeout=timeout)

    async def post_json(self, url: str, json: dict) -> httpx.Response:
        attempt = 0
        while True:
            try:
                resp = await self._client.post(url, json=json)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    attempt += 1
                    await asyncio.sleep(self.backoff * attempt)
                    continue
                return resp
            except (httpx.TransportError, httpx.HTTPError):
                if attempt >= self.max_retries:
                    raise
                attempt += 1
                await asyncio.sleep(self.backoff * attempt)

    async def aclose(self):
        await self._client.aclose()


HTTP = HttpClient(timeout=HTTP_TIMEOUT)

# Tiny semantic search (TF-IDF over sections) + fuzzy title fallback
_token_re = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _token_re.findall(text)]


def _tf(tokens: List[str]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = float(len(tokens) or 1)
    return {t: c / total for t, c in counts.items()}


def _idf(docs_tokens: List[List[str]]) -> Dict[str, float]:
    df: Dict[str, int] = {}
    for toks in docs_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    N = float(len(docs_tokens) or 1)
    return {t: math.log((N + 1.0) / (df_t + 1.0)) + 1.0 for t, df_t in df.items()}


def _tfidf_vec(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = _tf(tokens)
    return {t: tf.get(t, 0.0) * idf.get(t, 0.0) for t in tf}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    dot = 0.0
    for t, va in a.items():
        vb = b.get(t)
        if vb:
            dot += va * vb
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (na * nb)


def _split_guide_into_sections(guide_text: str) -> List[Dict[str, str]]:
    """Split on lines that start with '### ' into sections: [{title, content}]."""
    if not guide_text.strip():
        return []
    lines = guide_text.splitlines()
    sections = []
    current_title = None
    current_content: List[str] = []

    for line in lines:
        if line.startswith("### "):
            if current_title is not None:
                sections.append({"title": current_title, "content": "\n".join(current_content).strip()})
            current_title = line.lstrip("# ").strip()
            current_content = []
        else:
            current_content.append(line)

    if current_title is not None:
        sections.append({"title": current_title, "content": "\n".join(current_content).strip()})

    return sections


def _semantic_search(query: str, guide_text: str) -> Tuple[Optional[str], Optional[str], float]:
    """Return (title, content, score) of best-matching section using TF-IDF + cosine."""
    sections = _split_guide_into_sections(guide_text)
    if not sections:
        return None, None, 0.0

    docs_tokens = []
    for s in sections:
        combined = f"{s['title']}\n{s['content']}"
        toks = _tokenize(combined)
        docs_tokens.append(toks)

    idf = _idf(docs_tokens)
    section_vecs = [_tfidf_vec(toks, idf) for toks in docs_tokens]

    q_tokens = _tokenize(query)
    q_vec = _tfidf_vec(q_tokens, idf)

    best_idx, best_score = -1, 0.0
    for i, vec in enumerate(section_vecs):
        score = _cosine(q_vec, vec)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx >= 0 and best_score >= MIN_GUIDE_SCORE:
        title = sections[best_idx]["title"]
        content = sections[best_idx]["content"][:MAX_GUIDE_SECTION_CHARS]
        return title, content, best_score

    # Fallbacks (weaker score so we can decide to prefer AI)
    titles_raw = [f"### {s['title']}" for s in sections]
    match = difflib.get_close_matches(f"### {query.lower().strip()}", [t.lower() for t in titles_raw], n=1, cutoff=0.6)
    if match:
        matched_title = next((t for t in titles_raw if t.lower() == match[0]), None)
        if matched_title:
            idx = titles_raw.index(matched_title)
            return sections[idx]["title"], sections[idx]["content"][:MAX_GUIDE_SECTION_CHARS], 0.12

    for i, s in enumerate(sections):
        if query.lower() in s["title"].lower():
            return s["title"], s["content"][:MAX_GUIDE_SECTION_CHARS], 0.12

    return None, None, 0.0


# Humanizer – turn "Menu > Schemes > New" into friendly steps; remove "*"

_step_sep = re.compile(r"\s*>\s*")


def humanize(text: str, style: str = HUMANIZE_STYLE) -> str:
    """Make terse guide lines friendly for chat: remove '*' and turn 'A > B > C' into steps."""
    if not text:
        return text
    txt = text.replace("*", "").strip()

    # Breadcrumb → steps
    if ">" in txt:
        parts = [p.strip(" .") for p in _step_sep.split(txt) if p.strip()]
        if len(parts) >= 2:
            if style == "numbered":
                lines = [f"{i}. {p}." for i, p in enumerate(parts, 1)]
                return "Here’s how:\n" + "\n".join(lines)
            else:
                if len(parts) == 2:
                    return f"Go to {parts[0]}, then {parts[1]}."
                return "Start at " + ", then ".join(parts[:-1]) + f", and finally {parts[-1]}."
    # Ensure natural end
    if not txt.endswith((".", "!", "?")):
        txt += "."
    return txt


# Action
class ActionAskGrok(Action):
    def name(self) -> str:
        return "action_ask_grok"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_message: str = (tracker.latest_message.get("text") or "").strip()
        sender_id: str = getattr(tracker, "sender_id", "anonymous")

        if not user_message:
            dispatcher.utter_message(text="Could you please rephrase your question?")
            return []

        # Guardrail: truncate overly long user input
        if len(user_message) > MAX_USER_CHARS:
            user_message = user_message[:MAX_USER_CHARS]

        # Load guide (async file I/O)
        guide_text = await self._load_guide_text_async(GUIDE_PATH)

        # Short memory
        conversation_history: List[Tuple[str, str]] = tracker.get_slot("conversation_history") or []
        memory_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in conversation_history[-3:]])
        if len(memory_context) > MAX_MEMORY_CHARS:
            memory_context = memory_context[-MAX_MEMORY_CHARS:]

        # Cache key includes sender to avoid cross-user collisions
        cache_key = self._make_cache_key(user_message, sender_id)

        # Ensure Redis ready (if configured)
        await _ensure_redis_ready()

        # Try cache
        answer = await self._redis_get(cache_key)
        if answer:
            dispatcher.utter_message(text=answer)
            history = conversation_history + [(user_message, answer)]
            return [SlotSet("last_question", user_message), SlotSet("conversation_history", history)]

        # Prefer AI for broad/WH questions
        q_lower = user_message.lower()
        is_wh = bool(re.match(r"^\s*(who|what|why|when|where|which|whom|whose)\b", q_lower))
        is_broad = "use kurasa" in q_lower or "what can i do" in q_lower or "how do i use" in q_lower

        title = guide_response = None
        score = 0.0
        if not (is_wh or is_broad):
            title, guide_response, score = _semantic_search(user_message, guide_text)

        if guide_response:
            paraphrased = await self._get_paraphrased_answer(user_message, guide_response)
            final_text = humanize(paraphrased)
            await self._redis_set(cache_key, final_text, CACHE_TTL_SECONDS)
            dispatcher.utter_message(text=final_text)
            history = conversation_history + [(user_message, final_text)]
            return [SlotSet("last_question", user_message), SlotSet("conversation_history", history)]

        # Fallback to AI with short memory
        prompt = f"{memory_context}\n\nUser: {user_message}\nKurasa AI, please respond helpfully:".lstrip()
        fallback_response = await self._ask_grok(prompt)
        final_text = humanize(fallback_response)
        await self._redis_set(cache_key, final_text, CACHE_TTL_SECONDS)

        dispatcher.utter_message(text=final_text)
        history = conversation_history + [(user_message, final_text)]
        return [SlotSet("last_question", user_message), SlotSet("conversation_history", history)]

    # Helpers
    async def _load_guide_text_async(self, path: str) -> str:
        loop = asyncio.get_running_loop()

        def _read():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                logger.warning("Guide file not found at %s", path)
                return ""
            except Exception as e:
                logger.error("Error reading guide file: %s", e)
                return ""

        return await loop.run_in_executor(None, _read)

    def _make_cache_key(self, user_message: str, sender_id: str) -> str:
        raw = f"{CACHE_PREFIX}|{sender_id}|{user_message}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    async def _redis_get(self, key: str) -> Optional[str]:
        if not redis_client:
            return None
        try:
            val = await redis_client.get(key)  # type: ignore
            return val.decode("utf-8") if val else None
        except Exception as e:
            logger.debug("Redis get failed: %s", e)
            return None

    async def _redis_set(self, key: str, value: str, ttl: int) -> None:
        if not redis_client:
            return
        try:
            await redis_client.setex(key, ttl, value)  # type: ignore
        except Exception as e:
            logger.debug("Redis set failed: %s", e)

    async def _get_paraphrased_answer(self, question: str, guide_text: str) -> str:
        """Paraphrase guide text into a warm tone; avoid arrows/asterisks."""
        if not KURASA_API_TOKEN:
            return humanize(guide_text)

        payload = {
            "message": (
                f"User asked: {question}\n"
                f"Guide says: {guide_text}\n\n"
                "Rewrite for a teacher in a warm, natural tone. "
                "If there is a sequence of clicks, format as short numbered steps on new lines. "
                "Avoid using '>' arrows and avoid asterisks. Keep it concise."
            ),
            "ai": "grok",
            "token": KURASA_API_TOKEN,
            "max_output_tokens": 500,
        }

        try:
            res = await HTTP.post_json(KURASA_API_URL, json=payload)
            if res.is_success:
                data = res.json()
                # Humanize once more to catch any symbols
                return humanize(data.get("response") or guide_text)
            logger.warning("Paraphrase non-2xx: %s %s", res.status_code, res.text[:200])
        except Exception as e:
            logger.debug("[Paraphrasing failed] %s", e)
        return humanize(guide_text)

    async def _ask_grok(self, prompt: str) -> str:
        """Fallback AI call."""
        if not KURASA_API_TOKEN:
            logger.warning("KURASA_API_TOKEN is missing; returning default fallback.")
            return "Sorry, I don’t have a good answer."

        payload = {
            "message": prompt,
            "ai": "grok",
            "token": KURASA_API_TOKEN,
            "max_output_tokens": 1500,
        }

        try:
            res = await HTTP.post_json(KURASA_API_URL, json=payload)
            if res.is_success:
                data = res.json()
                return data.get("response", "Sorry, I don’t have a good answer.")
            logger.warning("Fallback AI non-2xx: %s %s", res.status_code, res.text[:200])
        except Exception as e:
            logger.debug("[Fallback AI error] %s", e)
        return "Sorry, I couldn’t get that info."
