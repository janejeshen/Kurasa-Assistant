import os
import hashlib
import difflib
import requests
from dotenv import load_dotenv
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# Load .env variables
load_dotenv()

# Optional Redis cache
redis_client = None
if os.getenv("REDIS_URL"):
    try:
        import redis
        redis_client = redis.Redis.from_url(os.getenv("REDIS_URL"))
        redis_client.ping()
        print("✅ Redis connected")
    except Exception as e:
        print(f"[Redis disabled] {e}")

class ActionAskGrok(Action):
    def name(self):
        return "action_ask_grok"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_message = tracker.latest_message.get("text", "").strip()

        guide_path = os.getenv("KURASA_GUIDE_PATH", os.path.join(os.path.dirname(__file__), "kurasa_guide.txt"))
        try:
            with open(guide_path, "r", encoding="utf-8") as f:
                guide_text = f.read()
        except FileNotFoundError:
            guide_text = ""

        last_question = tracker.get_slot("last_question")
        conversation_history = tracker.get_slot("conversation_history") or []

        # Check if response is already in Redis
        cache_key = hashlib.sha256(f"{user_message}|kurasa".encode("utf-8")).hexdigest()
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                answer = cached.decode("utf-8")
                dispatcher.utter_message(text=answer)
                history = conversation_history + [(user_message, answer)]
                return [SlotSet("last_question", user_message), SlotSet("conversation_history", history)]

        # Try to find from guide
        title, guide_response = self.get_guide_section(user_message, guide_text)
        if guide_response:
            paraphrased = self.get_paraphrased_answer(user_message, guide_response)
            if redis_client:
                redis_client.setex(cache_key, 86400, paraphrased)
            dispatcher.utter_message(text=paraphrased)
            history = conversation_history + [(user_message, paraphrased)]
            return [SlotSet("last_question", user_message), SlotSet("conversation_history", history)]

        # Fall back to AI with memory
        memory_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in conversation_history[-3:]])
        prompt = f"{memory_context}\n\nUser: {user_message}\nKurasa AI, please respond helpfully:"

        fallback_response = self.ask_grok(prompt)
        if redis_client:
            redis_client.setex(cache_key, 86400, fallback_response)

        dispatcher.utter_message(text=fallback_response)
        history = conversation_history + [(user_message, fallback_response)]
        return [SlotSet("last_question", user_message), SlotSet("conversation_history", history)]

    def get_guide_section(self, query, guide_text):
        lines = guide_text.splitlines()
        titles = [line for line in lines if line.startswith("### ")]
        match = difflib.get_close_matches(f"### {query.lower().strip()}", [t.lower() for t in titles], n=1, cutoff=0.6)
        if not match:
            return None, None

        matched_title = next((t for t in titles if t.lower() == match[0]), None)
        if not matched_title:
            return None, None

        start = lines.index(matched_title)
        content = []
        for line in lines[start + 1:]:
            if line.startswith("### "):
                break
            content.append(line)
        return matched_title.strip("# "), "\n".join(content).strip()

    def get_paraphrased_answer(self, question, guide_text):
        try:
            token = os.getenv("KURASA_API_TOKEN")
            payload = {
                "message": f"User asked: {question}\nGuide says: {guide_text}\n\nRephrase this to be accurate, friendly, and clear.",
                "ai": "grok",
                "token": token,
                "max_output_tokens": 1000
            }
            res = requests.post("https://research.kurasa.co/api/prompt-ai-with-tokens", json=payload)
            if res.status_code == 200:
                return res.json().get("response", guide_text)
        except Exception as e:
            print(f"[Paraphrasing failed] {e}")
        return guide_text

    def ask_grok(self, prompt):
        try:
            token = os.getenv("KURASA_API_TOKEN")
            payload = {
                "message": prompt,
                "ai": "grok",
                "token": token,
                "max_output_tokens": 1500
            }
            res = requests.post("https://research.kurasa.co/api/prompt-ai-with-tokens", json=payload)
            if res.status_code == 200:
                return res.json().get("response", "Sorry, I don’t have a good answer.")
        except Exception as e:
            print(f"[Fallback AI error] {e}")
        return "Sorry, I couldn’t get that info."

