import os
import openai
import hashlib
import difflib
from dotenv import load_dotenv
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from openai import OpenAIError
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()

# Set up OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Set up embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Optional Redis caching
redis_client = None
if os.getenv("REDIS_URL"):
    try:
        import redis
        from redis.exceptions import RedisError
        redis_url = os.getenv("REDIS_URL")
        redis_client = redis.Redis.from_url(redis_url)
        redis_client.ping()
        print("✅ Redis cache enabled")
    except (ImportError, RedisError) as e:
        print(f"[WARNING] Redis cache disabled: {e}")
else:
    print("[INFO] Redis cache disabled (REDIS_URL not set)")

# Vague keyword clarifier
VAGUE_KEYWORD_MAP = {
    "schemes": [
        "How to Check for Schemes",
        "How to Prepare a Scheme of Work",
        "How to Download a Scheme",
        "How to Share a Scheme",
        "How to Receive a Shared Scheme",
        "How to Clone a Scheme",
        "How to Delete a Lesson Plan or Scheme"
    ],
    "lesson": [
        "How to Evaluate Lesson Plans",
        "How to Create a Lesson Plan",
        "How to View Lesson Plans",
        "How to Edit Lesson Plan Date/Time",
        "How to Add Remarks for Past Lessons",
        "How to Delete a Lesson Plan or Scheme",
        "How to Download a Lesson Plan",
        "How to Share a Lesson Plan",
        "How to Receive Shared Lesson Plan"
    ],
    "attendance": [
        "How to Check Day's Attendance",
        "How to Mark the Attendance Register"
    ],
    "marks": [
        "How to Mark the Attendance Register",
        "How to Upload Summative Scores",
        "How to Check the Summative Marklist"
    ]
    # Add more if needed
}

class ActionAskGPT(Action):
    def name(self):
        return "action_ask_gpt"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_message = tracker.latest_message.get("text", "").strip()

        guide_path = os.getenv(
            "KURASA_GUIDE_PATH",
            os.path.join(os.path.dirname(__file__), "kurasa_guide.txt")
        )

        try:
            with open(guide_path, "r", encoding="utf-8") as file:
                guide_text = file.read()

            # Store and retrieve previous vague interaction
            last_topic = tracker.get_slot("last_topic")
            if last_topic and last_topic in VAGUE_KEYWORD_MAP:
                options = VAGUE_KEYWORD_MAP[last_topic]
                matches = difflib.get_close_matches(user_message.strip().lower(), [opt.lower() for opt in options], n=1, cutoff=0.6)
                if matches:
                    user_message = matches[0]
                    dispatcher.utter_message(text=f"Thanks! Here's what I found for: {user_message}")
                else:
                    dispatcher.utter_message(text="Thanks for clarifying! Let me check that for you.")

            # 1. Vague Keyword Handling
            for vague, options in VAGUE_KEYWORD_MAP.items():
                if vague in user_message.lower():
                    choices = "\n".join(f"- {opt}" for opt in options)
                    dispatcher.utter_message(
                        text=f"Just to clarify, were you asking about one of the following regarding '{vague}'?\n{choices}\nLet me know which one you'd like help with."
                    )
                    return [
                        {"event": "slot", "name": "last_topic", "value": vague}
                    ]

            # 2. Try direct fuzzy match from guide
            answer = self.extract_answer_from_guide(user_message, guide_text)
            if answer:
                dispatcher.utter_message(text=answer)
                return []

            # 3. Check Redis cache
            cache_key = hashlib.sha256(f"{user_message}|{guide_text}".encode("utf-8")).hexdigest()
            if redis_client:
                cached = redis_client.get(cache_key)
                if cached:
                    dispatcher.utter_message(text=cached.decode("utf-8"))
                    return []

            # 4. Fallback to GPT
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Kurasa Bot, a friendly and knowledgeable assistant that helps users navigate and use the Kurasa app. "
                            "Always speak in a warm, clear, and helpful tone. "
                            "If the question is out of scope, say: 'Sorry, I don't have that information. Please contact support.'"
                            "Try to sound natural and supportive — like a helpful human assistant would."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"""
Kurasa Guide:
{guide_text}

User Question:
{user_message}
"""
                    }
                ]
            )
            gpt_reply = response.choices[0].message.content.strip()

            if redis_client:
                try:
                    redis_client.setex(cache_key, 86400, gpt_reply)
                except Exception as e:
                    print(f"[WARNING] Redis cache write failed: {e}")

            dispatcher.utter_message(text=gpt_reply)

        except FileNotFoundError:
            dispatcher.utter_message(text="Sorry, I couldn't access the knowledge base.")
        except OpenAIError as e:
            dispatcher.utter_message(text="I'm having trouble processing that right now.")
        except Exception as e:
            dispatcher.utter_message(text="Oops! Something went wrong. Please try again shortly.")

        return []

    def extract_answer_from_guide(self, query, guide_text):
        lines = guide_text.splitlines()
        titles = [line.strip() for line in lines if line.strip().startswith("### ")]

        matches = difflib.get_close_matches(f"### {query.strip().lower()}", [t.lower() for t in titles], n=1, cutoff=0.6)
        if not matches:
            return None

        matched_title = [t for t in titles if t.lower() == matches[0]][0]
        start_index = lines.index(matched_title)
        answer_lines = []

        for line in lines[start_index + 1:]:
            if line.strip().startswith("### "):
                break
            answer_lines.append(line)

        return f"Here's what I found for you:\n{matched_title}\n{''.join(answer_lines).strip()}"
