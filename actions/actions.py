import os
import openai
import hashlib
import difflib
from dotenv import load_dotenv
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from openai import OpenAIError

# Load .env
load_dotenv()

# Securely get OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize Redis client if available
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


class ActionAskGPT(Action):
    def name(self):
        return "action_ask_gpt"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_message = tracker.latest_message.get("text")

        guide_path = os.getenv(
            "KURASA_GUIDE_PATH",
            os.path.join(os.path.dirname(__file__), "kurasa_guide.txt")
        )

        try:
            with open(guide_path, "r", encoding="utf-8") as file:
                guide_content = file.read()

            # Try to find a matching answer from the guide first
            answer = self.find_in_guide(user_message, guide_content)

            if answer:
                print("📘 Answer found in Kurasa guide.")
                dispatcher.utter_message(text=answer)
                return []

            # Use hash for cache key
            cache_key = hashlib.sha256(
                f"{user_message}|{guide_content}".encode("utf-8")
            ).hexdigest()

            # Try Redis cache
            if redis_client:
                cached_response = redis_client.get(cache_key)
                if cached_response:
                    print("💾 Loaded from Redis cache.")
                    dispatcher.utter_message(text=cached_response.decode("utf-8"))
                    return []

            # No match, fallback to GPT
            print("⚠️ Not found in guide. Falling back to GPT.")
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Kurasa App Assistant. ONLY answer based on the guide provided. "
                            "If you cannot find the answer, reply: 'Sorry, I don't have that information. Please contact support.'"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"""
Kurasa Guide:
{guide_content}

User Question:
{user_message}
"""
                    }
                ]
            )
            answer = response.choices[0].message.content

            # Save in Redis
            if redis_client:
                try:
                    redis_client.setex(cache_key, 86400, answer)
                except Exception as e:
                    print(f"[WARNING] Redis cache set failed: {e}")

            dispatcher.utter_message(text=answer)

        except FileNotFoundError:
            print(f"[ERROR] Guide file not found at: {guide_path}")
            dispatcher.utter_message(
                text="Sorry, I couldn't access the knowledge base. Please contact support."
            )
        except OpenAIError as e:
            print(f"[ERROR] OpenAI API error: {e}")
            dispatcher.utter_message(
                text="Sorry, I'm having trouble processing your request. Please try again later."
            )
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            dispatcher.utter_message(
                text="Sorry, I couldn't answer that right now. Please try again later or contact support."
            )

        return []

    def find_in_guide(self, query, guide_text):
        """
        Tries to find the most relevant answer in the guide based on fuzzy title match.
        """
        lines = guide_text.splitlines()
        titles = [line.strip() for line in lines if line.strip().startswith("### ")]

        matches = difflib.get_close_matches(f"### {query.strip().lower()}", [t.lower() for t in titles], n=1, cutoff=0.5)
        if not matches:
            return None

        matched_title = [t for t in titles if t.lower() == matches[0]][0]
        start_index = lines.index(matched_title)
        answer_lines = []

        for line in lines[start_index + 1:]:
            if line.strip().startswith("### "):
                break
            answer_lines.append(line)

        return "\n".join([matched_title] + answer_lines).strip()
