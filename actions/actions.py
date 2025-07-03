import os
import openai
import redis
import hashlib
from dotenv import load_dotenv
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# 🔐 Load environment variables from .env file
load_dotenv()

# 🗝️ Get OpenAI API key securely
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🔌 Connect to Redis using the REDIS_URL from .env (or fallback to local Redis)
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(redis_url)

class ActionAskGPT(Action):
    def name(self):
        return "action_ask_gpt"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # 🧠 Get the latest message the user sent to the bot
        user_message = tracker.latest_message.get("text")

        # 📄 Load the Kurasa guide file (from environment variable or local fallback)
        guide_path = os.getenv(
            "KURASA_GUIDE_PATH",
            os.path.join(os.path.dirname(__file__), "kurasa_guide.txt")
        )

        try:
            # 📖 Read the guide content
            with open(guide_path, "r", encoding="utf-8") as file:
                knowledge_base = file.read()

            # 🔑 Create a unique Redis key by hashing the message + guide
            cache_key = hashlib.sha256(
                f"{user_message}|{knowledge_base}".encode("utf-8")
            ).hexdigest()

            # 🧾 Try to get cached answer from Redis
            cached_response = redis_client.get(cache_key)

            if cached_response:
                # ✅ If cached, decode and use it
                answer = cached_response.decode("utf-8")
                print("💾 Loaded from Redis cache.")
            else:
                # 🤖 If not cached, send question to OpenAI
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a Kurasa App Assistant. Only answer based on the guide provided. "
                                "If unsure, say: 'Sorry, I don't have that information. Please contact support.'"
                            )
                        },
                        {
                            "role": "user",
                            "content": f"""
Kurasa Guide:
{knowledge_base}

User Question:
{user_message}
"""
                        }
                    ]
                )

                # 🧠 Extract the GPT-generated answer
                answer = response.choices[0].message.content

                # 🧊 Store answer in Redis for 24 hours (86400 seconds)
                redis_client.setex(cache_key, 86400, answer)

            # 🗣️ Send the final answer to the user
            dispatcher.utter_message(text=answer)

        except Exception as e:
            # ❌ If anything goes wrong, log the error and reply with a fallback message
            print(f"[ERROR] GPT or Redis failed: {e}")
            dispatcher.utter_message(
                text="Sorry, I couldn't answer that right now. Please try again later or contact support."
            )

        # No events to return (empty list)
        return []
