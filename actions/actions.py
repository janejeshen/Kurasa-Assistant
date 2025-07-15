import os
import openai
import hashlib
from dotenv import load_dotenv
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from openai import OpenAIError

# Loading environment variables from .env file
load_dotenv()

# Getting OpenAI API key securely
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize redis_client as None by default
redis_client = None

# Only try to set up Redis if REDIS_URL is explicitly provided
if os.getenv("REDIS_URL"):
    try:
        import redis
        from redis.exceptions import RedisError
        redis_url = os.getenv("REDIS_URL")
        redis_client = redis.Redis.from_url(redis_url)
        redis_client.ping()  # Test the connection
        print("✅ Redis cache enabled")
    except (ImportError, RedisError) as e:
        print(f"[WARNING] Redis cache disabled: {e}")
        redis_client = None
else:
    print("[INFO] Redis cache disabled (REDIS_URL not set)")

class ActionAskGPT(Action):
    def name(self):
        return "action_ask_gpt"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # Getting the latest message the user sent to the bot
        user_message = tracker.latest_message.get("text")

        # Loading the Kurasa guide file (from environment variable or local fallback)
        guide_path = os.getenv(
            "KURASA_GUIDE_PATH",
            os.path.join(os.path.dirname(__file__), "kurasa_guide.txt")
        )

        try:
            # Reading the guide content
            with open(guide_path, "r", encoding="utf-8") as file:
                knowledge_base = file.read()

            answer = None
            if redis_client:
                try:
                    # Creating a unique Redis key by hashing the message + guide
                    cache_key = hashlib.sha256(
                        f"{user_message}|{knowledge_base}".encode("utf-8")
                    ).hexdigest()
                    
                    # Trying to get cached answer from Redis
                    cached_response = redis_client.get(cache_key)
                    if cached_response:
                        # If cached, decode and use it
                        answer = cached_response.decode("utf-8")
                        print("💾 Loaded from Redis cache.")
                except Exception as e:
                    print(f"[WARNING] Redis operation failed: {e}")

            if not answer:
                # If not cached or Redis unavailable, send question to OpenAI
                response = openai.ChatCompletion.create(
                    model="gpt-4",  # Using a valid model name
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

                # Extracting the GPT-generated answer
                answer = response.choices[0].message.content

                # Try to store in Redis if available
                if redis_client:
                    try:
                        redis_client.setex(cache_key, 86400, answer)
                    except Exception as e:
                        print(f"[WARNING] Redis cache set failed: {e}")

            # Sending the final answer to the user
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
