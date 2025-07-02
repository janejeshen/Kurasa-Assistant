import openai
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

class ActionAskGPT(Action):
    def name(self) -> str:
        return "action_ask_gpt"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:

        user_message = tracker.latest_message.get('text')

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Kurasa app support. Answer clearly and concisely."},
                {"role": "user", "content": user_message}
            ]
        )

        reply = response.choices[0].message.content
        dispatcher.utter_message(text=reply)
        return []