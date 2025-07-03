import pytest
from unittest.mock import patch, MagicMock
from actions.actions import ActionAskGPT
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.interfaces import Tracker

@pytest.fixture
def tracker():
    return Tracker(
        sender_id="test_user",
        slots={},
        latest_message={"text": "How do I upload documents?"},
        events=[],
        paused=False,
        followup_action=None,
        active_loop={},
        latest_action_name=None
    )

@pytest.fixture
def dispatcher():
    return CollectingDispatcher()

@patch("actions.actions.openai.ChatCompletion.create")
@patch("actions.actions.redis_client")
def test_action_ask_gpt(mock_redis, mock_openai, tracker, dispatcher):
    # Simulate Redis miss
    mock_redis.get.return_value = None

    # Simulate OpenAI response
    mock_openai.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="You can upload documents via the Documents tab."))]
    )

    action = ActionAskGPT()
    events = action.run(dispatcher, tracker, domain={})

    # Check dispatcher output
    assert len(dispatcher.messages) == 1
    assert "upload documents" in dispatcher.messages[0]["text"]
