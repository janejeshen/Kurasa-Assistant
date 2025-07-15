import pytest
from unittest.mock import patch, MagicMock, mock_open
from actions.actions import ActionAskGPT
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.interfaces import Tracker
import builtins

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

@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("actions.actions.openai.ChatCompletion.create")
@patch("actions.actions.redis_client", None)  # Disable Redis for this test
@patch("builtins.open", new_callable=mock_open, read_data="### How to Upload Documents\nGo to Documents tab and upload files.\n")
@patch("os.path.join", return_value="dummy_path/kurasa_guide.txt")
def test_action_ask_gpt(
    mock_path_join,
    mock_file_open,
    mock_redis,
    mock_openai,
    tracker,
    dispatcher
):
    # Simulate OpenAI response
    mock_openai.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="You can upload documents via the Documents tab."))]
    )

    action = ActionAskGPT()
    events = action.run(dispatcher, tracker, domain={})

    # Check that GPT was called
    mock_openai.assert_called_once()

    # Check dispatcher output
    assert len(dispatcher.messages) == 1
    assert "upload documents" in dispatcher.messages[0]["text"].lower()
