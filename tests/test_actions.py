import os
import asyncio
from unittest.mock import patch, MagicMock, mock_open, AsyncMock
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk import Tracker


# Helpers & fakes

def make_tracker(text: str, sender_id: str = "test_user", history=None) -> Tracker:
    return Tracker(
        sender_id=sender_id,
        slots={"conversation_history": history or []},
        latest_message={"text": text},
        events=[],
        paused=False,
        followup_action=None,
        active_loop={},
        latest_action_name=None,
    )


class DummyRedis:
    """Minimal async Redis stub with .get/.setex."""
    def __init__(self):
        self.store = {}

    async def get(self, key: str):
        val = self.store.get(key)
        # Simulate real Redis returning bytes
        if isinstance(val, str):
            return val.encode("utf-8")
        return val

    async def setex(self, key: str, ttl: int, value: str):
        # Ignore ttl in the stub
        self.store[key] = value


class ExplodingRedis:
    """Always raises to ensure Redis failures don't break action flow."""
    async def get(self, key: str):
        raise RuntimeError("boom")

    async def setex(self, key: str, ttl: int, value: str):
        raise RuntimeError("boom")


def extract_slot_names(events):
    """Return slot names (keys) regardless of SDK representation."""
    names = set()
    for e in events:
        for attr in ("key", "name", "slot_name"):
            if hasattr(e, attr):
                val = getattr(e, attr, None)
                if val:
                    names.add(val)
        if hasattr(e, "as_dict"):
            try:
                d = e.as_dict()
                for k in ("key", "name"):
                    if d.get(k):
                        names.add(d[k])
            except Exception:
                pass
        if isinstance(e, dict):
            for k in ("key", "name"):
                if e.get(k):
                    names.add(e[k])
        s = str(e)
        for token in ("last_question", "conversation_history"):
            if token in s:
                names.add(token)
    return names


# Tests

def test_paraphrase_and_cache_write():
    # Prepare environment
    with patch.dict(os.environ, {"KURASA_API_TOKEN": "test-token"}, clear=False):
        # Import after env patch so module-level constants pick up values
        from actions.actions import ActionAskGrok

        # Prepare Redis stub instance we can assert on
        redis_stub = DummyRedis()

        # Mock paraphrase API response
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "response": "You can upload documents via the Documents tab."
        }

        with patch("actions.actions.redis_client", redis_stub):
            with patch("builtins.open", mock_open(
                read_data="### How to Upload Documents\nGo to Documents tab and upload files.\n"
            )):
                with patch("actions.actions.HTTP.post_json", new=AsyncMock(return_value=mock_response)) as http_mock:
                    dispatcher = CollectingDispatcher()
                    tracker = make_tracker("How do I upload documents?")
                    action = ActionAskGrok()

                    events = asyncio.run(action.run(dispatcher, tracker, domain={}))

    # Dispatched message is paraphrased
    assert len(dispatcher.messages) == 1
    assert "documents" in dispatcher.messages[0]["text"].lower()

    # Slots set
    names = extract_slot_names(events)
    assert "last_question" in names
    assert "conversation_history" in names

    # Cache wrote something
    key = action._make_cache_key("How do I upload documents?", "test_user")
    assert redis_stub.store.get(key) == "You can upload documents via the Documents tab."

    # API was called
    http_mock.assert_awaited()


def test_cache_hit_shortcuts_api_call():
    with patch.dict(os.environ, {"KURASA_API_TOKEN": "test-token"}, clear=False):
        from actions.actions import ActionAskGrok

        redis_stub = DummyRedis()
        dispatcher = CollectingDispatcher()
        tracker = make_tracker("How do I upload documents?")
        action = ActionAskGrok()

        cache_key = action._make_cache_key("How do I upload documents?", "test_user")
        redis_stub.store[cache_key] = "CACHED: use the Documents tab."

        with patch("actions.actions.redis_client", redis_stub):
            # If HTTP is called, we fail
            with patch("actions.actions.HTTP.post_json", new=AsyncMock(side_effect=AssertionError("Should not call HTTP"))):
                events = asyncio.run(action.run(dispatcher, tracker, domain={}))

    assert len(dispatcher.messages) == 1
    assert "cached" in dispatcher.messages[0]["text"].lower()

    names = extract_slot_names(events)
    assert "last_question" in names
    assert "conversation_history" in names


def test_fallback_when_no_guide():
    with patch.dict(os.environ, {"KURASA_API_TOKEN": "test-token"}, clear=False):
        from actions.actions import ActionAskGrok

        # Empty guide => force fallback path
        with patch.object(ActionAskGrok, "_load_guide_text_async", new=AsyncMock(return_value="")):
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.json.return_value = {"response": "Fallback answer from Grok."}

            with patch("actions.actions.redis_client", None):
                with patch("actions.actions.HTTP.post_json", new=AsyncMock(return_value=mock_response)) as http_mock:
                    dispatcher = CollectingDispatcher()
                    tracker = make_tracker("This is something not in the guide")
                    action = ActionAskGrok()
                    events = asyncio.run(action.run(dispatcher, tracker, domain={}))

    assert len(dispatcher.messages) == 1
    assert "fallback answer" in dispatcher.messages[0]["text"].lower()
    http_mock.assert_awaited()

    names = extract_slot_names(events)
    assert "last_question" in names
    assert "conversation_history" in names


def test_missing_token_returns_guide_text_no_api_call():
    with patch.dict(os.environ, {"KURASA_API_TOKEN": ""}, clear=False):  # missing/empty token
        from actions.actions import ActionAskGrok

        with patch("actions.actions.redis_client", None):
            with patch("builtins.open", mock_open(
                read_data="### How to Upload Documents\nGo to Documents tab and upload files.\n"
            )):
                # If HTTP is called, we fail (no token means we shouldn't call)
                with patch("actions.actions.HTTP.post_json", new=AsyncMock(side_effect=AssertionError("Should not call HTTP"))):
                    dispatcher = CollectingDispatcher()
                    tracker = make_tracker("How do I upload documents?")
                    action = ActionAskGrok()
                    events = asyncio.run(action.run(dispatcher, tracker, domain={}))

    assert len(dispatcher.messages) == 1
    # Should be the raw guide text (not paraphrased)
    assert "go to documents tab" in dispatcher.messages[0]["text"].lower()

    names = extract_slot_names(events)
    assert "last_question" in names
    assert "conversation_history" in names


def test_empty_message_politely_asks_to_rephrase():
    with patch.dict(os.environ, {"KURASA_API_TOKEN": "test-token"}, clear=False):
        from actions.actions import ActionAskGrok

        with patch("actions.actions.redis_client", None):
            dispatcher = CollectingDispatcher()
            tracker = make_tracker("")  # empty message triggers early return
            action = ActionAskGrok()
            events = asyncio.run(action.run(dispatcher, tracker, domain={}))

    assert len(dispatcher.messages) == 1
    assert "rephrase your question" in dispatcher.messages[0]["text"].lower()
    # No slot changes expected
    assert events == []


def test_redis_failures_do_not_break_flow():
    with patch.dict(os.environ, {"KURASA_API_TOKEN": "test-token"}, clear=False):
        from actions.actions import ActionAskGrok

        with patch("actions.actions.redis_client", ExplodingRedis()):
            with patch("builtins.open", mock_open(
                read_data="### How to Upload Documents\nGo to Documents tab and upload files.\n"
            )):
                mock_response = MagicMock()
                mock_response.is_success = True
                mock_response.json.return_value = {
                    "response": "You can upload documents via the Documents tab."
                }

                with patch("actions.actions.HTTP.post_json", new=AsyncMock(return_value=mock_response)):
                    dispatcher = CollectingDispatcher()
                    tracker = make_tracker("How do I upload documents?")
                    action = ActionAskGrok()
                    events = asyncio.run(action.run(dispatcher, tracker, domain={}))

    # We should still get a response despite Redis exploding
    assert len(dispatcher.messages) == 1
    assert "documents" in dispatcher.messages[0]["text"].lower()
    names = extract_slot_names(events)
    assert "last_question" in names
    assert "conversation_history" in names
