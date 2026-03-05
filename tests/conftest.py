import os

import pytest

from fish_audio_sdk import Session, WebSocketSession, AsyncWebSocketSession

APIKEY = os.environ.get("APIKEY", "")


def pytest_collection_modifyitems(config, items):
    """Skip all tests if APIKEY is not set."""
    if not APIKEY:
        skip_marker = pytest.mark.skip(reason="APIKEY environment variable not set")
        for item in items:
            item.add_marker(skip_marker)


@pytest.fixture
def session():
    if not APIKEY:
        pytest.skip("APIKEY environment variable not set")
    return Session(APIKEY)


@pytest.fixture
def sync_websocket():
    if not APIKEY:
        pytest.skip("APIKEY environment variable not set")
    return WebSocketSession(APIKEY)


@pytest.fixture
def async_websocket():
    if not APIKEY:
        pytest.skip("APIKEY environment variable not set")
    return AsyncWebSocketSession(APIKEY)
