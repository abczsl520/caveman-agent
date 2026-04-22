"""Test vision/multimodal content pipeline."""
from unittest.mock import patch, MagicMock
from caveman.agent.loop_engines import build_user_content, _download_image_as_data_uri


def test_build_user_content_no_attachments():
    assert build_user_content("hello") == "hello"
    assert build_user_content("hello", None) == "hello"
    assert build_user_content("hello", []) == "hello"


def test_build_user_content_non_image():
    atts = [{"url": "https://x.com/file.pdf", "content_type": "application/pdf"}]
    assert build_user_content("check this", atts) == "check this"


@patch("caveman.agent.loop_engines._download_image_as_data_uri")
def test_build_user_content_with_image_downloaded(mock_dl):
    mock_dl.return_value = "data:image/png;base64,abc123"
    atts = [{"url": "https://cdn.discord.com/img.png", "content_type": "image/png"}]
    result = build_user_content("what is this?", atts)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"type": "text", "text": "what is this?"}
    assert result[1]["type"] == "image_url"
    assert result[1]["image_url"]["url"] == "data:image/png;base64,abc123"


@patch("caveman.agent.loop_engines._download_image_as_data_uri")
def test_build_user_content_download_fails_fallback_url(mock_dl):
    mock_dl.return_value = None  # download failed
    atts = [{"url": "https://cdn.discord.com/img.png", "content_type": "image/png"}]
    result = build_user_content("what is this?", atts)
    assert isinstance(result, list)
    assert result[1]["image_url"]["url"] == "https://cdn.discord.com/img.png"


@patch("caveman.agent.loop_engines._download_image_as_data_uri")
def test_build_user_content_mixed(mock_dl):
    mock_dl.return_value = "data:image/jpeg;base64,xyz"
    atts = [
        {"url": "https://cdn.discord.com/img.png", "content_type": "image/png"},
        {"url": "https://cdn.discord.com/file.txt", "content_type": "text/plain"},
        {"url": "https://cdn.discord.com/img2.jpg", "content_type": "image/jpeg"},
    ]
    result = build_user_content("analyze", atts)
    assert isinstance(result, list)
    assert len(result) == 3  # text + 2 images (non-image skipped)
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "image_url"
    assert result[2]["type"] == "image_url"
