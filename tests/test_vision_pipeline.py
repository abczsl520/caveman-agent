"""Test vision/multimodal content pipeline."""
from caveman.agent.loop_engines import build_user_content


def test_build_user_content_no_attachments():
    assert build_user_content("hello") == "hello"
    assert build_user_content("hello", None) == "hello"
    assert build_user_content("hello", []) == "hello"


def test_build_user_content_non_image():
    atts = [{"url": "https://x.com/file.pdf", "content_type": "application/pdf"}]
    assert build_user_content("check this", atts) == "check this"


def test_build_user_content_with_image():
    atts = [{"url": "https://cdn.discord.com/img.png", "content_type": "image/png"}]
    result = build_user_content("what is this?", atts)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"type": "text", "text": "what is this?"}
    assert result[1]["type"] == "image_url"
    assert result[1]["image_url"]["url"] == "https://cdn.discord.com/img.png"


def test_build_user_content_mixed():
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
