"""Tests for link understanding module."""
import pytest

from caveman.gateway.link_understanding import (
    extract_urls,
    format_link_context,
    _clean_url,
    _is_fetchable,
    _extract_readable,
    _decode_entities,
)


class TestExtractUrls:
    def test_bare_url(self):
        urls = extract_urls("Check out https://example.com for more info")
        assert urls == ["https://example.com"]

    def test_markdown_link(self):
        urls = extract_urls("See [docs](https://docs.python.org/3/) for details")
        assert urls == ["https://docs.python.org/3/"]

    def test_multiple_urls(self):
        msg = "Visit https://a.com and https://b.com and https://c.com"
        urls = extract_urls(msg)
        assert len(urls) == 3

    def test_max_links(self):
        msg = "https://a.com https://b.com https://c.com https://d.com"
        urls = extract_urls(msg, max_links=2)
        assert len(urls) == 2

    def test_dedup(self):
        msg = "https://example.com is great, visit https://example.com again"
        urls = extract_urls(msg)
        assert urls == ["https://example.com"]

    def test_empty_message(self):
        assert extract_urls("") == []
        assert extract_urls("   ") == []
        assert extract_urls("no urls here") == []

    def test_skip_images(self):
        urls = extract_urls("Look at https://img.com/photo.jpg")
        assert urls == []

    def test_skip_blocked_hosts(self):
        assert extract_urls("http://localhost:3000/api") == []
        assert extract_urls("http://127.0.0.1/secret") == []
        assert extract_urls("http://169.254.169.254/metadata") == []

    def test_skip_private_ips(self):
        assert extract_urls("http://192.168.1.1/admin") == []
        assert extract_urls("http://10.0.0.1/internal") == []

    def test_mixed_markdown_and_bare(self):
        msg = "See [API](https://api.example.com) and also https://docs.example.com"
        urls = extract_urls(msg)
        assert len(urls) == 2
        assert "https://api.example.com" in urls
        assert "https://docs.example.com" in urls


class TestCleanUrl:
    def test_trailing_punctuation(self):
        assert _clean_url("https://example.com.") == "https://example.com"
        assert _clean_url("https://example.com,") == "https://example.com"
        assert _clean_url("https://example.com)") == "https://example.com"

    def test_trailing_quotes(self):
        assert _clean_url("'https://example.com'") == "https://example.com"
        assert _clean_url('"https://example.com"') == "https://example.com"

    def test_normal_url(self):
        assert _clean_url("https://example.com/path?q=1") == "https://example.com/path?q=1"


class TestIsFetchable:
    def test_normal_url(self):
        assert _is_fetchable("https://example.com") is True

    def test_ftp(self):
        assert _is_fetchable("ftp://files.example.com") is False

    def test_image(self):
        assert _is_fetchable("https://img.com/photo.png") is False

    def test_blocked(self):
        assert _is_fetchable("http://localhost:8080") is False

    def test_private_ip(self):
        assert _is_fetchable("http://192.168.0.1") is False


class TestExtractReadable:
    def test_simple_html(self):
        html = "<html><head><title>Test Page</title></head><body><p>Hello world</p></body></html>"
        title, text = _extract_readable(html)
        assert title == "Test Page"
        assert "Hello world" in text

    def test_strips_scripts(self):
        html = "<body><script>alert('xss')</script><p>Content</p></body>"
        _, text = _extract_readable(html)
        assert "alert" not in text
        assert "Content" in text

    def test_strips_style(self):
        html = "<body><style>.x{color:red}</style><p>Visible</p></body>"
        _, text = _extract_readable(html)
        assert "color" not in text
        assert "Visible" in text

    def test_no_title(self):
        html = "<body><p>No title here</p></body>"
        title, _ = _extract_readable(html)
        assert title == ""


class TestDecodeEntities:
    def test_common_entities(self):
        assert _decode_entities("&amp;") == "&"
        assert _decode_entities("&lt;") == "<"
        assert _decode_entities("&gt;") == ">"

    def test_numeric_entities(self):
        assert _decode_entities("&#65;") == "A"
        assert _decode_entities("&#x41;") == "A"


class TestFormatLinkContext:
    def test_empty(self):
        assert format_link_context([]) == ""

    def test_success(self):
        results = [{"url": "https://example.com", "title": "Example", "content": "Hello", "error": None}]
        ctx = format_link_context(results)
        assert "Example" in ctx
        assert "Hello" in ctx
        assert "[Link content auto-fetched" in ctx

    def test_error(self):
        results = [{"url": "https://fail.com", "error": "timeout"}]
        ctx = format_link_context(results)
        assert "timeout" in ctx

    def test_mixed(self):
        results = [
            {"url": "https://ok.com", "title": "OK", "content": "Good", "error": None},
            {"url": "https://fail.com", "error": "HTTP 404"},
        ]
        ctx = format_link_context(results)
        assert "Good" in ctx
        assert "HTTP 404" in ctx
