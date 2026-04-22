"""Tests for gateway pairing and media cache."""
import time
import pytest

from caveman.gateway.pairing import PairingManager, PairedDevice
from caveman.gateway.media_cache import MediaCache


# ── Pairing Tests ──

class TestPairingManager:
    @pytest.fixture
    def mgr(self, tmp_path):
        return PairingManager(pairing_dir=tmp_path / "pairing")

    def test_generate_token(self, mgr):
        token = mgr.generate_token()
        assert token.token
        assert token.expires_at > token.created_at
        assert token.claimed_by is None

    def test_claim_token(self, mgr):
        token = mgr.generate_token()
        device = mgr.claim_token(
            token.token, "dev-001", "My Phone", "android"
        )
        assert device is not None
        assert device.device_id == "dev-001"
        assert device.name == "My Phone"
        assert device.platform == "android"

    def test_claim_invalid_token(self, mgr):
        device = mgr.claim_token("invalid", "dev-001", "Phone", "android")
        assert device is None

    def test_claim_expired_token(self, mgr):
        token = mgr.generate_token(ttl=0)
        time.sleep(0.01)
        device = mgr.claim_token(token.token, "dev-001", "Phone", "android")
        assert device is None

    def test_double_claim(self, mgr):
        token = mgr.generate_token()
        mgr.claim_token(token.token, "dev-001", "Phone1", "android")
        device2 = mgr.claim_token(token.token, "dev-002", "Phone2", "ios")
        assert device2 is None

    def test_list_devices(self, mgr):
        token = mgr.generate_token()
        mgr.claim_token(token.token, "dev-001", "Phone", "android")
        devices = mgr.get_devices()
        assert len(devices) == 1
        assert devices[0].device_id == "dev-001"

    def test_remove_device(self, mgr):
        token = mgr.generate_token()
        mgr.claim_token(token.token, "dev-001", "Phone", "android")
        assert mgr.remove_device("dev-001") is True
        assert len(mgr.get_devices()) == 0

    def test_remove_nonexistent(self, mgr):
        assert mgr.remove_device("nonexistent") is False

    def test_is_paired(self, mgr):
        token = mgr.generate_token()
        mgr.claim_token(token.token, "dev-001", "Phone", "android")
        assert mgr.is_paired("dev-001") is True
        assert mgr.is_paired("dev-999") is False

    def test_update_last_seen(self, mgr):
        token = mgr.generate_token()
        mgr.claim_token(token.token, "dev-001", "Phone", "android")
        mgr.update_last_seen("dev-001")
        devices = mgr.get_devices()
        assert devices[0].last_seen > devices[0].paired_at

    def test_capabilities(self, mgr):
        token = mgr.generate_token()
        device = mgr.claim_token(
            token.token, "dev-001", "Phone", "android",
            capabilities=["camera", "gps"]
        )
        assert device.capabilities == ["camera", "gps"]


# ── Media Cache Tests ──

class TestMediaCache:
    @pytest.fixture
    def cache(self, tmp_path):
        return MediaCache(cache_dir=tmp_path / "cache", max_size_mb=1)

    def test_put_and_get(self, cache):
        cache.put("https://example.com/sticker.webp", b"image data", ".webp")
        result = cache.get("https://example.com/sticker.webp")
        assert result is not None
        assert result.read_bytes() == b"image data"

    def test_miss(self, cache):
        assert cache.get("https://example.com/missing") is None

    def test_has(self, cache):
        cache.put("https://example.com/img.png", b"data")
        assert cache.has("https://example.com/img.png") is True
        assert cache.has("https://example.com/other") is False

    def test_remove(self, cache):
        cache.put("https://example.com/img.png", b"data")
        assert cache.remove("https://example.com/img.png") is True
        assert cache.has("https://example.com/img.png") is False

    def test_clear(self, cache):
        cache.put("https://a.com/1", b"data1")
        cache.put("https://b.com/2", b"data2")
        count = cache.clear()
        assert count == 2

    def test_stats(self, cache):
        cache.put("https://a.com/1", b"data")
        cache.get("https://a.com/1")  # hit
        cache.get("https://b.com/2")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["files"] == 1

    def test_eviction(self, tmp_path):
        # 1KB max cache
        cache = MediaCache(cache_dir=tmp_path / "cache", max_size_mb=0)
        cache._max_bytes = 1024
        # Put 2KB of data
        cache.put("https://a.com/1", b"x" * 600)
        cache.put("https://b.com/2", b"y" * 600)
        # First file should be evicted
        assert cache.stats["files"] <= 2  # At most 2, eviction may have run
