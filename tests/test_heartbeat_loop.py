"""Test that tool heartbeat loops and calls touch_activity.

Regression test for the bug where _heartbeat ran only once (sleep 15s,
send message, exit), causing idle_shutdown to misfire during long tool
execution (e.g. coding_agent 15min > idle_shutdown 5min).
"""
import unittest


class TestHeartbeatLoop(unittest.TestCase):

    def test_heartbeat_is_loop_not_oneshot(self):
        """_heartbeat must contain 'while True' and 'touch_activity'."""
        src = open("caveman/gateway/task_runner_helpers.py").read()
        start = src.find("async def _heartbeat(name: str):")
        assert start != -1, "_heartbeat function not found"
        end = src.find("\n    ctx.tool_heartbeat", start)
        body = src[start:end]
        assert "while True:" in body, \
            "_heartbeat must be a loop (while True), not one-shot"
        assert "ctx.touch_activity()" in body, \
            "_heartbeat must call ctx.touch_activity() to prevent idle misfire"

    def test_idle_shutdown_config_value(self):
        """Verify idle_shutdown in default.yaml matches our expectation (300s)."""
        import yaml
        with open("caveman/config/default.yaml") as f:
            cfg = yaml.safe_load(f)
        idle = cfg["gateway"]["timeouts"]["idle_shutdown"]
        assert idle == 300, f"idle_shutdown should be 300s, got {idle}"

    def test_heartbeat_interval_shorter_than_idle(self):
        """Heartbeat interval (15s) must be much shorter than idle_shutdown (300s)."""
        src = open("caveman/gateway/task_runner_helpers.py").read()
        import re
        match = re.search(r'_heartbeat.*?await asyncio\.sleep\((\d+\.?\d*)\)', src, re.DOTALL)
        assert match, "Could not find sleep interval in _heartbeat"
        interval = float(match.group(1))
        assert interval <= 30, f"Heartbeat interval {interval}s too long, must be <= 30s"

        import yaml
        with open("caveman/config/default.yaml") as f:
            cfg = yaml.safe_load(f)
        idle = cfg["gateway"]["timeouts"]["idle_shutdown"]
        assert interval < idle / 5, \
            f"Heartbeat interval ({interval}s) must be < idle_shutdown/5 ({idle/5}s)"


if __name__ == "__main__":
    unittest.main()
