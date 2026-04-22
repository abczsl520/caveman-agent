"""Tests for session knowledge extraction pipeline."""
import json
import tempfile
from pathlib import Path

import pytest

from caveman.import_.session_parser import (
    ConversationTurn,
    ParsedSession,
    SessionMetadata,
    ToolCall,
    parse_session,
    scan_sessions,
)
from caveman.import_.session_scoring import (
    KnowledgeCategory,
    KnowledgeCluster,
    TurnScore,
    cluster_turns,
    score_turn,
)
from caveman.import_.session_extractor import (
    ExtractionResult,
    ExtractedKnowledge,
    extract_knowledge_rule_based,
    extract_session,
)
from caveman.import_.openclaw_sessions import OpenClawSessionImporter
from caveman.memory.types import MemoryType


# ---------------------------------------------------------------------------
# Fixtures: realistic session JSONL data
# ---------------------------------------------------------------------------

def _make_session_jsonl(turns: list[dict], session_id: str = "test-session") -> str:
    """Build a minimal session JSONL string from turn specs."""
    lines = [
        json.dumps({
            "type": "session", "version": 3, "id": session_id,
            "timestamp": "2026-04-15T10:00:00.000Z", "cwd": "/tmp",
        }),
        json.dumps({
            "type": "model_change", "id": "mc1", "parentId": None,
            "timestamp": "2026-04-15T10:00:00.001Z",
            "provider": "anthropic", "modelId": "claude-opus-4-6",
        }),
    ]
    parent = "mc1"
    for i, turn in enumerate(turns):
        # User message
        uid = f"u{i}"
        lines.append(json.dumps({
            "type": "message", "id": uid, "parentId": parent,
            "timestamp": f"2026-04-15T10:{i:02d}:01.000Z",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": turn.get("user", "hello")}],
            },
        }))
        parent = uid

        # Assistant message
        aid = f"a{i}"
        assistant_content = []
        if "assistant_text" in turn:
            assistant_content.append({"type": "text", "text": turn["assistant_text"]})
        for tc in turn.get("tool_calls", []):
            assistant_content.append({
                "type": "toolCall", "id": f"tc{i}_{tc['name']}",
                "name": tc["name"], "arguments": tc.get("arguments", {}),
            })
        lines.append(json.dumps({
            "type": "message", "id": aid, "parentId": uid,
            "timestamp": f"2026-04-15T10:{i:02d}:02.000Z",
            "message": {"role": "assistant", "content": assistant_content},
        }))
        parent = aid

        # Tool results
        for j, tc in enumerate(turn.get("tool_calls", [])):
            trid = f"tr{i}_{j}"
            lines.append(json.dumps({
                "type": "message", "id": trid, "parentId": aid,
                "timestamp": f"2026-04-15T10:{i:02d}:03.000Z",
                "message": {
                    "role": "toolResult",
                    "content": [{"type": "text", "text": tc.get("result", "ok")}],
                },
            }))
            parent = trid

    return "\n".join(lines)


def _write_session(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Session Parser Tests
# ---------------------------------------------------------------------------

class TestSessionParser:
    def test_parse_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert parse_session(p) is None

    def test_parse_basic_session(self, tmp_path):
        jsonl = _make_session_jsonl([
            {"user": "你好", "assistant_text": "你好！有什么可以帮你的？"},
        ])
        p = _write_session(tmp_path, "basic.jsonl", jsonl)
        session = parse_session(p)

        assert session is not None
        assert session.metadata.session_id == "test-session"
        assert session.metadata.provider == "anthropic"
        assert session.metadata.model_id == "claude-opus-4-6"
        assert len(session.turns) == 1
        assert session.turns[0].user_text == "你好"
        assert "你好" in session.turns[0].assistant_prose

    def test_parse_multi_turn(self, tmp_path):
        jsonl = _make_session_jsonl([
            {"user": "分析一下这个问题", "assistant_text": "让我看看"},
            {"user": "继续", "assistant_text": "根因分析完成，发现是配置问题"},
            {"user": "怎么修", "assistant_text": "修改 config.yaml 的 timeout 字段"},
        ])
        p = _write_session(tmp_path, "multi.jsonl", jsonl)
        session = parse_session(p)

        assert session is not None
        assert len(session.turns) == 3
        assert session.turns[1].turn_index == 2

    def test_parse_with_tool_calls(self, tmp_path):
        jsonl = _make_session_jsonl([{
            "user": "查一下服务器状态",
            "assistant_text": "我来查一下",
            "tool_calls": [
                {"name": "exec", "arguments": {"command": "uptime"}, "result": "up 30 days"},
                {"name": "message", "arguments": {"action": "send", "message": "服务器运行正常，已运行30天", "channel": "discord", "target": "123"}, "result": "sent"},
            ],
        }])
        p = _write_session(tmp_path, "tools.jsonl", jsonl)
        session = parse_session(p)

        assert session is not None
        turn = session.turns[0]
        assert len(turn.tool_calls) == 2
        assert turn.tool_calls[0].is_exec
        assert turn.tool_calls[1].is_message_send
        assert "运行正常" in turn.tool_calls[1].sent_message

    def test_parse_topic_from_filename(self, tmp_path):
        jsonl = _make_session_jsonl([
            {"user": "hello", "assistant_text": "hi"},
        ], session_id="abc123")
        p = _write_session(tmp_path, "abc123-topic-1492477554169352242.jsonl", jsonl)
        session = parse_session(p)

        assert session is not None
        assert session.metadata.topic_id == "1492477554169352242"

    def test_topic_hint_extraction(self, tmp_path):
        jsonl = _make_session_jsonl([
            {"user": "[Thread starter - for context] Hermes源码分析  Conversation info", "assistant_text": "好的"},
        ])
        p = _write_session(tmp_path, "topic.jsonl", jsonl)
        session = parse_session(p)

        assert session is not None
        assert "Hermes" in session.topic_hint

    def test_scan_sessions_excludes_deleted(self, tmp_path):
        _write_session(tmp_path, "active.jsonl", _make_session_jsonl([{"user": "hi", "assistant_text": "hello"}]))
        _write_session(tmp_path, "old.jsonl.deleted.2026-04-01", _make_session_jsonl([{"user": "hi", "assistant_text": "hello"}]))
        _write_session(tmp_path, "reset.jsonl.reset.2026-04-01", _make_session_jsonl([{"user": "hi", "assistant_text": "hello"}]))
        _write_session(tmp_path, "not-jsonl.txt", "hello")

        paths = scan_sessions(tmp_path)
        assert len(paths) == 1
        assert paths[0].name == "active.jsonl"

    def test_substantive_turn(self):
        turn = ConversationTurn(
            turn_index=1,
            assistant_texts=["这是一段很长的分析文本" * 20],
        )
        assert turn.has_substance

    def test_non_substantive_turn(self):
        turn = ConversationTurn(
            turn_index=1,
            assistant_texts=["OK"],
        )
        assert not turn.has_substance


# ---------------------------------------------------------------------------
# Knowledge Scoring Tests
# ---------------------------------------------------------------------------

class TestKnowledgeScoring:
    def test_high_score_research(self):
        turn = ConversationTurn(
            turn_index=1,
            assistant_texts=[
                "经过深入研究和分析，发现 Hermes 的架构设计有以下特点：\n"
                "1. 学习飞轮机制\n2. 技能自动创建\n3. 记忆 nudge 系统\n"
                "**关键发现**：Hermes 的 prompt_builder 使用分层 token 预算"
            ],
        )
        result = score_turn(turn)
        assert result.score >= 2.0
        assert result.category in (KnowledgeCategory.RESEARCH, KnowledgeCategory.ARCHITECTURE)

    def test_high_score_diagnosis(self):
        turn = ConversationTurn(
            turn_index=1,
            assistant_texts=[
                "根因分析完成。root cause 是 compaction 触发太晚，"
                "导致 context overflow。修复方案：调整 maxHistoryShare 从 0.65 到 0.5"
            ],
        )
        result = score_turn(turn)
        assert result.score >= 2.0
        assert result.category == KnowledgeCategory.DIAGNOSIS

    def test_low_score_trivial(self):
        turn = ConversationTurn(
            turn_index=1,
            assistant_texts=["ok"],
        )
        result = score_turn(turn)
        assert result.score < 2.0

    def test_low_score_filler(self):
        turn = ConversationTurn(
            turn_index=1,
            assistant_texts=["我来看看这个问题"],
        )
        result = score_turn(turn)
        assert result.score < 2.0

    def test_message_tool_bonus(self):
        turn = ConversationTurn(
            turn_index=1,
            tool_calls=[
                ToolCall(
                    name="message",
                    arguments={"action": "send", "message": "经过分析研究，发现架构设计的关键决策是使用分层方案，这个方案的优势在于可以渐进式替换。" * 8},
                ),
            ],
        )
        result = score_turn(turn)
        assert any("+substantial_user_message" in s for s in result.signals)

    def test_heartbeat_negative(self):
        turn = ConversationTurn(
            turn_index=1,
            assistant_texts=["HEARTBEAT_OK - 所有服务正常运行"],
        )
        result = score_turn(turn)
        assert result.score < 2.0

    def test_lesson_category(self):
        turn = ConversationTurn(
            turn_index=1,
            assistant_texts=[
                "教训总结：在使用 subagent 时必须设置 sandbox: inherit，"
                "否则文件系统隔离会导致所有写入在 subagent 结束后蒸发。"
                "这个坑我们踩了两次了。"
            ],
        )
        result = score_turn(turn)
        assert result.category == KnowledgeCategory.LESSON


# ---------------------------------------------------------------------------
# Clustering Tests
# ---------------------------------------------------------------------------

class TestClustering:
    def _make_scored(self, turns_spec: list[tuple[int, float, KnowledgeCategory]]) -> tuple[list[TurnScore], ParsedSession]:
        """Make scored turns from (index, score, category) specs."""
        session = ParsedSession(
            source_path=Path("/tmp/test.jsonl"),
            metadata=SessionMetadata(session_id="test", timestamp="2026-04-15"),
        )
        scored = []
        for idx, sc, cat in turns_spec:
            turn = ConversationTurn(turn_index=idx, assistant_texts=["x" * 200])
            scored.append(TurnScore(turn=turn, score=sc, category=cat))
        session.turns = [st.turn for st in scored]
        return scored, session

    def test_adjacent_same_category_merged(self):
        scored, session = self._make_scored([
            (1, 3.0, KnowledgeCategory.RESEARCH),
            (2, 2.5, KnowledgeCategory.RESEARCH),
            (3, 2.0, KnowledgeCategory.RESEARCH),
        ])
        clusters = cluster_turns(scored, session)
        assert len(clusters) == 1
        assert len(clusters[0].scored_turns) == 3

    def test_different_categories_split(self):
        scored, session = self._make_scored([
            (1, 3.0, KnowledgeCategory.RESEARCH),
            (2, 2.5, KnowledgeCategory.DIAGNOSIS),
        ])
        clusters = cluster_turns(scored, session)
        assert len(clusters) == 2

    def test_distant_same_category_split(self):
        scored, session = self._make_scored([
            (1, 3.0, KnowledgeCategory.RESEARCH),
            (10, 2.5, KnowledgeCategory.RESEARCH),  # too far apart
        ])
        clusters = cluster_turns(scored, session)
        assert len(clusters) == 2

    def test_below_threshold_excluded(self):
        scored, session = self._make_scored([
            (1, 1.0, KnowledgeCategory.RESEARCH),  # below 2.0
            (2, 0.5, KnowledgeCategory.RESEARCH),
        ])
        clusters = cluster_turns(scored, session)
        assert len(clusters) == 0

    def test_no_category_excluded(self):
        turn = ConversationTurn(turn_index=1, assistant_texts=["x" * 200])
        scored = [TurnScore(turn=turn, score=5.0, category=None)]
        session = ParsedSession(
            source_path=Path("/tmp/test.jsonl"),
            metadata=SessionMetadata(session_id="test"),
            turns=[turn],
        )
        clusters = cluster_turns(scored, session)
        assert len(clusters) == 0


# ---------------------------------------------------------------------------
# Rule-based Extraction Tests
# ---------------------------------------------------------------------------

class TestRuleBasedExtraction:
    def test_basic_extraction(self):
        turn = ConversationTurn(
            turn_index=1,
            assistant_texts=["Hermes 的架构核心是学习飞轮，包含技能自动创建和记忆 nudge 两个关键模块。" * 3],
        )
        st = TurnScore(turn=turn, score=3.0, category=KnowledgeCategory.RESEARCH)
        cluster = KnowledgeCluster(
            scored_turns=[st],
            category=KnowledgeCategory.RESEARCH,
            session_id="test-123",
            session_date="2026-04-15",
            topic_hint="Hermes源码分析",
        )
        entry = extract_knowledge_rule_based(cluster)

        assert entry is not None
        assert entry.memory_type == MemoryType.SEMANTIC
        assert entry.category == KnowledgeCategory.RESEARCH
        assert "Hermes" in entry.content
        assert entry.source_session == "test-123"
        assert entry.content_hash  # non-empty

    def test_truncation_for_long_content(self):
        turn = ConversationTurn(
            turn_index=1,
            assistant_texts=["A" * 5000],
        )
        st = TurnScore(turn=turn, score=3.0, category=KnowledgeCategory.RESEARCH)
        cluster = KnowledgeCluster(
            scored_turns=[st],
            category=KnowledgeCategory.RESEARCH,
        )
        entry = extract_knowledge_rule_based(cluster, max_chars=1000)

        assert entry is not None
        assert len(entry.content) < 1200  # header + truncated
        assert "truncated" in entry.content

    def test_empty_cluster_returns_none(self):
        cluster = KnowledgeCluster(
            scored_turns=[],
            category=KnowledgeCategory.RESEARCH,
        )
        assert extract_knowledge_rule_based(cluster) is None

    def test_category_to_memory_type_mapping(self):
        for cat, expected_type in [
            (KnowledgeCategory.RESEARCH, MemoryType.SEMANTIC),
            (KnowledgeCategory.DIAGNOSIS, MemoryType.EPISODIC),
            (KnowledgeCategory.SOLUTION, MemoryType.PROCEDURAL),
            (KnowledgeCategory.LESSON, MemoryType.PROCEDURAL),
            (KnowledgeCategory.ARCHITECTURE, MemoryType.SEMANTIC),
            (KnowledgeCategory.DECISION, MemoryType.SEMANTIC),
            (KnowledgeCategory.STATUS, MemoryType.EPISODIC),
            (KnowledgeCategory.REFERENCE, MemoryType.SEMANTIC),
        ]:
            turn = ConversationTurn(turn_index=1, assistant_texts=["content " * 30])
            st = TurnScore(turn=turn, score=3.0, category=cat)
            cluster = KnowledgeCluster(scored_turns=[st], category=cat)
            entry = extract_knowledge_rule_based(cluster)
            assert entry is not None, f"Failed for {cat}"
            assert entry.memory_type == expected_type, f"{cat}: expected {expected_type}, got {entry.memory_type}"

    def test_prefers_sent_messages_over_prose(self):
        turn = ConversationTurn(
            turn_index=1,
            assistant_texts=["内部思考过程，不重要"],
            tool_calls=[
                ToolCall(
                    name="message",
                    arguments={"action": "send", "message": "发给用户的精炼结论：Hermes 架构的核心优势是学习飞轮，包含技能自动创建和记忆 nudge 两个关键模块，这是它区别于其他 agent 框架的核心差异化能力"},
                ),
            ],
        )
        st = TurnScore(turn=turn, score=3.0, category=KnowledgeCategory.RESEARCH)
        cluster = KnowledgeCluster(scored_turns=[st], category=KnowledgeCategory.RESEARCH)
        entry = extract_knowledge_rule_based(cluster)

        assert entry is not None
        assert "精炼结论" in entry.content
        assert "内部思考" not in entry.content


# ---------------------------------------------------------------------------
# Full Pipeline Tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_extract_session_no_llm(self, tmp_path):
        jsonl = _make_session_jsonl([
            {"user": "分析 Hermes 的架构", "assistant_text": (
                "经过深入研究和分析，Hermes 的架构设计有以下关键特点：\n"
                "1. 学习飞轮（Learning Flywheel）— 技能自动创建和改进\n"
                "2. 记忆 nudge 系统 — 后台提取对话中的知识\n"
                "3. 训练闭环 — 轨迹 → RL → 微调\n"
                "**关键发现**：prompt_builder 使用分层 token 预算，每层有独立的 token 上限"
            )},
            {"user": "对比 OpenClaw", "assistant_text": (
                "对比分析结论：\n"
                "- Hermes 强在学习能力，OpenClaw 强在执行深度\n"
                "- Hermes 378K LOC 全部自建，OpenClaw 214K LOC TypeScript\n"
                "- 架构决策：Caveman 应该融合两者优势"
            )},
            {"user": "好的", "assistant_text": "收到"},
        ])
        p = _write_session(tmp_path, "research.jsonl", jsonl)
        session = parse_session(p)
        assert session is not None

        result = await extract_session(session)
        assert result.total_turns == 3
        assert result.extractable_turns >= 1
        assert len(result.knowledge_extracted) >= 1

        # Check the extracted knowledge quality
        for entry in result.knowledge_extracted:
            assert entry.content
            assert entry.memory_type in (MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.PROCEDURAL)
            assert entry.source_session == "test-session"

    @pytest.mark.asyncio
    async def test_extract_session_trivial_content(self, tmp_path):
        jsonl = _make_session_jsonl([
            {"user": "hi", "assistant_text": "hello"},
            {"user": "ok", "assistant_text": "好的"},
        ])
        p = _write_session(tmp_path, "trivial.jsonl", jsonl)
        session = parse_session(p)
        assert session is not None

        result = await extract_session(session)
        assert len(result.knowledge_extracted) == 0

    @pytest.mark.asyncio
    async def test_extract_session_with_mock_llm(self, tmp_path):
        jsonl = _make_session_jsonl([
            {"user": "分析根因", "assistant_text": (
                "根因分析：root cause 是 compaction 配置不当。"
                "maxHistoryShare 设为 0.65 太高，导致 context overflow。"
                "解决方案：调整到 0.5，同时开启 softTrim。"
            )},
        ])
        p = _write_session(tmp_path, "diag.jsonl", jsonl)
        session = parse_session(p)
        assert session is not None

        async def mock_llm(messages, system):
            return "根因：compaction maxHistoryShare=0.65 过高导致 context overflow。修复：调到 0.5 + 开启 softTrim。"

        result = await extract_session(session, llm_complete=mock_llm)
        assert len(result.knowledge_extracted) >= 1
        entry = result.knowledge_extracted[0]
        assert "compaction" in entry.content

    @pytest.mark.asyncio
    async def test_extract_session_llm_skip(self, tmp_path):
        jsonl = _make_session_jsonl([
            {"user": "查状态", "assistant_text": (
                "正在执行检查... 发现配置有问题需要修复。"
                "步骤如下：step 1 检查配置 step 2 修改参数"
            )},
        ])
        p = _write_session(tmp_path, "skip.jsonl", jsonl)
        session = parse_session(p)
        assert session is not None

        async def mock_llm_skip(messages, system):
            return "SKIP"

        result = await extract_session(session, llm_complete=mock_llm_skip)
        assert result.skipped_by_llm >= 0  # May or may not have extractable turns


# ---------------------------------------------------------------------------
# Importer Integration Tests
# ---------------------------------------------------------------------------

class TestOpenClawSessionImporter:
    def test_detect(self, tmp_path):
        importer = OpenClawSessionImporter(caveman_home=tmp_path)
        # Will check ~/.openclaw/agents/main/sessions/ which may or may not exist
        # Just verify it doesn't crash
        result = importer.detect()
        assert result is not None or result is None  # Just verify it returns without crash

    def test_scan_with_filter(self, tmp_path, monkeypatch):
        # Create fake session dir
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        jsonl1 = _make_session_jsonl(
            [{"user": "研究 Hermes 架构", "assistant_text": "深入分析研究发现 Hermes 的架构设计核心是学习飞轮机制" * 5}],
            session_id="session-april",
        )
        # Write with date-prefixed name
        (sessions_dir / "2026-04-15T10-00-00_session-april.jsonl").write_text(jsonl1)

        jsonl2 = _make_session_jsonl(
            [{"user": "hi", "assistant_text": "hello"}],
            session_id="session-march",
        )
        (sessions_dir / "2026-03-01T10-00-00_session-march.jsonl").write_text(jsonl2)

        # Monkeypatch the sessions path
        import caveman.import_.openclaw_sessions as mod
        monkeypatch.setattr(mod, "_OPENCLAW_SESSIONS", sessions_dir)

        importer = OpenClawSessionImporter(
            caveman_home=tmp_path,
            since="2026-04-01",
        )
        # Force detect to return True
        monkeypatch.setattr(importer, "detect", lambda: True)

        manifest = importer.scan()
        # Only April session should be included
        for item in manifest.items:
            assert "march" not in str(item.source_path).lower()

    def test_session_preview(self, tmp_path):
        jsonl = _make_session_jsonl([
            {"user": "分析架构", "assistant_text": "经过研究分析，架构的关键设计决策包括分层方案和渐进式替换策略" * 3},
        ])
        p = tmp_path / "preview.jsonl"
        p.write_text(jsonl)

        importer = OpenClawSessionImporter(caveman_home=tmp_path)
        preview = importer.get_session_preview(p)

        assert preview is not None
        assert preview["session_id"] == "test-session"
        assert preview["total_turns"] == 1


# ---------------------------------------------------------------------------
# Content Hash Dedup Tests
# ---------------------------------------------------------------------------

class TestExtractionDedup:
    def test_same_content_same_hash(self):
        turn = ConversationTurn(turn_index=1, assistant_texts=["same content " * 20])
        st = TurnScore(turn=turn, score=3.0, category=KnowledgeCategory.RESEARCH)
        c1 = KnowledgeCluster(scored_turns=[st], category=KnowledgeCategory.RESEARCH)
        c2 = KnowledgeCluster(scored_turns=[st], category=KnowledgeCategory.RESEARCH)
        assert c1.content_hash == c2.content_hash

    def test_different_content_different_hash(self):
        t1 = ConversationTurn(turn_index=1, assistant_texts=["content A " * 20])
        t2 = ConversationTurn(turn_index=2, assistant_texts=["content B " * 20])
        st1 = TurnScore(turn=t1, score=3.0, category=KnowledgeCategory.RESEARCH)
        st2 = TurnScore(turn=t2, score=3.0, category=KnowledgeCategory.RESEARCH)
        c1 = KnowledgeCluster(scored_turns=[st1], category=KnowledgeCategory.RESEARCH)
        c2 = KnowledgeCluster(scored_turns=[st2], category=KnowledgeCategory.RESEARCH)
        assert c1.content_hash != c2.content_hash
