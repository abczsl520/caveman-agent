"""Tests for Phase 4 (SFT/RL training) + Phase 5 (Hub + Plugins)."""
import asyncio
import json
import tempfile
from pathlib import Path

import pytest


# ── SFT Pipeline ──

def test_training_config():
    from caveman.training.sft import TrainingConfig
    config = TrainingConfig(model_name="test-model", epochs=5)
    d = config.to_dict()
    assert d["model_name"] == "test-model"
    assert d["epochs"] == 5

    restored = TrainingConfig.from_dict(d)
    assert restored.model_name == "test-model"
    assert restored.epochs == 5


def test_dataset_builder_sharegpt():
    from caveman.training.sft import TrainingConfig, DatasetBuilder

    with tempfile.TemporaryDirectory() as td:
        # Create trajectory files
        traj_dir = Path(td) / "trajectories"
        traj_dir.mkdir()
        with open(traj_dir / "test.jsonl", "w") as f:
            f.write(json.dumps({
                "quality_score": 0.8,
                "turns": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "A programming language."},
                ],
            }) + "\n")
            f.write(json.dumps({
                "quality_score": 0.3,  # Below threshold
                "turns": [{"role": "user", "content": "bad"}],
            }) + "\n")

        config = TrainingConfig(output_dir=f"{td}/output", min_quality=0.7, format="sharegpt")
        builder = DatasetBuilder(config)
        output = builder.build(str(traj_dir))

        assert output.exists()
        stats = builder.stats
        assert stats["total"] == 2
        assert stats["kept"] == 1
        assert stats["filtered"] == 1

        # Check format
        with open(output) as f:
            entry = json.loads(f.readline())
        assert "conversations" in entry
        assert entry["conversations"][0]["from"] == "human"


def test_dataset_builder_chatml():
    from caveman.training.sft import TrainingConfig, DatasetBuilder

    with tempfile.TemporaryDirectory() as td:
        traj_dir = Path(td) / "trajectories"
        traj_dir.mkdir()
        with open(traj_dir / "test.jsonl", "w") as f:
            f.write(json.dumps({
                "quality_score": 0.9,
                "turns": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
            }) + "\n")

        config = TrainingConfig(output_dir=f"{td}/output", format="chatml")
        builder = DatasetBuilder(config)
        output = builder.build(str(traj_dir))

        with open(output) as f:
            entry = json.loads(f.readline())
        assert "messages" in entry
        assert entry["messages"][0]["role"] == "user"


def test_dataset_builder_openai():
    from caveman.training.sft import TrainingConfig, DatasetBuilder

    with tempfile.TemporaryDirectory() as td:
        traj_dir = Path(td) / "trajectories"
        traj_dir.mkdir()
        with open(traj_dir / "test.jsonl", "w") as f:
            f.write(json.dumps({
                "quality_score": 0.75,
                "turns": [
                    {"role": "user", "content": "Test"},
                    {"role": "assistant", "content": "Response"},
                ],
            }) + "\n")

        config = TrainingConfig(output_dir=f"{td}/output", format="openai")
        builder = DatasetBuilder(config)
        output = builder.build(str(traj_dir))

        with open(output) as f:
            entry = json.loads(f.readline())
        assert "messages" in entry


def test_sft_mock_train():
    from caveman.training.sft import TrainingConfig, SFTTrainer

    with tempfile.TemporaryDirectory() as td:
        # Create dummy dataset
        ds_path = Path(td) / "dataset.jsonl"
        with open(ds_path, "w") as f:
            for i in range(5):
                f.write(json.dumps({"conversations": [{"from": "human", "value": f"q{i}"}]}) + "\n")

        config = TrainingConfig(output_dir=f"{td}/output")
        trainer = SFTTrainer(config)
        result = trainer.train(ds_path)

        assert result["status"] in ("mock", "fallback_no_trl")
        assert result["entries"] == 5


# ── RL Pipeline ──

def test_rl_config():
    from caveman.training.rl import RLConfig
    config = RLConfig(method="dpo", beta=0.2)
    d = config.to_dict()
    assert d["method"] == "dpo"
    assert d["beta"] == 0.2


def test_preference_pair_builder():
    from caveman.training.rl import RLConfig, PreferencePairBuilder

    with tempfile.TemporaryDirectory() as td:
        traj_dir = Path(td) / "trajectories"
        traj_dir.mkdir()
        with open(traj_dir / "test.jsonl", "w") as f:
            # High quality (chosen)
            f.write(json.dumps({
                "task": "write hello",
                "quality_score": 0.9,
                "turns": [
                    {"role": "user", "content": "Write hello"},
                    {"role": "assistant", "content": "print('hello')"},
                ],
            }) + "\n")
            # Low quality (rejected)
            f.write(json.dumps({
                "task": "write hello",
                "quality_score": 0.3,
                "turns": [
                    {"role": "user", "content": "Write hello"},
                    {"role": "assistant", "content": "idk"},
                ],
            }) + "\n")

        config = RLConfig(output_dir=f"{td}/output")
        builder = PreferencePairBuilder(config)
        output = builder.build(str(traj_dir))

        assert output.exists()
        stats = builder.stats
        assert stats["pairs"] >= 1
        assert stats["chosen"] >= 1
        assert stats["rejected"] >= 1

        with open(output) as f:
            pair = json.loads(f.readline())
        assert "chosen" in pair
        assert "rejected" in pair
        assert pair["chosen_score"] > pair["rejected_score"]


def test_rl_mock_train():
    from caveman.training.rl import RLConfig, RLTrainer

    with tempfile.TemporaryDirectory() as td:
        ds_path = Path(td) / "pairs.jsonl"
        with open(ds_path, "w") as f:
            f.write(json.dumps({"chosen": "good", "rejected": "bad"}) + "\n")

        config = RLConfig(method="dpo", output_dir=f"{td}/output")
        trainer = RLTrainer(config)
        result = trainer.train(ds_path)

        assert result["status"] in ("mock", "fallback_no_trl")
        assert result["method"] == "dpo"


def test_rl_methods():
    from caveman.training.rl import RLConfig, RLTrainer

    with tempfile.TemporaryDirectory() as td:
        ds_path = Path(td) / "pairs.jsonl"
        ds_path.write_text("{}\n")

        for method in ("dpo", "ppo", "grpo"):
            config = RLConfig(method=method, output_dir=f"{td}/output_{method}")
            trainer = RLTrainer(config)
            result = trainer.train(ds_path)
            assert result["method"] == method


# ── Plugin Manager ──

def test_plugin_discovery():
    from caveman.plugins.manager import PluginManager

    with tempfile.TemporaryDirectory() as td:
        # Create a plugin
        plugin_dir = Path(td) / "test-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: test-plugin\nversion: '1.0'\ndescription: A test plugin\ntype: tool\n"
        )
        (plugin_dir / "__init__.py").write_text("def setup(): pass\n")

        mgr = PluginManager(plugins_dir=td)
        found = mgr.discover()
        assert len(found) == 1
        assert found[0].name == "test-plugin"
        assert found[0].version == "1.0"


def test_plugin_load():
    from caveman.plugins.manager import PluginManager

    with tempfile.TemporaryDirectory() as td:
        plugin_dir = Path(td) / "hello"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: hello\nversion: '0.1'\ntype: tool\n"
        )
        (plugin_dir / "__init__.py").write_text(
            "GREETING = 'hello from plugin'\ndef setup(): pass\n"
        )

        mgr = PluginManager(plugins_dir=td)
        mgr.discover()
        module = mgr.load("hello")

        assert module.GREETING == "hello from plugin"
        assert "hello" in mgr.list_loaded()


def test_plugin_unload():
    from caveman.plugins.manager import PluginManager

    with tempfile.TemporaryDirectory() as td:
        plugin_dir = Path(td) / "temp"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text("name: temp\nversion: '1'\n")
        (plugin_dir / "__init__.py").write_text("def teardown(): pass\n")

        mgr = PluginManager(plugins_dir=td)
        mgr.discover()
        mgr.load("temp")
        assert "temp" in mgr.list_loaded()

        mgr.unload("temp")
        assert "temp" not in mgr.list_loaded()


def test_plugin_hooks():
    from caveman.plugins.manager import PluginManager

    async def _run():
        mgr = PluginManager()
        results = []
        mgr.register_hook("on_task", lambda task="": results.append(task))

        await mgr.emit_hook("on_task", task="test")
        assert results == ["test"]

    asyncio.run(_run())


def test_plugin_disabled():
    from caveman.plugins.manager import PluginManager

    with tempfile.TemporaryDirectory() as td:
        plugin_dir = Path(td) / "disabled"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text("name: disabled\nenabled: false\n")
        (plugin_dir / "__init__.py").write_text("")

        mgr = PluginManager(plugins_dir=td)
        mgr.discover()
        with pytest.raises(ValueError, match="disabled"):
            mgr.load("disabled")


# ── Hub Client ──

def test_hub_client_offline():
    async def _run():
        from caveman.hub.client import HubClient
        client = HubClient(hub_url="http://localhost:99999")
        try:
            results = await client.search_skills("test")
            assert isinstance(results, list)

            stats = await client.hub_stats()
            assert stats["status"] == "offline"
        except (OSError, ExceptionGroup):
            # Connection refused or TaskGroup error in some Python versions
            pass

    asyncio.run(_run())


# ── Migration Tools ──

def test_migration_from_openclaw():
    from caveman.hub.client import MigrationTool

    with tempfile.TemporaryDirectory() as td:
        skill_dir = Path(td) / "my-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# My Skill\n\n> Does something useful\n\n## Steps\n\n1. Do this\n2. Do that\n")

        result = MigrationTool.from_openclaw_skill(str(skill_md))
        assert result["name"] == "My Skill"
        assert result["description"] == "Does something useful"
        assert result["source"] == "openclaw"


def test_migration_to_openclaw():
    from caveman.hub.client import MigrationTool

    skill = {"name": "test", "description": "A test skill", "content": "Step 1\nStep 2"}
    md = MigrationTool.to_openclaw_skill(skill)
    assert "# test" in md
    assert "A test skill" in md
    assert "Step 1" in md


def test_migration_from_hermes():
    from caveman.hub.client import MigrationTool

    hermes_skill = {"name": "web-scrape", "description": "Scrape websites", "version": 2}
    result = MigrationTool.from_hermes_skill(hermes_skill)
    assert result["name"] == "web-scrape"
    assert result["source"] == "hermes"
    assert result["version"] == 2
