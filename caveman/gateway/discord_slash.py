"""Discord slash command registration — extracted from discord_gw.py.

Registers native Discord Application Commands (slash commands)
including Chinese aliases for bilingual support.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

# Chinese aliases — top 40 most useful (guild limit: 100 per guild)
ZH_ALIASES = {
    "帮助": "help", "状态": "status", "模型": "model",
    "命令": "commands", "记忆": "memory", "工具": "tools",
    "技能": "skills", "诊断": "doctor", "自检": "selftest",
    "护盾": "shield", "飞轮": "flywheel", "配置": "config",
    "搜索": "search", "历史": "history", "清除": "clear",
    "重置": "reset", "导出": "export", "导入": "import",
    "备份": "backup", "恢复": "restore",
    "任务": "task", "进度": "progress", "停止": "stop",
    "继续": "continue", "重试": "retry",
    "分析": "analyze", "优化": "optimize", "审计": "audit",
    "测试": "test", "部署": "deploy",
    "笔记": "note", "待办": "todo", "提醒": "remind",
    "翻译": "translate", "总结": "summarize",
    "代码": "code", "运行": "run", "调试": "debug",
    "文件": "file", "目录": "ls", "编辑": "edit",
}


async def sync_slash_commands(client, locale: str = "en") -> None:
    """Register native Discord Application Commands."""
    try:
        import discord
        from discord import app_commands
        from caveman.commands.dispatcher import dispatch
        from caveman.commands.registry import COMMAND_REGISTRY
    except ImportError:
        return

    tree = client.tree
    guilds = client.guilds

    for cmd_def in COMMAND_REGISTRY:
        if not hasattr(cmd_def, 'name'):
            continue
        cmd_name = cmd_def.name
        desc = (cmd_def.description or cmd_name)[:100]

        def _make_handler(name: str):
            async def _handler(interaction: discord.Interaction, args: str = ""):
                await interaction.response.defer()
                result = await dispatch(f"/{name} {args}".strip())
                text = result.get("text", str(result))[:2000]
                if text:
                    await interaction.followup.send(text)
                else:
                    await interaction.followup.send("命令已接收；没有额外输出。")
            return _handler

        handler = app_commands.Command(
            name=cmd_def.name, description=desc, callback=_make_handler(cmd_name),
        )
        handler.add_check(lambda i: True)
        for guild in guilds:
            tree.add_command(handler, guild=guild, override=True)

    # Chinese aliases
    if locale in ("zh", "zh-CN", "zh-TW"):
        for zh_name, en_name in ZH_ALIASES.items():
            cmd_def = next((c for c in COMMAND_REGISTRY if c.name == en_name), None)
            if not cmd_def:
                continue
            desc = (cmd_def.description or en_name)[:100]

            def _make_zh_handler(name: str):
                async def _zh_handler(interaction: discord.Interaction, args: str = ""):
                    await interaction.response.defer()
                    result = await dispatch(f"/{name} {args}".strip())
                    text = result.get("text", str(result))[:2000]
                    if text:
                        await interaction.followup.send(text)
                    else:
                        await interaction.followup.send("命令已接收；没有额外输出。")
                return _zh_handler

            zh_handler = app_commands.Command(
                name=zh_name, description=desc, callback=_make_zh_handler(en_name),
            )
            zh_handler.add_check(lambda i: True)
            for guild in guilds:
                tree.add_command(zh_handler, guild=guild, override=True)

    # Language switch commands
    @tree.command(name="language", description="Switch language / 切换语言", guilds=guilds)
    async def _language_handler(interaction: discord.Interaction, lang: str):
        await interaction.response.send_message(f"🌐 Language set to: {lang}")

    @tree.command(name="语言", description="切换语言 / Switch language", guilds=guilds)
    async def _language_zh_handler(interaction: discord.Interaction, lang: str):
        await interaction.response.send_message(f"🌐 语言已切换为: {lang}")

    # Sync
    try:
        tree.clear_commands(guild=None)
        await tree.sync()
        logger.info("Cleared global commands")
        for guild in guilds:
            synced = await tree.sync(guild=guild)
            logger.info("Synced %d slash commands to guild %s", len(synced), guild.name)
    except Exception as e:
        logger.warning("Failed to sync slash commands: %s", e)
