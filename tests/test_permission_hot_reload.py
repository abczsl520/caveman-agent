from __future__ import annotations

from enum import Enum

import pytest

from caveman.security.permissions import PermissionLevel, PermissionManager


@pytest.mark.asyncio
async def test_permission_manager_accepts_semantically_equal_reloaded_auto_enum():
    """Hot reload can leave existing managers holding old PermissionLevel objects.

    The permission check must compare semantic values, not Enum object identity;
    otherwise a gateway SIGUSR2 reload turns old AUTO into effective ASK and
    rejects tools because gateway sessions have no approval callback.
    """

    class ReloadedPermissionLevel(Enum):
        AUTO = "auto"
        ASK = "ask"
        DENY = "deny"

    manager = PermissionManager({"bash_write": ReloadedPermissionLevel.AUTO})

    assert await manager.request("bash_write", "bash({})") is True


@pytest.mark.asyncio
async def test_permission_manager_accepts_plain_string_levels_from_config():
    manager = PermissionManager({"bash_write": "auto", "bash_sudo": "deny"})

    assert await manager.request("bash_write", "bash({})") is True
    assert await manager.request("bash_sudo", "sudo") is False


def test_permission_manager_check_normalizes_unknown_to_ask():
    manager = PermissionManager({})

    assert manager.check("missing") is PermissionLevel.ASK
