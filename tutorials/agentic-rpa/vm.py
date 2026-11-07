"""SSH and SFTP to the target VM (rungs 2-4). To drive a Windows machine, this is the module to
swap for OpenSSH-on-Windows or WinRM; nothing above it changes."""

from __future__ import annotations

import contextlib
import shlex
from typing import AsyncIterator

import asyncssh

import config


@contextlib.asynccontextmanager
async def connect() -> AsyncIterator[asyncssh.SSHClientConnection]:
    if config.SSH_KEY:
        keys = [asyncssh.import_private_key(config.SSH_KEY)]
    elif config.SSH_KEY_PATH:
        keys = [asyncssh.read_private_key(config.SSH_KEY_PATH)]
    else:
        raise RuntimeError("set RPA_SSH_KEY (key text) or RPA_SSH_KEY_PATH")
    # Toy demo: the VM's host key changes on every rebuild, so host key checking is off.
    # A real deployment pins known_hosts.
    async with asyncssh.connect(
        config.VM_HOST, port=config.SSH_PORT, username=config.SSH_USER,
        client_keys=keys, agent_path=None, known_hosts=None, connect_timeout=20,
    ) as conn:
        yield conn


async def run(command: str, timeout: float = 300, check: bool = False) -> dict:
    async with connect() as conn:
        res = await conn.run(command, timeout=timeout, check=False)
    out = {
        "command": command,
        "exit_code": res.exit_status,
        "stdout": (res.stdout or "")[-20_000:],
        "stderr": (res.stderr or "")[-5_000:],
    }
    if check and res.exit_status != 0:
        raise RuntimeError(f"`{command}` exited {res.exit_status}: {out['stderr'] or out['stdout']}")
    return out


async def get(remote: str, local: str) -> None:
    async with connect() as conn, conn.start_sftp_client() as sftp:
        await sftp.get(remote, local)


async def put(local: str, remote: str) -> None:
    async with connect() as conn, conn.start_sftp_client() as sftp:
        await sftp.put(local, remote)


def q(s: str) -> str:
    return shlex.quote(s)
