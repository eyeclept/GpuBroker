"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:
    Steam Proton process detection and post-game cooldown tests.
"""
# Imports
import importlib
import json
import logging
import os
import shutil

import pytest
from fastapi.testclient import TestClient

# Globals
logger = logging.getLogger(__name__)

_WRAPPER_COMM = b"steam-launch-wr\n"
_WRAPPER_CMDLINE = (
    b"/home/user/.steam/steam-launch-wrapper\0"
    b"SteamLaunch\0AppId=2001120\0/mnt/Proton Hotfix/proton\0"
)


def _patch_proc(
    broker_app, monkeypatch, tmp_path, procs, *, foreign=False, unreadable_cmdline_pid=None,
):
    """Point the Proton scan at a fake procfs tree under tmp_path.

    Writes <tmp>/<pid>/comm and <tmp>/<pid>/cmdline as bytes. Real files
    are owned by the test user. Does not patch os.stat, os.scandir, or open.
    """
    proc_root = tmp_path / "proc"
    if proc_root.exists():
        shutil.rmtree(proc_root)
    proc_root.mkdir()
    for pid, fields in procs.items():
        pid_dir = proc_root / str(pid)
        pid_dir.mkdir()
        (pid_dir / "comm").write_bytes(fields["comm"])
        cmdline_path = pid_dir / "cmdline"
        cmdline_path.write_bytes(fields["cmdline"])
        if pid == unreadable_cmdline_pid:
            cmdline_path.chmod(0o000)
    monkeypatch.setattr(broker_app, "_PROC_ROOT", str(proc_root))
    if foreign:
        # geteuid is not used by pytest tmp handling, so this patch does
        # not break pytest internals the way a patched os.stat did.
        monkeypatch.setattr(broker_app.os, "geteuid", lambda: os.getuid() ^ 1)


def _lease(caller, preemptible=True):
    return {
        "caller": caller, "mb": 1000, "acquired_at": 0.0, "last_seen": 0.0,
        "priority": "P2", "preemptible": preemptible, "pid": None,
        "preempt_flagged_at": None, "preempt_reason": None,
    }


# Functions
def test_scan_matches_only_steam_proton_wrapper(broker_app, monkeypatch, tmp_path):
    """Input: Steam, a Proton game, a native game. Output: only Proton matches."""
    procs = {
        1: {
            "comm": b"steam\n",
            "cmdline": b"/usr/bin/steam\0SteamLaunch\0AppId=2001120\0/proton\0",
        },
        2: {"comm": _WRAPPER_COMM, "cmdline": _WRAPPER_CMDLINE},
        3: {
            "comm": _WRAPPER_COMM,
            "cmdline": (
                b"/home/user/.steam/steam-launch-wrapper\0"
                b"SteamLaunch\0AppId=300\0/native-game\0"
            ),
        },
    }
    _patch_proc(broker_app, monkeypatch, tmp_path, procs)
    assert broker_app._steam_proton_running() is True
    procs[2] = {
        "comm": _WRAPPER_COMM,
        "cmdline": (
            b"/home/user/.steam/steam-launch-wrapper\0"
            b"SteamLaunch\0AppId=2001120\0/native-game\0"
        ),
    }
    _patch_proc(broker_app, monkeypatch, tmp_path, procs)
    assert broker_app._steam_proton_running() is False


def test_invalid_utf8_comm_does_not_abort_scan(broker_app, monkeypatch, tmp_path):
    """Input: unrelated pid with invalid UTF-8 comm plus a real wrapper. Output: scan finds the wrapper."""
    procs = {
        1: {"comm": b"\xff\xfe\n", "cmdline": b"/bin/weird\0"},
        2: {"comm": _WRAPPER_COMM, "cmdline": _WRAPPER_CMDLINE},
    }
    _patch_proc(broker_app, monkeypatch, tmp_path, procs)
    assert broker_app._steam_proton_running() is True


@pytest.mark.skipif(os.geteuid() == 0, reason="chmod 000 on cmdline does not block root")
def test_permission_error_on_cmdline_does_not_abort_scan(broker_app, monkeypatch, tmp_path):
    """Input: PermissionError on one pid cmdline plus a real wrapper. Output: scan finds the wrapper."""
    procs = {
        1: {"comm": _WRAPPER_COMM, "cmdline": _WRAPPER_CMDLINE},
        2: {"comm": _WRAPPER_COMM, "cmdline": _WRAPPER_CMDLINE},
    }
    _patch_proc(
        broker_app, monkeypatch, tmp_path, procs,
        unreadable_cmdline_pid=1,
    )
    assert broker_app._steam_proton_running() is True


def test_empty_cmdline_for_matching_comm_returns_false(broker_app, monkeypatch, tmp_path):
    """Input: matching comm and empty cmdline. Output: False, no raise."""
    procs = {1: {"comm": _WRAPPER_COMM, "cmdline": b""}}
    _patch_proc(broker_app, monkeypatch, tmp_path, procs)
    assert broker_app._steam_proton_running() is False


def test_wrapper_must_be_owned_by_this_euid(broker_app, monkeypatch, tmp_path):
    """Input: matching wrappers at this euid and another uid. Output: only the same-uid wrapper counts."""
    procs = {
        1: {"comm": _WRAPPER_COMM, "cmdline": _WRAPPER_CMDLINE},
        2: {"comm": _WRAPPER_COMM, "cmdline": _WRAPPER_CMDLINE},
    }
    _patch_proc(broker_app, monkeypatch, tmp_path, procs)
    assert broker_app._steam_proton_running() is True
    _patch_proc(broker_app, monkeypatch, tmp_path, {1: procs[1]}, foreign=True)
    assert broker_app._steam_proton_running() is False


@pytest.mark.asyncio
async def test_heartbeat_reason_differs_for_proton_and_fairness(broker_app, monkeypatch):
    """Input: Proton-flagged lease and fairness-flagged lease. Output: distinct reason texts from the constants."""
    monkeypatch.setattr(broker_app, "_steam_proton_running", lambda: True)
    broker_app._leases["proton"] = _lease("proton-holder")
    await broker_app._refresh_proton_state()

    broker_app._leases["fair"] = _lease("fair-holder")
    flagged = broker_app._maybe_flag_for_preemption(
        "P0", 0.0, {"fair": broker_app._leases["fair"]},
    )
    assert flagged is not None

    client = TestClient(broker_app.app)
    proton = client.post("/heartbeat", json={"lease_id": "proton"}).json()
    fair = client.post("/heartbeat", json={"lease_id": "fair"}).json()
    assert proton["preempt"] is True
    assert fair["preempt"] is True
    assert proton["reason"] == broker_app.PREEMPT_PROTON_REASON_TEXT
    assert fair["reason"] == broker_app.PREEMPT_REASON_TEXT
    assert proton["reason"] != fair["reason"]
    assert "proton-holder" not in proton["reason"]
    assert "fair-holder" not in fair["reason"]


@pytest.mark.asyncio
async def test_game_blocks_all_acquires_and_cooldown_survives_ledger_roundtrip(
    broker_app, mock_env, monkeypatch,
):
    """Input: Proton starts and exits. Output: acquires stay blocked for 30 minutes.

    Cooldown remaining is recomputed from the persisted `proton_game_cleared_at`.
    This test does not call `startup`.
    """
    # The shared fixture mocks ledger writes. Restore the real writer for
    # this persistence and restart transition test.
    broker_app = importlib.reload(broker_app)
    running = {"value": True}
    now = {"value": 1000.0}
    monkeypatch.setattr(broker_app, "_steam_proton_running", lambda: running["value"])
    monkeypatch.setattr(broker_app.time, "time", lambda: now["value"])
    await broker_app._refresh_proton_state()
    assert broker_app._proton_game_active is True
    assert json.loads(open(mock_env["ledger_path"]).read())["proton_game_active"] is True

    # An existing preemptible holder yields even before the game uses VRAM.
    broker_app._leases["old"] = {"caller": "old", "mb": 1000, "preemptible": True,
                                 "preempt_flagged_at": None, "last_seen": 0}
    await broker_app._refresh_proton_state()
    assert broker_app._leases["old"]["preempt_flagged_at"] is not None

    client = TestClient(broker_app.app)
    for mb in (0, 2000):
        response = client.post("/acquire", json={"caller": f"new-{mb}", "mb": mb,
                                                 "preemptible": False, "timeout_s": 1})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1800"
        assert response.json()["detail"] == "unmanaged_gpu_load"

    running["value"] = False
    now["value"] = 1100.0
    await broker_app._refresh_proton_state()
    assert broker_app._proton_game_active is False
    assert broker_app._proton_retry_after_seconds() == 1800
    now["value"] = 2899.0
    assert broker_app._proton_retry_after_seconds() == 1
    now["value"] = 2900.0
    assert broker_app._proton_retry_after_seconds() == 0
    broker_app._gpu_total_mb = 12282.0
    broker_app._ollama_mb = 0.0
    async def used_mb():
        return 1500.0
    monkeypatch.setattr(broker_app, "_live_used_mb", used_mb)
    response = client.post("/acquire", json={"caller": "after-cooldown", "mb": 2000, "timeout_s": 1})
    assert response.status_code == 200

    # A later broker restart during the cooldown must not grant early.
    now["value"] = 2000.0
    with open(mock_env["ledger_path"]) as f:
        saved = json.load(f)
    assert saved["proton_game_cleared_at"] == 1100.0
    broker_app._proton_game_cleared_at = saved["proton_game_cleared_at"]
    assert broker_app._proton_retry_after_seconds() == 900


@pytest.mark.asyncio
async def test_restart_while_game_was_active_starts_full_cooldown_if_now_gone(
    broker_app, mock_env, monkeypatch,
):
    """Input: ledger says a game was active before restart. Output: 30-minute block if it ended while down."""
    broker_app = importlib.reload(broker_app)
    with open(mock_env["ledger_path"], "w") as f:
        json.dump({"leases": {}, "proton_game_active": True, "proton_game_cleared_at": None}, f)
    monkeypatch.setattr(broker_app, "_nvidia_smi_query", lambda field: 12282.0)
    monkeypatch.setattr(broker_app, "_load_non_lease_samples", lambda: None)
    monkeypatch.setattr(broker_app, "_steam_proton_running", lambda: False)
    monkeypatch.setattr(broker_app.time, "time", lambda: 1234.0)
    monkeypatch.setattr(broker_app.asyncio, "create_task", lambda task: task.close())
    await broker_app.startup()
    assert broker_app._proton_game_active is False
    assert broker_app._proton_game_uncertain is False
    assert broker_app._proton_retry_after_seconds() == 1800
    assert json.loads(open(mock_env["ledger_path"]).read())["proton_game_cleared_at"] == 1234.0
