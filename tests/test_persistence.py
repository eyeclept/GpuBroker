"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

Unit tests for GpuBroker persistence: the shared atomic JSON writer
(S-21), ledger restore of wrong-shape JSON (S-24), and batched
non-lease sample writes (P-04). Real files under tmp_path. No live GPU
or network.
"""

import asyncio
import importlib
import json
import logging
import os
import stat
import sys
from unittest.mock import AsyncMock, patch

import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _good_lease(caller="ok_caller", mb=1000.0):
    return {
        "caller": caller,
        "mb": mb,
        "acquired_at": 0.0,
        "last_seen": 0.0,
    }


def _patch_startup_deps(broker_app, monkeypatch):
    monkeypatch.setattr(broker_app, "_nvidia_smi_query", lambda field: 12282.0)
    monkeypatch.setattr(broker_app, "_load_non_lease_samples", lambda: None)
    monkeypatch.setattr(broker_app, "_steam_proton_running", lambda: False)
    monkeypatch.setattr(broker_app.asyncio, "create_task", lambda task: task.close())


async def _n_non_lease_polls(srv, n):
    remaining = n

    async def fake_sleep(_interval):
        nonlocal remaining
        remaining -= 1
        if remaining <= 0:
            raise asyncio.CancelledError

    with patch.object(srv.asyncio, "sleep", fake_sleep):
        with pytest.raises(asyncio.CancelledError):
            await srv._poll_non_lease_usage()


def test_writer_does_not_follow_fixed_tmp_symlink(broker_app, mock_env, tmp_path):
    """Test: the writer ignores a planted <path>.tmp symlink, writes mode 0600, and leaves no unique .tmp files."""
    broker_app = importlib.reload(broker_app)
    dest = mock_env["ledger_path"]
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched")
    planted = tmp_path / (os.path.basename(dest) + ".tmp")
    planted.symlink_to(victim)

    broker_app._leases["L1"] = _good_lease("c1", 512.0)
    broker_app._persist_ledger_sync()

    assert victim.read_text() == "untouched"
    assert planted.is_symlink()
    assert not os.path.islink(dest)
    saved = json.loads(open(dest).read())
    assert saved["leases"]["L1"]["caller"] == "c1"
    assert stat.S_IMODE(os.stat(dest).st_mode) == 0o600
    leftovers = list(tmp_path.glob(os.path.basename(dest) + ".*.tmp"))
    assert leftovers == []


def test_writer_removes_temp_file_when_json_fails(broker_app, mock_env, tmp_path):
    """Test: a JSON serialization failure deletes the unique temp file and does not create the target."""
    broker_app = importlib.reload(broker_app)
    dest = mock_env["ledger_path"]
    broker_app._leases["L1"] = {"caller": "c1", "mb": object()}
    with pytest.raises(TypeError):
        broker_app._persist_ledger_sync()
    assert not os.path.exists(dest)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload, keep_ids",
    [
        (["not", "a", "dict"], []),
        (
            {
                "leases": ["x"],
                "proton_game_active": False,
                "proton_game_cleared_at": None,
            },
            [],
        ),
        (
            {
                "leases": {
                    "bad": "string-lease",
                    "good": _good_lease("ok_caller", 1000.0),
                },
                "proton_game_active": False,
                "proton_game_cleared_at": None,
            },
            ["good"],
        ),
        (
            {
                "leases": {
                    "nocaller": {
                        "mb": 1000.0,
                        "acquired_at": 0.0,
                        "last_seen": 0.0,
                    },
                    "good": _good_lease("ok_caller", 500.0),
                },
                "proton_game_active": False,
                "proton_game_cleared_at": None,
            },
            ["good"],
        ),
    ],
    ids=["top-level-list", "leases-list", "lease-is-string", "lease-without-caller"],
)
async def test_startup_malformed_ledger_shapes(
    broker_app, mock_env, monkeypatch, payload, keep_ids,
):
    """Test: startup with each wrong-shape ledger completes, keeps valid leases, and skips bad ones."""
    with open(mock_env["ledger_path"], "w") as f:
        json.dump(payload, f)
    _patch_startup_deps(broker_app, monkeypatch)
    await broker_app.startup()
    assert set(broker_app._leases) == set(keep_ids)
    for lease_id in keep_ids:
        lease = broker_app._leases[lease_id]
        assert broker_app._caller_to_lease[lease["caller"]] == lease_id
    if not keep_ids:
        assert broker_app._caller_to_lease == {}


@pytest.mark.asyncio
async def test_sampler_persists_every_ten_samples(broker_app, monkeypatch):
    """Test: the sampler writes the sample file on the 10th sample, not on samples 1-9."""
    monkeypatch.setattr(broker_app, "_live_used_mb", AsyncMock(return_value=2000.0))
    monkeypatch.setattr(broker_app, "_read_lock_state", lambda: "unlocked")
    path = broker_app.NON_LEASE_USAGE_PATH
    every = broker_app.NON_LEASE_PERSIST_EVERY
    assert every == 10

    await _n_non_lease_polls(broker_app, every - 1)
    assert not os.path.exists(path)
    assert len(broker_app._non_lease_samples) == every - 1

    broker_app._non_lease_samples.clear()
    await _n_non_lease_polls(broker_app, every)
    assert os.path.exists(path)
    saved = json.loads(open(path).read())
    assert len(saved["samples"]) == every
    assert len(broker_app._non_lease_samples) == every
