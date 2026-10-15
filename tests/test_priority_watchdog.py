"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

Unit tests for GpuBroker server.py's priority-based fairness preemption
(Mechanism 1) and broken-client watchdog (Mechanism 2), Assist/TODO.md
section 8's "Priority-based preemption" item. Tests the standalone,
directly-callable pieces (_effective_priority, _weakest_preemptible_holder,
_maybe_flag_for_preemption, _validate_priority_config,
_classify_reclaim_reasons, _reclaim_broken_lease), the infinite
_sweep_expired_leases loop's exception handling, and the one-shot
notification delivered through acquire(). No live GPU or network;
nvidia-smi/subprocess calls are mocked the same way tests/test_server.py
already mocks them.
"""

import asyncio
import logging
import signal
import time
from unittest.mock import AsyncMock, patch

import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _make_lease(caller="c", mb=1000.0, acquired_at=0.0, last_seen=0.0,
                 priority="P2", preemptible=False, pid=None, preempt_flagged_at=None):
    """Build a lease dict with every field acquire() now writes, defaults matching AcquireRequest's."""
    return {
        "caller": caller, "mb": mb, "acquired_at": acquired_at, "last_seen": last_seen,
        "priority": priority, "preemptible": preemptible, "pid": pid,
        "preempt_flagged_at": preempt_flagged_at,
    }


_CONTAINER_CGROUP = (
    "0::/user.slice/user-1000.slice/user@1000.service/user.slice/"
    "libpod-d1ecaf800b2d.scope/container\n"
)
_HOST_CGROUP = (
    "0::/user.slice/user-1000.slice/user@1000.service/app.slice/app-gnome.scope\n"
)


def _stat_contents(pid, comm, start_time):
    after = ["R"] + ["0"] * 18 + [str(start_time)]
    return f"{pid} ({comm}) " + " ".join(after) + "\n"


def _write_proc_pid(proc_root, pid, *, cgroup, nspids, start_time=100, comm="python3"):
    pid_dir = proc_root / str(pid)
    pid_dir.mkdir(parents=True, exist_ok=True)
    (pid_dir / "cgroup").write_text(cgroup)
    nspid_line = "NSpid:\t" + "\t".join(str(n) for n in nspids) + "\n"
    (pid_dir / "status").write_text(f"Name:\t{comm}\n{nspid_line}")
    (pid_dir / "stat").write_text(_stat_contents(pid, comm, start_time))


def _fake_proc(broker_app, monkeypatch, tmp_path, entries):
    """Point _PROC_ROOT at tmp_path/proc and write per-pid files.

    entries: host_pid -> dict with cgroup, nspids, optional start_time, comm.
    """
    proc_root = tmp_path / "proc"
    proc_root.mkdir(exist_ok=True)
    for pid, spec in entries.items():
        _write_proc_pid(
            proc_root,
            pid,
            cgroup=spec["cgroup"],
            nspids=spec["nspids"],
            start_time=spec.get("start_time", 100),
            comm=spec.get("comm", "python3"),
        )
    monkeypatch.setattr(broker_app, "_PROC_ROOT", str(proc_root))
    return proc_root


def _container_gpu(pid=4242, nspid=7, start_time=100, name="stuck_proc", used_mb=2048.0):
    return {"pid": pid, "process_name": name, "used_mb": used_mb}, {
        pid: {
            "cgroup": _CONTAINER_CGROUP,
            "nspids": [pid, nspid],
            "start_time": start_time,
            "comm": name,
        },
    }

# Tests: Aging (_effective_priority)

def test_effective_priority_at_zero_wait_equals_base(broker_app):
    """Test: no wait yet -> effective priority is exactly the tier's base."""
    result = broker_app._effective_priority(broker_app.TIER_BASE_PRIORITY["P2"], 0.0)
    assert result == broker_app.TIER_BASE_PRIORITY["P2"]


def test_effective_priority_ages_linearly_partway(broker_app):
    """Test: half the aging window adds half a tier-step."""
    base = broker_app.TIER_BASE_PRIORITY["P2"]
    half_window = broker_app.PRIORITY_AGING_WINDOW_SECONDS / 2
    result = broker_app._effective_priority(base, half_window)
    assert result == pytest.approx(base + broker_app.TIER_STEP / 2)


def test_effective_priority_capped_at_next_tier_parity(broker_app):
    """Test: full aging reaches exactly the next tier's base value, not more."""
    base_p2 = broker_app.TIER_BASE_PRIORITY["P2"]
    result = broker_app._effective_priority(base_p2, broker_app.PRIORITY_AGING_WINDOW_SECONDS)
    assert result == pytest.approx(broker_app.TIER_BASE_PRIORITY["P1"])


def test_effective_priority_never_exceeds_cap_beyond_aging_window(broker_app):
    """Test: waiting far longer than the aging window does not climb past the cap."""
    base = broker_app.TIER_BASE_PRIORITY["P2"]
    result = broker_app._effective_priority(base, broker_app.PRIORITY_AGING_WINDOW_SECONDS * 100)
    assert result == pytest.approx(base + broker_app.TIER_STEP)


# Tests: Weakest Preemptible Holder / Non-Preemptible Never Targeted

def test_weakest_preemptible_holder_none_when_no_leases(broker_app):
    """Test: an empty ledger has no preemption target."""
    assert broker_app._weakest_preemptible_holder({}) is None


def test_weakest_preemptible_holder_ignores_non_preemptible(broker_app):
    """Test: a non-preemptible holder is never returned, even as the only lease."""
    leases = {"L1": _make_lease(preemptible=False, priority="P3")}
    assert broker_app._weakest_preemptible_holder(leases) is None


def test_weakest_preemptible_holder_picks_lowest_tier(broker_app):
    """Test: among preemptible holders, the lowest-tier one is the target."""
    leases = {
        "L1": _make_lease(caller="a", preemptible=True, priority="P0", acquired_at=0.0),
        "L2": _make_lease(caller="b", preemptible=True, priority="P3", acquired_at=10.0),
        "L3": _make_lease(caller="c", preemptible=False, priority="P3", acquired_at=5.0),
    }
    lease_id, lease = broker_app._weakest_preemptible_holder(leases)
    assert lease_id == "L2"
    assert lease["caller"] == "b"


def test_weakest_preemptible_holder_tiebreak_oldest_first(broker_app):
    """Test: same-tier preemptible holders break ties on the oldest acquired_at."""
    leases = {
        "newer": _make_lease(caller="new", preemptible=True, priority="P2", acquired_at=100.0),
        "older": _make_lease(caller="old", preemptible=True, priority="P2", acquired_at=10.0),
    }
    lease_id, lease = broker_app._weakest_preemptible_holder(leases)
    assert lease_id == "older"


# Tests: Preemption Flagging (_maybe_flag_for_preemption)

def test_maybe_flag_for_preemption_triggers_when_aged_priority_exceeds_holder(broker_app):
    """Test: a same-tier waiter fully aged out exceeds holder+bonus and gets flagged."""
    leases = {"L1": _make_lease(caller="holder", preemptible=True, priority="P2")}
    result = broker_app._maybe_flag_for_preemption("P2", broker_app.PRIORITY_AGING_WINDOW_SECONDS, leases)
    assert result is not None
    lease_id, lease = result
    assert lease_id == "L1"
    assert lease["preempt_flagged_at"] is not None


def test_maybe_flag_for_preemption_no_trigger_before_aging_sufficient(broker_app):
    """Test: a same-tier waiter with zero wait has not aged past holder+bonus yet."""
    leases = {"L1": _make_lease(caller="holder", preemptible=True, priority="P2")}
    result = broker_app._maybe_flag_for_preemption("P2", 0.0, leases)
    assert result is None
    assert leases["L1"]["preempt_flagged_at"] is None


def test_maybe_flag_for_preemption_never_targets_non_preemptible(broker_app):
    """Test: no matter how long the waiter has aged, a non-preemptible-only ledger flags nothing."""
    leases = {"L1": _make_lease(caller="holder", preemptible=False, priority="P3")}
    result = broker_app._maybe_flag_for_preemption("P0", broker_app.PRIORITY_AGING_WINDOW_SECONDS * 10, leases)
    assert result is None
    assert leases["L1"]["preempt_flagged_at"] is None


def test_maybe_flag_for_preemption_higher_tier_waiter_does_not_need_full_aging(broker_app):
    """Test: a genuinely higher-tier waiter exceeds a lower-tier holder immediately, no aging needed."""
    leases = {"L1": _make_lease(caller="holder", preemptible=True, priority="P3")}
    result = broker_app._maybe_flag_for_preemption("P1", 0.0, leases)
    assert result is not None


def test_maybe_flag_for_preemption_lower_tier_waiter_never_preempts_higher_tier_holder(broker_app):
    """Test: a lower-tier waiter, even fully aged, cannot preempt a holder a full tier above it."""
    leases = {"L1": _make_lease(caller="holder", preemptible=True, priority="P1")}
    result = broker_app._maybe_flag_for_preemption("P2", broker_app.PRIORITY_AGING_WINDOW_SECONDS * 100, leases)
    assert result is None


def test_maybe_flag_for_preemption_does_not_reset_already_flagged_timer(broker_app):
    """Test: a second qualifying waiter does not re-flag (and so cannot reset the grace timer of) an already-flagged holder."""
    leases = {"L1": _make_lease(caller="holder", preemptible=True, priority="P2")}
    first = broker_app._maybe_flag_for_preemption("P2", broker_app.PRIORITY_AGING_WINDOW_SECONDS, leases)
    assert first is not None
    flagged_at_first = leases["L1"]["preempt_flagged_at"]

    second = broker_app._maybe_flag_for_preemption("P0", broker_app.PRIORITY_AGING_WINDOW_SECONDS, leases)
    assert second is None
    assert leases["L1"]["preempt_flagged_at"] == flagged_at_first


# Tests: Incumbency Bonus Startup Assertion

def test_validate_priority_config_accepts_default(broker_app):
    """Test: the default INCUMBENCY_BONUS/TIER_STEP config passes validation."""
    broker_app._validate_priority_config()  # must not raise


def test_validate_priority_config_rejects_bonus_equal_to_tier_step(broker_app, monkeypatch):
    """Test: a bonus equal to a full tier-step is rejected -- it could starve a same-tier waiter forever."""
    monkeypatch.setattr(broker_app, "INCUMBENCY_BONUS", broker_app.TIER_STEP)
    with pytest.raises(AssertionError):
        broker_app._validate_priority_config()


def test_validate_priority_config_rejects_bonus_above_tier_step(broker_app, monkeypatch):
    """Test: a bonus larger than a tier-step is rejected."""
    monkeypatch.setattr(broker_app, "INCUMBENCY_BONUS", broker_app.TIER_STEP * 2)
    with pytest.raises(AssertionError):
        broker_app._validate_priority_config()


def test_validate_priority_config_rejects_negative_bonus(broker_app, monkeypatch):
    """Test: a negative bonus is rejected."""
    monkeypatch.setattr(broker_app, "INCUMBENCY_BONUS", -1.0)
    with pytest.raises(AssertionError):
        broker_app._validate_priority_config()


# Tests: Watchdog Reason Classification (_classify_reclaim_reasons)

def test_classify_reclaim_reasons_healthy_lease_is_empty(broker_app):
    """Test: a fresh, unflagged lease triggers no reclaim reason."""
    now = 1000.0
    lease = _make_lease(acquired_at=now, last_seen=now)
    assert broker_app._classify_reclaim_reasons(lease, now) == []


def test_classify_reclaim_reasons_heartbeat_stale(broker_app):
    """Test: last_seen older than LEASE_TTL_SECONDS triggers heartbeat_stale, independent of age."""
    now = 1000.0
    lease = _make_lease(acquired_at=now - 5, last_seen=now - broker_app.LEASE_TTL_SECONDS - 1)
    assert broker_app._classify_reclaim_reasons(lease, now) == ["heartbeat_stale"]


def test_classify_reclaim_reasons_lease_age(broker_app):
    """An old non-preemptible lease triggers lease_age despite a fresh heartbeat."""
    now = 10000.0
    lease = _make_lease(acquired_at=now - broker_app.MAX_LEASE_AGE_SECONDS - 1, last_seen=now - 1)
    assert broker_app._classify_reclaim_reasons(lease, now) == ["lease_age"]


def test_classify_reclaim_reasons_old_preemptible_fresh_heartbeat_is_not_reclaimed(broker_app):
    now = 10000.0
    lease = _make_lease(
        acquired_at=now - broker_app.MAX_LEASE_AGE_SECONDS - 1,
        last_seen=now - 1,
        preemptible=True,
    )
    assert broker_app._classify_reclaim_reasons(lease, now) == []


def test_classify_reclaim_reasons_old_preemptible_stale_heartbeat_is_reclaimed(broker_app):
    now = 10000.0
    lease = _make_lease(
        acquired_at=now - broker_app.MAX_LEASE_AGE_SECONDS - 1,
        last_seen=now - broker_app.LEASE_TTL_SECONDS - 1,
        preemptible=True,
    )
    assert set(broker_app._classify_reclaim_reasons(lease, now)) == {
        "heartbeat_stale", "lease_age",
    }

def test_classify_reclaim_reasons_preempt_grace_expired(broker_app):
    """Test: a flagged lease past PREEMPT_GRACE_SECONDS without self-eviction triggers preempt_grace_expired."""
    now = 1000.0
    lease = _make_lease(
        acquired_at=now - 5, last_seen=now - 1,
        preemptible=True, preempt_flagged_at=now - broker_app.PREEMPT_GRACE_SECONDS - 1,
    )
    assert broker_app._classify_reclaim_reasons(lease, now) == ["preempt_grace_expired"]


def test_classify_reclaim_reasons_preempt_flagged_within_grace_is_not_reclaimed(broker_app):
    """Test: a flagged lease still inside its grace period is not yet a reclaim candidate."""
    now = 1000.0
    lease = _make_lease(
        acquired_at=now - 5, last_seen=now - 1,
        preemptible=True, preempt_flagged_at=now - 1,
    )
    assert broker_app._classify_reclaim_reasons(lease, now) == []


def test_classify_reclaim_reasons_both_stale_and_aged_independently_trigger(broker_app):
    """Test: heartbeat staleness and lease age are independent triggers and both can fire on the same lease."""
    now = 100000.0
    lease = _make_lease(
        acquired_at=now - broker_app.MAX_LEASE_AGE_SECONDS - 1,
        last_seen=now - broker_app.LEASE_TTL_SECONDS - 1,
    )
    reasons = broker_app._classify_reclaim_reasons(lease, now)
    assert set(reasons) == {"heartbeat_stale", "lease_age"}


# Tests: Watchdog Kill Sequence (_reclaim_broken_lease), PID-Reuse Avoidance

@pytest.mark.asyncio
async def test_reclaim_broken_lease_no_pid_is_ledger_only_delete(broker_app):
    """Test: a lease with no reported pid is removed from the ledger without any kill attempt."""
    broker_app._leases["L1"] = _make_lease(caller="c1", pid=None)
    broker_app._caller_to_lease["c1"] = "L1"

    with patch.object(broker_app, "_list_gpu_compute_processes") as mock_list, \
         patch.object(broker_app, "_kill_process") as mock_kill:
        await broker_app._reclaim_broken_lease("L1", dict(broker_app._leases["L1"]), ["heartbeat_stale"])
        mock_list.assert_not_called()
        mock_kill.assert_not_called()

    assert "L1" not in broker_app._leases
    assert "c1" not in broker_app._caller_to_lease
    assert "c1" in broker_app._pending_notifications



@pytest.mark.asyncio
async def test_reclaimed_zero_size_proxy_lease_does_not_queue_watchdog_notice(broker_app):
    """A zero-size request holds no GPU memory, so reclaim must not queue a GPU warning."""
    caller = "oaie-proxy:ollama:one-request"
    broker_app._leases["L0"] = _make_lease(caller=caller, mb=0.0, pid=None)
    broker_app._caller_to_lease[caller] = "L0"

    await broker_app._reclaim_broken_lease(
        "L0", dict(broker_app._leases["L0"]), ["heartbeat_stale"],
    )

    assert "L0" not in broker_app._leases
    assert caller not in broker_app._pending_notifications

@pytest.mark.asyncio
async def test_reclaim_broken_lease_pid_not_in_compute_apps_sends_no_signal(broker_app, caplog):
    """Test: PID-reuse-avoidance cross-check -- a reported pid absent from a fresh
    compute-apps listing is treated as already-exited: ledger-only delete, no signal sent."""
    broker_app._leases["L1"] = _make_lease(caller="c1", pid=4242)
    broker_app._caller_to_lease["c1"] = "L1"

    with patch.object(broker_app, "_list_gpu_compute_processes", return_value=[]) as mock_list, \
         patch.object(broker_app, "_kill_process") as mock_kill:
        with caplog.at_level(logging.WARNING, logger="server"):
            await broker_app._reclaim_broken_lease("L1", dict(broker_app._leases["L1"]), ["lease_age"])
        mock_list.assert_called_once()
        mock_kill.assert_not_called()

    assert "L1" not in broker_app._leases
    assert "c1" in broker_app._pending_notifications
    assert any("no container GPU match" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_reclaim_broken_lease_pid_found_kills_and_verifies(
    broker_app, caplog, monkeypatch, tmp_path,
):
    """Test: a pid confirmed as a live GPU compute process is signaled, and its
    disappearance afterward is logged at WARNING (not ERROR)."""
    live_process, entries = _container_gpu()
    _fake_proc(broker_app, monkeypatch, tmp_path, entries)
    broker_app._leases["L1"] = _make_lease(caller="c1", pid=4242)
    broker_app._caller_to_lease["c1"] = "L1"

    with patch.object(broker_app, "_list_gpu_compute_processes", side_effect=[[live_process], []]), \
         patch.object(broker_app, "_kill_process", return_value="already_exited") as mock_kill:
        with caplog.at_level(logging.WARNING, logger="server"):
            await broker_app._reclaim_broken_lease("L1", dict(broker_app._leases["L1"]), ["heartbeat_stale"])
        mock_kill.assert_called_once_with(4242, "stuck_proc")

    assert "L1" not in broker_app._leases
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


@pytest.mark.asyncio
async def test_reclaim_broken_lease_still_present_after_kill_logs_error(
    broker_app, caplog, monkeypatch, tmp_path,
):
    """Test: memory still not free after the kill sequence is logged at ERROR, and the
    ledger entry is still removed rather than left stuck forever."""
    live_process, entries = _container_gpu()
    _fake_proc(broker_app, monkeypatch, tmp_path, entries)
    broker_app._leases["L1"] = _make_lease(caller="c1", pid=4242)
    broker_app._caller_to_lease["c1"] = "L1"

    with patch.object(broker_app, "_list_gpu_compute_processes", side_effect=[[live_process], [live_process]]), \
         patch.object(broker_app, "_kill_process", return_value="already_exited"):
        with caplog.at_level(logging.WARNING, logger="server"):
            await broker_app._reclaim_broken_lease("L1", dict(broker_app._leases["L1"]), ["lease_age"])

    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert "L1" not in broker_app._leases


@pytest.mark.asyncio
async def test_reclaim_broken_lease_verify_query_failure_logs_unverified(
    broker_app, caplog, monkeypatch, tmp_path,
):
    """Test: a failed post-kill process query logs could not verify, not confirmed cleared."""
    live_process, entries = _container_gpu()
    _fake_proc(broker_app, monkeypatch, tmp_path, entries)
    broker_app._leases["L1"] = _make_lease(caller="c1", pid=4242)
    broker_app._caller_to_lease["c1"] = "L1"

    with patch.object(
        broker_app,
        "_list_gpu_compute_processes",
        side_effect=[[live_process], RuntimeError("nvidia-smi down")],
    ), patch.object(broker_app, "_kill_process", return_value="already_exited"):
        with caplog.at_level(logging.WARNING, logger="server"):
            await broker_app._reclaim_broken_lease(
                "L1", dict(broker_app._leases["L1"]), ["heartbeat_stale"],
            )

    messages = [r.getMessage() for r in caplog.records]
    assert any("could not verify" in m for m in messages)
    assert not any("confirmed cleared" in m for m in messages)
    assert any(
        r.levelno == logging.WARNING and "could not verify" in r.getMessage()
        for r in caplog.records
    )
    assert "L1" not in broker_app._leases



@pytest.mark.asyncio
async def test_reclaim_broken_lease_leaves_notification_naming_the_reason(broker_app):
    """Test: the queued notification text reflects the reason(s) that triggered the reclaim."""
    broker_app._leases["L1"] = _make_lease(caller="c1", pid=None)
    broker_app._caller_to_lease["c1"] = "L1"

    await broker_app._reclaim_broken_lease("L1", dict(broker_app._leases["L1"]), ["lease_age"])

    notice, _created = broker_app._pending_notifications["c1"]
    assert "lease-age" in notice or "minute" in notice
    assert "force-terminated" in notice


@pytest.mark.asyncio
async def test_reclaim_broken_lease_skips_already_released_lease(broker_app):
    """Test: if the caller released the lease before the reclaim runs (race window), nothing is mutated."""
    # lease already removed from the live ledger -- only the snapshot remains
    snapshot = _make_lease(caller="c1", pid=None)

    await broker_app._reclaim_broken_lease("L1", snapshot, ["heartbeat_stale"])

    assert "c1" not in broker_app._pending_notifications
    assert "L1" not in broker_app._leases


@pytest.mark.asyncio
async def test_reclaim_registered_host_pid_of_container_gpu_is_signaled(
    broker_app, monkeypatch, tmp_path,
):
    """Registered host pid of a container GPU process is signaled."""
    live_process, entries = _container_gpu()
    _fake_proc(broker_app, monkeypatch, tmp_path, entries)
    broker_app._leases["L1"] = _make_lease(caller="c1", pid=4242)
    broker_app._caller_to_lease["c1"] = "L1"

    with patch.object(broker_app, "_list_gpu_compute_processes", side_effect=[[live_process], []]), \
         patch.object(broker_app, "_kill_process", return_value="already_exited") as mock_kill:
        await broker_app._reclaim_broken_lease("L1", dict(broker_app._leases["L1"]), ["heartbeat_stale"])

    mock_kill.assert_called_once_with(4242, "stuck_proc")
    assert "L1" not in broker_app._leases


@pytest.mark.asyncio
async def test_reclaim_namespace_pid_resolves_to_host_pid_and_signals(
    broker_app, monkeypatch, tmp_path,
):
    """FreeToken case: innermost NSpid matches, host pid is signaled."""
    live_process, entries = _container_gpu(pid=4242, nspid=7)
    _fake_proc(broker_app, monkeypatch, tmp_path, entries)
    broker_app._leases["L1"] = _make_lease(caller="c1", pid=7)
    broker_app._caller_to_lease["c1"] = "L1"

    with patch.object(broker_app, "_list_gpu_compute_processes", side_effect=[[live_process], []]), \
         patch.object(broker_app, "_kill_process", return_value="already_exited") as mock_kill:
        await broker_app._reclaim_broken_lease("L1", dict(broker_app._leases["L1"]), ["heartbeat_stale"])

    mock_kill.assert_called_once_with(4242, "stuck_proc")
    assert "L1" not in broker_app._leases


@pytest.mark.asyncio
async def test_reclaim_host_gpu_process_is_never_signaled(
    broker_app, caplog, monkeypatch, tmp_path,
):
    """Registered pid matching a non-container GPU process is ledger-only."""
    live_process = {"pid": 4242, "process_name": "blender", "used_mb": 2048.0}
    _fake_proc(broker_app, monkeypatch, tmp_path, {
        4242: {"cgroup": _HOST_CGROUP, "nspids": [4242], "comm": "blender"},
    })
    broker_app._leases["L1"] = _make_lease(caller="c1", pid=4242)
    broker_app._caller_to_lease["c1"] = "L1"

    with patch.object(broker_app, "_list_gpu_compute_processes", return_value=[live_process]), \
         patch.object(broker_app, "_kill_process") as mock_kill:
        with caplog.at_level(logging.WARNING, logger="server"):
            await broker_app._reclaim_broken_lease("L1", dict(broker_app._leases["L1"]), ["heartbeat_stale"])

    mock_kill.assert_not_called()
    assert "L1" not in broker_app._leases
    assert any("refusing to signal host process" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_reclaim_ambiguous_nspid_is_ledger_only(
    broker_app, caplog, monkeypatch, tmp_path,
):
    """Two container GPU processes sharing innermost NSpid: no signal."""
    procs = [
        {"pid": 1000, "process_name": "a", "used_mb": 100.0},
        {"pid": 2000, "process_name": "b", "used_mb": 100.0},
    ]
    _fake_proc(broker_app, monkeypatch, tmp_path, {
        1000: {"cgroup": _CONTAINER_CGROUP, "nspids": [1000, 7], "comm": "a"},
        2000: {"cgroup": _CONTAINER_CGROUP, "nspids": [2000, 7], "comm": "b"},
    })
    broker_app._leases["L1"] = _make_lease(caller="c1", pid=7)
    broker_app._caller_to_lease["c1"] = "L1"

    with patch.object(broker_app, "_list_gpu_compute_processes", return_value=procs), \
         patch.object(broker_app, "_kill_process") as mock_kill:
        with caplog.at_level(logging.WARNING, logger="server"):
            await broker_app._reclaim_broken_lease("L1", dict(broker_app._leases["L1"]), ["heartbeat_stale"])

    mock_kill.assert_not_called()
    assert "L1" not in broker_app._leases
    assert any("ambiguous" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_reclaim_heartbeat_between_snapshot_and_kill_keeps_lease(
    broker_app, monkeypatch, tmp_path,
):
    """A heartbeat after the sweep snapshot aborts the kill and keeps the lease."""
    live_process, entries = _container_gpu()
    _fake_proc(broker_app, monkeypatch, tmp_path, entries)
    now = time.monotonic()
    broker_app._leases["L1"] = _make_lease(
        caller="c1", pid=4242, acquired_at=now,
        last_seen=now - broker_app.LEASE_TTL_SECONDS - 1,
    )
    broker_app._caller_to_lease["c1"] = "L1"

    def list_and_heartbeat():
        broker_app._leases["L1"]["last_seen"] = time.monotonic()
        return [live_process]

    with patch.object(broker_app, "_list_gpu_compute_processes", side_effect=list_and_heartbeat), \
         patch.object(broker_app, "_kill_process") as mock_kill:
        await broker_app._reclaim_broken_lease("L1", dict(broker_app._leases["L1"]), ["heartbeat_stale"])

    mock_kill.assert_not_called()
    assert "L1" in broker_app._leases
    assert broker_app._caller_to_lease["c1"] == "L1"


@pytest.mark.asyncio
async def test_reclaim_release_between_snapshot_and_kill_sends_no_signal(
    broker_app, monkeypatch, tmp_path,
):
    """A release after the sweep snapshot aborts the kill."""
    live_process, entries = _container_gpu()
    _fake_proc(broker_app, monkeypatch, tmp_path, entries)
    now = time.monotonic()
    broker_app._leases["L1"] = _make_lease(
        caller="c1", pid=4242, acquired_at=now,
        last_seen=now - broker_app.LEASE_TTL_SECONDS - 1,
    )
    broker_app._caller_to_lease["c1"] = "L1"

    def list_and_release():
        broker_app._leases.pop("L1", None)
        broker_app._caller_to_lease.pop("c1", None)
        return [live_process]

    with patch.object(broker_app, "_list_gpu_compute_processes", side_effect=list_and_release), \
         patch.object(broker_app, "_kill_process") as mock_kill:
        await broker_app._reclaim_broken_lease(
            "L1",
            {"caller": "c1", "mb": 1000.0, "acquired_at": now, "last_seen": now, "pid": 4242},
            ["heartbeat_stale"],
        )

    mock_kill.assert_not_called()
    assert "L1" not in broker_app._leases
    assert "c1" not in broker_app._pending_notifications


@pytest.mark.asyncio
async def test_kill_grace_skips_sigkill_when_start_time_changes(
    broker_app, monkeypatch, tmp_path,
):
    """Start time change between SIGTERM and SIGKILL skips SIGKILL."""
    proc_root = _fake_proc(broker_app, monkeypatch, tmp_path, {
        4242: {
            "cgroup": _CONTAINER_CGROUP,
            "nspids": [4242, 7],
            "start_time": 100,
            "comm": "reused",
        },
    })
    monkeypatch.setattr(broker_app, "ADMIN_KILL_GRACE_SECONDS", 0.0)
    calls = []

    def fake_kill(pid, name, sig=signal.SIGTERM):
        calls.append(sig)
        if sig == signal.SIGTERM:
            _write_proc_pid(
                proc_root, 4242,
                cgroup=_CONTAINER_CGROUP, nspids=[4242, 7],
                start_time=200, comm="reused",
            )
        return "terminated"

    with patch.object(broker_app, "_kill_process", fake_kill), \
         patch.object(broker_app, "_process_alive", return_value=True):
        result = await broker_app._kill_targets_with_grace([
            {"pid": 4242, "process_name": "reused", "start_time": 100},
        ])

    assert calls == [signal.SIGTERM]
    assert result[0]["signal_result"] == "terminated"


@pytest.mark.asyncio
async def test_kill_grace_returns_early_when_process_exits(broker_app):
    """A process that exits 0.2 s after SIGTERM does not wait the full grace."""
    term_at = {}

    def fake_kill(pid, name, sig=signal.SIGTERM):
        term_at["t"] = time.monotonic()
        return "terminated"

    def fake_alive(pid):
        return time.monotonic() - term_at["t"] < 0.2

    with patch.object(broker_app, "_kill_process", fake_kill), \
         patch.object(broker_app, "_process_alive", fake_alive):
        t0 = time.monotonic()
        result = await broker_app._kill_targets_with_grace([
            {"pid": 4242, "process_name": "gone", "start_time": 1},
        ])
        elapsed = time.monotonic() - t0

    assert elapsed < 1.0
    assert broker_app.ADMIN_KILL_GRACE_SECONDS >= 3.0
    assert result[0]["signal_result"] == "terminated"


def test_proc_start_time_parses_after_last_paren(broker_app, monkeypatch, tmp_path):
    """Field 22 is read after the last ')' so comm may contain spaces and parens."""
    proc_root = tmp_path / "proc"
    pid_dir = proc_root / "99"
    pid_dir.mkdir(parents=True)
    after = ["S"] + ["0"] * 18 + ["12345"]
    (pid_dir / "stat").write_text("99 (foo (bar) baz) " + " ".join(after) + "\n")
    monkeypatch.setattr(broker_app, "_PROC_ROOT", str(proc_root))
    assert broker_app._proc_start_time(99) == 12345


# Tests: One-Shot Notification Delivery Through acquire()

def test_acquire_grant_delivers_and_clears_pending_notification(broker_app):
    """Test: a queued watchdog notice is delivered exactly once on the caller's next grant, then cleared."""
    from fastapi.testclient import TestClient

    broker_app._pending_notifications["test_client"] = (
        "your previous lease was force-terminated...",
        time.monotonic(),
    )
    client = TestClient(broker_app.app)

    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0):
        resp1 = client.post("/acquire", json={"caller": "test_client", "mb": 500.0, "timeout_s": 5.0})
        assert resp1.status_code == 200
        assert resp1.json()["warning"] == "your previous lease was force-terminated..."
        assert "test_client" not in broker_app._pending_notifications

        # release and acquire again -- the notice must not resurface.
        lease_id = resp1.json()["lease_id"]
        client.post("/release", json={"lease_id": lease_id})
        resp2 = client.post("/acquire", json={"caller": "test_client", "mb": 500.0, "timeout_s": 5.0})
        assert resp2.status_code == 200
        assert "warning" not in resp2.json()


def test_acquire_grant_without_notification_has_no_warning_field(broker_app):
    """Test: a normal grant with no pending notice carries no warning field at all."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0):
        resp = client.post("/acquire", json={"caller": "clean_caller", "mb": 500.0, "timeout_s": 5.0})
        assert resp.status_code == 200
        assert "warning" not in resp.json()


# Tests: Priority Tier Validation and Lease Fields on Grant

def test_acquire_rejects_unknown_priority_tier(broker_app):
    """Test: an invalid priority tier is rejected with 400, not queued."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    resp = client.post("/acquire", json={"caller": "c", "mb": 100.0, "priority": "P9"})
    assert resp.status_code == 400


def test_acquire_grant_stores_priority_preemptible_and_pid_on_lease(broker_app):
    """Test: a granted lease records the caller's declared priority/preemptible/pid."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0):
        resp = client.post("/acquire", json={
            "caller": "c", "mb": 100.0, "timeout_s": 5.0,
            "priority": "P0", "preemptible": True, "pid": 9999,
        })
        assert resp.status_code == 200
        lease = broker_app._leases[resp.json()["lease_id"]]
        assert lease["priority"] == "P0"
        assert lease["preemptible"] is True
        assert lease["pid"] == 9999
        assert lease["preempt_flagged_at"] is None


def test_deduplicated_acquire_registers_pid_on_existing_lease(broker_app):
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0):
        first = client.post("/acquire", json={
            "caller": "freetoken", "mb": 100.0, "timeout_s": 5.0,
            "preemptible": True,
        })
        lease_id = first.json()["lease_id"]
        assert broker_app._leases[lease_id]["pid"] is None

        second = client.post("/acquire", json={
            "caller": "freetoken", "mb": 100.0, "timeout_s": 5.0,
            "preemptible": True, "pid": 4242,
        })

    assert second.status_code == 200
    assert second.json()["lease_id"] == lease_id
    assert broker_app._leases[lease_id]["pid"] == 4242


def test_acquire_defaults_priority_p2_and_not_preemptible(broker_app):
    """Test: a caller that omits the new fields gets today's implicit behavior -- normal tier, not preemptible."""
    from fastapi.testclient import TestClient

    client = TestClient(broker_app.app)
    with patch.object(broker_app, "_nvidia_smi_query", return_value=2048.0):
        resp = client.post("/acquire", json={"caller": "legacy_client", "mb": 100.0, "timeout_s": 5.0})
        assert resp.status_code == 200
        lease = broker_app._leases[resp.json()["lease_id"]]
        assert lease["priority"] == "P2"
        assert lease["preemptible"] is False
        assert lease["pid"] is None


# Tests: heartbeat() preempt/reason Fields

def test_heartbeat_reports_preempt_false_when_not_flagged(broker_app):
    """Test: an ordinary lease's heartbeat reports preempt=False and an empty reason."""
    from fastapi.testclient import TestClient

    broker_app._leases["L1"] = _make_lease(caller="c1")
    client = TestClient(broker_app.app)
    resp = client.post("/heartbeat", json={"lease_id": "L1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["preempt"] is False
    assert body["reason"] == ""


def test_heartbeat_reports_preempt_true_with_generic_reason_when_flagged(broker_app):
    """Test: a flagged lease's heartbeat reports preempt=True with the fixed, generic reason text -- never naming the waiter."""
    from fastapi.testclient import TestClient

    broker_app._leases["L1"] = _make_lease(caller="c1", preemptible=True, preempt_flagged_at=time.monotonic())
    client = TestClient(broker_app.app)
    resp = client.post("/heartbeat", json={"lease_id": "L1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["preempt"] is True
    assert body["reason"] == broker_app.PREEMPT_REASON_TEXT
    assert "c1" not in body["reason"]


def test_heartbeat_preempt_flag_persists_across_calls(broker_app):
    """Test: preempt=True stays True on every subsequent heartbeat -- there is no un-flagging."""
    from fastapi.testclient import TestClient

    broker_app._leases["L1"] = _make_lease(caller="c1", preemptible=True, preempt_flagged_at=time.monotonic())
    client = TestClient(broker_app.app)
    for _ in range(3):
        resp = client.post("/heartbeat", json={"lease_id": "L1"})
        assert resp.json()["preempt"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# Tests: fairness age across consumer backoff retries

async def _two_short_acquires(srv, monkeypatch, memory_seconds):
    """Holder P2 preemptible blocks a waiter; waiter acquires twice with a pause between."""
    import asyncio
    from unittest.mock import AsyncMock

    from fastapi import HTTPException

    monkeypatch.setattr(srv, "_live_used_mb", AsyncMock(return_value=1500.0))
    monkeypatch.setattr(srv, "_refresh_proton_state", AsyncMock())
    monkeypatch.setattr(srv, "ACQUIRE_POLL_MIN_SECONDS", 0.01)
    monkeypatch.setattr(srv, "ACQUIRE_POLL_MAX_SECONDS", 0.01)
    monkeypatch.setattr(srv, "PRIORITY_AGING_WINDOW_SECONDS", 0.2)
    monkeypatch.setattr(srv, "WAITER_MEMORY_SECONDS", memory_seconds)
    srv._ollama_mb = 0.0
    # budget 12288 - 1536 = 10752; 3000 + 8382 does not fit.
    srv._leases["L"] = _make_lease(caller="laya", mb=3000.0, preemptible=True,
                                   acquired_at=time.monotonic(), last_seen=time.monotonic())
    srv._caller_to_lease["laya"] = "L"
    errors = []
    for _ in range(2):
        try:
            await srv.acquire(srv.AcquireRequest(caller="freetoken", mb=8382.0, timeout_s=0.06,
                                                 preemptible=True))
        except HTTPException as exc:
            errors.append(exc)
        await asyncio.sleep(0.06)
    return errors


@pytest.mark.asyncio
async def test_wait_age_survives_backoff_retry_and_preempts_holder(broker_app, monkeypatch):
    """Test: two 0.06 s attempts add up past the 0.1 s same-tier threshold and flag the holder."""
    errors = await _two_short_acquires(broker_app, monkeypatch, memory_seconds=120.0)
    assert [e.status_code for e in errors] == [504, 504]
    assert errors[0].headers["Retry-After"] == str(int(broker_app.ACQUIRE_POLL_MAX_SECONDS))
    assert broker_app._leases["L"]["preempt_flagged_at"] is not None


@pytest.mark.asyncio
async def test_wait_age_resets_after_memory_window(broker_app, monkeypatch):
    """Test: a retry after the memory window starts at zero age; no attempt alone reaches the threshold."""
    errors = await _two_short_acquires(broker_app, monkeypatch, memory_seconds=0.0)
    assert [e.status_code for e in errors] == [504, 504]
    assert broker_app._leases["L"]["preempt_flagged_at"] is None


# Tests: Sweep loop resilience (_sweep_expired_leases)

async def _n_sweeps(srv, ticks):
    """Run ticks of _sweep_expired_leases, then end the loop via CancelledError on sleep."""
    side = [None] * ticks + [asyncio.CancelledError]
    with patch.object(srv.asyncio, "sleep", AsyncMock(side_effect=side)):
        with pytest.raises(asyncio.CancelledError):
            await srv._sweep_expired_leases()


@pytest.mark.asyncio
async def test_sweep_continues_after_one_reclaim_fails(broker_app, caplog):
    """Two stale leases: first reclaim raises, second still reclaims in the same pass."""
    now = time.monotonic()
    stale = now - broker_app.LEASE_TTL_SECONDS - 1
    broker_app._leases["L-fail"] = _make_lease(caller="c-fail", last_seen=stale, acquired_at=stale)
    broker_app._leases["L-ok"] = _make_lease(caller="c-ok", last_seen=stale, acquired_at=stale)
    broker_app._caller_to_lease["c-fail"] = "L-fail"
    broker_app._caller_to_lease["c-ok"] = "L-ok"

    orig = broker_app._reclaim_broken_lease

    async def flaky(lease_id, lease, reasons):
        if lease_id == "L-fail":
            raise RuntimeError("nvidia-smi failed")
        await orig(lease_id, lease, reasons)

    with caplog.at_level(logging.ERROR, logger="server"):
        with patch.object(broker_app, "_reclaim_broken_lease", flaky):
            await _n_sweeps(broker_app, 1)

    assert "L-fail" in broker_app._leases
    assert "L-ok" not in broker_app._leases
    assert any("L-fail" in r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)


@pytest.mark.asyncio
async def test_sweep_survives_classify_error_and_reclaims_same_pass(broker_app, caplog):
    """One lease whose classify always raises does not stop a second stale lease in the same pass."""
    now = time.monotonic()
    stale = now - broker_app.LEASE_TTL_SECONDS - 1
    broker_app._leases["L-boom"] = _make_lease(caller="c-boom", last_seen=stale, acquired_at=stale)
    broker_app._leases["L-ok"] = _make_lease(caller="c-ok", last_seen=stale, acquired_at=stale)
    broker_app._caller_to_lease["c-boom"] = "L-boom"
    broker_app._caller_to_lease["c-ok"] = "L-ok"

    orig = broker_app._classify_reclaim_reasons

    def flaky(lease, now_ts):
        if lease.get("caller") == "c-boom":
            raise RuntimeError("classify boom")
        return orig(lease, now_ts)

    with caplog.at_level(logging.ERROR, logger="server"):
        with patch.object(broker_app, "_classify_reclaim_reasons", flaky):
            await _n_sweeps(broker_app, 1)

    assert "L-boom" in broker_app._leases
    assert "L-ok" not in broker_app._leases
    assert any("L-boom" in r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)


@pytest.mark.asyncio
async def test_sweep_cancelled_error_from_sleep_ends_task(broker_app):
    """CancelledError from sleep still ends the sweep task."""
    await _n_sweeps(broker_app, 0)
