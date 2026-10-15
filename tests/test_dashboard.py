"""
Author: Richard Baldwin
Date:   2026
Email:  eyeclept@pm.me

Description:

Unit tests for tools/dashboard.py. Covers snapshot log read/write, lease-age
tracking, sparkline/age formatting, and a static read-only guarantee check --
all against mocked HTTP responses, no live broker required.
"""
import re
import inspect
import json
import logging
import os
import sys

import httpx
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import dashboard  # noqa: E402


# Tests: Read-only guarantee (static source scan, see dashboard.py module docstring)

def _source_without_docstrings():
    """Module source with every triple-quoted docstring span stripped, so the
    guarantee scans below check only executable code, not the prose in this
    file's own docstrings that names the forbidden endpoints/token by way of
    explaining the guarantee."""
    source = inspect.getsource(dashboard)
    return re.sub(r'"""[\s\S]*?"""', "", source)


def test_dashboard_never_calls_post():
    """Test: dashboard.py code contains no `.post(` call."""
    assert ".post(" not in _source_without_docstrings()


def test_dashboard_never_references_mutating_endpoints():
    """Test: dashboard.py code never mentions a mutating broker endpoint."""
    code = _source_without_docstrings()
    for endpoint in ("/acquire", "/release", "/heartbeat", "/admin/clear"):
        assert endpoint not in code, f"dashboard.py must not reference {endpoint}"


def test_dashboard_never_references_admin_token():
    """Test: dashboard.py code never reads GPU_BROKER_ADMIN_TOKEN."""
    assert "ADMIN_TOKEN" not in _source_without_docstrings()


# Tests: Snapshot Log Read/Write

def test_append_and_read_history_round_trip(tmp_path):
    """Test: appended snapshots read back in order via read_history."""
    log_path = str(tmp_path / "history.jsonl")
    snap1 = {"ts": "t1", "reserved_mb": 100.0}
    snap2 = {"ts": "t2", "reserved_mb": 200.0}

    dashboard.append_snapshot(log_path, snap1)
    dashboard.append_snapshot(log_path, snap2)

    rows = dashboard.read_history(log_path)
    assert rows == [snap1, snap2]


def test_read_history_missing_file_returns_empty(tmp_path):
    """Test: read_history on a nonexistent log returns []."""
    assert dashboard.read_history(str(tmp_path / "nope.jsonl")) == []


def test_read_history_respects_limit(tmp_path):
    """Test: read_history(limit=N) returns only the most recent N rows."""
    log_path = str(tmp_path / "history.jsonl")
    for i in range(5):
        dashboard.append_snapshot(log_path, {"ts": f"t{i}", "i": i})

    rows = dashboard.read_history(log_path, limit=2)
    assert [r["i"] for r in rows] == [3, 4]


def test_read_history_skips_unparsable_lines(tmp_path):
    """Test: a corrupt line is skipped, not raised, and valid rows still return."""
    log_path = tmp_path / "history.jsonl"
    log_path.write_text('{"ts": "t0", "i": 0}\nnot json\n{"ts": "t1", "i": 1}\n')

    rows = dashboard.read_history(str(log_path))
    assert [r["i"] for r in rows] == [0, 1]


def test_append_snapshot_trims_to_retention_cap(tmp_path):
    """Test: the log is trimmed once it exceeds max_samples by more than one batch."""
    log_path = str(tmp_path / "history.jsonl")
    max_samples = 10
    # trim_batch = max(1, 10 // 10) = 1, so trimming triggers once len > 11
    for i in range(15):
        dashboard.append_snapshot(log_path, {"i": i}, max_samples=max_samples)

    with open(log_path) as f:
        lines = f.readlines()
    assert len(lines) <= max_samples + 1
    # the most recent sample must always survive trimming
    assert json.loads(lines[-1])["i"] == 14


def test_append_below_threshold_does_not_read_file(tmp_path, monkeypatch):
    """Test: appending N rows below the trim threshold never reads the log."""
    log_path = str(tmp_path / "history.jsonl")
    read_calls = []
    real_open = open

    def counting_open(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and "r" in mode and "a" not in mode and "w" not in mode:
            read_calls.append((file, mode))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(dashboard, "open", counting_open, raising=False)
    max_samples = 10
    count = 0
    for i in range(11):
        count = dashboard.append_snapshot(
            log_path, {"i": i}, max_samples=max_samples, row_count=count
        )
    assert count == 11
    assert read_calls == []


def test_append_crossing_threshold_trims_to_cap_and_keeps_newest(tmp_path):
    """Test: crossing the trim threshold rewrites to max_samples and keeps the newest row."""
    log_path = str(tmp_path / "history.jsonl")
    max_samples = 10
    count = 0
    for i in range(12):
        count = dashboard.append_snapshot(
            log_path, {"i": i}, max_samples=max_samples, row_count=count
        )
    with open(log_path) as f:
        lines = f.readlines()
    assert len(lines) == max_samples
    assert json.loads(lines[0])["i"] == 2
    assert json.loads(lines[-1])["i"] == 11
    assert count == max_samples


def test_trim_does_not_follow_planted_tmp_symlink(tmp_path):
    """Test: trim writes a unique temp file and does not follow <log>.tmp."""
    log_path = tmp_path / "history.jsonl"
    planted_target = tmp_path / "planted_target"
    planted_target.write_text("untouched\n")
    (tmp_path / "history.jsonl.tmp").symlink_to(planted_target)

    max_samples = 10
    count = 0
    for i in range(12):
        count = dashboard.append_snapshot(
            str(log_path), {"i": i}, max_samples=max_samples, row_count=count
        )

    assert planted_target.read_text() == "untouched\n"
    lines = log_path.read_text().splitlines()
    assert len(lines) == max_samples
    assert json.loads(lines[-1])["i"] == 11
    assert count == max_samples

# Tests: snapshot_from_status

def test_snapshot_from_status_computes_free_mb():
    """Test: free_mb = budget_mb - reserved_mb - ollama_mb, matching server.py's admission gate."""
    payload = {
        "budget_mb": 7418.0,
        "gpu_total_mb": 12282.0,
        "safety_margin_mb": 4864,
        "reserved_mb": 1000.0,
        "ollama_mb": 2000.0,
        "nvidia_smi_used_mb": 5000.0,
        "leases": {"lease-1": {}, "lease-2": {}},
    }
    snap = dashboard.snapshot_from_status(payload, healthy=True)
    assert snap["free_mb"] == pytest.approx(7418.0 - 1000.0 - 2000.0)
    assert snap["healthy"] is True
    assert snap["lease_count"] == 2
    assert "ts" in snap


def test_snapshot_from_status_keeps_none_nvidia_smi_used_mb():
    """Test: snapshot_from_status keeps nvidia_smi_used_mb=None from /status."""
    payload = {
        "budget_mb": 7418.0,
        "gpu_total_mb": 12282.0,
        "safety_margin_mb": 4864,
        "reserved_mb": 1000.0,
        "ollama_mb": 2000.0,
        "nvidia_smi_used_mb": None,
        "leases": {},
    }
    snap = dashboard.snapshot_from_status(payload, healthy=True)
    assert snap["nvidia_smi_used_mb"] is None


# Tests: Lease Age Tracking

def test_update_lease_ages_new_lease_starts_at_zero():
    """Test: a lease observed for the first time reports age 0.0."""
    tracking = {}
    leases = {"lease-1": {"caller": "c", "mb": 100.0, "last_seen": 50.0}}
    ages = dashboard.update_lease_ages(tracking, leases, now_wall=1000.0)
    assert ages["lease-1"] == 0.0


def test_update_lease_ages_grows_without_new_heartbeat():
    """Test: age grows with elapsed wall time when last_seen does not advance."""
    tracking = {}
    leases = {"lease-1": {"caller": "c", "mb": 100.0, "last_seen": 50.0}}
    dashboard.update_lease_ages(tracking, leases, now_wall=1000.0)

    ages = dashboard.update_lease_ages(tracking, leases, now_wall=1010.0)
    assert ages["lease-1"] == pytest.approx(10.0)


def test_update_lease_ages_resets_on_new_heartbeat():
    """Test: age resets to 0 once the broker reports a higher last_seen (a real heartbeat)."""
    tracking = {}
    leases = {"lease-1": {"caller": "c", "mb": 100.0, "last_seen": 50.0}}
    dashboard.update_lease_ages(tracking, leases, now_wall=1000.0)
    dashboard.update_lease_ages(tracking, leases, now_wall=1010.0)

    leases_heartbeat = {"lease-1": {"caller": "c", "mb": 100.0, "last_seen": 60.0}}
    ages = dashboard.update_lease_ages(tracking, leases_heartbeat, now_wall=1015.0)
    assert ages["lease-1"] == 0.0


def test_update_lease_ages_drops_released_leases():
    """Test: a lease_id no longer present in `leases` is dropped from tracking."""
    tracking = {}
    leases = {"lease-1": {"caller": "c", "mb": 100.0, "last_seen": 50.0}}
    dashboard.update_lease_ages(tracking, leases, now_wall=1000.0)

    dashboard.update_lease_ages(tracking, {}, now_wall=1010.0)
    assert tracking == {}


# Tests: Sparkline / Age Formatting

def test_render_sparkline_empty():
    """Test: an empty series renders as an empty string."""
    assert dashboard.render_sparkline([]) == ""


def test_render_sparkline_flat_series_no_division_by_zero():
    """Test: identical values render the same glyph, not a ZeroDivisionError."""
    result = dashboard.render_sparkline([5.0, 5.0, 5.0])
    assert len(result) == 3
    assert len(set(result)) == 1


def test_render_sparkline_length_matches_input():
    """Test: sparkline emits exactly one glyph per input value, low to high spans full glyph range."""
    values = [0.0, 25.0, 50.0, 75.0, 100.0]
    result = dashboard.render_sparkline(values)
    assert len(result) == len(values)
    assert result[0] == dashboard.SPARK_CHARS[0]
    assert result[-1] == dashboard.SPARK_CHARS[-1]


def test_format_age_seconds():
    """Test: sub-minute ages format as whole seconds."""
    assert dashboard.format_age(42.4) == "42s"


def test_format_age_minutes():
    """Test: sub-hour ages format as minutes and seconds."""
    assert dashboard.format_age(125.0) == "2m 5s"


def test_format_age_hours():
    """Test: hour-scale ages format as hours and minutes."""
    assert dashboard.format_age(3725.0) == "1h 2m"


def test_format_age_negative_clamped_to_zero():
    """Test: a negative age (clock skew) clamps to 0s rather than showing a negative duration."""
    assert dashboard.format_age(-5.0) == "0s"


# Tests: HTTP Polling (mocked transport, no live broker)

def _client_with_mock_transport(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_status_returns_parsed_json():
    """Test: fetch_status GETs /status and returns the parsed body."""
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/status"
        return httpx.Response(200, json={"budget_mb": 100.0})

    with _client_with_mock_transport(handler) as client:
        result = dashboard.fetch_status(client, "http://broker.test")
    assert result == {"budget_mb": 100.0}


def test_fetch_status_raises_on_http_error():
    """Test: fetch_status propagates httpx.HTTPError on a non-2xx response."""
    def handler(request):
        return httpx.Response(500)

    with _client_with_mock_transport(handler) as client:
        with pytest.raises(httpx.HTTPError):
            dashboard.fetch_status(client, "http://broker.test")


def test_run_makes_one_http_request_per_tick(tmp_path, monkeypatch):
    """Test: the poll loop GETs /status once per tick and never GETs /health."""
    requests = []
    status_body = {
        "budget_mb": 7418.0,
        "gpu_total_mb": 12282.0,
        "safety_margin_mb": 4864,
        "reserved_mb": 1000.0,
        "ollama_mb": 2000.0,
        "nvidia_smi_used_mb": 5000.0,
        "leases": {},
    }

    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(200, json=status_body)

    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    class FakeLive:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, *args, **kwargs):
            pass

    ticks = {"n": 0}

    def fake_sleep(_seconds):
        ticks["n"] += 1
        if ticks["n"] >= 2:
            raise RuntimeError("stop-loop")

    monkeypatch.setattr(dashboard.httpx, "Client", fake_client)
    monkeypatch.setattr(dashboard, "Live", FakeLive)
    monkeypatch.setattr(dashboard.time, "sleep", fake_sleep)

    args = dashboard.parse_args([
        "--url", "http://broker.test",
        "--interval", "0.01",
        "--log-path", str(tmp_path / "history.jsonl"),
    ])
    with pytest.raises(RuntimeError, match="stop-loop"):
        dashboard.run(args)

    assert ticks["n"] == 2
    assert requests == ["/status", "/status"]


# Tests: Rendering (smoke -- must not raise on healthy or failed-poll frames)

def test_render_dashboard_smoke_with_data():
    """Test: render_dashboard builds a renderable from a full status payload."""
    payload = {
        "budget_mb": 7418.0,
        "gpu_total_mb": 12282.0,
        "safety_margin_mb": 4864,
        "reserved_mb": 1000.0,
        "ollama_mb": 2000.0,
        "nvidia_smi_used_mb": 5000.0,
        "leases": {"lease-1": {"caller": "flux", "mb": 1000.0, "last_seen": 5.0, "acquired_at": 5.0}},
    }
    ages = {"lease-1": 12.0}
    history = [dashboard.snapshot_from_status(payload, True)]
    group = dashboard.render_dashboard("http://broker.test", payload, True, ages, history)
    assert group is not None


def test_render_dashboard_null_nvidia_smi_used_mb():
    """Test: render_dashboard does not raise when nvidia_smi_used_mb is None."""
    payload = {
        "budget_mb": 7418.0,
        "gpu_total_mb": 12282.0,
        "safety_margin_mb": 4864,
        "reserved_mb": 1000.0,
        "ollama_mb": 2000.0,
        "nvidia_smi_used_mb": None,
        "leases": {},
    }
    history = [dashboard.snapshot_from_status(payload, True)]
    group = dashboard.render_dashboard("http://broker.test", payload, True, {}, history)
    assert group is not None


def test_render_dashboard_history_mixed_none_nvidia_smi_used_mb():
    """Test: the history panel renders when nvidia_smi_used_mb mixes None and numbers."""
    payload = {
        "budget_mb": 7418.0,
        "gpu_total_mb": 12282.0,
        "safety_margin_mb": 4864,
        "reserved_mb": 1000.0,
        "ollama_mb": 2000.0,
        "nvidia_smi_used_mb": 5000.0,
        "leases": {},
    }
    snap_num = dashboard.snapshot_from_status(payload, True)
    snap_none = dict(snap_num)
    snap_none["nvidia_smi_used_mb"] = None
    history = [snap_none, snap_num, snap_none]
    group = dashboard.render_dashboard("http://broker.test", payload, True, {}, history)
    assert group is not None


def test_render_dashboard_smoke_on_poll_failure():
    """Test: render_dashboard builds an error-state renderable when status_payload is None."""
    group = dashboard.render_dashboard("http://broker.test", None, False, {}, [])
    assert group is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
