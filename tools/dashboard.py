"""
Author: Richard Baldwin
Date:   2026
Email: eyeclept@pm.me

Description:

Read-only terminal dashboard for GpuBroker (Assist/TODO.md section 8,
"Status dashboard TUI"). Polls GET /status on an interval, renders live
budget/reservation/lease state, and appends every polled snapshot to a
local JSONL log so a usage-over-time view (reserved/used/free MB per
sample) can render alongside the live snapshot.

Read-only guarantee: this file calls httpx's `.get()` exactly once, against
`/status`, and nowhere constructs or sends a POST. It has no
code path that reads GPU_BROKER_ADMIN_TOKEN and cannot reach /admin/clear,
/acquire, /release, or /heartbeat. tests/test_dashboard.py asserts this
statically (source scan for the mutating endpoint strings and for any
`.post(` call) so a future edit that adds a write path fails the test suite
instead of silently drifting from this guarantee.

Library choice: `rich` (Live + Table + Panel), not `textual`. This dashboard
is a single passive view with no keybindings, tabs, or user input beyond
Ctrl-C to quit -- textual's widget/screen/event-loop machinery buys nothing
here and is a heavier dependency than the job needs. `rich.live.Live`
re-rendering a `rich.table.Table`/`rich.panel.Panel` group at the poll
interval is the whole feature. httpx is reused rather than adding `requests`
since it is already this project's HTTP client (server.py, TI-Template/).

Log storage choice: flat append-only JSONL (one JSON object per line), not
SQLite. Samples are a handful of floats per timestamp, written by a single
process and read back sequentially for a trend view -- there is no
concurrent-writer, indexed-query, or transactional-multi-row need that would
justify SQLite's schema/migration overhead. JSONL is stdlib-only, trivially
appendable, and human-inspectable with `cat`/`tail`, matching this project's
existing flat-JSON ledger convention (server.py `_persist_ledger_sync`),
including its unique-temp + fsync + os.replace atomic-write pattern.

Retention: the log is trimmed to the most recent MAX_LOG_SAMPLES (10000)
rows once it exceeds that by more than one trim batch. At the default 5s
poll interval that is roughly 14 hours of rolling history -- enough to cover
"what happened tonight" on a dashboard meant to be left running, without
unbounded disk growth. Row count is kept in memory after one startup pass;
the file is reread only when the count crosses the trim threshold.

Poll interval: 5.0s default (--interval). GET /status normally answers from
the broker's cached nvidia-smi reading and runs no subprocess; with a stale
cache it takes one reading outside the broker lock. 5s feels live without
meaningfully loading the broker or the GPU driver.

File location: tools/dashboard.py. It is a pure API client with no
broker-side change, so it lives as a standalone script rather than inside
server.py or its own project.

Usage:
    python3 tools/dashboard.py --url http://localhost:8792
    python3 tools/dashboard.py --url http://localhost:8798 --interval 2 \\
        --log-path data/dashboard_history_test.jsonl

Dependencies (already installed on this host): rich, httpx.
"""

# Imports

import argparse
import collections
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone

import httpx
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Globals

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"))
logger = logging.getLogger(__name__)

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)

DEFAULT_URL = "http://localhost:8792"
DEFAULT_INTERVAL_SECONDS = 5.0
DEFAULT_LOG_PATH = os.path.join(_REPO_ROOT, "data", "dashboard_history.jsonl")
MAX_LOG_SAMPLES = 10000
HISTORY_POINTS = 60  # samples shown in the usage-over-time sparklines

SPARK_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"  # ascending block glyphs


# Functions

def parse_args(argv=None):
    """
    Input:      argv - list[str] or None (None reads sys.argv)
    Output:     argparse.Namespace with url, interval, log_path
    Details:
        CLI flags for the dashboard. --url is the broker base URL (point it
        at the live broker for display verification, or the test instance
        on :8798 for interaction/write-path testing done through other
        clients -- this dashboard itself never writes).
    """
    parser = argparse.ArgumentParser(
        description="Read-only live/history dashboard for a GpuBroker instance."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"broker base URL (default: {DEFAULT_URL})")
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL_SECONDS,
        help=f"poll interval in seconds (default: {DEFAULT_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--log-path", default=DEFAULT_LOG_PATH,
        help=f"JSONL snapshot log path (default: {DEFAULT_LOG_PATH})",
    )
    return parser.parse_args(argv)


def fetch_status(client, base_url):
    """
    Input:      client - httpx.Client; base_url - broker base URL
    Output:     dict - parsed GET /status JSON body
    Details:
        Raises httpx.HTTPError on network failure or non-2xx status. The
        only write-adjacent word in this function is "raises" -- this
        method is GET only, see module docstring's read-only guarantee.
    """
    resp = client.get(f"{base_url}/status", timeout=5.0)
    resp.raise_for_status()
    return resp.json()


def snapshot_from_status(status_payload, healthy):
    """
    Input:      status_payload - dict, parsed GET /status body;
                healthy - bool, True when GET /status succeeded
    Output:     dict - one flat log row: wall-clock ts, every field the live
                and history views need, plus derived free_mb
    Details:
        free_mb mirrors the headroom term server.py's own admission formula
        computes (see acquire(): reserved_total = reserved_mb + ollama_mb;
        fits = reserved_total + requested_mb <= budget_mb), so the history
        view's "free" line means the same thing the broker gates on.
    """
    reserved_mb = status_payload["reserved_mb"]
    ollama_mb = status_payload["ollama_mb"]
    budget_mb = status_payload["budget_mb"]
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "healthy": healthy,
        "budget_mb": budget_mb,
        "gpu_total_mb": status_payload["gpu_total_mb"],
        "safety_margin_mb": status_payload["safety_margin_mb"],
        "reserved_mb": reserved_mb,
        "ollama_mb": ollama_mb,
        "nvidia_smi_used_mb": status_payload["nvidia_smi_used_mb"],
        "free_mb": budget_mb - reserved_mb - ollama_mb,
        "lease_count": len(status_payload.get("leases", {})),
    }


def _count_log_rows(log_path):
    """
    Input:      log_path - JSONL snapshot log path
    Output:     int - physical line count, or 0 if the file does not exist
    Details:
        One cheap pass used at dashboard startup, and by append_snapshot
        when the caller does not pass a row_count.
    """
    if not os.path.exists(log_path):
        return 0
    with open(log_path) as handle:
        return sum(1 for _ in handle)


def _atomic_write_text(path, text):
    """
    Input:      path - destination file; text - str to write
    Output:     None
    Details:
        Writes text through a unique temp file in the same directory
        (tempfile.mkstemp, O_EXCL, mode 0600), flush+fsync, then
        os.replace onto path. On any exception, removes the temp file
        (logs a cleanup failure) and re-raises. Copied from
        server._atomic_write_json; this module must not import server.py.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(
        dir=directory,
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        logger.exception("atomic write to %s failed", path)
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.warning("failed to remove temp file %s after write error", tmp_path)
        raise


def _trim_snapshot_log(log_path, max_samples):
    """
    Input:      log_path - JSONL snapshot log path; max_samples - rows to keep
    Output:     None
    Details:
        Reads the file once and atomically rewrites the most recent
        max_samples physical lines.
    """
    with open(log_path) as handle:
        lines = handle.readlines()
    _atomic_write_text(log_path, "".join(lines[-max_samples:]))


def append_snapshot(log_path, snapshot, max_samples=MAX_LOG_SAMPLES, row_count=None):
    """
    Input:      log_path - JSONL snapshot log path; snapshot - dict, one row;
                max_samples - retention cap (see module docstring);
                row_count - known physical line count, or None to count now
    Output:     int - physical line count after append and any trim
    Details:
        Appends one JSON line and increments row_count. Reads and rewrites
        the file only when row_count exceeds max_samples + max_samples//10.
        Trim writes a unique temp file in the same directory, fsyncs, and
        os.replace()s it into place (same pattern as server._atomic_write_json).
        Callers that already hold a count (the poll loop) must pass it so
        each append does not reread the log.
    """
    if row_count is None:
        row_count = _count_log_rows(log_path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a") as handle:
        handle.write(json.dumps(snapshot) + "\n")
    row_count += 1
    trim_batch = max(1, max_samples // 10)
    if row_count > max_samples + trim_batch:
        _trim_snapshot_log(log_path, max_samples)
        row_count = max_samples
    return row_count


def read_history(log_path, limit=None):
    """
    Input:      log_path - JSONL snapshot log path; limit - max most-recent
                rows to return, or None for all
    Output:     list[dict] - parsed snapshot rows, oldest to newest; []
                when the log does not exist yet
    Details:
        Skips (and logs) any line that fails to parse rather than raising --
        this log is diagnostic history, not state the broker depends on.
    """
    if not os.path.exists(log_path):
        return []
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping unparsable snapshot log line in %s", log_path)
    if limit is not None:
        rows = rows[-limit:]
    return rows


def update_lease_ages(tracking, leases, now_wall):
    """
    Input:      tracking - dict[lease_id -> {"last_seen_mono", "since_wall"}],
                mutated in place, persists across polls; leases - dict from
                the /status payload's "leases" field; now_wall - time.time()
                at the moment of this poll
    Output:     dict[lease_id -> float] - seconds since each lease's last
                observed heartbeat
    Details:
        /status reports last_seen as the broker's own time.monotonic()
        reading, which has no client-relatable "now" -- a single snapshot
        cannot convert it to a wall-clock age. Instead this tracks, per
        lease, the wall-clock time this dashboard last saw last_seen
        increase (a real heartbeat, since monotonic time only moves
        forward) and reports elapsed wall time since then. A lease's age
        reads 0.0 the first poll this dashboard observes it, since no prior
        reference exists -- this under-reports true age for leases that
        existed before the dashboard started polling. Lease ids no longer
        present in `leases` are dropped from `tracking`.
    """
    for lease_id in set(tracking) - set(leases):
        del tracking[lease_id]

    ages = {}
    for lease_id, lease in leases.items():
        last_seen_mono = lease["last_seen"]
        entry = tracking.get(lease_id)
        if entry is None or last_seen_mono > entry["last_seen_mono"]:
            entry = {"last_seen_mono": last_seen_mono, "since_wall": now_wall}
            tracking[lease_id] = entry
        ages[lease_id] = now_wall - entry["since_wall"]
    return ages


def render_sparkline(values):
    """
    Input:      values - list[float], oldest to newest
    Output:     str - one block-glyph character per value
    Details:
        Empty input renders as an empty string. A flat series (max == min,
        including a single value) renders as the middle glyph repeated,
        rather than dividing by zero.
    """
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return SPARK_CHARS[len(SPARK_CHARS) // 2] * len(values)
    scale = len(SPARK_CHARS) - 1
    return "".join(SPARK_CHARS[min(int((v - lo) / span * scale), scale)] for v in values)


def format_age(seconds):
    """
    Input:      seconds - float
    Output:     str - human-readable "Ns"/"Nm Ns"/"Nh Nm"
    Details:
        Keeps the leases table narrow; heartbeats are TTL-bounded (default
        180s, see GPU_BROKER_LEASE_TTL_SECONDS) so hours-scale ages only
        show up on a dashboard that outlives a very long-lived lease.
    """
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes:.0f}m {secs:.0f}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours:.0f}h {mins:.0f}m"


def _format_mb(value):
    """
    Input:      value - float MB, or None when nvidia-smi has no reading
    Output:     str - whole-number MB, or "n/a" when value is None
    """
    if value is None:
        return "n/a"
    return f"{value:.0f}"


def build_leases_table(leases, ages):
    """
    Input:      leases - dict from the /status payload's "leases" field;
                ages - dict from update_lease_ages()
    Output:     rich.table.Table
    Details:
        One row per lease: caller, lease id (short form), reserved MB, and
        age since last heartbeat. Sorted oldest-heartbeat-first so the
        leases closest to sweeper reclaim (see GPU_BROKER_LEASE_TTL_SECONDS)
        sort to the top.
    """
    table = Table(title="Leases", expand=True)
    table.add_column("Caller")
    table.add_column("Lease ID")
    table.add_column("MB", justify="right")
    table.add_column("Age (since last heartbeat)", justify="right")

    rows = sorted(leases.items(), key=lambda kv: -ages.get(kv[0], 0.0))
    for lease_id, lease in rows:
        table.add_row(
            str(lease["caller"]),
            lease_id[:8],
            f"{lease['mb']:.0f}",
            format_age(ages.get(lease_id, 0.0)),
        )
    if not rows:
        table.add_row("(no active leases)", "", "", "")
    return table


def build_budget_panel(url, snapshot, healthy):
    """
    Input:      url - broker base URL; snapshot - dict from
                snapshot_from_status(), or None on poll failure;
                healthy - bool, True when GET /status succeeded
    Output:     rich.panel.Panel
    Details:
        Renders budget_mb/reserved_mb/ollama_mb/nvidia_smi_used_mb and
        derived free headroom. nvidia_smi_used_mb displays as n/a when
        None. Renders an error state, not a crash, when the most recent
        poll failed (snapshot is None).
    """
    health_text = Text("healthy", style="bold green") if healthy else Text("unreachable/unhealthy", style="bold red")
    if snapshot is None:
        body = Group(Text(f"broker: {url}"), Text("status: "), health_text, Text("\nno data yet (poll failed)"))
        return Panel(body, title="GpuBroker Status", border_style="red")

    lines = [
        Text.assemble("broker: ", (url, "bold")),
        Text.assemble("status: ", health_text),
        Text(""),
        Text.assemble("budget_mb:          ", (f"{snapshot['budget_mb']:.0f}", "bold")),
        Text.assemble("gpu_total_mb:       ", f"{snapshot['gpu_total_mb']:.0f}"),
        Text.assemble("safety_margin_mb:   ", f"{snapshot['safety_margin_mb']:.0f}"),
        Text.assemble("reserved_mb:        ", f"{snapshot['reserved_mb']:.0f}"),
        Text.assemble("ollama_mb:          ", f"{snapshot['ollama_mb']:.0f}"),
        Text.assemble("nvidia_smi_used_mb: ", _format_mb(snapshot["nvidia_smi_used_mb"])),
        Text.assemble("free_mb (headroom): ", (f"{snapshot['free_mb']:.0f}", "bold cyan")),
    ]
    return Panel(Group(*lines), title="GpuBroker Status", border_style="green" if healthy else "yellow")


def build_history_panel(history):
    """
    Input:      history - list[dict], oldest to newest, from snapshot_from_status()
    Output:     rich.panel.Panel
    Details:
        One sparkline per tracked quantity (reserved/ollama/used/free MB)
        over the most recent samples, each labeled with its current value
        and the min/max across the shown window for scale. None values are
        omitted from sparkline and min/max; an all-None series uses the
        empty sparkline. A None current value displays as n/a.
    """
    if not history:
        return Panel(Text("no samples logged yet"), title="Usage Over Time")

    series = {
        "reserved_mb": [s["reserved_mb"] for s in history],
        "ollama_mb": [s["ollama_mb"] for s in history],
        "nvidia_smi_used_mb": [s["nvidia_smi_used_mb"] for s in history],
        "free_mb": [s["free_mb"] for s in history],
    }
    lines = []
    for name, values in series.items():
        numeric = [v for v in values if v is not None]
        spark = render_sparkline(numeric)
        now_text = _format_mb(values[-1])
        min_text = _format_mb(min(numeric) if numeric else None)
        max_text = _format_mb(max(numeric) if numeric else None)
        lines.append(
            Text.assemble(
                f"{name:<20}", (spark, "cyan"),
                f"  now={now_text}  min={min_text}  max={max_text}",
            )
        )
    lines.append(Text(f"\n{len(history)} sample(s) shown (most recent {HISTORY_POINTS})"))
    return Panel(Group(*lines), title="Usage Over Time")


def render_dashboard(url, status_payload, healthy, ages, history):
    """
    Input:      url - broker base URL; status_payload - dict or None on poll
                failure; healthy - bool; ages - dict from update_lease_ages();
                history - list[dict] from snapshot_from_status()
    Output:     rich.console.Group - the full renderable for this frame
    Details:
        Composes the budget panel, leases table, and history panel into one
        renderable for Live.update(). Builds snapshot_from_status() inline
        so failed polls (status_payload is None) still render a coherent
        error frame instead of raising.
    """
    snapshot = snapshot_from_status(status_payload, healthy) if status_payload is not None else None
    budget_panel = build_budget_panel(url, snapshot, healthy)
    leases_table = build_leases_table(status_payload.get("leases", {}) if status_payload else {}, ages)
    history_panel = build_history_panel(history)
    return Group(budget_panel, leases_table, history_panel)


def run(args):
    """
    Input:      args - argparse.Namespace from parse_args()
    Output:     None (runs until KeyboardInterrupt)
    Details:
        Main poll loop. Seeds in-memory history from the existing log tail
        so a restarted dashboard's sparklines are not empty, counts log
        rows once, then on each poll: GET /status (success is healthy,
        failure is unhealthy), log the snapshot (only on a successful
        /status poll -- a failed poll has nothing new to log), update
        lease-age tracking, and re-render.
    """
    lease_tracking = {}
    row_count = _count_log_rows(args.log_path)
    history = collections.deque(read_history(args.log_path, limit=HISTORY_POINTS), maxlen=HISTORY_POINTS)

    with httpx.Client() as client:
        frame = render_dashboard(args.url, None, False, {}, list(history))
        with Live(frame, refresh_per_second=4, screen=False) as live:
            while True:
                status_payload = None
                healthy = False
                try:
                    status_payload = fetch_status(client, args.url)
                    healthy = True
                except httpx.HTTPError as exc:
                    logger.warning("poll failed against %s: %s", args.url, exc)

                ages = {}
                if status_payload is not None:
                    snapshot = snapshot_from_status(status_payload, healthy)
                    row_count = append_snapshot(args.log_path, snapshot, row_count=row_count)
                    history.append(snapshot)
                    ages = update_lease_ages(lease_tracking, status_payload.get("leases", {}), time.time())

                live.update(render_dashboard(args.url, status_payload, healthy, ages, list(history)))
                time.sleep(args.interval)


def main():
    """
    Input:      None
    Output:     None
    Details:
        Entry point. Ctrl-C exits cleanly with no traceback.
    """
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        logger.warning("dashboard stopped by KeyboardInterrupt")


if __name__ == "__main__":
    main()
