#!/usr/bin/env python3
"""Generate a regular 5-minute ESNet delta-counter CSV from raw IF-MIB snapshots.

Key properties
--------------
* Operates on raw cumulative IF-MIB HC counters.
* Uses a fixed set of physical interfaces determined from the first usable snapshot.
* Requires all requested counters to be present for an interface observation.
* Computes deltas per interface before aggregation.
* Uses ifCounterDiscontinuityTime and negative-delta checks to reject resets.
* Bridges short missing observations using the next valid cumulative reading and
  distributes the measured total delta across the covered 5-minute bins.
* Emits a row only when every expected interface contributes to that bin, so all
  metrics are aggregated over exactly the same interface set.
* Never bridges gaps longer than --max-bridge-minutes (default 15 minutes).

The resulting counter columns are per-5-minute DELTAS, not lifetime counters.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

METRICS = (
    "ifHCOutOctets",
    "ifHCOutUcastPkts",
    "ifHCOutMulticastPkts",
    "ifHCOutBroadcastPkts",
    "ifHCInOctets",
    "ifHCInUcastPkts",
    "ifHCInMulticastPkts",
    "ifHCInBroadcastPkts",
)

FILE_TS_RE = re.compile(r"\.(\d{14})$")
IFNAME_RE = re.compile(r"^IF-MIB::ifName\.(\d+)\s+=\s+STRING:\s*(.*)$")
DISC_RE = re.compile(
    r"^IF-MIB::ifCounterDiscontinuityTime\.(\d+)\s+=\s+Timeticks:\s*\((\d+)\)"
)
COUNTER_RE = re.compile(
    r"^IF-MIB::(" + "|".join(map(re.escape, METRICS))
    + r")\.(\d+)\s+=\s+Counter(?:32|64):\s*(\d+)\s*$"
)


@dataclass
class InterfaceSnapshot:
    name: Optional[str] = None
    discontinuity_ticks: Optional[int] = None
    counters: Dict[str, int] = field(default_factory=dict)


@dataclass
class Snapshot:
    timestamp: datetime
    path: Path
    interfaces: Dict[int, InterfaceSnapshot]


def timestamp_from_path(path: Path) -> datetime:
    m = FILE_TS_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot extract YYYYMMDDhhmmss timestamp from {path}")
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")


def parse_snapshot(path: Path) -> Snapshot:
    interfaces: Dict[int, InterfaceSnapshot] = {}

    def iface(idx: int) -> InterfaceSnapshot:
        return interfaces.setdefault(idx, InterfaceSnapshot())

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")

            m = IFNAME_RE.match(line)
            if m:
                iface(int(m.group(1))).name = m.group(2).strip()
                continue

            m = DISC_RE.match(line)
            if m:
                iface(int(m.group(1))).discontinuity_ticks = int(m.group(2))
                continue

            m = COUNTER_RE.match(line)
            if m:
                metric, idx, value = m.groups()
                iface(int(idx)).counters[metric] = int(value)

    return Snapshot(timestamp_from_path(path), path, interfaces)


def discover_snapshots(root: Path, device: str) -> list[Path]:
    paths = [p for p in root.rglob(f"{device}.ifmib.*") if p.is_file()]
    paths.sort(key=timestamp_from_path)
    return paths


def regularize_5min(ts: datetime) -> datetime:
    minute = ts.minute - (ts.minute % 5)
    return ts.replace(minute=minute, second=0, microsecond=0)


def complete_interface(obs: Optional[InterfaceSnapshot], expected_name: str) -> bool:
    return bool(
        obs is not None
        and obs.name == expected_name
        and all(metric in obs.counters for metric in METRICS)
    )


def split_integer_total(total: int, n: int) -> list[int]:
    """Split nonnegative integer total into n bins exactly, differing by at most 1."""
    q, r = divmod(total, n)
    return [q + (1 if i < r else 0) for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate regular 5-minute per-interval IF-MIB counter deltas."
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--device", default="r-nova-fd-01")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--interface-regex",
        default=r"^Ethernet",
        help="Physical interfaces to aggregate (default: ^Ethernet).",
    )
    ap.add_argument(
        "--max-bridge-minutes",
        type=int,
        default=15,
        help="Maximum gap bridged from cumulative counters (default: 15).",
    )
    ap.add_argument(
        "--interval-seconds",
        type=int,
        default=300,
        help="Target regular interval in seconds (default: 300).",
    )
    ap.add_argument(
        "--timestamp-tolerance-seconds",
        type=int,
        default=30,
        help="Allowed deviation from an integer number of intervals (default: 30).",
    )
    args = ap.parse_args()

    if not args.root.exists():
        ap.error(f"root does not exist: {args.root}")
    if args.interval_seconds <= 0:
        ap.error("--interval-seconds must be positive")
    if args.max_bridge_minutes * 60 < args.interval_seconds:
        ap.error("--max-bridge-minutes must cover at least one interval")

    try:
        iface_re = re.compile(args.interface_regex)
    except re.error as exc:
        ap.error(f"invalid --interface-regex: {exc}")

    paths = discover_snapshots(args.root, args.device)
    if len(paths) < 2:
        ap.error(f"found only {len(paths)} snapshots; need at least 2")

    print(f"Found {len(paths):,} snapshots for {args.device}")
    print(f"First: {paths[0]}")
    print(f"Last : {paths[-1]}")
    print(f"Interface filter: {args.interface_regex}")

    # Find the first snapshot with at least one fully populated selected interface.
    first: Optional[Snapshot] = None
    expected: Dict[int, str] = {}
    first_pos = 0
    for pos, path in enumerate(paths):
        snap = parse_snapshot(path)
        candidates = {
            idx: obs.name
            for idx, obs in snap.interfaces.items()
            if obs.name
            and iface_re.search(obs.name)
            and all(m in obs.counters for m in METRICS)
        }
        if candidates:
            first = snap
            expected = candidates
            first_pos = pos
            break

    if first is None:
        ap.error("no snapshot contains a complete selected physical interface")

    print(f"Expected physical interfaces: {len(expected)}")

    # Last fully populated observation for every expected interface. Missing/partial
    # snapshots do NOT replace this state; a later valid observation may bridge them.
    last_valid: Dict[int, tuple[datetime, InterfaceSnapshot]] = {}
    for idx, name in expected.items():
        obs = first.interfaces.get(idx)
        if complete_interface(obs, name):
            last_valid[idx] = (first.timestamp, obs)  # type: ignore[arg-type]

    # Per regularized bin: exact totals and the interfaces that contributed.
    bin_totals: Dict[datetime, Dict[str, int]] = defaultdict(
        lambda: {m: 0 for m in METRICS}
    )
    bin_contributors: Dict[datetime, set[int]] = defaultdict(set)

    stats = defaultdict(int)
    stats["snapshots"] = len(paths)
    stats["expected_interfaces"] = len(expected)
    max_bridge_seconds = args.max_bridge_minutes * 60

    for path in paths[first_pos + 1 :]:
        snap = parse_snapshot(path)

        for idx, name in expected.items():
            obs = snap.interfaces.get(idx)
            if not complete_interface(obs, name):
                stats["incomplete_interface_observations"] += 1
                continue

            assert obs is not None
            prev_state = last_valid.get(idx)
            if prev_state is None:
                last_valid[idx] = (snap.timestamp, obs)
                continue

            prev_ts, prev = prev_state
            elapsed = (snap.timestamp - prev_ts).total_seconds()

            # Always advance state on a valid observation. If this gap cannot be
            # trusted/bridged, future deltas start from this observation instead.
            def reset_state(reason: str) -> None:
                stats[reason] += 1
                last_valid[idx] = (snap.timestamp, obs)

            if elapsed <= 0:
                reset_state("nonpositive_elapsed")
                continue

            n_intervals = int(round(elapsed / args.interval_seconds))
            timing_error = abs(elapsed - n_intervals * args.interval_seconds)
            if (
                n_intervals < 1
                or elapsed > max_bridge_seconds
                or timing_error > args.timestamp_tolerance_seconds
            ):
                reset_state("unbridged_interface_gaps")
                continue

            if (
                prev.discontinuity_ticks is not None
                and obs.discontinuity_ticks is not None
                and prev.discontinuity_ticks != obs.discontinuity_ticks
            ):
                reset_state("counter_discontinuities")
                continue

            deltas: Dict[str, int] = {}
            bad = False
            for metric in METRICS:
                delta = obs.counters[metric] - prev.counters[metric]
                if delta < 0:
                    bad = True
                    break
                deltas[metric] = delta

            if bad:
                reset_state("negative_delta_intervals")
                continue

            end_bin = regularize_5min(snap.timestamp)
            bins = [
                end_bin - timedelta(seconds=args.interval_seconds * k)
                for k in range(n_intervals - 1, -1, -1)
            ]

            pieces = {
                metric: split_integer_total(total, n_intervals)
                for metric, total in deltas.items()
            }
            for j, ts in enumerate(bins):
                # An interface may only contribute once to a given bin. Duplicate
                # timestamps (e.g. ambiguous local-clock files) are rejected.
                if idx in bin_contributors[ts]:
                    stats["duplicate_bin_contributions"] += 1
                    continue
                for metric in METRICS:
                    bin_totals[ts][metric] += pieces[metric][j]
                bin_contributors[ts].add(idx)

            if n_intervals > 1:
                stats["bridged_interface_intervals"] += 1
                stats["bridged_5min_bins"] += n_intervals

            last_valid[idx] = (snap.timestamp, obs)

    expected_count = len(expected)
    rows = []
    incomplete_bins = 0
    for ts in sorted(bin_totals):
        coverage = len(bin_contributors[ts])
        if coverage != expected_count:
            incomplete_bins += 1
            continue
        row = {"date": ts.isoformat(sep=" ")}
        row.update(bin_totals[ts])
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["date", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    # Final regular-grid diagnostics.
    non5 = 0
    longest = 0
    current = 0
    prev_ts: Optional[datetime] = None
    for row in rows:
        ts = datetime.fromisoformat(row["date"])
        if prev_ts is None or ts - prev_ts == timedelta(seconds=args.interval_seconds):
            current += 1
        else:
            non5 += 1
            current = 1
        longest = max(longest, current)
        prev_ts = ts

    print(f"Wrote {len(rows):,} complete rows to {args.output}")
    print("Diagnostics:")
    for key in (
        "expected_interfaces",
        "incomplete_interface_observations",
        "bridged_interface_intervals",
        "bridged_5min_bins",
        "counter_discontinuities",
        "negative_delta_intervals",
        "unbridged_interface_gaps",
        "nonpositive_elapsed",
        "duplicate_bin_contributions",
    ):
        print(f"  {key}: {stats[key]:,}")
    print(f"  incomplete_output_bins_dropped: {incomplete_bins:,}")
    print(f"  non_5min_output_gaps: {non5:,}")
    print(f"  longest_contiguous_5min_run: {longest:,} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
