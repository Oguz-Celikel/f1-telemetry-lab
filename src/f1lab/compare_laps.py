"""Compare two drivers' fastest laps from a FastF1 session.

Loads a session (laps + car telemetry), picks each driver's fastest lap and
renders a speed-vs-distance comparison with a time-delta trace underneath.

Run as a module::

    python -m f1lab --year 2026 --gp Silverstone --session R --drivers VER NOR
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import fastf1
import matplotlib
import numpy as np
from numpy.typing import NDArray

from f1lab import native
from f1lab.analysis import (
    FastestLap,
    build_output_path,
    compute_delta_time,
    distance_and_time,
    fastest_lap_telemetry,
    format_lap_time,
)
from f1lab.corners import detect_corners

matplotlib.use("Agg")  # headless rendering; must be selected before pyplot is imported

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(".fastf1-cache")
DEFAULT_OUTPUT_DIR = Path("output")

# Colors from the validated reference palette (dataviz skill): categorical
# slots 1-2 for the two drivers, neutral ink/chrome tokens for everything else.
COLOR_DRIVER_1 = "#2a78d6"
COLOR_DRIVER_2 = "#1baf7a"
COLOR_SURFACE = "#fcfcfb"
COLOR_INK = "#0b0b0b"
COLOR_INK_SECONDARY = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"


def enable_cache(cache_dir: Path | None = None) -> Path:
    """Enable FastF1's on-disk cache; honours the ``FASTF1_CACHE`` env var."""
    resolved = cache_dir or Path(os.environ.get("FASTF1_CACHE", str(DEFAULT_CACHE_DIR)))
    resolved.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(resolved))
    return resolved


def load_session(year: int, gp: str, session_name: str) -> fastf1.core.Session:
    """Fetch and load a session (laps + car telemetry, no weather/messages)."""
    session = fastf1.get_session(year, gp, session_name)
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    return session


def plot_comparison(
    lap_1: FastestLap,
    lap_2: FastestLap,
    delta_s: NDArray[np.float64],
    title: str,
    out_path: Path,
) -> None:
    """Render both speed traces plus the time delta and save as a PNG."""
    import matplotlib.pyplot as plt  # deferred so matplotlib.use("Agg") above always wins

    dist_1, _ = distance_and_time(lap_1.telemetry)
    dist_2, _ = distance_and_time(lap_2.telemetry)
    speed_1 = np.asarray(lap_1.telemetry["Speed"], dtype=np.float64)
    speed_2 = np.asarray(lap_2.telemetry["Speed"], dtype=np.float64)

    fig, (ax_speed, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(11.0, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0]},
        layout="constrained",
    )
    fig.set_facecolor(COLOR_SURFACE)

    for ax in (ax_speed, ax_delta):
        ax.set_facecolor(COLOR_SURFACE)
        ax.grid(color=COLOR_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLOR_BASELINE)
        ax.spines["bottom"].set_color(COLOR_BASELINE)
        ax.tick_params(colors=COLOR_MUTED, labelcolor=COLOR_INK_SECONDARY)

    label_1 = f"{lap_1.driver}  {format_lap_time(lap_1.lap_time_s)}"
    label_2 = f"{lap_2.driver}  {format_lap_time(lap_2.lap_time_s)}"
    ax_speed.plot(dist_1, speed_1, color=COLOR_DRIVER_1, linewidth=1.8, label=label_1)
    ax_speed.plot(dist_2, speed_2, color=COLOR_DRIVER_2, linewidth=1.8, label=label_2)

    # Apexes detected on the reference driver's trace, as subtle markers.
    corners = detect_corners(dist_1, speed_1)
    if len(corners) > 0:
        ax_speed.plot(
            dist_1[corners.apex_indices],
            speed_1[corners.apex_indices],
            linestyle="none",
            marker="v",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor=COLOR_MUTED,
            label="apex",
        )
    ax_speed.set_ylabel("Speed (km/h)", color=COLOR_INK_SECONDARY)
    legend = ax_speed.legend(loc="lower right", frameon=False)
    for text in legend.get_texts():
        text.set_color(COLOR_INK)

    # Direct labels at the line ends (ink, not series color), nudged apart so
    # they stay readable even when both traces finish at similar speeds.
    faster_end = speed_1[-1] >= speed_2[-1]
    for dist, speed, lap, above in (
        (dist_1, speed_1, lap_1, faster_end),
        (dist_2, speed_2, lap_2, not faster_end),
    ):
        ax_speed.annotate(
            lap.driver,
            xy=(float(dist[-1]), float(speed[-1])),
            xytext=(6, 8 if above else -8),
            textcoords="offset points",
            color=COLOR_INK,
            fontsize=9,
            fontweight="bold",
            va="center",
        )

    ax_delta.plot(dist_1, delta_s, color=COLOR_INK_SECONDARY, linewidth=1.6)
    ax_delta.axhline(0.0, color=COLOR_BASELINE, linewidth=1.0)
    ax_delta.set_ylabel(f"Delta (s)\n+ = {lap_2.driver} behind", color=COLOR_INK_SECONDARY)
    ax_delta.set_xlabel("Lap distance (m)", color=COLOR_INK_SECONDARY)

    fig.suptitle(title, color=COLOR_INK, fontsize=13, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """CLI definition; kept separate from ``main`` so tests can exercise it."""
    parser = argparse.ArgumentParser(
        prog="f1lab",
        description="Compare two drivers' fastest laps in an F1 session.",
    )
    parser.add_argument("--year", type=int, required=True, help="Season year, e.g. 2026")
    parser.add_argument(
        "--gp", required=True, help='Grand Prix name or location, e.g. "Silverstone"'
    )
    parser.add_argument(
        "--session", default="R", help="Session code: R, Q, S, FP1, FP2, FP3 (default: R)"
    )
    parser.add_argument(
        "--drivers",
        nargs=2,
        required=True,
        metavar=("DRIVER1", "DRIVER2"),
        help="Two three-letter driver codes, e.g. VER NOR",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory the PNG is written to (default: output)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    driver_1, driver_2 = (code.upper() for code in args.drivers)

    cache_dir = enable_cache()
    LOGGER.info("FastF1 cache: %s", cache_dir)
    LOGGER.info("Analysis engine: %s", native.backend_name())

    try:
        session = load_session(args.year, args.gp, args.session)
        lap_1 = fastest_lap_telemetry(session, driver_1)
        lap_2 = fastest_lap_telemetry(session, driver_2)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 1

    dist_1, time_1 = distance_and_time(lap_1.telemetry)
    dist_2, time_2 = distance_and_time(lap_2.telemetry)
    delta_s = compute_delta_time(dist_1, time_1, dist_2, time_2)

    event_name = str(session.event["EventName"])
    title = f"{event_name} {args.year} — {session.name}: fastest lap comparison"
    out_path = build_output_path(
        args.output_dir, args.year, event_name, args.session, driver_1, driver_2
    )
    plot_comparison(lap_1, lap_2, delta_s, title, out_path)
    LOGGER.info("Saved %s", out_path)
    return 0
