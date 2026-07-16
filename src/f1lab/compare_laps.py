"""Compare two drivers' fastest laps from a FastF1 session.

Loads a session (laps + car telemetry), picks each driver's fastest lap and
renders the time delta above every telemetry channel the cars recorded — speed,
throttle, brake, gear, RPM and DRS — all sharing one distance axis, so a gain in
the delta lines up vertically with the inputs that produced it.

Run as a module::

    python -m f1lab --year 2026 --gp Silverstone --session R --drivers VER NOR
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fastf1
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray

from f1lab import native
from f1lab.analysis import (
    FastestLap,
    build_output_path,
    channel,
    compute_delta_time,
    distance_and_time,
    fastest_lap_telemetry,
    format_lap_time,
    has_signal,
)
from f1lab.corners import detect_corners
from f1lab.logs import log_uncaught_exceptions, setup_logging
from f1lab.paths import cache_directory

matplotlib.use("Agg")  # headless rendering; must be selected before pyplot is imported

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("output")


@dataclass(frozen=True)
class Theme:
    """Every colour one background mode needs, chrome and defaults together.

    Dark mode is not the light palette inverted: each theme's driver colours
    were validated separately against its own surface (lightness band, chroma,
    colour-vision separation, contrast), because a hue that reads well on
    near-white can be illegibly bright on near-black.
    """

    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    baseline: str
    driver_1: str
    driver_2: str


# Categorical slots 1-2 for the two drivers, neutral ink/chrome tokens for
# everything else — from the validated reference palette (dataviz skill).
LIGHT = Theme(
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    driver_1="#2a78d6",
    driver_2="#1baf7a",
)
DARK = Theme(
    surface="#1a1a19",
    ink="#f5f5f4",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2e2d2b",
    baseline="#52514e",
    driver_1="#3987e5",
    driver_2="#18a875",
)


@dataclass(frozen=True)
class PlotOptions:
    """What the viewer chose: which panels, whose colours, which background.

    ``channels`` is None for "everything the session recorded"; otherwise only
    the named channels are drawn. The delta has its own flag — it is computed
    from both laps rather than recorded by either, so it is not a channel.
    Driver colours default to the theme's validated pair and can be overridden
    per driver from the app.
    """

    channels: frozenset[str] | None = None
    show_delta: bool = True
    theme: Theme = LIGHT
    driver_1_color: str | None = None
    driver_2_color: str | None = None

    @property
    def color_1(self) -> str:
        return self.driver_1_color or self.theme.driver_1

    @property
    def color_2(self) -> str:
        return self.driver_2_color or self.theme.driver_2


def enable_cache(cache_dir: Path | None = None) -> Path:
    """Enable FastF1's on-disk cache; resolution lives in :mod:`f1lab.paths`."""
    resolved = cache_dir or cache_directory()
    resolved.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(resolved))
    return resolved


def load_session(year: int, gp: str, session_name: str) -> fastf1.core.Session:
    """Fetch and load a session (laps + car telemetry, no weather/messages)."""
    session = fastf1.get_session(year, gp, session_name)
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    return session


@dataclass(frozen=True)
class Panel:
    """One row of the figure: a telemetry channel and how to draw it.

    Panels are data rather than code so the figure can be assembled in a single
    loop, and so a channel that carries no signal in a given season can simply
    be dropped from the list.
    """

    column: str
    label: str
    height: float
    # "line"   — a continuous quantity (speed, throttle, RPM)
    # "step"   — a quantity that only takes whole values (gear)
    # "state"  — an on/off channel (brake, DRS); see _draw_state
    style: str
    # A required panel is drawn even if the channel never varies. Speed is the
    # reference trace — the apexes and the driver labels live on it — so the
    # figure would lose its anchor without it.
    required: bool = False


# Delta sits on top: it is the answer, and the channels below it are the
# explanation. Speed gets the most height because it carries the most shape.
# Anything binary gets a short strip — it has two values to show.
PANELS: tuple[Panel, ...] = (
    Panel("Speed", "Speed\n(km/h)", 2.6, "line", required=True),
    Panel("Throttle", "Throttle\n(%)", 1.1, "line"),
    Panel("Brake", "Brake", 0.55, "state"),
    Panel("nGear", "Gear", 0.9, "step"),
    Panel("RPM", "RPM", 1.1, "line"),
    Panel("DRS", "DRS", 0.55, "state"),
)

DELTA_HEIGHT = 1.0


def _style_axis(ax: Any, theme: Theme) -> None:
    """Recessive chrome: the data should be the only thing that carries weight."""
    ax.set_facecolor(theme.surface)
    ax.grid(color=theme.grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(theme.baseline)
    ax.spines["bottom"].set_color(theme.baseline)
    ax.tick_params(colors=theme.muted, labelcolor=theme.ink_secondary)


def _draw_state(
    ax: Any,
    lanes: Sequence[tuple[str, NDArray[np.float64], NDArray[np.float64]]],
    options: PlotOptions,
) -> None:
    """Draw an on/off channel as one lane per driver.

    Overlapping two binary step lines — which is the obvious thing to do, and
    what most telemetry plots do — produces a tangle exactly where the channels
    disagree, which is the only place worth looking. Giving each driver their
    own lane means "who is braking where" is legible at a glance, and the two
    can be compared vertically.
    """
    for lane, (_driver, distance, values) in enumerate(reversed(lanes)):
        color = options.color_1 if lane == len(lanes) - 1 else options.color_2
        # A filled band between lane and lane+0.7, present only where the
        # channel is on. The 0.3 gap keeps the two lanes from touching.
        ax.fill_between(
            distance,
            lane,
            lane + 0.7,
            where=values > 0.5,
            color=color,
            linewidth=0,
            step="post",
        )
    ax.set_ylim(-0.15, len(lanes))
    ax.set_yticks([lane + 0.35 for lane in range(len(lanes))])
    ax.set_yticklabels([driver for driver, _, _ in reversed(lanes)], fontsize=8)
    ax.grid(False)


PanelData = tuple[Panel, NDArray[np.float64], NDArray[np.float64]]


def select_panels(
    telemetry_1: pd.DataFrame,
    telemetry_2: pd.DataFrame,
    channels: frozenset[str] | None = None,
) -> list[PanelData]:
    """Which panels these two laps support, with the values to draw in each.

    A channel is skipped when the cars did not record it, when neither lap
    varies in it — a flat panel takes height from the ones that have something
    to say — or when the viewer deselected it (``channels``). ``required``
    exempts a panel from the flat-signal rule only: speed is drawn even when
    it never varies, but the viewer can still choose to hide it.
    """
    panels: list[PanelData] = []
    for panel in PANELS:
        if channels is not None and panel.column not in channels:
            continue
        try:
            values_1 = channel(telemetry_1, panel.column)
            values_2 = channel(telemetry_2, panel.column)
        except KeyError:
            continue
        if panel.required or has_signal(values_1, values_2):
            panels.append((panel, values_1, values_2))
    return panels


def build_comparison_figure(
    lap_1: FastestLap,
    lap_2: FastestLap,
    delta_s: NDArray[np.float64],
    title: str,
    options: PlotOptions | None = None,
) -> Figure:
    """Build the comparison figure: the delta above every available channel.

    All panels share one distance axis, so a feature in the speed trace lines up
    vertically with the throttle, brake and gear that produced it. Each channel
    keeps its own y-scale in its own panel — two units never share an axis.

    Channels that carry no signal are dropped: 2026 cars report DRS as a
    constant zero (the regulations replaced it with active aerodynamics), and a
    flat panel would only take space away from the ones that say something.

    Built on a bare ``Figure`` rather than through pyplot: pyplot keeps global
    state and assumes it owns the backend, which breaks the moment the same
    figure has to live inside a Qt window. The CLI saves this figure to a PNG;
    the GUI hands it to a canvas — same function, no global anything.
    """
    options = options or PlotOptions()
    theme = options.theme
    dist_1, _ = distance_and_time(lap_1.telemetry)
    dist_2, _ = distance_and_time(lap_2.telemetry)
    panels = select_panels(lap_1.telemetry, lap_2.telemetry, options.channels)

    # Everything deselected leaves nothing to draw; a figure with zero rows
    # helps nobody, so the delta comes back as the floor.
    show_delta = options.show_delta or not panels

    # Row order: speed leads when shown — it is the trace everything else
    # explains — with the delta directly under it, then the input channels.
    # Without speed, the delta takes the top.
    speed_rows = [p for p in panels if p[0].column == "Speed"]
    other_rows = [p for p in panels if p[0].column != "Speed"]
    rows: list[PanelData | None] = [*speed_rows, *([None] if show_delta else []), *other_rows]

    heights = [DELTA_HEIGHT if row is None else row[0].height for row in rows]
    fig = Figure(figsize=(11.5, 1.25 + 1.35 * len(heights)), layout="constrained")
    axes_grid = fig.subplots(
        len(heights),
        1,
        sharex=True,
        gridspec_kw={"height_ratios": heights},
        # Without this, a figure with a single row returns a bare Axes rather
        # than an array, and the loop below would have to special-case it.
        squeeze=False,
    )
    axes = axes_grid[:, 0]
    fig.set_facecolor(theme.surface)
    for ax in axes:
        _style_axis(ax, theme)

    for ax, row in zip(axes, rows, strict=True):
        if row is None:
            # The delta, in neutral ink rather than a driver colour: it
            # belongs to neither of them.
            ax.plot(dist_1, delta_s, color=theme.ink_secondary, linewidth=1.6)
            ax.axhline(0.0, color=theme.baseline, linewidth=1.0)
            ax.set_ylabel(f"Delta (s)\n+ = {lap_2.driver} behind", color=theme.ink_secondary)
            continue
        panel, values_1, values_2 = row
        if panel.style == "state":
            _draw_state(
                ax,
                [(lap_1.driver, dist_1, values_1), (lap_2.driver, dist_2, values_2)],
                options,
            )
        else:
            drawstyle = "steps-post" if panel.style == "step" else "default"
            for distance, values, color in (
                (dist_1, values_1, options.color_1),
                (dist_2, values_2, options.color_2),
            ):
                ax.plot(distance, values, color=color, linewidth=1.5, drawstyle=drawstyle)
            if panel.style == "step":
                # A step channel only ever takes whole values: a tick reading
                # "2.5" would label a gear that cannot exist.
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(panel.label, color=theme.ink_secondary)

        if panel.column == "Speed":
            _annotate_speed(ax, lap_1, lap_2, dist_1, dist_2, values_1, values_2, options)

    # Identity must survive any selection: the legend normally lives on the
    # speed panel, but with speed hidden the colours would mean nothing — so
    # it moves to the topmost axes instead.
    if not speed_rows:
        _legend(axes[0], lap_1, lap_2, options, with_apex_marker=False)

    axes[-1].set_xlabel("Lap distance (m)", color=theme.ink_secondary)
    fig.suptitle(title, color=theme.ink, fontsize=13, fontweight="bold")
    return fig


def plot_comparison(
    lap_1: FastestLap,
    lap_2: FastestLap,
    delta_s: NDArray[np.float64],
    title: str,
    out_path: Path,
) -> None:
    """Build the comparison figure and save it as a PNG — the CLI's output path."""
    fig = build_comparison_figure(lap_1, lap_2, delta_s, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)


def _legend(
    ax: Any,
    lap_1: FastestLap,
    lap_2: FastestLap,
    options: PlotOptions,
    *,
    with_apex_marker: bool,
) -> None:
    """Driver identity and lap times, once per figure.

    The legend carries the lap times, so the headline number is where the eye
    already goes for identity — and the two colours mean the same thing all
    the way down the figure.
    """
    theme = options.theme
    handles = [
        Line2D([], [], color=options.color_1, linewidth=1.8),
        Line2D([], [], color=options.color_2, linewidth=1.8),
    ]
    labels = [
        f"{lap_1.driver}  {format_lap_time(lap_1.lap_time_s)}",
        f"{lap_2.driver}  {format_lap_time(lap_2.lap_time_s)}",
    ]
    if with_apex_marker:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="v",
                markersize=6,
                markerfacecolor="none",
                markeredgecolor=theme.muted,
            )
        )
        labels.append("apex")
    legend = ax.legend(handles, labels, loc="lower right", frameon=False)
    for text in legend.get_texts():
        text.set_color(theme.ink)


def _annotate_speed(
    ax: Any,
    lap_1: FastestLap,
    lap_2: FastestLap,
    dist_1: NDArray[np.float64],
    dist_2: NDArray[np.float64],
    speed_1: NDArray[np.float64],
    speed_2: NDArray[np.float64],
    options: PlotOptions,
) -> None:
    """Apexes, identity and lap times — the speed panel is the figure's anchor."""
    theme = options.theme
    # Apexes detected on the reference driver's trace, as subtle markers.
    corners = detect_corners(dist_1, speed_1)
    if len(corners) > 0:
        ax.plot(
            dist_1[corners.apex_indices],
            speed_1[corners.apex_indices],
            linestyle="none",
            marker="v",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor=theme.muted,
            label="apex",
        )

    _legend(ax, lap_1, lap_2, options, with_apex_marker=len(corners) > 0)

    # Direct labels at the line ends (ink, not series colour), nudged apart so
    # they stay readable even when both traces finish at similar speeds.
    faster_end = speed_1[-1] >= speed_2[-1]
    for dist, speed, lap, above in (
        (dist_1, speed_1, lap_1, faster_end),
        (dist_2, speed_2, lap_2, not faster_end),
    ):
        ax.annotate(
            lap.driver,
            xy=(float(dist[-1]), float(speed[-1])),
            xytext=(6, 8 if above else -8),
            textcoords="offset points",
            color=theme.ink,
            fontsize=9,
            fontweight="bold",
            va="center",
        )


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
    log_path = setup_logging("f1lab.log")
    log_uncaught_exceptions(LOGGER)
    args = build_parser().parse_args(argv)
    driver_1, driver_2 = (code.upper() for code in args.drivers)

    cache_dir = enable_cache()
    LOGGER.info("Log file: %s", log_path)
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
