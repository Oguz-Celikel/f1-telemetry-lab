"""Tests for the CLI and plotting layer.

FastF1 is imported here (``f1lab.compare_laps`` pulls it in), but nothing in
these tests touches the network: the parser tests are pure string handling, and
the plot is rendered from synthetic telemetry. Session loading — the one truly
I/O-bound step — is the only thing left uncovered, and it is deliberately kept
to a thin wrapper so there is little there to break.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fastf1
import numpy as np
import pandas as pd
import pytest

from f1lab import compare_laps
from f1lab.analysis import FastestLap
from f1lab.compare_laps import (
    build_parser,
    enable_cache,
    main,
    plot_comparison,
    select_panels,
)


def test_parser_parses_all_arguments() -> None:
    # Note the lowercase driver codes: the parser passes them through untouched.
    # Upper-casing belongs to main(), and keeping that boundary explicit is the
    # point of testing the parser separately.
    args = build_parser().parse_args(
        ["--year", "2026", "--gp", "Silverstone", "--session", "Q", "--drivers", "ver", "nor"]
    )
    assert args.year == 2026
    assert args.gp == "Silverstone"
    assert args.session == "Q"
    assert args.drivers == ["ver", "nor"]


def test_parser_defaults() -> None:
    # The defaults are part of the CLI's contract with the user, so they are
    # pinned here rather than left to whatever argparse was last told.
    args = build_parser().parse_args(["--year", "2026", "--gp", "Monza", "--drivers", "LEC", "PIA"])
    assert args.session == "R"
    assert args.output_dir == Path("output")


def test_parser_requires_exactly_two_drivers() -> None:
    # A comparison needs two laps. argparse reports the error and exits, so
    # SystemExit is the expected failure, not ValueError.
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--year", "2026", "--gp", "Monza", "--drivers", "LEC"])


def _fastest_lap(
    driver: str,
    base_speed_ms: float,
    *,
    corners: bool = False,
    channels: bool = False,
    drs: bool = False,
) -> FastestLap:
    """A synthetic lap in the shape the plotting code expects.

    The flags switch on the things the figure only draws when the data supports
    them, so each can be tested for rather than hoped for:

    ``corners`` gives the speed trace real dips, so the apex markers have
    something to mark. ``channels`` adds throttle, brake, gear and RPM, so the
    lower panels appear. ``drs`` adds a DRS column that actually opens — without
    it the channel is flat and the panel is dropped, which is what happens with
    real 2026 data.
    """
    distance = np.linspace(0.0, 5000.0, 200, dtype=np.float64)
    speed_kmh = np.full(distance.shape, base_speed_ms * 3.6)
    if corners:
        # Four slow points per lap, deep enough to clear the detector's
        # prominence threshold.
        speed_kmh = speed_kmh - 120.0 * np.abs(np.sin(4.0 * np.pi * distance / 5000.0))
    speed_ms = speed_kmh / 3.6
    # Integrate the time from the speed the car is actually doing, so the lap is
    # physically consistent even when it has corners.
    steps = np.diff(distance, prepend=0.0)
    time_s = np.cumsum(steps / speed_ms)
    frame: dict[str, object] = {
        "Distance": distance,
        "Time": pd.to_timedelta(time_s, unit="s"),
        "Speed": speed_kmh,
    }
    if channels:
        slow = speed_kmh < speed_kmh.mean()
        frame["Throttle"] = np.where(slow, 0.0, 100.0)
        frame["Brake"] = slow  # bool, as FastF1 reports it
        frame["nGear"] = np.clip((speed_kmh / 45.0).astype(np.int64), 1, 8)
        frame["RPM"] = 6000.0 + 40.0 * speed_kmh
        # A DRS column that never opens: 8 means "eligible, not activated".
        frame["DRS"] = np.full(distance.shape, 8, dtype=np.int64)
    if drs:
        # Codes >= 10 mean the flap is open; open it on the second half.
        frame["DRS"] = np.where(distance > 2500.0, 12, 8).astype(np.int64)
    return FastestLap(driver=driver, lap_time_s=float(time_s[-1]), telemetry=pd.DataFrame(frame))


def test_plot_comparison_writes_png(tmp_path: Path) -> None:
    """The whole rendering path runs end to end and produces a file.

    A modest assertion on purpose: comparing pixels would break on every
    matplotlib upgrade and on every deliberate styling change. What this
    guards is that the figure can actually be built and saved — that no axis,
    label or annotation raises — and that the output directory is created when
    it does not exist (hence the nested ``plots/`` path).
    """
    lap_1 = _fastest_lap("VER", 55.0)
    lap_2 = _fastest_lap("NOR", 52.0)
    delta = np.linspace(0.0, 1.5, 200, dtype=np.float64)
    out_path = tmp_path / "plots" / "comparison.png"
    plot_comparison(lap_1, lap_2, delta, "Test GP 2026 — Race", out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_comparison_marks_apexes(tmp_path: Path) -> None:
    """A lap with corners exercises the apex-marker branch of the plot.

    The flat lap above never reaches it: no corners, no markers. Passing a
    cornering trace through the same code is what proves the markers can be
    drawn at all — the detector runs inside plot_comparison, so a failure there
    would otherwise only surface in a real run.
    """
    lap_1 = _fastest_lap("VER", 55.0, corners=True)
    lap_2 = _fastest_lap("NOR", 52.0, corners=True)
    delta = np.linspace(0.0, 1.5, 200, dtype=np.float64)
    out_path = tmp_path / "with_corners.png"
    plot_comparison(lap_1, lap_2, delta, "Test GP 2026 — Race", out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


class FakeSession:
    """The slice of a loaded FastF1 session that ``main`` reads."""

    def __init__(self, laps: dict[str, FastestLap]) -> None:
        self._laps = laps
        self.name = "Race"
        self.event = {"EventName": "British Grand Prix"}

    def fastest_for(self, driver: str) -> FastestLap:
        return self._laps[driver]


@pytest.fixture
def offline_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeSession:
    """Replace the two functions in ``main`` that reach the network or disk.

    Everything else — argument parsing, engine selection, the delta computation,
    corner detection, rendering — runs for real. Only ``load_session`` (an HTTP
    download) and ``enable_cache`` (which would create a cache directory in the
    repository) are stubbed, so the test covers the orchestration rather than
    the mock.
    """
    session = FakeSession(
        {
            "VER": _fastest_lap("VER", 55.0, corners=True),
            "NOR": _fastest_lap("NOR", 52.0, corners=True),
        }
    )
    monkeypatch.setattr(compare_laps, "enable_cache", lambda: tmp_path / "cache")
    monkeypatch.setattr(compare_laps, "load_session", lambda year, gp, session_name: session)
    monkeypatch.setattr(
        compare_laps,
        "fastest_lap_telemetry",
        lambda sess, driver: sess.fastest_for(driver),
    )
    return session


def test_main_renders_a_plot(offline_main: FakeSession, tmp_path: Path) -> None:
    """The full CLI path, end to end: arguments in, PNG on disk, exit code 0."""
    exit_code = main(
        [
            "--year",
            "2026",
            "--gp",
            "Silverstone",
            "--drivers",
            "ver",  # lowercase on purpose: main upper-cases them
            "nor",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    # The filename follows the deterministic naming contract, built from the
    # event name the session reported rather than the --gp argument.
    out_path = tmp_path / "2026_british_grand_prix_r_ver_vs_nor.png"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_main_reports_a_missing_driver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A driver with no valid lap is a clean exit code 1, not a traceback."""

    def raise_missing(session: Any, driver: str) -> FastestLap:
        raise ValueError(f"No valid fastest lap found for driver {driver!r}")

    monkeypatch.setattr(compare_laps, "enable_cache", lambda: tmp_path / "cache")
    monkeypatch.setattr(compare_laps, "load_session", lambda *args: object())
    monkeypatch.setattr(compare_laps, "fastest_lap_telemetry", raise_missing)

    exit_code = main(["--year", "2026", "--gp", "Monza", "--drivers", "VER", "XXX"])
    assert exit_code == 1


def test_enable_cache_honours_the_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """FASTF1_CACHE decides where the cache lives, and the directory is created."""
    cache_dir = tmp_path / "telemetry-cache"
    monkeypatch.setenv("FASTF1_CACHE", str(cache_dir))
    # Stub out FastF1's own cache registration: this test is about the path
    # resolution, not about FastF1's internals.
    monkeypatch.setattr(fastf1.Cache, "enable_cache", lambda path: None)

    resolved = enable_cache()
    assert resolved == cache_dir
    assert cache_dir.is_dir()


class TestPanelSelection:
    """Which channels get a panel — the decision, tested apart from the drawing.

    Pulling the choice out of plot_comparison is what makes it testable at all:
    reading panels back off a rendered figure would mean re-deriving the rule in
    the test, which tests a copy of the logic rather than the logic.
    """

    def test_all_recorded_channels_get_a_panel(self) -> None:
        lap_1 = _fastest_lap("VER", 55.0, corners=True, channels=True, drs=True)
        lap_2 = _fastest_lap("NOR", 52.0, corners=True, channels=True, drs=True)
        columns = [panel.column for panel, _, _ in select_panels(lap_1.telemetry, lap_2.telemetry)]
        assert columns == ["Speed", "Throttle", "Brake", "nGear", "RPM", "DRS"]

    def test_a_flat_channel_is_dropped(self) -> None:
        # The real case this guards: 2026 cars report DRS as a constant — the
        # regulations replaced it with active aerodynamics — so the panel would
        # otherwise be an empty strip stealing height from the rest.
        lap_1 = _fastest_lap("VER", 55.0, corners=True, channels=True)
        lap_2 = _fastest_lap("NOR", 52.0, corners=True, channels=True)
        columns = [panel.column for panel, _, _ in select_panels(lap_1.telemetry, lap_2.telemetry)]
        assert "DRS" not in columns
        assert "Speed" in columns  # everything else survives

    def test_missing_channels_are_skipped(self) -> None:
        # Older or partial telemetry may simply not carry a column.
        lap_1 = _fastest_lap("VER", 55.0, corners=True)
        lap_2 = _fastest_lap("NOR", 52.0, corners=True)
        columns = [panel.column for panel, _, _ in select_panels(lap_1.telemetry, lap_2.telemetry)]
        assert columns == ["Speed"]

    def test_speed_survives_even_when_flat(self) -> None:
        # Speed is the figure's anchor: the apexes, the legend and the driver
        # labels all live on it, so the has-signal rule must not remove it.
        lap_1 = _fastest_lap("VER", 55.0)  # constant speed
        lap_2 = _fastest_lap("NOR", 52.0)
        columns = [panel.column for panel, _, _ in select_panels(lap_1.telemetry, lap_2.telemetry)]
        assert columns == ["Speed"]

    def test_one_driver_using_drs_is_enough(self) -> None:
        # If either lap varies in a channel, the comparison is worth drawing.
        lap_1 = _fastest_lap("VER", 55.0, corners=True, channels=True, drs=True)
        lap_2 = _fastest_lap("NOR", 52.0, corners=True, channels=True)  # flat DRS
        columns = [panel.column for panel, _, _ in select_panels(lap_1.telemetry, lap_2.telemetry)]
        assert "DRS" in columns


def test_plot_renders_every_panel(tmp_path: Path) -> None:
    """The full figure — all six channels plus the delta — renders and saves.

    Each channel keeps its own y-scale in its own panel: speed, throttle and RPM
    share nothing but the distance they were measured at, and putting two units
    on one axis is the standard way to invent a correlation that is not there.
    """
    lap_1 = _fastest_lap("VER", 55.0, corners=True, channels=True, drs=True)
    lap_2 = _fastest_lap("NOR", 52.0, corners=True, channels=True, drs=True)
    delta = np.linspace(0.0, 1.5, 200, dtype=np.float64)
    out_path = tmp_path / "all_panels.png"
    plot_comparison(lap_1, lap_2, delta, "Test GP 2026 — Race", out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_options_filter_the_panels(tmp_path: Path) -> None:
    """Deselected panels do not appear — and identity survives losing speed.

    With only throttle chosen the axes are delta + throttle, and the legend —
    which normally lives on the speed panel — moves to the top, so the driver
    colours stay identified whatever the selection.
    """
    from f1lab.compare_laps import PlotOptions, build_comparison_figure

    lap_1 = _fastest_lap("VER", 55.0, corners=True, channels=True)
    lap_2 = _fastest_lap("NOR", 52.0, corners=True, channels=True)
    delta = np.linspace(0.0, 1.5, 200, dtype=np.float64)

    everything = build_comparison_figure(lap_1, lap_2, delta, "t")
    only_throttle = build_comparison_figure(
        lap_1, lap_2, delta, "t", PlotOptions(channels=frozenset({"Throttle"}))
    )
    assert len(everything.axes) > 3
    assert len(only_throttle.axes) == 2  # delta + throttle
    assert only_throttle.axes[0].get_legend() is not None

    no_delta = build_comparison_figure(
        lap_1, lap_2, delta, "t", PlotOptions(channels=frozenset({"Speed"}), show_delta=False)
    )
    assert len(no_delta.axes) == 1  # speed alone

    # Nothing selected at all: the delta returns rather than an empty figure.
    nothing = build_comparison_figure(
        lap_1, lap_2, delta, "t", PlotOptions(channels=frozenset(), show_delta=False)
    )
    assert len(nothing.axes) == 1
    assert nothing.axes[0].get_legend() is not None


def test_dark_theme_styles_the_whole_figure() -> None:
    """DARK is a validated palette of its own, not the light theme inverted."""
    from matplotlib.colors import to_rgba

    from f1lab.compare_laps import DARK, PlotOptions, build_comparison_figure

    lap_1 = _fastest_lap("VER", 55.0)
    lap_2 = _fastest_lap("NOR", 52.0)
    delta = np.linspace(0.0, 1.5, 200, dtype=np.float64)
    figure = build_comparison_figure(lap_1, lap_2, delta, "t", PlotOptions(theme=DARK))

    assert figure.get_facecolor() == to_rgba(DARK.surface)
    assert figure.axes[0].get_facecolor() == to_rgba(DARK.surface)
    # The traces wear the dark theme's validated driver colours.
    assert figure.axes[0].lines[0].get_color() == DARK.driver_1  # speed is the top row


def test_custom_driver_colours_override_the_theme() -> None:
    from f1lab.compare_laps import PlotOptions, build_comparison_figure

    lap_1 = _fastest_lap("VER", 55.0)
    lap_2 = _fastest_lap("NOR", 52.0)
    delta = np.linspace(0.0, 1.5, 200, dtype=np.float64)
    figure = build_comparison_figure(
        lap_1,
        lap_2,
        delta,
        "t",
        PlotOptions(driver_1_color="#123456", driver_2_color="#654321"),
    )
    speed_ax = figure.axes[0]  # speed is the top row
    assert speed_ax.lines[0].get_color() == "#123456"
    assert speed_ax.lines[1].get_color() == "#654321"
