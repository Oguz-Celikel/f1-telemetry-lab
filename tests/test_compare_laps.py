"""Tests for the CLI and plotting layer.

FastF1 is imported here (``f1lab.compare_laps`` pulls it in), but nothing in
these tests touches the network: the parser tests are pure string handling, and
the plot is rendered from synthetic telemetry. Session loading — the one truly
I/O-bound step — is the only thing left uncovered, and it is deliberately kept
to a thin wrapper so there is little there to break.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1lab.analysis import FastestLap
from f1lab.compare_laps import build_parser, plot_comparison


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


def _fastest_lap(driver: str, base_speed_ms: float) -> FastestLap:
    """A constant-speed lap in the shape the plotting code expects."""
    distance = np.linspace(0.0, 5000.0, 200, dtype=np.float64)
    telemetry = pd.DataFrame(
        {
            "Distance": distance,
            "Time": pd.to_timedelta(distance / base_speed_ms, unit="s"),
            "Speed": np.full(distance.shape, base_speed_ms * 3.6),
        }
    )
    lap_time_s = float(distance[-1] / base_speed_ms)
    return FastestLap(driver=driver, lap_time_s=lap_time_s, telemetry=telemetry)


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
