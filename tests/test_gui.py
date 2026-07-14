"""Tests for the desktop app, driven through fake loaders — no network, no Qt
event loop of its own (pytest-qt supplies one).

The window takes its three outside-world functions (schedule, session, roster)
as an injected ``Loaders`` value, which is what makes these tests possible: the
fakes below answer instantly, so the tests exercise the real widget chain —
year refills the Grand Prix list, loading fills the rosters, comparing embeds a
figure — without downloading a byte.

The module is skipped when PySide6 is not installed, mirroring how the parity
suite skips without the C++ extension: the ``gui`` extra is optional, and the
test suite must pass either way.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6", reason="the gui extra is not installed")

from f1lab.analysis import FastestLap
from f1lab.gui.app import Loaders, MainWindow


def _lap(driver: str, base_speed_ms: float) -> FastestLap:
    """A small cornering lap, physically consistent, cheap to build."""
    distance = np.linspace(0.0, 3000.0, 120, dtype=np.float64)
    speed_kmh = base_speed_ms * 3.6 - 100.0 * np.abs(np.sin(2.0 * np.pi * distance / 3000.0))
    steps = np.diff(distance, prepend=0.0)
    time_s = np.cumsum(steps / (speed_kmh / 3.6))
    telemetry = pd.DataFrame(
        {
            "Distance": distance,
            "Time": pd.to_timedelta(time_s, unit="s"),
            "Speed": speed_kmh,
        }
    )
    return FastestLap(driver=driver, lap_time_s=float(time_s[-1]), telemetry=telemetry)


class FakeSession:
    """The attributes the window and fastest_lap_telemetry read."""

    name = "Race"
    event: ClassVar[dict[str, str]] = {"EventName": "British Grand Prix"}

    def __init__(self) -> None:
        self._laps = {"VER": _lap("VER", 60.0), "NOR": _lap("NOR", 58.0)}
        self.laps = self  # quacks just enough for fastest_lap_telemetry

    # The two pick_* calls fastest_lap_telemetry chains on session.laps:
    def pick_drivers(self, driver: str) -> Any:
        self._picked = driver
        return self

    def pick_fastest(self) -> Any:
        lap = self._laps[self._picked]

        class _Lap:
            def __init__(self, inner: FastestLap) -> None:
                self._inner = inner

            def __getitem__(self, key: str) -> pd.Timedelta:
                assert key == "LapTime"
                return pd.Timedelta(self._inner.lap_time_s, unit="s")

            def get_car_data(self) -> Any:
                outer = self

                class _CarData:
                    def add_distance(self) -> pd.DataFrame:
                        return outer._inner.telemetry

                return _CarData()

        return _Lap(lap)


FAKE_LOADERS = Loaders(
    schedule=lambda year: [f"Grand Prix of {year}", "British Grand Prix"],
    session=lambda year, gp, code: FakeSession(),
    roster=lambda session: [("VER", "Max Verstappen"), ("NOR", "Lando Norris")],
)


@pytest.fixture
def window(qtbot: Any) -> MainWindow:
    win = MainWindow(loaders=FAKE_LOADERS)
    qtbot.addWidget(win)
    # The constructor kicks off the initial schedule fetch; let it land.
    qtbot.waitUntil(lambda: win.gp_box.count() > 0, timeout=2000)
    return win


def test_year_change_refills_the_grand_prix_list(window: MainWindow, qtbot: Any) -> None:
    """Picking a season repopulates the calendar — the first link of the chain."""
    assert window.gp_box.count() == 2
    assert "British Grand Prix" in [window.gp_box.itemText(i) for i in range(2)]
    # Drivers stay disabled until a session is loaded: the roster belongs to a
    # session, and pretending otherwise is how you offer HAM in a 2026 seat.
    assert not window.driver_1_box.isEnabled()
    assert not window.compare_button.isEnabled()


def test_loading_a_session_fills_the_rosters(window: MainWindow, qtbot: Any) -> None:
    window.load_button.click()
    qtbot.waitUntil(lambda: window.driver_1_box.count() > 0, timeout=2000)
    codes = [window.driver_1_box.itemData(i) for i in range(window.driver_1_box.count())]
    assert codes == ["VER", "NOR"]
    # The second box preselects the second driver, so the default comparison is
    # never a driver against themselves.
    assert window.driver_2_box.currentData() == "NOR"
    assert window.compare_button.isEnabled()


def test_compare_embeds_a_figure(window: MainWindow, qtbot: Any) -> None:
    """The full click path: load, compare, and a real figure lands on the canvas."""
    window.load_button.click()
    qtbot.waitUntil(lambda: window.compare_button.isEnabled(), timeout=2000)

    before = window._canvas
    window.compare_button.click()
    qtbot.waitUntil(lambda: window._canvas is not before, timeout=4000)

    # The embedded figure is the real multi-panel comparison: delta + speed.
    assert len(window._canvas.figure.axes) >= 2


def test_same_driver_twice_is_refused(window: MainWindow, qtbot: Any) -> None:
    window.load_button.click()
    qtbot.waitUntil(lambda: window.compare_button.isEnabled(), timeout=2000)
    window.driver_2_box.setCurrentIndex(0)  # same as driver 1

    before = window._canvas
    window.compare_button.click()
    assert "different drivers" in window.statusBar().currentMessage()
    assert window._canvas is before  # nothing was rendered


def test_gui_entry_point_reports_missing_qt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the gui extra, f1lab-gui explains itself instead of tracebacking."""
    import builtins

    from f1lab import gui

    real_import = builtins.__import__

    def refuse_qt(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("f1lab.gui.app") or name.startswith("PySide6"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_qt)
    assert gui.main() == 1


def test_loader_failure_lands_in_the_status_bar(qtbot: Any) -> None:
    """A failing fetch reports itself and re-enables the controls.

    The schedule loader raising is what a dead network looks like to the
    window. The failure must surface where the user can read it — and the
    window must come back to life, or the app is soft-locked after one bad
    request.
    """

    def broken_schedule(year: int) -> list[str]:
        raise RuntimeError("no route to the FastF1 servers")

    loaders = Loaders(
        schedule=broken_schedule,
        session=FAKE_LOADERS.session,
        roster=FAKE_LOADERS.roster,
    )
    win = MainWindow(loaders=loaders)
    qtbot.addWidget(win)

    qtbot.waitUntil(lambda: "no route" in win.statusBar().currentMessage(), timeout=2000)
    # Recovered: the user can change the year and try again.
    assert win.year_box.isEnabled()
    assert win.load_button.isEnabled()
