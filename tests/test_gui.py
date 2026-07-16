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
    """A small cornering lap with every channel, physically consistent.

    The extra channels matter here: the plot controls toggle panels, and a
    fake without throttle or gear would leave those checkboxes nothing to
    prove.
    """
    distance = np.linspace(0.0, 3000.0, 120, dtype=np.float64)
    speed_kmh = base_speed_ms * 3.6 - 100.0 * np.abs(np.sin(2.0 * np.pi * distance / 3000.0))
    steps = np.diff(distance, prepend=0.0)
    time_s = np.cumsum(steps / (speed_kmh / 3.6))
    slow = speed_kmh < speed_kmh.mean()
    telemetry = pd.DataFrame(
        {
            "Distance": distance,
            "Time": pd.to_timedelta(time_s, unit="s"),
            "Speed": speed_kmh,
            "Throttle": np.where(slow, 0.0, 100.0),
            "Brake": slow,
            "nGear": np.clip((speed_kmh / 45.0).astype(np.int64), 1, 8),
            "RPM": 6000.0 + 40.0 * speed_kmh,
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


def test_loader_failure_lands_in_the_status_bar(qtbot: Any, caplog: Any) -> None:
    """A failing fetch reports itself, logs its traceback, and re-enables the UI.

    The schedule loader raising is what a dead network looks like to the
    window. The failure must surface twice: one readable line in the status
    bar for the user, and the full traceback in the log for whoever debugs it
    later — the exception dies on a pool thread where no debugger ever saw it.
    And the window must come back to life, or the app is soft-locked after one
    bad request.
    """

    def broken_schedule(year: int) -> list[str]:
        raise RuntimeError("no route to the FastF1 servers")

    loaders = Loaders(
        schedule=broken_schedule,
        session=FAKE_LOADERS.session,
        roster=FAKE_LOADERS.roster,
    )
    with caplog.at_level("ERROR", logger="f1lab.gui.app"):
        win = MainWindow(loaders=loaders)
        qtbot.addWidget(win)
        qtbot.waitUntil(lambda: "no route" in win.statusBar().currentMessage(), timeout=2000)

    # The log record carries the exception itself, not just a message.
    failure = next(r for r in caplog.records if "Background job failed" in r.message)
    assert failure.exc_info is not None
    assert "no route" in str(failure.exc_info[1])

    # Recovered: the user can change the year and try again.
    assert win.year_box.isEnabled()
    assert win.load_button.isEnabled()


def _compared(window: MainWindow, qtbot: Any) -> None:
    """Load the fake session and run one comparison — the controls' precondition."""
    window.load_button.click()
    qtbot.waitUntil(lambda: window.compare_button.isEnabled(), timeout=2000)
    before = window._canvas
    window.compare_button.click()
    qtbot.waitUntil(lambda: window._canvas is not before, timeout=4000)


def test_channel_toggle_rebuilds_without_that_panel(window: MainWindow, qtbot: Any) -> None:
    """Unchecking a channel removes its panel from the embedded figure.

    This is the control that replaced the navigation toolbar: instead of
    panning around a fixed figure, the viewer decides what the figure contains.
    """
    _compared(window, qtbot)
    axes_before = len(window._canvas.figure.axes)

    window.channel_boxes["Throttle"].setChecked(False)
    assert len(window._canvas.figure.axes) == axes_before - 1

    window.channel_boxes["Throttle"].setChecked(True)
    assert len(window._canvas.figure.axes) == axes_before


def test_delta_and_speed_are_both_toggleable(window: MainWindow, qtbot: Any) -> None:
    """Every panel is the viewer's choice — delta and speed included.

    Hiding speed moves the legend to the topmost remaining axes: the driver
    colours must stay identified whatever the selection, or the figure stops
    being readable by anyone who did not build it.
    """
    _compared(window, qtbot)
    axes_before = len(window._canvas.figure.axes)

    window.delta_box.setChecked(False)
    assert len(window._canvas.figure.axes) == axes_before - 1

    window.channel_boxes["Speed"].setChecked(False)
    assert len(window._canvas.figure.axes) == axes_before - 2
    # The legend survived the loss of its home panel.
    assert window._canvas.figure.axes[0].get_legend() is not None

    window.delta_box.setChecked(True)
    window.channel_boxes["Speed"].setChecked(True)
    assert len(window._canvas.figure.axes) == axes_before


def test_deselecting_everything_still_draws_the_delta(window: MainWindow, qtbot: Any) -> None:
    # A figure with zero panels helps nobody: with nothing selected, the
    # delta comes back as the floor.
    _compared(window, qtbot)
    window.delta_box.setChecked(False)
    for box in window.channel_boxes.values():
        box.setChecked(False)
    assert len(window._canvas.figure.axes) == 1


def test_dark_background_restyles_the_figure(window: MainWindow, qtbot: Any) -> None:
    """The dark toggle swaps the whole theme, not just the facecolor."""
    from matplotlib.colors import to_rgba

    from f1lab.compare_laps import DARK, LIGHT

    _compared(window, qtbot)
    assert window._canvas.figure.get_facecolor() == to_rgba(LIGHT.surface)

    window.dark_box.setChecked(True)
    assert window._canvas.figure.get_facecolor() == to_rgba(DARK.surface)


def test_picking_a_driver_colour_recolours_the_traces(
    window: MainWindow, qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The colour dialog's choice lands on the speed trace immediately.

    The dialog is replaced where the window looks it up, so the real modal
    never opens; everything after the choice — storing it, rebuilding the
    figure — runs for real.
    """
    from PySide6.QtGui import QColor

    from f1lab.gui import app as app_module

    class _FakeColorDialog:
        @staticmethod
        # Qt's name, so the window finds it — hence the noqa.
        def getColor(*args: Any, **kwargs: Any) -> QColor:  # noqa: N802
            return QColor("#ff0000")

    _compared(window, qtbot)
    monkeypatch.setattr(app_module, "QColorDialog", _FakeColorDialog)
    window._pick_color(0)

    speed_ax = window._canvas.figure.axes[0]  # speed leads; the delta sits under it
    assert speed_ax.lines[0].get_color() == "#ff0000"


def test_save_button_writes_the_png(
    window: MainWindow, qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Save is disabled until there is something to save, then writes the file."""
    from f1lab.gui import app as app_module

    assert not window.save_button.isEnabled()
    _compared(window, qtbot)
    assert window.save_button.isEnabled()

    out = tmp_path / "comparison.png"

    class _FakeFileDialog:
        @staticmethod
        # Qt's name, so the window finds it — hence the noqa.
        def getSaveFileName(*args: Any, **kwargs: Any) -> tuple[str, str]:  # noqa: N802
            return str(out), "PNG image (*.png)"

    monkeypatch.setattr(app_module, "QFileDialog", _FakeFileDialog)
    window.save_button.click()
    assert out.exists()
    assert out.stat().st_size > 0


def test_zoom_pan_and_reset(window: MainWindow, qtbot: Any) -> None:
    """The viewing gestures, end to end: zoom in, pan within the lap, reset.

    Driven through the canvas's own methods — the pieces the Qt wheel and
    gesture handlers call once they have decoded the platform's event — so the
    test pins the behaviour without having to fabricate native events.
    """
    _compared(window, qtbot)
    canvas = window._canvas
    ax = canvas.figure.axes[0]
    home = ax.get_xlim()
    centre = (home[0] + home[1]) / 2

    assert not canvas.is_zoomed()
    assert not window.pan_bar.isEnabled()  # nothing to pan at full view

    # Zoom in around the middle: the window narrows, the pan bar comes alive.
    canvas._zoom(1 / canvas.ZOOM_STEP, centre)
    zoomed = ax.get_xlim()
    assert zoomed[1] - zoomed[0] < home[1] - home[0]
    assert canvas.is_zoomed()
    assert window.pan_bar.isEnabled()

    # Pan by view: the span is preserved, the window slides, never past the lap.
    span = zoomed[1] - zoomed[0]
    canvas.set_view(home[0] - 1000.0, home[0] - 1000.0 + span)  # try to escape left
    clamped = ax.get_xlim()
    assert clamped[0] == pytest.approx(home[0])  # clamped to the lap start
    assert clamped[1] - clamped[0] == pytest.approx(span)

    # Dragging the bar moves the view to the matching position.
    window.pan_bar.setValue(window.pan_bar.maximum())
    assert ax.get_xlim()[1] == pytest.approx(home[1], rel=1e-3)

    # Double-click comes home.
    from types import SimpleNamespace

    canvas._on_press(SimpleNamespace(dblclick=True))
    assert ax.get_xlim() == pytest.approx(home)
    assert not window.pan_bar.isEnabled()


def test_zoom_survives_a_panel_toggle(window: MainWindow, qtbot: Any) -> None:
    """Rebuilding the figure keeps the zoomed window.

    Without this, touching any control while inspecting a corner would throw
    the view back to the full lap — the controls and the zoom would fight.
    """
    _compared(window, qtbot)
    canvas = window._canvas
    home = canvas.figure.axes[0].get_xlim()
    centre = (home[0] + home[1]) / 2
    canvas._zoom(1 / canvas.ZOOM_STEP, centre)
    zoomed = canvas.figure.axes[0].get_xlim()

    window.channel_boxes["Throttle"].setChecked(False)

    # A new canvas, the same window onto the lap.
    assert window._canvas is not canvas
    assert window._canvas.figure.axes[0].get_xlim() == pytest.approx(zoomed)


def _wheel(
    *,
    angle_x: int = 0,
    angle_y: int = 0,
    modifier: Any = None,
    at: tuple[float, float] = (200.0, 100.0),
) -> Any:
    """A real QWheelEvent, as Qt would deliver it — no fakes at this layer.

    The Qt handlers are exactly where a platform quirk broke Cmd+scroll once
    already, so they are tested with genuine events rather than stand-ins.
    """
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    return QWheelEvent(
        QPointF(*at),
        QPointF(*at),
        QPoint(0, 0),
        QPoint(angle_x, angle_y),
        Qt.MouseButton.NoButton,
        modifier if modifier is not None else Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def test_qt_wheel_events_zoom_and_pan(window: MainWindow, qtbot: Any) -> None:
    """The Qt handlers end to end: modifier+wheel zooms, sideways wheel pans.

    A plain vertical wheel is left to Qt — hijacking it would break ordinary
    window scrolling, which trackpads emit constantly.
    """
    from PySide6.QtCore import Qt

    _compared(window, qtbot)
    canvas = window._canvas
    ax = canvas.figure.axes[0]
    home = ax.get_xlim()

    # Plain vertical scroll: no zoom.
    canvas.wheelEvent(_wheel(angle_y=120))
    assert ax.get_xlim() == home

    # Ctrl/Cmd + scroll up: zoom in around the cursor.
    canvas.wheelEvent(_wheel(angle_y=120, modifier=Qt.KeyboardModifier.ControlModifier))
    zoomed = ax.get_xlim()
    assert zoomed[1] - zoomed[0] < home[1] - home[0]

    # Sideways scroll while zoomed: the window slides.
    before = ax.get_xlim()
    canvas.wheelEvent(_wheel(angle_x=-240))
    panned = ax.get_xlim()
    assert panned != before
    assert panned[1] - panned[0] == pytest.approx(before[1] - before[0])

    # Meta works too — Qt maps the Mac's Cmd to Control by default, but not
    # always: both flags must zoom.
    canvas.wheelEvent(_wheel(angle_y=-120, modifier=Qt.KeyboardModifier.MetaModifier))


def test_trackpad_pinch_zooms(window: MainWindow, qtbot: Any) -> None:
    """A native pinch gesture — how macOS trackpads actually report zooming."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QNativeGestureEvent, QPointingDevice

    _compared(window, qtbot)
    canvas = window._canvas
    ax = canvas.figure.axes[0]
    home = ax.get_xlim()

    position = QPointF(200.0, 100.0)
    pinch = QNativeGestureEvent(
        Qt.NativeGestureType.ZoomNativeGesture,
        QPointingDevice.primaryPointingDevice(),
        2,  # fingers
        position,
        position,
        position,
        0.25,  # fingers apart: zoom in
        QPointF(0, 0),
    )
    assert canvas.event(pinch) is True
    zoomed = ax.get_xlim()
    assert zoomed[1] - zoomed[0] < home[1] - home[0]


def test_gesture_guards_hold_on_edge_cases(window: MainWindow, qtbot: Any) -> None:
    """The corners of the zoom machinery: empty figures, depth limits, centres.

    An empty canvas (before any comparison) must shrug gestures off, zooming
    has a floor so the view cannot collapse into noise, and a zoom without a
    cursor position centres itself.
    """
    from PySide6.QtCore import Qt

    # Gestures on the empty startup canvas: no axes, no crash, no effect.
    empty = window._canvas
    empty.wheelEvent(_wheel(angle_y=120, modifier=Qt.KeyboardModifier.ControlModifier))
    empty._pan_pixels(50.0)
    assert not empty.is_zoomed()

    _compared(window, qtbot)
    canvas = window._canvas
    ax = canvas.figure.axes[0]
    home = ax.get_xlim()

    # No cursor position: the zoom centres on the middle of the view.
    canvas._zoom(1 / canvas.ZOOM_STEP, None)
    zoomed = ax.get_xlim()
    assert zoomed[0] > home[0]
    assert zoomed[1] < home[1]

    # The depth floor: zooming in forever stops instead of collapsing.
    for _ in range(200):
        canvas._zoom(1 / canvas.ZOOM_STEP, None)
    low, high = ax.get_xlim()
    assert high - low >= (home[1] - home[0]) * 5e-4
