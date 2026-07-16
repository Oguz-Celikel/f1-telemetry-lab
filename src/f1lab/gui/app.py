"""The Qt window: pick a season, a Grand Prix, a session and two drivers.

Layout of the module, top to bottom:

* ``Loaders`` — the three functions that talk to the outside world (schedule,
  session, roster). They are injected into the window so the tests can hand in
  fakes and drive the whole UI without a network connection, exactly the way
  ``main()`` in the CLI is tested.
* ``Worker`` — FastF1 downloads take seconds to minutes, and any of that on the
  Qt main thread freezes the window. Every fetch runs in the global thread pool
  and reports back through signals, which Qt delivers on the main thread.
* ``MainWindow`` — the widgets and the chain between them: picking a year
  refills the Grand Prix list, loading a session fills the driver lists from
  the cars that actually ran in it, comparing embeds the figure in the window.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import fastf1
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QColor, QNativeGestureEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollBar,
    QVBoxLayout,
    QWidget,
)

from f1lab import native
from f1lab.analysis import compute_delta_time, distance_and_time, fastest_lap_telemetry
from f1lab.compare_laps import (
    DARK,
    LIGHT,
    PANELS,
    PlotOptions,
    build_comparison_figure,
    enable_cache,
    load_session,
)
from f1lab.logs import log_uncaught_exceptions, setup_logging

LOGGER = logging.getLogger(__name__)

# FastF1's telemetry coverage starts here; earlier seasons have no car data.
FIRST_SEASON = 2018
LAST_SEASON = 2026
SESSIONS = ("R", "Q", "S", "FP1", "FP2", "FP3")


@dataclass(frozen=True)
class Loaders:
    """The window's only contact with the outside world, as three functions.

    ``schedule`` returns the Grand Prix names of a season; ``session`` returns
    a loaded FastF1 session; ``roster`` lists (code, full name) for the drivers
    who ran in it. The defaults hit the network through FastF1; the tests
    replace them with fakes, so every click path can be exercised offline.
    """

    schedule: Callable[[int], list[str]]
    session: Callable[[int, str, str], Any]
    roster: Callable[[Any], list[tuple[str, str]]]


# The real loaders are thin wrappers around FastF1's network calls — testing
# them would mean mocking the very line each one consists of.
def _real_schedule(year: int) -> list[str]:  # pragma: no cover
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return [str(name) for name in schedule["EventName"]]


def _real_session(year: int, gp: str, session_code: str) -> Any:  # pragma: no cover
    enable_cache()
    return load_session(year, gp, session_code)


def _real_roster(session: Any) -> list[tuple[str, str]]:  # pragma: no cover
    results = session.results
    return [(str(row["Abbreviation"]), str(row["FullName"])) for _, row in results.iterrows()]


REAL_LOADERS = Loaders(schedule=_real_schedule, session=_real_session, roster=_real_roster)


class _WorkerSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Worker(QRunnable):
    """Run one blocking job off the main thread and signal the outcome.

    The result crosses back to the GUI thread through the signal — the worker
    itself never touches a widget, which is the whole thread-safety story: Qt
    widgets may only be used from the thread that created them.
    """

    def __init__(self, job: Callable[[], object]) -> None:
        super().__init__()
        self.signals = _WorkerSignals()
        self._job = job

    @Slot()
    def run(self) -> None:
        try:
            result = self._job()
        except Exception as exc:
            # The status bar gets one readable line; the log file gets the full
            # traceback. Without this the stack is gone — the exception dies
            # here, on a pool thread, where no debugger ever saw it.
            LOGGER.exception("Background job failed")
            self.signals.failed.emit(str(exc))
        else:
            self.signals.done.emit(result)


class ZoomableCanvas(FigureCanvasQTAgg):
    """A figure canvas with the viewing gestures this app actually needs.

    Cmd+scroll (or a trackpad pinch) zooms the shared distance axis around
    the cursor, and a double-click restores the full lap. This replaces
    matplotlib's modal zoom tool, which is easy to enter and hard to leave —
    and because every panel shares the x axis, one zoom moves them all.
    Zooming never escapes the lap: the limits clamp to the full-lap view.
    """

    ZOOM_STEP = 1.2

    # Fires whenever the visible x window changes (zoom, pan, reset), so the
    # window can keep the pan bar under the plot in step with the gestures.
    view_changed = Signal()

    def __init__(self, figure: Figure) -> None:
        super().__init__(figure)
        # The full-lap limits, captured before any gesture touches them —
        # this is what a double-click comes home to.
        self._home_xlim: tuple[float, float] | None = (
            figure.axes[0].get_xlim() if figure.axes else None
        )
        self.mpl_connect("button_press_event", self._on_press)

    # -- the view, for whoever needs to mirror or restore it ----------------

    def view(self) -> tuple[float, float, float, float] | None:
        """(home_low, home_high, low, high), or None for an empty figure."""
        if not self.figure.axes or self._home_xlim is None:
            return None
        low, high = self.figure.axes[0].get_xlim()
        return (*self._home_xlim, low, high)

    def is_zoomed(self) -> bool:
        view = self.view()
        if view is None:
            return False
        home_low, home_high, low, high = view
        return (high - low) < (home_high - home_low) * 0.999

    def set_view(self, low: float, high: float) -> None:
        """Show exactly this x window (clamped to the lap) — used to carry the
        zoom across a rebuild, so changing a panel does not lose your place."""
        if not self.figure.axes or self._home_xlim is None:
            return
        home_low, home_high = self._home_xlim
        span = min(high - low, home_high - home_low)
        low = max(home_low, min(low, home_high - span))
        self.figure.axes[0].set_xlim(low, low + span)
        self.draw_idle()
        self.view_changed.emit()

    def _on_press(self, event: Any) -> None:
        if event.dblclick:
            self._reset_view()

    def wheelEvent(self, event: Any) -> None:  # noqa: N802 — Qt's name
        """Scrolling, straight from Qt rather than through matplotlib.

        matplotlib's scroll event names the held modifier differently across
        platforms and versions ("ctrl", "cmd", "control"…), which is how a
        Cmd+scroll can silently match nothing. Qt's own modifier flags are
        unambiguous, so the gesture is decided here: modifier+scroll zooms,
        a horizontal scroll pans a zoomed view, anything else stays Qt's.
        """
        # Qt maps the Mac's Cmd to ControlModifier by default; accepting Meta
        # too covers both keys on every platform.
        zoom_modifiers = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        if event.modifiers() & zoom_modifiers:
            delta = event.angleDelta().y() or event.pixelDelta().y()
            if delta:
                factor = 1 / self.ZOOM_STEP if delta > 0 else self.ZOOM_STEP
                self._zoom(factor, self._xdata_at(event.position()))
            event.accept()
            return
        # Two-finger sideways scroll pans a zoomed view — the natural trackpad
        # follow-up to a pinch. Vertical scrolls pass through untouched.
        pan_px = event.pixelDelta().x() or event.angleDelta().x() // 8
        if self.is_zoomed() and pan_px:
            self._pan_pixels(-pan_px)
            event.accept()
            return
        super().wheelEvent(event)

    def event(self, e: Any) -> bool:
        # macOS trackpad pinches arrive as native gestures, not scroll events;
        # matplotlib never sees them, so the Qt widget handles them itself.
        if (
            isinstance(e, QNativeGestureEvent)
            and e.gestureType() == Qt.NativeGestureType.ZoomNativeGesture
        ):
            # Pinch deltas are small (±~0.1); apart means in, together means out.
            if 1.0 + e.value() > 0.1:
                self._zoom(1.0 / (1.0 + e.value()), self._xdata_at(e.position()))
            return True
        return bool(super().event(e))

    def _xdata_at(self, position: Any) -> float | None:
        """The distance under the cursor, or None when it is off the axes."""
        if not self.figure.axes:
            return None
        x, y = self.mouseEventCoords(position)
        return float(self.figure.axes[0].transData.inverted().transform((x, y))[0])

    def _pan_pixels(self, pixels: float) -> None:
        """Slide the visible window sideways by a screen distance."""
        view = self.view()
        if view is None:
            return
        _, _, low, high = view
        width = self.figure.axes[0].get_window_extent().width or 1.0
        shift = pixels / width * (high - low)
        self.set_view(low + shift, high + shift)

    def _zoom(self, factor: float, center: float | None) -> None:
        """Scale the x window by ``factor`` around ``center``, clamped to the lap."""
        if not self.figure.axes or self._home_xlim is None:
            return
        ax = self.figure.axes[0]  # sharex: one set_xlim moves every panel
        low, high = ax.get_xlim()
        if center is None:
            center = (low + high) / 2  # cursor outside the axes: zoom the middle
        home_low, home_high = self._home_xlim
        new_low = max(center - (center - low) * factor, home_low)
        new_high = min(center + (high - center) * factor, home_high)
        if new_high - new_low < (home_high - home_low) * 1e-3:
            return  # deep enough — going further would just show noise
        ax.set_xlim(new_low, new_high)
        self.draw_idle()
        self.view_changed.emit()

    def _reset_view(self) -> None:
        if self.figure.axes and self._home_xlim is not None:
            self.figure.axes[0].set_xlim(self._home_xlim)
            self.draw_idle()
            self.view_changed.emit()


class MainWindow(QMainWindow):
    """Year → Grand Prix → session → two drivers → compare."""

    def __init__(self, loaders: Loaders = REAL_LOADERS) -> None:
        super().__init__()
        self._loaders = loaders
        self._pool = QThreadPool.globalInstance()
        self._session: Any = None
        # The last comparison's data, kept so the plot controls can rebuild
        # the figure instantly — no download, no recomputation.
        self._last_comparison: tuple[Any, Any, Any, str] | None = None
        # None means "the theme's validated default" for that driver.
        self._driver_colors: list[str | None] = [None, None]
        # Python would garbage-collect a running worker whose only reference
        # lives in C++; keeping them here pins them until they finish.
        self._workers: list[Worker] = []

        self.setWindowTitle("F1 Telemetry Lab")
        self.year_box = QComboBox()
        self.gp_box = QComboBox()
        self.session_box = QComboBox()
        self.driver_1_box = QComboBox()
        self.driver_2_box = QComboBox()
        self.load_button = QPushButton("Load session")
        self.compare_button = QPushButton("Compare")

        for year in range(LAST_SEASON, FIRST_SEASON - 1, -1):
            self.year_box.addItem(str(year), year)
        self.session_box.addItems(SESSIONS)

        selectors = QHBoxLayout()
        for label, widget in (
            ("Year", self.year_box),
            ("Grand Prix", self.gp_box),
            ("Session", self.session_box),
        ):
            selectors.addWidget(QLabel(label))
            selectors.addWidget(widget, 1 if widget is self.gp_box else 0)
        selectors.addWidget(self.load_button)

        drivers = QHBoxLayout()
        for label, widget in (
            ("Driver 1", self.driver_1_box),
            ("Driver 2", self.driver_2_box),
        ):
            drivers.addWidget(QLabel(label))
            drivers.addWidget(widget, 1)
        drivers.addWidget(self.compare_button)

        self._canvas = ZoomableCanvas(Figure())

        # The canvas and its controls side by side: the plot takes the space,
        # the controls keep a fixed strip on the right.
        # Under the plot: a pan bar that comes alive when zoomed in. It mirrors
        # the visible window (the handle is the viewport) and dragging it slides
        # the view — the mouse-first counterpart to sideways trackpad scrolling.
        self.pan_bar = QScrollBar(Qt.Orientation.Horizontal)
        self.pan_bar.setEnabled(False)
        self.pan_bar.valueChanged.connect(self._pan_bar_moved)
        self._canvas.view_changed.connect(self._sync_pan_bar)

        plot_column = QVBoxLayout()
        plot_column.addWidget(self._canvas, 1)
        plot_column.addWidget(self.pan_bar)

        content = QHBoxLayout()
        content.addLayout(plot_column, 1)
        content.addWidget(self._build_controls())

        column = QVBoxLayout()
        column.addLayout(selectors)
        column.addLayout(drivers)
        column.addLayout(content, 1)
        container = QWidget()
        container.setLayout(column)
        self.setCentralWidget(container)
        self.resize(1280, 800)

        self.year_box.currentIndexChanged.connect(self._refresh_schedule)
        self.load_button.clicked.connect(self._load_session)
        self.compare_button.clicked.connect(self._compare)

        self._set_downstream_enabled(False)
        self._refresh_schedule()

    def _build_controls(self) -> QWidget:
        """The strip that replaced matplotlib's navigation toolbar.

        That toolbar answers questions nobody asks of this app — pan history,
        subplot margins, axis editors — and its modal zoom is easy to get
        stuck in. These controls answer the questions people actually have:
        which channels, whose colours, which background, save it. Every
        change rebuilds the figure from the kept comparison data, which is
        fast enough that a "reset view" concept never needs to exist.
        """
        channels_group = QGroupBox("Panels")
        channels_layout = QVBoxLayout(channels_group)
        # The checkboxes mirror the figure's row order: speed on top, the
        # delta directly under it, then the input channels. Delta is not a
        # telemetry channel — it is computed from both laps — so it gets its
        # own checkbox rather than a slot in the channel dict.
        self.delta_box = QCheckBox("Delta")
        self.delta_box.setChecked(True)
        self.delta_box.toggled.connect(self._redraw)

        self.channel_boxes: dict[str, QCheckBox] = {}
        for panel in PANELS:
            box = QCheckBox(panel.column)
            box.setChecked(True)
            box.toggled.connect(self._redraw)
            channels_layout.addWidget(box)
            self.channel_boxes[panel.column] = box
            if panel.column == "Speed":
                channels_layout.addWidget(self.delta_box)

        appearance_group = QGroupBox("Appearance")
        appearance_layout = QVBoxLayout(appearance_group)
        self.color_buttons: list[QPushButton] = []
        for index in (0, 1):
            button = QPushButton(f"Driver {index + 1} colour…")
            button.clicked.connect(lambda _=False, i=index: self._pick_color(i))
            appearance_layout.addWidget(button)
            self.color_buttons.append(button)
        self.dark_box = QCheckBox("Dark background")
        self.dark_box.toggled.connect(self._redraw)
        appearance_layout.addWidget(self.dark_box)

        self.save_button = QPushButton("Save as PNG…")
        self.save_button.clicked.connect(self._save_figure)
        self.save_button.setEnabled(False)  # nothing to save until a comparison ran

        controls = QWidget()
        controls.setFixedWidth(190)
        layout = QVBoxLayout(controls)
        layout.addWidget(channels_group)
        layout.addWidget(appearance_group)
        layout.addStretch(1)
        layout.addWidget(self.save_button)
        return controls

    # --- async plumbing -----------------------------------------------------

    def _spawn(
        self,
        job: Callable[[], object],
        on_done: Callable[[Any], None],
        busy_message: str,
    ) -> None:
        """Run ``job`` in the pool; deliver its result (or error) to the UI."""
        self.statusBar().showMessage(busy_message)
        worker = Worker(job)
        self._workers.append(worker)

        def finished(result: Any) -> None:
            self._workers.remove(worker)
            on_done(result)

        def failed(message: str) -> None:
            self._workers.remove(worker)
            self.statusBar().showMessage(message)
            self._set_busy(False)

        worker.signals.done.connect(finished)
        worker.signals.failed.connect(failed)
        self._set_busy(True)
        self._pool.start(worker)

    def _set_busy(self, busy: bool) -> None:
        # One fetch at a time: a second click mid-download would race the first.
        self.load_button.setEnabled(not busy)
        self.compare_button.setEnabled(not busy and self._session is not None)
        self.year_box.setEnabled(not busy)

    def _set_downstream_enabled(self, enabled: bool) -> None:
        # The driver pickers mean nothing until a session has been loaded —
        # the roster is *of* the session, which is the point of the app.
        self.driver_1_box.setEnabled(enabled)
        self.driver_2_box.setEnabled(enabled)
        self.compare_button.setEnabled(enabled)

    # --- the chain: year → schedule → session → roster → figure -------------

    def _refresh_schedule(self) -> None:
        year = self.year_box.currentData()
        self.gp_box.clear()
        self._session = None
        self._set_downstream_enabled(False)
        self._spawn(
            lambda: self._loaders.schedule(year),
            self._schedule_ready,
            f"Fetching the {year} calendar…",
        )

    def _schedule_ready(self, names: Any) -> None:
        self.gp_box.addItems(list(names))
        self._set_busy(False)
        self.statusBar().showMessage("Pick a Grand Prix and load the session.")

    def _load_session(self) -> None:
        year = self.year_box.currentData()
        gp = self.gp_box.currentText()
        code = self.session_box.currentText()
        self._spawn(
            lambda: self._loaders.session(year, gp, code),
            self._session_ready,
            f"Loading {gp} {year} {code} — the first time downloads telemetry…",
        )

    def _session_ready(self, session: Any) -> None:
        self._session = session
        self.driver_1_box.clear()
        self.driver_2_box.clear()
        for code, name in self._loaders.roster(session):
            self.driver_1_box.addItem(f"{code} — {name}", code)
            self.driver_2_box.addItem(f"{code} — {name}", code)
        if self.driver_2_box.count() > 1:
            self.driver_2_box.setCurrentIndex(1)
        self._set_downstream_enabled(True)
        self._set_busy(False)
        self.statusBar().showMessage("Session loaded — pick two drivers and compare.")

    def _compare(self) -> None:
        driver_1 = self.driver_1_box.currentData()
        driver_2 = self.driver_2_box.currentData()
        if driver_1 == driver_2:
            self.statusBar().showMessage("Pick two different drivers.")
            return
        session = self._session

        def job() -> tuple[Any, Any, Any, str]:
            lap_1 = fastest_lap_telemetry(session, driver_1)
            lap_2 = fastest_lap_telemetry(session, driver_2)
            dist_1, time_1 = distance_and_time(lap_1.telemetry)
            dist_2, time_2 = distance_and_time(lap_2.telemetry)
            delta_s = compute_delta_time(dist_1, time_1, dist_2, time_2)
            event = str(session.event["EventName"])
            year = self.year_box.currentData()
            title = f"{event} {year} — {session.name}: fastest lap comparison"
            return lap_1, lap_2, delta_s, title

        self._spawn(job, self._comparison_ready, f"Comparing {driver_1} and {driver_2}…")

    def _comparison_ready(self, result: Any) -> None:
        self._last_comparison = result
        self.save_button.setEnabled(True)
        self._redraw()
        self._set_busy(False)
        lap_1, lap_2, _, _ = result
        self.statusBar().showMessage(
            f"{lap_1.driver} vs {lap_2.driver} — Cmd+scroll or pinch zooms, "
            "the bar below pans, double-click resets."
        )

    # --- plot controls -------------------------------------------------------

    def _plot_options(self) -> PlotOptions:
        """The controls, read into one value the figure builder understands."""
        return PlotOptions(
            channels=frozenset(
                column for column, box in self.channel_boxes.items() if box.isChecked()
            ),
            show_delta=self.delta_box.isChecked(),
            theme=DARK if self.dark_box.isChecked() else LIGHT,
            driver_1_color=self._driver_colors[0],
            driver_2_color=self._driver_colors[1],
        )

    def _redraw(self) -> None:
        """Rebuild the figure from the kept data with the current options.

        Cheap by design: the telemetry and the delta are already computed, so
        this is pure drawing — which is what makes live controls viable
        without threading. A zoomed view survives the rebuild: losing your
        place because you toggled a channel would make the controls and the
        zoom fight each other.
        """
        if self._last_comparison is None:
            return  # nothing compared yet; the options apply from the first plot
        zoomed_view = self._canvas.view() if self._canvas.is_zoomed() else None
        lap_1, lap_2, delta_s, title = self._last_comparison
        # The figure is built on the GUI thread: matplotlib objects are not
        # thread-safe once a canvas owns them. The slow part (the data) is
        # already done by the time we get here.
        figure = build_comparison_figure(lap_1, lap_2, delta_s, title, self._plot_options())
        self._show_figure(figure)
        if zoomed_view is not None:
            self._canvas.set_view(zoomed_view[2], zoomed_view[3])
        self._sync_pan_bar()

    # --- panning -------------------------------------------------------------

    PAN_BAR_RESOLUTION = 10_000  # scrollbar units per full lap; plenty smooth

    def _sync_pan_bar(self) -> None:
        """Mirror the canvas's visible window onto the bar under the plot.

        The handle *is* the viewport: its size is the visible fraction of the
        lap, its position the view's start. Signals are blocked while writing
        so the mirror never feeds back into the view it mirrors.
        """
        bar = self.pan_bar
        view = self._canvas.view()
        bar.blockSignals(True)
        if view is None or not self._canvas.is_zoomed():
            bar.setRange(0, 0)
            bar.setEnabled(False)
        else:
            home_low, home_high, low, high = view
            home_span = home_high - home_low
            page = max(1, round((high - low) / home_span * self.PAN_BAR_RESOLUTION))
            bar.setRange(0, self.PAN_BAR_RESOLUTION - page)
            bar.setPageStep(page)
            bar.setValue(round((low - home_low) / home_span * self.PAN_BAR_RESOLUTION))
            bar.setEnabled(True)
        bar.blockSignals(False)

    def _pan_bar_moved(self, value: int) -> None:
        view = self._canvas.view()
        if view is None:
            return
        home_low, home_high, low, high = view
        home_span = home_high - home_low
        new_low = home_low + value / self.PAN_BAR_RESOLUTION * home_span
        self._canvas.set_view(new_low, new_low + (high - low))

    def _pick_color(self, index: int) -> None:
        theme = self._plot_options().theme
        current = self._driver_colors[index] or (theme.driver_1, theme.driver_2)[index]
        chosen = QColorDialog.getColor(QColor(current), self, f"Driver {index + 1} colour")
        if chosen.isValid():
            self._driver_colors[index] = chosen.name()
            self._redraw()

    def _save_figure(self) -> None:
        if self._last_comparison is None:
            return
        lap_1, lap_2, _, _ = self._last_comparison
        suggested = f"{lap_1.driver}_vs_{lap_2.driver}.png".lower()
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", suggested, "PNG image (*.png)")
        if path:
            self._canvas.figure.savefig(path, dpi=150)
            self.statusBar().showMessage(f"Saved {path}")

    def _show_figure(self, figure: Figure) -> None:
        """Swap the embedded canvas for one owning the new figure.

        A canvas is married to its figure at construction, so showing a new
        figure means a new canvas — replacing the widget in the layout is
        simpler and more reliable than transplanting axes between figures.
        """
        old_canvas = self._canvas
        self._canvas = ZoomableCanvas(figure)
        # The pan bar mirrors whichever canvas is current, so every new canvas
        # is wired to it before it is shown.
        self._canvas.view_changed.connect(self._sync_pan_bar)
        layout = old_canvas.parentWidget().layout()
        assert layout is not None  # the canvas always sits in a layout; see __init__
        # replaceWidget searches nested layouts by default, so calling it on
        # the container's layout finds the canvas inside the content row.
        layout.replaceWidget(old_canvas, self._canvas)
        old_canvas.deleteLater()
        # A synchronous draw, not draw_idle: the slow work (the data) is done,
        # rendering takes a fraction of a second, and the user should never see
        # a blank canvas between the swap and the next paint event.
        self._canvas.draw()


def run() -> int:  # pragma: no cover — app.exec() blocks until the window closes
    """Create the application and hand control to Qt's event loop."""
    # A windowed app has no console worth reading, so the log file is where
    # its errors live — including anything that escapes a Qt slot, which Qt
    # would otherwise print to a stderr nobody sees and carry on.
    log_path = setup_logging("f1lab-gui.log")
    log_uncaught_exceptions(LOGGER)
    # The cache is configured before the first FastF1 call of any kind —
    # otherwise the initial schedule fetch installs FastF1's own default
    # cache and our directory only takes over from the second request on.
    cache_dir = enable_cache()
    LOGGER.info("Log file: %s", log_path)
    LOGGER.info("FastF1 cache: %s", cache_dir)
    LOGGER.info("Analysis engine: %s", native.backend_name())
    app = QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
