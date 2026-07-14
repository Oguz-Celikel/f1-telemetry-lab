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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import fastf1
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from f1lab.analysis import compute_delta_time, distance_and_time, fastest_lap_telemetry
from f1lab.compare_laps import build_comparison_figure, enable_cache, load_session

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
            self.signals.failed.emit(str(exc))
        else:
            self.signals.done.emit(result)


class MainWindow(QMainWindow):
    """Year → Grand Prix → session → two drivers → compare."""

    def __init__(self, loaders: Loaders = REAL_LOADERS) -> None:
        super().__init__()
        self._loaders = loaders
        self._pool = QThreadPool.globalInstance()
        self._session: Any = None
        # Python would garbage-collect a running worker whose only reference
        # lives in C++; keeping them here pins them until they finish.
        self._workers: list[Worker] = []

        self.setWindowTitle("f1lab — fastest lap comparison")
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

        self._canvas = FigureCanvasQTAgg(Figure())
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        column = QVBoxLayout()
        column.addLayout(selectors)
        column.addLayout(drivers)
        column.addWidget(self._toolbar)
        column.addWidget(self._canvas, 1)
        container = QWidget()
        container.setLayout(column)
        self.setCentralWidget(container)
        self.resize(1150, 800)

        self.year_box.currentIndexChanged.connect(self._refresh_schedule)
        self.load_button.clicked.connect(self._load_session)
        self.compare_button.clicked.connect(self._compare)

        self._set_downstream_enabled(False)
        self._refresh_schedule()

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
        lap_1, lap_2, delta_s, title = result
        # The figure is built on the GUI thread: matplotlib objects are not
        # thread-safe once a canvas owns them. The slow part (the data) is
        # already done by the time we get here.
        figure = build_comparison_figure(lap_1, lap_2, delta_s, title)
        self._show_figure(figure)
        self._set_busy(False)
        self.statusBar().showMessage(
            f"{lap_1.driver} vs {lap_2.driver} — pan and zoom with the toolbar."
        )

    def _show_figure(self, figure: Figure) -> None:
        """Swap the embedded canvas for one owning the new figure.

        A canvas is married to its figure at construction, so showing a new
        figure means a new canvas (and a toolbar pointing at it) — replacing
        the widgets in the layout is simpler and more reliable than trying to
        transplant axes between figures.
        """
        layout = self.centralWidget().layout()
        assert layout is not None  # the central widget always has one; see __init__
        old_canvas, old_toolbar = self._canvas, self._toolbar
        self._canvas = FigureCanvasQTAgg(figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.replaceWidget(old_toolbar, self._toolbar)
        layout.replaceWidget(old_canvas, self._canvas)
        old_toolbar.deleteLater()
        old_canvas.deleteLater()
        # A synchronous draw, not draw_idle: the slow work (the data) is done,
        # rendering takes a fraction of a second, and the user should never see
        # a blank canvas between the swap and the next paint event.
        self._canvas.draw()


def run() -> int:  # pragma: no cover — app.exec() blocks until the window closes
    """Create the application and hand control to Qt's event loop."""
    app = QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
