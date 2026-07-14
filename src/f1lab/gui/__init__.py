"""Desktop app for lap comparisons — ``f1lab-gui`` on the command line.

Qt lives behind this gate: the package promises to work without the ``gui``
extra installed, so nothing here may import PySide6 at module level. The real
application is in :mod:`f1lab.gui.app`, imported only once the entry point
actually runs.
"""

from __future__ import annotations


def main() -> int:
    """Launch the desktop app; explain what to install when Qt is missing."""
    try:
        from f1lab.gui.app import run
    except ImportError:  # pragma: no cover — depends on how f1lab was installed
        print(
            "The desktop app needs Qt, which is an optional extra.\n"
            "Install it with:\n\n"
            '    pip install "f1lab[gui]"\n'
        )
        return 1
    return run()
