"""PyInstaller's entry script — the bundle's whole Python-side surface.

Kept to one import and one call on purpose: everything real lives in the
package, where the tests can reach it. This file only exists because
PyInstaller wants a script, not a module, as its entry point.
"""

from f1lab.gui import main

if __name__ == "__main__":
    raise SystemExit(main())
