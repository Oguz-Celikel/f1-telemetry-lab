# PyInstaller spec for the macOS app. Build with `just app`.
#
# One-folder mode wrapped in a BUNDLE: .app bundles are folders anyway, and
# one-file mode would unpack ~400 MB to a temp directory on every launch.
# PyInstaller ad-hoc-signs the binaries itself, which is what lets the app
# run at all on Apple Silicon; Gatekeeper's "unidentified developer" prompt
# on downloaded copies is a distribution problem (no paid Apple ID), not a
# build problem — see the README's install note.

import importlib.util
from importlib.metadata import version

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

APP_VERSION = version("f1lab")

# FastF1 ships data files (circuit metadata among them) that a plain module
# graph walk does not see. f1lab's own metadata rides along so the app can
# read its version at runtime for the window title.
datas = collect_data_files("fastf1") + copy_metadata("f1lab")

# The C++ engine has to be added by its file path. PyInstaller's static
# analysis cannot see it twice over: the import sits inside try/except, and an
# editable install serves the extension through a custom import finder that
# module-graph walks do not consult. Shipping the app without it would not
# fail — it would silently fall back to numpy, making the bundle slower than
# a checkout — so a missing extension aborts the build instead.
_native = importlib.util.find_spec("f1lab._native")
if _native is None or not _native.origin:
    raise SystemExit("f1lab._native is not importable — build/install the package first")
binaries = [(_native.origin, "f1lab")]

a = Analysis(
    ["launch.py"],
    datas=datas,
    binaries=binaries,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="F1 Telemetry Lab",
    console=False,  # a GUI app; stray console windows are a Windows-ism anyway
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="f1lab-app",
)

app = BUNDLE(
    coll,
    name="F1 Telemetry Lab.app",
    icon="icon.icns",
    bundle_identifier="com.oguzcelikel.f1telemetrylab",
    info_plist={
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": APP_VERSION,
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "MIT License — unofficial, not associated with Formula 1",
    },
)
