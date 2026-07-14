# f1-telemetry-lab — task runner.
# Run `just` with no arguments to list available tasks.

set shell := ["bash", "-euo", "pipefail", "-c"]

compose := "docker compose"
venv_python := ".venv/bin/python"

# List available tasks
default:
    @just --list

# One-time setup: local virtualenv (for your IDE) + working directories
init:
    python3 -m venv .venv
    {{venv_python}} -m pip install --upgrade pip
    mkdir -p output .fastf1-cache
    @echo "Done. Next: just dependencies"

# Install f1lab plus dev tools into the local virtualenv (editable)
dependencies:
    @test -x {{venv_python}} || { echo "No .venv found — run 'just init' first."; exit 1; }
    {{venv_python}} -m pip install --editable ".[dev]"

# Install the git hooks (ruff, clang-format, hygiene checks) — run once per clone
hooks:
    @test -x {{venv_python}} || { echo "No .venv found — run 'just init' first."; exit 1; }
    {{venv_python}} -m pre_commit install
    @echo "Hooks installed. They now run on every 'git commit'."

# Run the git hooks against every file, not just the staged ones
hooks-run:
    @test -x {{venv_python}} || { echo "No .venv found — run 'just init' first."; exit 1; }
    {{venv_python}} -m pre_commit run --all-files

# Fail with an actionable message when Docker is not usable. Private (leading
# underscore) so it stays out of `just --list`; every Docker task depends on
# `build`, so guarding here covers all of them.
#
# Two probes, because Docker Desktop has a half-started state: the daemon
# answers `docker info` while its API still 500s on the calls compose makes.
# `docker compose ls` fails exactly when our tasks would fail, so each state
# gets the message that actually fixes it.
_require-docker:
    @docker info >/dev/null 2>&1 || { \
        echo "Docker is not running. Start Docker Desktop:"; \
        echo ""; \
        echo "    open -a Docker"; \
        echo ""; \
        echo "It needs ~20s to come up, then re-run your command."; \
        exit 1; }
    @{{compose}} ls >/dev/null 2>&1 || { \
        echo "Docker is still starting up — the daemon is not answering yet."; \
        echo "Wait ~20s and re-run your command."; \
        echo "(If it never settles, quit and reopen Docker Desktop.)"; \
        exit 1; }

# Build the Docker image
build: _require-docker
    {{compose}} build f1lab

# Run all tests: Python (pytest + parity) and C++ (Catch2), inside Docker
test: test-py test-cpp

# Line-by-line coverage report you can click through (output/coverage/)
coverage: build
    {{compose}} run --rm f1lab pytest --cov-report=html:output/coverage
    @echo ""
    @echo "Report: output/coverage/index.html"
    @-open output/coverage/index.html 2>/dev/null || true

# Run the Python unit tests (with coverage) inside Docker
test-py: build
    {{compose}} run --rm f1lab pytest

# The .cpp-build directory below is a named volume, so the Catch2 download
# and object files survive between runs.
# Configure, build and run the C++ Catch2 suite inside Docker
test-cpp: build
    {{compose}} run --rm f1lab bash -c "\
        cmake -S cpp -B .cpp-build -G Ninja -DF1LAB_BUILD_TESTS=ON \
        && cmake --build .cpp-build \
        && ctest --test-dir .cpp-build --output-on-failure"

# Benchmark the numpy engine against the C++ engine (markdown table)
bench: build
    {{compose}} run --rm f1lab python -m f1lab.bench

# Static checks inside Docker: ruff (lint + formatting) and mypy
lint: build
    {{compose}} run --rm f1lab ruff check src tests
    {{compose}} run --rm f1lab ruff format --check src tests
    {{compose}} run --rm f1lab mypy src tests

# Compare two drivers' fastest laps. Blank line below keeps this out of
# `just --list` (a long parameterised signature renders it awkwardly there);
# `just examples` shows ready-to-run commands instead.

run year gp session="R" driver1="VER" driver2="NOR": build
    {{compose}} run --rm f1lab \
        python -m f1lab --year {{year}} --gp "{{gp}}" --session {{session}} \
        --drivers {{driver1}} {{driver2}}

# Show ready-to-run example commands
examples:
    @echo 'just run 2026 Silverstone Q VER NOR    # qualifying: Verstappen vs Norris'
    @echo 'just run 2026 Silverstone R VER NOR    # same duel, race pace'
    @echo 'just run 2026 Monza Q LEC PIA          # different GP and drivers'
    @echo 'just run 2026 Spa Q HAM RUS            # teammate comparison'
    @echo 'just run 2025 Suzuka R VER ALO         # earlier seasons work too'
    @echo 'just run 2026 "Abu Dhabi" Q ALO STR    # quote GP names with spaces'
    @echo ''
    @echo 'Arguments: year gp [session=R] [driver1=VER] [driver2=NOR]'
    @echo 'Sessions: R (race), Q (qualifying), S (sprint), FP1/FP2/FP3'

# Build a wheel into dist/ using the local virtualenv
package:
    @test -x {{venv_python}} || { echo "No .venv found — run 'just init' first."; exit 1; }
    {{venv_python}} -m pip wheel --no-deps --wheel-dir dist .

# Remove caches, build artifacts and generated plots (keeps .fastf1-cache)
clean:
    rm -rf dist build .cpp-build .pytest_cache .mypy_cache .ruff_cache .coverage src/*.egg-info
    find . -name "__pycache__" -type d -prune -exec rm -rf {} +
    rm -f output/*.png
    -{{compose}} down --remove-orphans
