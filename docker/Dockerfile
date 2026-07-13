FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    FASTF1_CACHE=/app/.fastf1-cache

RUN apt-get update \
    && apt-get install --yes --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc \
    CXX=g++

WORKDIR /app

COPY pyproject.toml README.md ./
RUN python -c "import tomllib, pathlib; project = tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']; print('\n'.join(project['dependencies'] + project['optional-dependencies']['dev']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt scikit-build-core pybind11 cmake ninja \
    && rm /tmp/requirements.txt

COPY cpp/ cpp/
COPY src/ src/
COPY tests/ tests/
RUN pip install --no-deps --no-build-isolation .

CMD ["python", "-m", "f1lab", "--help"]
