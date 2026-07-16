#!/usr/bin/env bash
# DepthForge – installer wrapper.
# Runs the installer with the CPython interpreter bundled in this folder,
# so no system Python is required.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/python/bin/python3"

if [ ! -x "$PY" ]; then
    echo "ERROR: bundled Python not found at $PY"
    echo "The archive looks incomplete — unpack it again."
    exit 1
fi

exec "$PY" "$HERE/bundle_install.py" "$@"
