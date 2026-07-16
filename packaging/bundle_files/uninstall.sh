#!/usr/bin/env bash
# DepthForge – removes the plugin from GIMP. The bundle folder stays.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/python/bin/python3" "$HERE/bundle_install.py" --uninstall "$@"
