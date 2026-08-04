#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ── EARLIEST POSSIBLE LOG – runs before any import ────────────────────────────
import sys as _sys, os as _os, tempfile as _tmp, datetime as _dt
try:
    _el = _os.path.join(_tmp.gettempdir(), "depthforge_early.log")
    with open(_el, "a", encoding="utf-8") as _f:
        _f.write(f"[{_dt.datetime.now():%H:%M:%S}] LOAD argv={_sys.argv}\n")
        _f.write(f"[{_dt.datetime.now():%H:%M:%S}]      file={__file__}  cwd={_os.getcwd()}\n")
except Exception as _e:
    pass   # silently ignore – we try again after gi imports
# ──────────────────────────────────────────────────────────────────────────────
"""
DepthForge – GIMP 3.x Plugin  (GIMP 3.2.x)

Architecture
------------
GIMP bundles Python 3.14 which has no binary wheels for numpy/opencv.
This plugin calls an EXTERNAL Python interpreter (project .venv) via
subprocess to generate the depth map, then loads the result back into GIMP.

Menu entries
------------
  Filters > DepthForge > Generate Depth Map…
  Filters > DepthForge > Diagnose (test configuration)
"""

import sys
import os
import struct
import zlib
import time
import tempfile
import subprocess
import traceback
import json as _json

import gi
gi.require_version("Gimp",   "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gtk",    "3.0")
gi.require_version("Gegl",   "0.4")

from gi.repository import Gimp, GimpUi, GLib, GObject, Gio, Gegl

PROC_NAME   = "plug-in-depthforge"
PROC_DIAG   = "plug-in-depthforge-diagnose"
PROC_UPDATE = "plug-in-depthforge-update"
MENU_PATH   = "<Image>/Filters/DepthForge"

# Where releases are published. Overridable per installation through the
# "update_repo" key in depthforge_install.json, so a fork or an internal mirror
# does not need a code change — and "check_updates": false in the same file
# turns the whole thing off for an air-gapped museum machine.
UPDATE_REPO_DEFAULT = "GrzegorzOle/DepthForge"
UPDATE_API_URL = "https://api.github.com/repos/{repo}/releases/latest"
UPDATE_CHECK_INTERVAL_S = 24 * 3600

# Resolve plugin directory robustly:
# In GIMP 3.x, __file__ may be a bare filename with no directory.
# sys.argv[0] is always the full absolute path to the script.
_PLUGIN_SCRIPT = os.path.realpath(sys.argv[0] if sys.argv else __file__)
_PLUGIN_DIR    = os.path.dirname(_PLUGIN_SCRIPT)

# Log to 3 locations to guarantee at least one is writable
_LOG_FILE  = os.path.join(_PLUGIN_DIR,           "depthforge_run.log")
_LOG_TEMP  = os.path.join(tempfile.gettempdir(),  "depthforge_run.log")
# Fallback: hardcoded project path (set by install.py)
_LOG_PROJ  = ""   # filled from install JSON below


def _log(msg: str):
    import datetime
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n"
    for path in (_LOG_FILE, _LOG_TEMP, _LOG_PROJ):
        if not path:
            continue
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            break           # stop after first successful write
        except Exception:
            pass


# Try to load project root from install JSON (written by install.py)
try:
    _cfg_path = os.path.join(_PLUGIN_DIR, "depthforge_install.json")
    if os.path.isfile(_cfg_path):
        import json as _j
        _d = _j.load(open(_cfg_path, encoding="utf-8"))
        _root = _d.get("project_root", "")
        if _root:
            _LOG_PROJ = os.path.join(_root, "depthforge_gimp.log")
except Exception:
    pass


# Write startup entry immediately when GIMP loads the plugin
_log(f"=== Plugin module loaded | argv0={sys.argv[0] if sys.argv else '?'}")
_log(f"    __file__   = {__file__}")
_log(f"    PLUGIN_DIR = {_PLUGIN_DIR}")
_log(f"    LOG_FILE   = {_LOG_FILE}")
_log(f"    LOG_TEMP   = {_LOG_TEMP}")
_log(f"    LOG_PROJ   = {_LOG_PROJ}")


# ─────────────────────────────────────────────────────────────────────────────
#  Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_project_root() -> str | None:
    """
    Find the DepthForge project root (directory that contains config.json).
    Priority:
      1. depthforge_install.json (always correct – written by install.py)
      2. Walk up from plugin directory looking for config.json
    """
    # 1. Install JSON – always has the correct path
    cfg_path = os.path.join(_PLUGIN_DIR, "depthforge_install.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data = _json.load(f)
            root = data.get("project_root", "").strip()
            if root and os.path.isfile(os.path.join(root, "config.json")):
                return root
        except Exception:
            pass

    # 2. Walk up from plugin directory
    here = _PLUGIN_DIR
    for _ in range(10):
        if os.path.isfile(os.path.join(here, "config.json")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def _find_external_python() -> str | None:
    """
    Priority:
      1. depthforge_install.json  (written by whichever installer wired this copy)
      2. DEPTHFORGE_PYTHON environment variable  (manual override)
      3. Auto-discovery of project .venv
      4. System Python

    The install JSON deliberately outranks the environment variable. The JSON
    sits next to this very file, so it always describes *this* installation,
    while DEPTHFORGE_PYTHON is machine-wide and outlives the install that set
    it: a stale value left by an earlier source install used to silently shadow
    a later bundle install forever, and the symptom surfaced far from the cause
    (a broken interpreter failing deep inside the helper subprocess).
    """
    env = os.environ.get("DEPTHFORGE_PYTHON", "").strip()

    # 1. Install config JSON
    cfg_path = os.path.join(_PLUGIN_DIR, "depthforge_install.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data = _json.load(f)
            venv_py = data.get("venv_python", "").strip()
            if venv_py and os.path.isfile(venv_py):
                if env and os.path.normcase(os.path.abspath(env)) != \
                        os.path.normcase(os.path.abspath(venv_py)):
                    _log(f"NOTE: DEPTHFORGE_PYTHON={env} disagrees with the "
                         f"installed configuration and is being ignored. "
                         f"Unset it if it is a leftover from an older install.")
                _log(f"Python found via install JSON: {venv_py}")
                return venv_py
            if venv_py:
                _log(f"Install JSON points at a missing interpreter: {venv_py}")
        except Exception as e:
            _log(f"Could not read install JSON: {e}")

    # 2. Env var override
    if env:
        if os.path.isfile(env):
            _log(f"Python found via env var: {env}")
            return env
        _log(f"DEPTHFORGE_PYTHON points at a missing file: {env}")

    # 3. Auto-discovery
    root = _find_project_root()
    if root:
        for rel in (
            os.path.join(".venv", "Scripts", "python.exe"),
            os.path.join(".venv", "bin",     "python3"),
            os.path.join(".venv", "bin",     "python"),
            os.path.join("venv",  "Scripts", "python.exe"),
            os.path.join("venv",  "bin",     "python3"),
        ):
            cand = os.path.join(root, rel)
            if os.path.isfile(cand):
                _log(f"Python found via auto-discovery: {cand}")
                return cand

    # 4. System fallback
    import shutil
    py = shutil.which("python3") or shutil.which("python")
    if py:
        _log(f"Python found via PATH: {py}")
    return py


def _child_env() -> dict:
    """Environment for every interpreter we spawn.

    GIMP 3.x ships its own Python and may export PYTHONHOME/PYTHONPATH for it.
    Inheriting those into a *different* interpreter makes it load GIMP's stdlib
    instead of its own, which surfaces as numpy failing to import its C
    extensions — the classic "works in a terminal, fails inside GIMP". Strip
    them so the child resolves its own installation, exactly as it does when
    launched from a shell.

    PYTHONIOENCODING pairs with the encoding="utf-8" we decode the pipes with:
    the helper prints Polish, and the default here would be the console code
    page (cp1250 on a Polish Windows), which mangles or raises on that text.
    """
    env = dict(os.environ)
    for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE"):
        env.pop(var, None)
    env["PYTHONIOENCODING"] = "utf-8"
    # Second layer under the private helper directory: keep the script's own
    # directory (and, for -c, the CWD) off sys.path entirely, so nothing there
    # can shadow a stdlib module. Python 3.11+; older versions ignore it.
    env["PYTHONSAFEPATH"] = "1"
    return env


# ─────────────────────────────────────────────────────────────────────────────
#  Update check
#
#  Everything here is best-effort by design. The target audience is museums,
#  the product is deliberately offline-capable and ships its models inside the
#  installer, so a machine with no route to github.com is a supported setup —
#  not an error state. No failure in this section may ever reach the user as
#  anything but a log line.
# ─────────────────────────────────────────────────────────────────────────────

def _install_config() -> dict:
    """Contents of depthforge_install.json, or {} when absent/unreadable."""
    try:
        with open(os.path.join(_PLUGIN_DIR, "depthforge_install.json"),
                  encoding="utf-8") as f:
            data = _json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _update_repo() -> str:
    return str(_install_config().get("update_repo", "")).strip() or UPDATE_REPO_DEFAULT


def _updates_enabled() -> bool:
    return _install_config().get("check_updates", True) is not False


def _installed_version() -> str:
    """Version of the DepthForge code this plugin drives.

    VERSION.json is written by the bundle builder and rewritten by the updater,
    so it is the value that actually advances after an in-place code update.
    config.json is the fallback for source checkouts, which have no VERSION.json
    — and the updater deliberately never touches config.json, because it is the
    one file a user may reasonably have edited.
    """
    root = _find_project_root()
    if not root:
        return ""
    try:
        with open(os.path.join(root, "VERSION.json"), encoding="utf-8") as f:
            v = str(_json.load(f).get("version", "")).strip()
        if v:
            return v
    except Exception:
        pass
    try:
        with open(os.path.join(root, "config.json"), encoding="utf-8") as f:
            return str(_json.load(f).get("project", {}).get("version", "")).strip()
    except Exception:
        return ""


def _parse_version(text: str) -> tuple:
    """'v1.7.10-rc1' → (1, 7, 10). Unparsable components count as 0."""
    parts = []
    for chunk in str(text or "").strip().lstrip("vV").split(".")[:4]:
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _is_newer(latest: str, installed: str) -> bool:
    if not latest or not installed:
        return False
    return _parse_version(latest) > _parse_version(installed)


def _update_cache_path() -> str:
    if os.path.isdir(_PLUGIN_DIR) and os.access(_PLUGIN_DIR, os.W_OK):
        return os.path.join(_PLUGIN_DIR, "depthforge_update.json")
    return os.path.join(tempfile.gettempdir(), "depthforge_update.json")


def _read_update_cache() -> dict:
    try:
        with open(_update_cache_path(), encoding="utf-8") as f:
            data = _json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_update_cache(data: dict) -> None:
    """Write the cache atomically — the writer is a daemon thread that GIMP may
    kill at any moment when the plug-in process exits, and a half-written cache
    would suppress every later check."""
    path = _update_cache_path()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        _log(f"could not write update cache: {e}")
        try:
            os.unlink(tmp)
        except Exception:
            pass


def _fetch_latest_release(timeout: float = 8.0):
    """Newest published (non-draft, non-prerelease) release, or None."""
    import urllib.request

    url = UPDATE_API_URL.format(repo=_update_repo())
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "DepthForge-plugin",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as e:
        _log(f"update check failed ({url}): {e}")
        return None

    tag = str(data.get("tag_name", "")).strip()
    if not tag:
        _log("update check: release has no tag_name")
        return None

    version = tag.lstrip("vV")
    assets = {}
    for a in (data.get("assets") or []):
        name = str(a.get("name", ""))
        if name:
            assets[name] = str(a.get("browser_download_url", ""))
    code_zip = f"DepthForge-{version}-code.zip"

    info = {
        "checked_at":   int(time.time()),
        "tag":          tag,
        "version":      version,
        "html_url":     str(data.get("html_url", "")),
        "notes":        (str(data.get("body") or "")).strip()[:600],
        "code_url":     assets.get(code_zip, ""),
        "code_sha_url": assets.get(code_zip + ".sha256", ""),
    }
    _log(f"update check: latest={version} code_pkg={'yes' if info['code_url'] else 'no'}")
    return info


def _start_background_update_check() -> None:
    """Refresh the cached release info without blocking anything.

    Deliberately *not* called from do_query_procedures(): that runs at GIMP
    startup for every installed plug-in, and a network round trip there would
    stall the splash screen on a machine with no route to github.com. The thread
    only writes the cache file; every dialog reads the cache, never the network.
    """
    if not _updates_enabled():
        return
    cache = _read_update_cache()
    try:
        age = time.time() - float(cache.get("checked_at", 0))
    except Exception:
        age = UPDATE_CHECK_INTERVAL_S + 1
    if age < UPDATE_CHECK_INTERVAL_S:
        return

    def _worker():
        info = _fetch_latest_release()
        if info is None:
            # Stamp the attempt anyway: without this an offline machine retries
            # on every single run and pays the timeout each time.
            stale = _read_update_cache()
            stale["checked_at"] = int(time.time())
            _write_update_cache(stale)
            return
        old = _read_update_cache()
        info["notified"] = (old.get("version") == info["version"]
                            and bool(old.get("notified")))
        _write_update_cache(info)

    try:
        import threading
        threading.Thread(target=_worker, daemon=True).start()
    except Exception as e:
        _log(f"could not start the update check thread: {e}")


def _code_update_possible() -> tuple:
    """(can_update, reason) for an in-place code update of this installation.

    No for an AppImage: project_root/src is a symlink into the squashfs mount,
    which is read-only by construction and lands at a fresh /tmp/.mount_XXXXXX
    on every launch. Those users get the new .AppImage instead — replacing the
    file is their update mechanism.
    """
    root = _find_project_root()
    if not root:
        return False, "nie znaleziono katalogu projektu DepthForge"
    src = os.path.join(root, "src")
    if os.path.islink(src):
        return False, "instalacja AppImage – pobierz nowy plik .AppImage"
    # A developer checkout is not an installation to overwrite: the update
    # replaces src/ wholesale, which would silently blow away uncommitted work.
    if os.path.isdir(os.path.join(root, ".git")):
        return False, "instalacja ze źródeł (repozytorium git) – użyj git pull"
    if not os.access(root, os.W_OK):
        return False, "katalog projektu jest tylko do odczytu"
    if os.path.isdir(src) and not os.access(src, os.W_OK):
        return False, "katalog src/ jest tylko do odczytu"
    if not os.access(_PLUGIN_DIR, os.W_OK):
        return False, "katalog wtyczki jest tylko do odczytu"
    if not _find_external_python():
        return False, "nie znaleziono interpretera DepthForge"
    return True, ""


def _open_url(url: str) -> None:
    if not url:
        return
    try:
        Gio.AppInfo.launch_default_for_uri(url, None)
    except Exception as e:
        _log(f"could not open {url}: {e}")


def _message_dialog(title: str, body: str, buttons=None) -> int:
    """Plain GTK message box. buttons = [(label, response), …]; default OK."""
    try:
        from gi.repository import Gtk
        dlg = Gtk.MessageDialog(
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.NONE,
            text=title,
        )
        dlg.format_secondary_text(body)
        for label, response in (buttons or [("OK", 0)]):
            dlg.add_button(label, response)
        dlg.set_default_response(0)
        result = dlg.run()
        dlg.destroy()
        return result
    except Exception as e:
        _log(f"dialog failed ({title}): {e}")
        return 0


def _apply_code_update(info: dict) -> tuple:
    """Run the update helper in the DepthForge interpreter. → (ok, detail)."""
    import shutil as _sh

    ext_python = _find_external_python()
    root = _find_project_root() or ""
    if not ext_python or not root:
        return False, "brak interpretera lub katalogu projektu"

    # Same private-directory rule as the pipeline helper: a script sitting
    # directly in %TEMP% would put that world-writable folder on sys.path.
    work = tempfile.mkdtemp(prefix="df_update_")
    script = os.path.join(work, "df_update.py")
    try:
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(_UPDATE_SCRIPT)
        cmd = [ext_python, script,
               info.get("code_url", ""), info.get("code_sha_url", ""),
               root, _PLUGIN_DIR, info.get("version", "")]
        _log(f"update: running {ext_python} on {info.get('code_url','')}")
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           env=_child_env(), timeout=900)
    except Exception as e:
        _log(f"update helper failed to run: {e}")
        return False, str(e)
    finally:
        _sh.rmtree(work, ignore_errors=True)

    # Full output to the log, tail to the user: as everywhere else here, the
    # real cause is printed last.
    _log(f"update helper exit={r.returncode}")
    if r.stdout.strip():
        _log("--- update stdout ---\n" + r.stdout.strip())
    if r.stderr.strip():
        _log("--- update stderr ---\n" + r.stderr.strip())

    if r.returncode == 0 and "UPDATE_OK:" in r.stdout:
        return True, r.stdout.strip()[-400:]
    return False, (r.stdout + "\n" + r.stderr).strip()[-700:]


def _offer_update(info: dict, installed: str) -> None:
    """Show what is available and act on the answer."""
    can_apply, reason = _code_update_possible()
    has_pkg = bool(info.get("code_url"))

    body = (f"Zainstalowana wersja: {installed or '?'}\n"
            f"Dostępna wersja: {info.get('version', '?')}\n")
    if not has_pkg:
        body += "\nTo wydanie nie zawiera pakietu aktualizacji kodu –\n" \
                "trzeba pobrać pełny instalator.\n"
    elif not can_apply:
        body += f"\nAktualizacja w miejscu niemożliwa: {reason}.\n"
    if info.get("notes"):
        body += "\n" + info["notes"]

    buttons = []
    if has_pkg and can_apply:
        buttons.append(("Aktualizuj teraz", 1))
    if info.get("html_url"):
        buttons.append(("Otwórz stronę wydania", 2))
    buttons.append(("Później", 0))

    answer = _message_dialog("DepthForge – dostępna aktualizacja", body, buttons)
    if answer == 2:
        _open_url(info.get("html_url", ""))
        return
    if answer != 1:
        return

    ok, detail = _apply_code_update(info)
    if ok:
        # Drop the cache so the next check re-reads the now-current version
        # instead of offering the update the user just installed.
        _write_update_cache({})
        _message_dialog(
            "DepthForge – zaktualizowano",
            f"Zainstalowano wersję {info.get('version', '?')}.\n\n"
            "Uruchom ponownie GIMP-a, żeby załadować nową wtyczkę.\n\n"
            f"Log: {_LOG_FILE}",
        )
    else:
        answer = _message_dialog(
            "DepthForge – aktualizacja nieudana",
            "Nic nie zostało zmienione – poprzednia wersja została przywrócona.\n\n"
            f"{detail}\n\nPełny log: {_LOG_FILE}",
            [("Otwórz stronę wydania", 2), ("OK", 0)],
        )
        if answer == 2:
            _open_url(info.get("html_url", ""))


def _maybe_notify_update() -> None:
    """One quiet nudge per new version, after a run the user already waited on."""
    try:
        if not _updates_enabled():
            return
        cache = _read_update_cache()
        latest = str(cache.get("version", ""))
        if cache.get("notified") or not _is_newer(latest, _installed_version()):
            return
        cache["notified"] = True
        _write_update_cache(cache)
        _offer_update(cache, _installed_version())
    except Exception as e:
        _log(f"update notification failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Minimal PNG writer  (pure stdlib – struct + zlib)
# ─────────────────────────────────────────────────────────────────────────────

def _write_png_rgb(path: str, width: int, height: int, rgb_bytes: bytes):
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    stride   = width * 3
    raw_rows = b"".join(b"\x00" + rgb_bytes[y * stride:(y + 1) * stride]
                        for y in range(height))
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n"
                 + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw_rows, 6))
                 + chunk(b"IEND", b""))


# ─────────────────────────────────────────────────────────────────────────────
#  External helper script  (runs inside project .venv)
# ─────────────────────────────────────────────────────────────────────────────

_HELPER_SCRIPT = r"""
# DepthForge GIMP helper – runs in project .venv (has numpy, opencv, openvino)
# Pipeline: Standard + OpenVINO MiDaS + OpenVINO DPT → ensemble fusion →
#   Visual:    postprocess_depth (CLAHE + Gaussian)
#   Tactile:   fill_holes → detail_overlay → multiscale_smooth (v9 presets)
#   Advanced:  jak taktylny, ale ze wszystkimi parametrami z argv
import sys, os

input_png         = sys.argv[1]
output_png        = sys.argv[2]
enhancement       = int(sys.argv[3])               if len(sys.argv) > 3  else 50
invert            = sys.argv[4].lower() == "true"  if len(sys.argv) > 4  else False
project_root      = sys.argv[5]                    if len(sys.argv) > 5  else ""
color_out_png     = sys.argv[6]                    if len(sys.argv) > 6  else ""
stl_out_path      = sys.argv[7]                    if len(sys.argv) > 7  else ""
tactile_mode      = sys.argv[8].lower() == "true"  if len(sys.argv) > 8  else False
advanced_mode     = sys.argv[9].lower() == "true"  if len(sys.argv) > 9  else False
detail_strength   = float(sys.argv[10])            if len(sys.argv) > 10 else 0.05
detail_blur_sigma = float(sys.argv[11])            if len(sys.argv) > 11 else 2.5
fine_sigma        = float(sys.argv[12])            if len(sys.argv) > 12 else 1.5
limb_sigma        = float(sys.argv[13])            if len(sys.argv) > 13 else 3.0
fill_holes        = sys.argv[14].lower() == "true" if len(sys.argv) > 14 else True
tactile_levels    = int(sys.argv[15])              if len(sys.argv) > 15 else 0

# ── Make DepthForge importable ────────────────────────────────────────────────
if project_root:
    for p in (os.path.join(project_root, "src"), project_root):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

try:
    import numpy as np
    import cv2
except ImportError as e:
    # numpy wraps *any* failure in its own "C-extensions failed" banner, which
    # says nothing about the real cause. Print the traceback too: it names the
    # file that actually got imported, which is how a stray module shadowing a
    # stdlib name becomes obvious instead of mysterious.
    import traceback
    sys.stderr.write("IMPORT_ERROR: %s\n\n" % e)
    sys.stderr.write("executable: %s\n" % sys.executable)
    sys.stderr.write("sys.path:\n%s\n\n" % "\n".join("  " + p for p in sys.path))
    traceback.print_exc()
    sys.exit(1)

img = cv2.imread(input_png)
if img is None:
    sys.exit(f"READ_ERROR: cannot read {input_png}")

# enhancement (0-100) → clahe_clip dla postprocess_depth (tylko tryb visual)
clahe_clip = enhancement / 25.0   # 0→0.0, 50→2.0, 100→4.0

depth_u8 = None

# ── 1. Pełny pipeline: Standard + MiDaS OV + DPT OV + ensemble ───────────────
try:
    from depth_pipeline import (
        fuse_depth_maps, postprocess_depth,
        normalize_f32_robust, DepthForge,
        fill_small_object_holes, apply_detail_overlay,
        prepare_for_touch_multiscale,
        quantize_depth_foreground_aware, smooth_quantized_boundaries,
    )
    import time

    # WAŻNE: config.json używa ścieżek względnych – ustawiamy CWD na project_root
    if project_root and os.path.isdir(project_root):
        os.chdir(project_root)
        print(f"CWD set to project_root: {project_root}", flush=True)
    cfg_path = os.path.join(project_root, "config.json") if project_root else "config.json"
    df = DepthForge(cfg_path if os.path.isfile(cfg_path) else None)
    de_cfg = df.config.get("model", {}).get("depth_estimation", {})

    midas_compiled = None
    dpt_compiled   = None
    if df.ov_core:
        midas_path = de_cfg.get("midas_model_path")
        if midas_path and os.path.isfile(midas_path):
            midas_compiled = df._load_openvino_model(midas_path)
            print(f"OpenVINO MiDaS loaded: {midas_path}", flush=True)
        else:
            print(f"OpenVINO MiDaS NOT found: {midas_path}", flush=True)
        dpt_path = de_cfg.get("dpt_model_path")
        if dpt_path and os.path.isfile(dpt_path):
            dpt_compiled = df._load_openvino_model(dpt_path)
            print(f"OpenVINO DPT loaded:   {dpt_path}", flush=True)
        else:
            print(f"OpenVINO DPT NOT found: {dpt_path}", flush=True)

    depth_maps_f32 = {}
    t0 = time.perf_counter()
    s = df.generate_depth_map_midas(img)
    if s is not None:
        if s.ndim == 3:
            s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
        depth_maps_f32["standard"] = normalize_f32_robust(s.astype(np.float32))
        print(f"  Standard:  {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)
    if midas_compiled:
        t0 = time.perf_counter()
        m = df._run_openvino_inference_f32(midas_compiled, img, "midas")
        if m is not None:
            depth_maps_f32["midas"] = m.astype(np.float32)
            print(f"  MiDaS OV:  {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)
    if dpt_compiled:
        t0 = time.perf_counter()
        d = df._run_openvino_inference_f32(dpt_compiled, img, "dpt")
        if d is not None:
            depth_maps_f32["dpt"] = d.astype(np.float32)
            print(f"  DPT OV:    {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)

    print(f"  Modele: {list(depth_maps_f32.keys())}", flush=True)
    if not depth_maps_f32:
        raise RuntimeError("Żaden model nie zwrócił mapy głębi")

    if len(depth_maps_f32) > 1:
        fused = fuse_depth_maps(depth_maps_f32)
        print(f"  Ensemble fusion: {list(depth_maps_f32.keys())}", flush=True)
    else:
        key = list(depth_maps_f32.keys())[0]
        fused = normalize_f32_robust(list(depth_maps_f32.values())[0])
        print(f"  Pojedynczy model: {key}", flush=True)

    # ── Post-processing ───────────────────────────────────────────────────────
    if not tactile_mode and not advanced_mode:
        # === TRYB WIZUALNY ====================================================
        print("  Tryb: WIZUALNY (CLAHE + Gaussian)", flush=True)
        depth_u8 = postprocess_depth(fused, clahe_clip=clahe_clip, sigma=0.7)

    else:
        # === TRYB TAKTYLNY lub ZAAWANSOWANY ===================================
        # W trybie zaawansowanym używamy parametrów z argv,
        # w trybie taktylnym (v9 preset) używamy stałych wartości
        if advanced_mode:
            _detail_str   = detail_strength
            _detail_blur  = detail_blur_sigma
            _fine_s       = fine_sigma
            _limb_s       = limb_sigma
            _fill         = fill_holes
            _levels       = tactile_levels
            print(f"  Tryb: ZAAWANSOWANY (parametry ręczne)", flush=True)
            print(f"    detail_strength={_detail_str}  blur_sigma={_detail_blur}", flush=True)
            print(f"    fine_sigma={_fine_s}  limb_sigma={_limb_s}", flush=True)
            print(f"    fill_holes={_fill}  levels={_levels}", flush=True)
        else:
            # v9 presets
            _detail_str  = 0.05
            _detail_blur = 2.5
            _fine_s      = 1.5
            _limb_s      = 3.0
            _fill        = True
            _levels      = 0
            print("  Tryb: TAKTYLNY (v9 presets)", flush=True)

        # Krok 1: fill holes
        if _fill:
            fused = fill_small_object_holes(fused, min_area=20, max_area=2000, kernel_size=5)
            print("  fill_holes OK", flush=True)

        # Krok 2: detail overlay
        if _detail_str > 0.0:
            fused = apply_detail_overlay(fused, img,
                                         strength=_detail_str,
                                         blur_sigma=_detail_blur)
            print(f"  detail_overlay({_detail_str}, {_detail_blur}) OK", flush=True)

        # Krok 3: wieloskalowe wygładzanie taktylne
        fused = prepare_for_touch_multiscale(
            fused, median_size=5, fine_sigma=_fine_s, limb_sigma=_limb_s
        )
        print(f"  multiscale_smooth(fine={_fine_s}, limb={_limb_s}) OK", flush=True)

        # Krok 4: kwantyzacja (opcjonalna)
        if _levels > 1:
            bg_levels = max(1, _levels // 3)
            fg_levels = _levels - bg_levels
            fused, _level_idx, _n_levels = quantize_depth_foreground_aware(
                fused,
                fg_threshold_pct=40.0,
                bg_levels=bg_levels,
                fg_levels=fg_levels,
            )
            fused = smooth_quantized_boundaries(
                fused, level_idx=_level_idx, n_levels=_n_levels, kernel_size=9
            )
            print(f"  quantize({bg_levels}bg+{fg_levels}fg) + boundary_smooth OK",
                  flush=True)

        # Krok 5: postprocess bez CLAHE
        depth_u8 = postprocess_depth(fused, clahe_clip=0.0, sigma=0.0)

    print("  Pipeline OK", flush=True)

except Exception as e:
    import traceback
    print(f"Pipeline error (fallback do syntetycznego): {e}", flush=True)
    traceback.print_exc()
    depth_u8 = None

# ── 2. Fallback: syntetyczny estymator ────────────────────────────────────────
if depth_u8 is None:
    print("Fallback: syntetyczny estymator głębi", flush=True)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    edges = cv2.GaussianBlur(edges, (9, 9), 0)
    def _n(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo + 1e-6)
    fused_fb = _n(255.0 - gray) * 0.70 + _n(edges) * 0.30
    fb_u8 = (fused_fb * 255).clip(0, 255).astype(np.uint8)
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        fb_u8 = clahe.apply(fb_u8)
    depth_u8 = fb_u8

# ── 3. Inwersja + dopasowanie rozmiaru ───────────────────────────────────────
if invert:
    depth_u8 = 255 - depth_u8

h, w = img.shape[:2]
if depth_u8.shape[:2] != (h, w):
    depth_u8 = cv2.resize(depth_u8, (w, h), interpolation=cv2.INTER_LINEAR)

ok = cv2.imwrite(output_png, depth_u8)
if not ok:
    sys.exit(f"WRITE_ERROR: cannot write {output_png}")
print(f"OK:{output_png}", flush=True)

# ── 4. Kolorowa mapa głębi (INFERNO) ─────────────────────────────────────────
if color_out_png:
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    ok_c = cv2.imwrite(color_out_png, depth_color)
    if ok_c:
        print(f"COLOR_OK:{color_out_png}", flush=True)
    else:
        print(f"COLOR_FAIL:{color_out_png}", flush=True)

# ── 5. Eksport STL ────────────────────────────────────────────────────────────
if stl_out_path:
    try:
        from depth_pipeline import depth_to_stl
        depth_to_stl(depth_u8, stl_out_path)
        print(f"STL_OK:{stl_out_path}", flush=True)
    except Exception as e:
        import traceback
        print(f"STL_FAIL:{e}", flush=True)
        traceback.print_exc()
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Update helper script  (runs in the DepthForge interpreter, stdlib only)
# ─────────────────────────────────────────────────────────────────────────────

_UPDATE_SCRIPT = r"""
# DepthForge – applies a code-only update package.
#
# Runs in the DepthForge interpreter (bundled runtime or project .venv), never
# in GIMP's Python, for two reasons: a multi-megabyte download must not block
# GIMP's UI thread, and only this interpreter can prove, after the swap, that
# the new code still imports against the installed numpy/opencv/openvino. That
# check is what makes the rollback below worth anything.
#
# Everything here is stdlib: the package may well contain the fix for whatever
# is broken in src/, so this script must not import from it.
import sys, os, json, hashlib, shutil, zipfile, tempfile, subprocess
import urllib.request

asset_url    = sys.argv[1] if len(sys.argv) > 1 else ""
sha_url      = sys.argv[2] if len(sys.argv) > 2 else ""
project_root = sys.argv[3] if len(sys.argv) > 3 else ""
plugin_dir   = sys.argv[4] if len(sys.argv) > 4 else ""
new_version  = sys.argv[5] if len(sys.argv) > 5 else ""

HEADERS = {"User-Agent": "DepthForge-updater"}


def fail(msg):
    print("UPDATE_FAIL: " + msg, flush=True)
    sys.exit(1)


def fetch(url, timeout=600):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_version(text):
    parts = []
    for chunk in str(text or "").strip().lstrip("vV").split(".")[:4]:
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def installed_version():
    for name, path_keys in (("VERSION.json", ("version",)),
                            ("config.json", ("project", "version"))):
        try:
            with open(os.path.join(project_root, name), encoding="utf-8") as fh:
                data = json.load(fh)
            for key in path_keys:
                data = data[key]
            if str(data).strip():
                return str(data).strip()
        except Exception:
            continue
    return ""


if not asset_url:
    fail("this release publishes no code update package")
if not project_root or not os.path.isdir(project_root):
    fail("project root does not exist: " + str(project_root))

work = tempfile.mkdtemp(prefix="df_update_work_")
try:
    print("downloading " + asset_url, flush=True)
    blob = fetch(asset_url)
    print("got %d bytes" % len(blob), flush=True)

    # The package replaces src/ wholesale, so an unverified download is an
    # arbitrary-code-execution hole even over HTTPS: a mirror, a caching proxy
    # or a truncated body would every one of them pass silently otherwise.
    if not sha_url:
        fail("release publishes no .sha256 for the code package")
    want = fetch(sha_url, timeout=60).decode("utf-8", "replace").split()[0].strip()
    got = hashlib.sha256(blob).hexdigest()
    if want.lower() != got.lower():
        fail("sha256 mismatch: expected %s, got %s" % (want, got))
    print("sha256 OK " + got, flush=True)

    zip_path = os.path.join(work, "code.zip")
    with open(zip_path, "wb") as fh:
        fh.write(blob)

    extract = os.path.join(work, "extract")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            norm = os.path.normpath(member)
            if (member.startswith("/") or member.startswith("\\")
                    or os.path.splitdrive(norm)[0]
                    or ".." in norm.replace("\\", "/").split("/")):
                fail("refusing zip member outside the tree: " + member)
        zf.extractall(extract)

    meta_path = os.path.join(extract, "code.json")
    if not os.path.isfile(meta_path):
        fail("package has no code.json - not a DepthForge code update")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    pkg_version = str(meta.get("version", "")).strip()
    if new_version and pkg_version != new_version:
        fail("package is %s but the release says %s" % (pkg_version, new_version))

    # The valve that keeps a code-only update honest: the runtime (CPython plus
    # numpy/opencv/openvino) is not shipped here, so a release that needs a new
    # or newer dependency must refuse and send the user to the full installer.
    needs = str(meta.get("requires_runtime", "")).strip()
    have = installed_version()
    if needs and have and parse_version(have) < parse_version(needs):
        fail("this update needs runtime %s but %s is installed - "
             "run the full DepthForge installer instead" % (needs, have))

    new_app = os.path.join(extract, "app")
    if not os.path.isdir(new_app):
        fail("package has no app/ payload")

    backup = os.path.join(work, "backup")
    os.makedirs(backup)
    replaced = []          # (live path, backup path, kind)

    def stash(live, key):
        dst = os.path.join(backup, key)
        if os.path.isdir(live) and not os.path.islink(live):
            shutil.copytree(live, dst, symlinks=True)
            replaced.append((live, dst, "dir"))
        elif os.path.exists(live):
            shutil.copy2(live, dst)
            replaced.append((live, dst, "file"))
        else:
            # Recorded too: a rollback has to remove files this update created,
            # or a failed run leaves a VERSION.json claiming the new number.
            replaced.append((live, None, "absent"))

    def restore():
        for live, dst, kind in reversed(replaced):
            try:
                if kind == "absent":
                    if os.path.isdir(live) and not os.path.islink(live):
                        shutil.rmtree(live, ignore_errors=True)
                    elif os.path.exists(live) or os.path.islink(live):
                        os.unlink(live)
                elif kind == "dir":
                    shutil.rmtree(live, ignore_errors=True)
                    shutil.copytree(dst, live, symlinks=True)
                else:
                    shutil.copy2(dst, live)
            except Exception as exc:
                print("ROLLBACK_WARN: %s: %s" % (live, exc), flush=True)

    try:
        # app/ maps onto project_root. Directories are replaced wholesale rather
        # than merged, so a module deleted upstream really disappears instead of
        # lingering on sys.path and shadowing whatever replaced it.
        for entry in sorted(os.listdir(new_app)):
            src = os.path.join(new_app, entry)
            dst = os.path.join(project_root, entry)
            stash(dst, "app__" + entry)
            if os.path.isdir(src):
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            print("updated " + dst, flush=True)

        new_plugin = os.path.join(extract, "plugin", "depthforge", "depthforge.py")
        if os.path.isfile(new_plugin) and plugin_dir:
            live_plugin = os.path.join(plugin_dir, "depthforge.py")
            stash(live_plugin, "plugin__depthforge.py")
            shutil.copy2(new_plugin, live_plugin)
            print("updated " + live_plugin, flush=True)

        # Prove the new code imports against the *installed* dependencies before
        # calling this a success. Without it a mismatched package would leave a
        # permanently broken install behind a cheerful dialog.
        env = dict(os.environ)
        for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE"):
            env.pop(var, None)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONSAFEPATH"] = "1"
        probe = ("import sys; sys.path.insert(0, sys.argv[1]); "
                 "import depth_forge, depth_pipeline; print('import OK')")
        r = subprocess.run(
            [sys.executable, "-c", probe, os.path.join(project_root, "src")],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env, cwd=project_root, timeout=300,
        )
        if r.returncode != 0:
            raise RuntimeError("the new code does not import:\n"
                               + (r.stderr or r.stdout or "")[-1200:])
        print("smoke test: " + r.stdout.strip(), flush=True)
    except Exception as exc:
        restore()
        fail("rolled back - " + str(exc))

    with open(os.path.join(project_root, "VERSION.json"), "w", encoding="utf-8") as fh:
        json.dump({"name": "DepthForge", "version": pkg_version,
                   "source": "code-update"}, fh, indent=2)

    print("UPDATE_OK:" + pkg_version, flush=True)
finally:
    shutil.rmtree(work, ignore_errors=True)
"""


# ─────────────────────────────────────────────────────────────────────────────
#  GIMP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_error(procedure, message: str):
    _log(f"ERROR returned to GIMP: {message[:200]}")
    return procedure.new_return_values(
        Gimp.PDBStatusType.EXECUTION_ERROR,
        GLib.Error.new_literal(GLib.quark_from_string("depthforge"), message, 0),
    )


def _load_result_image(path: str):
    """Load PNG file as a new GIMP image (GIMP 3.x compatible)."""
    gfile = Gio.File.new_for_path(path)
    # Try the simplest form first (GIMP 3.x)
    try:
        return Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, gfile)
    except TypeError:
        pass
    # Some builds need explicit extra args – try via PDB
    try:
        pdb    = Gimp.get_pdb()
        proc   = pdb.lookup_procedure("file-png-load")
        cfg    = proc.create_config()
        cfg.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
        cfg.set_property("file", gfile)
        result = proc.run(cfg)
        return result.index(1)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Plugin class
# ─────────────────────────────────────────────────────────────────────────────

class DepthForgePlugin(Gimp.PlugIn):

    def do_query_procedures(self):
        # No network call here, however tempting: this runs at GIMP startup for
        # every installed plug-in. See _start_background_update_check().
        return [PROC_NAME, PROC_DIAG, PROC_UPDATE]

    def do_set_i18n(self, procname):
        return False, None, None

    # ── Create procedures ─────────────────────────────────────────────────────
    def do_create_procedure(self, name):
        if name == PROC_DIAG:
            return self._create_diag_procedure(name)
        if name == PROC_UPDATE:
            return self._create_update_procedure(name)
        return self._create_main_procedure(name)

    def _create_main_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, self.run, None
        )
        procedure.set_image_types("RGB*, GRAY*")
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
        procedure.set_menu_label("Generate Depth Map\u2026")
        procedure.add_menu_path(MENU_PATH)
        procedure.set_documentation(
            "DepthForge \u2013 Depth Map Generator",
            "Generates a depth map from the active layer.",
            name,
        )
        procedure.set_attribution("DepthForge Team", "DepthForge Team", "2026")
        procedure.add_int_argument(
            "enhancement-level", "_Enhancement (0-100)",
            "CLAHE contrast enhancement (0=none, 100=max).",
            0, 100, 50, GObject.ParamFlags.READWRITE,
        )
        procedure.add_boolean_argument(
            "invert-depth", "_Invert depth",
            "Invert depth polarity.", False, GObject.ParamFlags.READWRITE,
        )
        procedure.add_boolean_argument(
            "color-depth-layer", "Warstwa _kolorowa (INFERNO)",
            "Zaznacz = warstwa głębi w kolorach INFERNO. "
            "Odznacz = warstwa głębi czarno-biała.",
            False, GObject.ParamFlags.READWRITE,
        )
        procedure.add_boolean_argument(
            "export-stl", "Export _STL (3D model)",
            "Export depth map as a watertight STL file for 3D printing.",
            False, GObject.ParamFlags.READWRITE,
        )
        procedure.add_boolean_argument(
            "tactile-mode", "_Tryb taktylny (v9 – fill-holes + detail + multiscale)",
            "Włącza pełny pipeline taktylny (jak indian_summer_tactile_v9): "
            "fill_holes → detail_overlay(0.05,2.5) → multiscale_smooth(1.5,3.0). "
            "Wyłącz dla trybu wizualnego z CLAHE.",
            False, GObject.ParamFlags.READWRITE,
        )
        # ── Tryb zaawansowany – ręczna kontrola wszystkich parametrów ─────────
        procedure.add_boolean_argument(
            "advanced-mode", "Tryb _zaawansowany (parametry ręczne)",
            "Gdy zaznaczony: ignoruje predefiniowane ustawienia v9 i używa "
            "poniższych parametrów ręcznych. Działa razem z trybem taktylnym.",
            False, GObject.ParamFlags.READWRITE,
        )
        procedure.add_double_argument(
            "detail-strength", "  Detail strength (0–1)",
            "Siła nakładki mikrodetali z luminancji obrazu. "
            "0 = wyłączona. Zalecane: 0.05–0.08 (taktylny), 0.10–0.20 (wizualny).",
            0.0, 1.0, 0.05, GObject.ParamFlags.READWRITE,
        )
        procedure.add_double_argument(
            "detail-blur-sigma", "  Detail blur sigma [px]",
            "Odcięcie częstotliwościowe nakładki. Wyższe = szersze pasma (kontury kończyn). "
            "Zalecane: 2.5 (taktylny), 1.2 (wizualny).",
            0.1, 15.0, 2.5, GObject.ParamFlags.READWRITE,
        )
        procedure.add_double_argument(
            "fine-sigma", "  Fine sigma [px] (multiscale)",
            "Usunięcie drobnego szumu tekstury (tkanina, trawa). "
            "Zalecane: 1.2–1.5.",
            0.1, 10.0, 1.5, GObject.ParamFlags.READWRITE,
        )
        procedure.add_double_argument(
            "limb-sigma", "  Limb sigma [px] (multiscale)",
            "Skala kończyn – filtr końcowy używa limb_sigma×0.5, "
            "żeby nie zlewać sąsiednich nóg/rąk. Zalecane: 2.5–3.5.",
            0.5, 15.0, 3.0, GObject.ParamFlags.READWRITE,
        )
        procedure.add_boolean_argument(
            "fill-holes", "  Fill holes (małe obiekty)",
            "Wypełnia płaskie wnętrza małych obiektów (zwierzęta, dalekie postaci) "
            "nierozpoznanych przez modele głębi.",
            True, GObject.ParamFlags.READWRITE,
        )
        procedure.add_int_argument(
            "tactile-levels", "  Tactile levels (0=ciągły)",
            "Kwantyzacja na N dyskretnych poziomów (0 = gradient ciągły). "
            "Zalecane dla druku muzealnego: 6 (2 tło + 4 pierwszoplan).",
            0, 12, 0, GObject.ParamFlags.READWRITE,
        )
        return procedure

    def _create_diag_procedure(self, name):
        procedure = Gimp.Procedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, self.run_diag, None
        )
        procedure.set_menu_label("Diagnose\u2026")
        procedure.add_menu_path(MENU_PATH)
        procedure.set_documentation(
            "DepthForge \u2013 Diagnose",
            "Tests the DepthForge configuration and shows a status report.",
            name,
        )
        procedure.set_attribution("DepthForge Team", "DepthForge Team", "2026")
        return procedure

    def _create_update_procedure(self, name):
        procedure = Gimp.Procedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, self.run_update, None
        )
        procedure.set_menu_label("Sprawdź aktualizacje…")
        procedure.add_menu_path(MENU_PATH)
        procedure.set_documentation(
            "DepthForge – Check for updates",
            "Checks GitHub for a newer DepthForge release and can install the "
            "code update in place.",
            name,
        )
        procedure.set_attribution("DepthForge Team", "DepthForge Team", "2026")
        return procedure

    # ── Check for updates ─────────────────────────────────────────────────────
    def run_update(self, procedure, run_mode, args, run_data):
        GimpUi.init("depthforge-update")
        installed = _installed_version()

        # Synchronous on purpose: the user asked, so waiting for the answer is
        # the expected behaviour. The automatic check is the threaded one.
        info = _fetch_latest_release(timeout=15.0)
        if info is None:
            _message_dialog(
                "DepthForge – sprawdzanie aktualizacji",
                "Nie udało się połączyć z GitHubem.\n\n"
                f"Repozytorium: {_update_repo()}\n"
                f"Zainstalowana wersja: {installed or '?'}\n\n"
                f"Szczegóły w logu: {_LOG_FILE}",
            )
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

        info["notified"] = True
        _write_update_cache(info)

        if _is_newer(info["version"], installed):
            _offer_update(info, installed)
        else:
            answer = _message_dialog(
                "DepthForge – wersja aktualna",
                f"Zainstalowana wersja: {installed or '?'}\n"
                f"Najnowsze wydanie: {info.get('version', '?')}\n\n"
                "Nie ma nic do zaktualizowania.",
                [("Otwórz stronę wydania", 2), ("OK", 0)],
            )
            if answer == 2:
                _open_url(info.get("html_url", ""))

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    # ── Diagnose ──────────────────────────────────────────────────────────────
    def run_diag(self, procedure, run_mode, args, run_data):
        GimpUi.init("depthforge-diag")
        lines = ["DepthForge – Diagnostic Report", "=" * 40]

        ext_py = _find_external_python()
        if ext_py:
            lines.append(f"[OK] External Python:\n     {ext_py}")
        else:
            lines.append("[FAIL] External Python: NOT FOUND")
            lines.append("       Re-run the DepthForge installer, or set the")
            lines.append("       DEPTHFORGE_PYTHON env var by hand")

        cfg_path = os.path.join(_PLUGIN_DIR, "depthforge_install.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    data = _json.load(f)
                lines.append(f"[OK] Install config: {cfg_path}")
                lines.append(f"     project_root = {data.get('project_root','?')}")
                lines.append(f"     venv_python  = {data.get('venv_python','?')}")
            except Exception as e:
                lines.append(f"[WARN] Install config unreadable: {e}")
        else:
            lines.append(f"[WARN] depthforge_install.json missing:\n     {cfg_path}")
            lines.append("       Re-run the DepthForge installer to create it")

        root = _find_project_root()
        lines.append(f"{'[OK]' if root else '[WARN]'} Project root: {root or 'not found'}")

        installed = _installed_version()
        lines.append(f"{'[OK]' if installed else '[WARN]'} Installed version: "
                     f"{installed or 'unknown'}")
        if _updates_enabled():
            can_apply, why = _code_update_possible()
            cache = _read_update_cache()
            lines.append(f"[INFO] Update repo: {_update_repo()}")
            if cache.get("version"):
                lines.append(f"[INFO] Last seen release: {cache['version']}")
            lines.append(
                "[OK] In-place code update: available" if can_apply
                else f"[INFO] In-place code update: unavailable – {why}"
            )
        else:
            lines.append('[INFO] Update check: disabled ("check_updates": false)')

        if ext_py:
            r = subprocess.run(
                [ext_py, "-c",
                 "import numpy, cv2; print(f'numpy {numpy.__version__}, "
                 "opencv {cv2.__version__}')"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=_child_env(), timeout=15,
            )
            if r.returncode == 0:
                lines.append(f"[OK] Python deps: {r.stdout.strip()}")
            else:
                # Tail, not head: the real exception is at the end.
                lines.append(f"[FAIL] Python deps:\n{r.stderr.strip()[-300:]}")

        env_val = os.environ.get("DEPTHFORGE_PYTHON", "")
        if not env_val:
            lines.append("[INFO] DEPTHFORGE_PYTHON env = (not set – using install JSON)")
        elif ext_py and os.path.normcase(os.path.abspath(env_val)) == \
                os.path.normcase(os.path.abspath(ext_py)):
            lines.append(f"[OK] DEPTHFORGE_PYTHON env = {env_val}")
        else:
            lines.append(f"[WARN] DEPTHFORGE_PYTHON env = {env_val}")
            lines.append("       Ignored – the install JSON takes precedence.")
            lines.append("       Likely a leftover from an older install; unset it.")

        lines.append("=" * 40)
        lines.append(f"Log file: {_LOG_FILE}")

        msg = "\n".join(lines)
        _log("Diagnose ran:\n" + msg)

        dialog = Gtk = None
        try:
            from gi.repository import Gtk as _Gtk
            Gtk = _Gtk
            dialog = Gtk.MessageDialog(
                message_type=Gtk.MessageType.INFO,
                buttons=Gtk.ButtonsType.OK,
                text="DepthForge Diagnostic",
            )
            dialog.format_secondary_text(msg)
            dialog.run()
            dialog.destroy()
        except Exception:
            pass

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    # ── Main run ──────────────────────────────────────────────────────────────
    def run(self, procedure, run_mode, image, drawables, config, run_data):
        _log(f"run() called, run_mode={run_mode}")

        enhancement  = config.get_property("enhancement-level")
        invert_depth = config.get_property("invert-depth")

        if run_mode == Gimp.RunMode.INTERACTIVE:
            GimpUi.init("depthforge")
            dialog = GimpUi.ProcedureDialog.new(
                procedure, config, "DepthForge \u2013 Depth Map Generator"
            )
            dialog.fill(None)
            ok = dialog.run()
            dialog.destroy()
            if not ok:
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CANCEL, GLib.Error()
                )
            enhancement  = config.get_property("enhancement-level")
            invert_depth = config.get_property("invert-depth")

        color_depth_layer = config.get_property("color-depth-layer")
        export_stl        = config.get_property("export-stl")
        tactile_mode      = config.get_property("tactile-mode")
        advanced_mode     = config.get_property("advanced-mode")
        detail_strength   = config.get_property("detail-strength")
        detail_blur_sigma = config.get_property("detail-blur-sigma")
        fine_sigma        = config.get_property("fine-sigma")
        limb_sigma        = config.get_property("limb-sigma")
        fill_holes        = config.get_property("fill-holes")
        tactile_levels    = config.get_property("tactile-levels")

        _log(f"params: enhancement={enhancement} invert={invert_depth} "
             f"color={color_depth_layer} stl={export_stl} tactile={tactile_mode} "
             f"advanced={advanced_mode} detail_str={detail_strength} "
             f"detail_blur={detail_blur_sigma} fine_s={fine_sigma} "
             f"limb_s={limb_sigma} fill_holes={fill_holes} levels={tactile_levels}")

        # Kicked off here, not at startup: the check runs in a daemon thread and
        # has the whole pipeline subprocess to finish in, so it never costs the
        # user a single millisecond of waiting.
        _start_background_update_check()

        ext_python = _find_external_python()
        if not ext_python:
            return _make_error(
                procedure,
                "DepthForge: nie znaleziono Pythona z numpy/opencv.\n\n"
                "Uruchom ponownie instalator DepthForge - zapisuje on obok\n"
                "wtyczki plik depthforge_install.json ze sciezka do interpretera.\n\n"
                "Ewentualnie ustaw recznie zmienna srodowiskowa:\n"
                "  DEPTHFORGE_PYTHON = sciezka\\do\\python.exe\n\n"
                f"(szukano w: {os.path.join(_PLUGIN_DIR, 'depthforge_install.json')})\n"
                f"Log: {_LOG_FILE}",
            )

        Gimp.context_push()
        image.undo_group_start()
        tmp_in = tmp_out = tmp_helper = tmp_color = None
        tmp_helper_dir = None

        # Duplikat obrazu do eksportu (spłaszczony – wszystkie warstwy złożone)
        export_image = None
        try:
            drawable = drawables[0] if drawables else image.get_active_drawable()
            w = image.get_width()
            h = image.get_height()
            _log(f"Image size: {w}x{h}, base_type: {image.get_base_type()}, "
                 f"layers: {len(image.get_layers())}")

            # Tworzymy spłaszczoną kopię całego obrazu (wszystkie widoczne warstwy złożone).
            # Dzięki temu model głębi widzi dokładnie to samo co widzisz na ekranie,
            # a nie tylko wybraną warstwę.
            try:
                export_image = image.duplicate()
                export_image.flatten()
                export_layers = export_image.get_layers()
                export_drawable = export_layers[0] if export_layers else export_image.get_active_drawable()
                _log("Using flattened duplicate for depth estimation (full composite)")
            except Exception as e:
                _log(f"Could not flatten duplicate ({e}) – falling back to active drawable")
                export_image = None
                export_drawable = drawable

            # Konwertuj do RGB jeśli obraz jest w trybie GRAY lub ma kanał alfa
            # (modele OpenVINO oczekują BGR/RGB 3-kanałowego wejścia)
            if export_image is not None:
                try:
                    base = export_image.get_base_type()
                    # Gimp.ImageBaseType.GRAY == 1
                    if int(base) != 0:   # 0 = RGB
                        pdb_conv = Gimp.get_pdb()
                        proc_conv = pdb_conv.lookup_procedure("gimp-image-convert-rgb")
                        if proc_conv:
                            cfg_conv = proc_conv.create_config()
                            cfg_conv.set_property("image", export_image)
                            proc_conv.run(cfg_conv)
                            export_layers = export_image.get_layers()
                            export_drawable = export_layers[0] if export_layers else export_image.get_active_drawable()
                            _log("Converted export image to RGB")
                except Exception as e:
                    _log(f"RGB conversion warning: {e}")

            # 1. Save image to temp PNG – GIMP 3.2.x multi-method export
            tmp_fd, tmp_in = tempfile.mkstemp(suffix=".png", prefix="df_in_")
            os.close(tmp_fd)
            saved = False
            pdb = Gimp.get_pdb()
            src_image   = export_image if export_image is not None else image
            src_drawable = export_drawable

            # Method A: file-png-export PDB (confirmed available in GIMP 3.2.4)
            if not saved:
                try:
                    proc = pdb.lookup_procedure("file-png-export")
                    if proc:
                        cfg = proc.create_config()
                        cfg.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
                        cfg.set_property("image",    src_image)
                        cfg.set_property("file",     Gio.File.new_for_path(tmp_in))
                        for prop in ("drawables", "drawable"):
                            try:
                                val = [src_drawable] if prop == "drawables" else src_drawable
                                cfg.set_property(prop, val)
                                break
                            except Exception:
                                pass
                        proc.run(cfg)
                        if os.path.isfile(tmp_in) and os.path.getsize(tmp_in) > 100:
                            saved = True
                            _log(f"Saved via file-png-export: {tmp_in}")
                except Exception as e:
                    _log(f"file-png-export failed: {e}")

            # Method B: Gimp.file_save
            if not saved:
                for args in (
                    (Gimp.RunMode.NONINTERACTIVE, src_image, src_drawable,
                     Gio.File.new_for_path(tmp_in)),
                    (Gimp.RunMode.NONINTERACTIVE, src_image,
                     Gio.File.new_for_path(tmp_in)),
                ):
                    try:
                        Gimp.file_save(*args)
                        if os.path.isfile(tmp_in) and os.path.getsize(tmp_in) > 100:
                            saved = True
                            _log(f"Saved via Gimp.file_save({len(args)} args): {tmp_in}")
                            break
                    except Exception as e:
                        _log(f"  file_save {len(args)} args: {e}")

            # Method C: pixel-by-pixel via get_pixel → pure-Python PNG writer
            if not saved:
                try:
                    _log("Falling back to pixel-by-pixel read (slow for large images)")
                    buf = bytearray(w * h * 3)
                    for y in range(h):
                        for x in range(w):
                            px = src_drawable.get_pixel(x, y)
                            data = px[1] if isinstance(px, tuple) else px
                            r = data[0] if len(data) > 0 else 0
                            g = data[1] if len(data) > 1 else r
                            b = data[2] if len(data) > 2 else r
                            off = (y * w + x) * 3
                            buf[off] = r; buf[off+1] = g; buf[off+2] = b
                    _write_png_rgb(tmp_in, w, h, bytes(buf))
                    if os.path.getsize(tmp_in) > 100:
                        saved = True
                        _log(f"Saved via pixel-by-pixel: {tmp_in}")
                except Exception as e:
                    _log(f"pixel-by-pixel failed: {e}")

            if not saved:
                raise RuntimeError(
                    "Nie mozna eksportowac obrazu do pliku tymczasowego.\n"
                    f"Sprawdz log: {_LOG_FILE}"
                )
            _log(f"Input PNG size: {os.path.getsize(tmp_in)} bytes")

            tmp_fd, tmp_out = tempfile.mkstemp(suffix=".png", prefix="df_out_")
            os.close(tmp_fd)

            # Zawsze tworzymy ścieżkę tmp_color – helper zawsze generuje wersję kolorową.
            # Wybór która trafi do GIMP zależy od color_depth_layer.
            tmp_color = ""
            fd_c, tmp_color = tempfile.mkstemp(suffix=".png", prefix="df_color_")
            os.close(fd_c)

            stl_out_path = ""
            if export_stl:
                # Try to save STL alongside the image, otherwise use temp dir
                try:
                    img_uri = image.get_uri()
                    if img_uri:
                        import urllib.request
                        img_path = urllib.request.url2pathname(
                            img_uri.replace("file://", "").replace("file:///", "/")
                        )
                        base_no_ext = os.path.splitext(img_path)[0]
                        stl_out_path = base_no_ext + "_depth.stl"
                    else:
                        raise ValueError("no URI")
                except Exception:
                    fd_s, stl_out_path = tempfile.mkstemp(suffix=".stl", prefix="df_stl_")
                    os.close(fd_s)
                _log(f"STL output path: {stl_out_path}")

            # 2. Write helper script
            # The helper gets a directory of its own, and this is not tidiness.
            # Python puts the script's directory first on sys.path, so writing
            # the helper straight into %TEMP% would put %TEMP% there — a
            # world-writable folder full of other people's scripts. One stray
            # inspect.py / types.py / json.py in it shadows the stdlib module of
            # that name for our process, and the failure surfaces as a baffling
            # "Importing the numpy C-extensions failed" (this happened: a leftover
            # %TEMP%\inspect.py that did `import duckdb`). An empty private
            # directory makes sys.path[0] harmless.
            tmp_helper_dir = tempfile.mkdtemp(prefix="df_helper_")
            tmp_helper = os.path.join(tmp_helper_dir, "df_helper.py")
            with open(tmp_helper, "w", encoding="utf-8") as fh:
                fh.write(_HELPER_SCRIPT)

            # 3. Call external Python
            project_root = _find_project_root() or ""
            cmd = [
                ext_python, tmp_helper,
                tmp_in, tmp_out,
                str(enhancement),
                str(invert_depth).lower(),
                project_root,
                tmp_color,
                stl_out_path,
                str(tactile_mode).lower(),
                str(advanced_mode).lower(),
                str(detail_strength),
                str(detail_blur_sigma),
                str(fine_sigma),
                str(limb_sigma),
                str(fill_holes).lower(),
                str(tactile_levels),
            ]
            _log(f"Subprocess cmd: {' '.join(cmd[:3])} ...")
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace",
                                    env=_child_env(), timeout=180)
            # Log both streams in full. Truncating here is how the real cause of
            # an import failure ("Original error was: ...", printed last) got
            # lost while the generic advice at the top survived.
            _log(f"Subprocess exit={result.returncode}")
            if result.stdout.strip():
                _log("--- helper stdout ---\n" + result.stdout.strip())
            if result.stderr.strip():
                _log("--- helper stderr ---\n" + result.stderr.strip())

            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "(no output)").strip()
                # Show the *tail*: Python puts the actual exception at the end,
                # and numpy's import banner is 600 characters of preamble.
                if len(detail) > 700:
                    detail = "[...]\n" + detail[-700:]
                raise RuntimeError(
                    f"Zewnetrzny Python zakonczyl sie bledem (kod {result.returncode}):\n\n"
                    f"{detail}\n\n"
                    f"Pelne wyjscie w logu: {_LOG_FILE}"
                )

            if not os.path.isfile(tmp_out) or os.path.getsize(tmp_out) == 0:
                raise RuntimeError(
                    f"Plik wynikowy nie istnieje lub jest pusty:\n{tmp_out}\n\n"
                    f"stdout: {result.stdout[:200]}"
                )

            # 5. Załaduj wynik do GIMP jako nową warstwę
            _log(f"Loading result: {tmp_out} (color_mode={color_depth_layer})")

            # Wybierz plik źródłowy: kolorowy (INFERNO) lub czarno-biały
            if color_depth_layer and tmp_color and os.path.isfile(tmp_color) \
                    and os.path.getsize(tmp_color) > 0:
                src_result = tmp_color
                layer_name = "Depth Map – INFERNO (DepthForge)"
                _log("Using COLOR (INFERNO) result")
            else:
                src_result = tmp_out
                layer_name = "Depth Map (DepthForge)"
                _log("Using GRAYSCALE result")

            result_img = _load_result_image(src_result)
            if result_img is None:
                raise RuntimeError(f"Nie mozna zaladowac pliku wynikowego: {src_result}")

            layers = result_img.get_layers()
            result_drawable = layers[0] if layers else result_img.get_active_item()
            new_layer = Gimp.Layer.new_from_drawable(result_drawable, image)
            new_layer.set_name(layer_name)
            image.insert_layer(new_layer, None, -1)
            result_img.delete()

            Gimp.displays_flush()
            _log(f"Done – layer '{layer_name}' added")

            # Powiadomienie o eksporcie STL
            if export_stl and stl_out_path and os.path.isfile(stl_out_path):
                _log(f"STL exported: {stl_out_path}")
                try:
                    from gi.repository import Gtk as _Gtk
                    stl_folder = os.path.dirname(stl_out_path)
                    stl_name   = os.path.basename(stl_out_path)

                    _dlg = _Gtk.MessageDialog(
                        message_type=_Gtk.MessageType.INFO,
                        buttons=_Gtk.ButtonsType.NONE,
                        text="DepthForge – Eksport STL zakończony",
                    )
                    _dlg.format_secondary_text(
                        f"Model 3D zapisany:\n{stl_out_path}"
                    )
                    # Przyciski: Otwórz folder  +  OK
                    _dlg.add_button("Otwórz folder", 1)
                    _dlg.add_button("OK",            0)
                    _dlg.set_default_response(0)

                    response = _dlg.run()
                    _dlg.destroy()

                    if response == 1:
                        # Otwórz folder w eksploratorze i zaznacz plik STL
                        import subprocess as _sp, platform as _pl
                        try:
                            if _pl.system() == "Windows":
                                # /select, zaznacza plik w Eksploratorze
                                _sp.Popen(["explorer", "/select,", stl_out_path])
                            elif _pl.system() == "Darwin":
                                _sp.Popen(["open", "-R", stl_out_path])
                            else:
                                _sp.Popen(["xdg-open", stl_folder])
                        except Exception as _oe:
                            _log(f"open folder failed: {_oe}")
                except Exception as _de:
                    _log(f"STL dialog error: {_de}")

        except Exception:
            tb = traceback.format_exc()
            _log(f"EXCEPTION:\n{tb}")
            image.undo_group_end()
            Gimp.context_pop()
            return _make_error(procedure,
                               f"DepthForge blad:\n\n{tb[:800]}\n\nLog: {_LOG_FILE}")
        finally:
            # Usuń tymczasowe pliki
            for tmp in (tmp_in, tmp_out, tmp_helper, tmp_color):
                try:
                    if tmp and os.path.isfile(tmp):
                        os.unlink(tmp)
                except Exception:
                    pass
            try:
                if tmp_helper_dir and os.path.isdir(tmp_helper_dir):
                    import shutil as _sh
                    _sh.rmtree(tmp_helper_dir, ignore_errors=True)
            except Exception:
                pass
            # Usuń spłaszczony duplikat obrazu (żeby nie zaśmiecać listy obrazów GIMP)
            try:
                if export_image is not None:
                    export_image.delete()
            except Exception:
                pass

        image.undo_group_end()
        Gimp.context_pop()

        # After the work, never before: the user came here for a depth map, and
        # this fires at most once per released version.
        _maybe_notify_update()

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


# ─────────────────────────────────────────────────────────────────────────────
Gimp.main(DepthForgePlugin.__gtype__, sys.argv)

