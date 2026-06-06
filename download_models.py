#!/usr/bin/env python3

"""
DepthForge – Model downloader
Downloads pre-converted OpenVINO IR models from GitHub Releases.

Usage:
    python download_models.py
    python download_models.py --release v0.1.0
"""

import argparse
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ── Release configuration ────────────────────────────────────────────────────
GITHUB_REPO    = "GrzegorzOle/DepthForge"
DEFAULT_RELEASE = "v0.1.0"

MODELS = [
    {
        "name":       "DPT Large – weights (.bin)",
        "filename":   "dpt_large.bin",
        "local_path": "models/dpt/openvino/dpt_large.bin",
        "size_mb":    652,
    },
    {
        "name":       "DPT Large – graph (.xml)",
        "filename":   "dpt_large.xml",
        "local_path": "models/dpt/openvino/dpt_large.xml",
        "size_mb":    1,
    },
    {
        "name":       "MiDaS v2.1 Small – weights (.bin)",
        "filename":   "midas_v21_small_256.bin",
        "local_path": "models/midas/openvino/midas_v21_small_256.bin",
        "size_mb":    32,
    },
    {
        "name":       "MiDaS v2.1 Small – graph (.xml)",
        "filename":   "midas_v21_small_256.xml",
        "local_path": "models/midas/openvino/midas_v21_small_256.xml",
        "size_mb":    1,
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_url(release: str, filename: str) -> str:
    return (
        f"https://github.com/{GITHUB_REPO}/releases/download"
        f"/{release}/{filename}"
    )


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(block_num * block_size, total_size)
    pct  = downloaded / total_size * 100
    done = int(pct / 2)
    bar  = "█" * done + "░" * (50 - done)
    mb   = downloaded / 1_048_576
    tot  = total_size / 1_048_576
    print(f"\r  [{bar}] {pct:5.1f}%  {mb:.1f}/{tot:.1f} MB", end="", flush=True)


def download_file(url: str, dest: Path, name: str, expected_mb: int) -> bool:
    """Download a single file with a progress bar. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        actual_mb = dest.stat().st_size / 1_048_576
        if actual_mb > expected_mb * 0.9:          # within 10 % of expected
            print(f"  ✓ Already present: {dest}  ({actual_mb:.0f} MB) – skipping")
            return True
        else:
            print(f"  ⚠ Incomplete file found ({actual_mb:.0f} MB), re-downloading…")

    print(f"\n  Downloading: {name}")
    print(f"  URL: {url}")
    print(f"  → {dest}")

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()   # newline after progress bar
        size_mb = dest.stat().st_size / 1_048_576
        print(f"  ✓ Done  ({size_mb:.1f} MB)")
        return True

    except urllib.error.HTTPError as e:
        print(f"\n  ✗ HTTP {e.code}: {e.reason}")
        if e.code == 404:
            print(f"    → Release '{DEFAULT_RELEASE}' not found or file not attached.")
            print(f"      Check: https://github.com/{GITHUB_REPO}/releases")
        return False

    except urllib.error.URLError as e:
        print(f"\n  ✗ Network error: {e.reason}")
        return False

    except KeyboardInterrupt:
        if dest.exists():
            dest.unlink()
        print("\n  Interrupted by user.")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download DepthForge OpenVINO models from GitHub Releases"
    )
    parser.add_argument(
        "--release", "-r",
        default=DEFAULT_RELEASE,
        help=f"GitHub release tag (default: {DEFAULT_RELEASE})"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["dpt", "midas", "all"],
        default="all",
        help="Which model to download (default: all)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  DepthForge – Model downloader")
    print(f"  Release : {args.release}")
    print(f"  Models  : {args.model}")
    print("=" * 60)

    # Filter models
    models_to_download = [
        m for m in MODELS
        if args.model == "all"
        or (args.model == "dpt"   and "dpt"   in m["filename"])
        or (args.model == "midas" and "midas" in m["filename"])
    ]

    results = {}
    for model in models_to_download:
        url  = build_url(args.release, model["filename"])
        dest = Path(model["local_path"])
        ok   = download_file(url, dest, model["name"], model["size_mb"])
        results[model["name"]] = ok

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "✓ OK    " if ok else "✗ FAILED"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  All models downloaded successfully.")
        print("  You can now run:  python benchmark.py")
    else:
        print("\n  Some downloads failed.")
        print(f"  Make sure release {args.release} exists and has model files attached:")
        print(f"  https://github.com/{GITHUB_REPO}/releases")
        sys.exit(1)


if __name__ == "__main__":
    main()

