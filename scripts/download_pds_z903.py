#!/usr/bin/env python
"""Download the pLIBS Z-903 collection from the official NASA PDS directory."""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = (
    "https://pds-geosciences.wustl.edu/speclib/"
    "urn-nasa-pds-libs_reference_database/"
)
DEFAULT_COLLECTION = "data_plibs_z903"
DEFAULT_METADATA = "libs_metadata.xlsx"
USER_AGENT = "atomic-spectrum-cnn-copilot/0.1 (NASA PDS data download)"


def fetch_bytes(url: str, *, timeout: int = 60) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def discover_spectral_urls(base_url: str, collection: str) -> list[str]:
    """Discover CSV products from the official collection inventory."""
    collection_url = urljoin(base_url, f"{collection}/")
    inventory_url = urljoin(collection_url, f"collection_{collection}_inventory.csv")
    try:
        inventory = fetch_bytes(inventory_url).decode("utf-8-sig")
        urls: list[str] = []
        for row in csv.reader(io.StringIO(inventory)):
            if not row:
                continue
            product_id = row[-1].strip()
            stem = product_id.rsplit("::", 1)[0].split(":")[-1]
            if stem.lower().startswith("plibs_z903_"):
                urls.append(urljoin(collection_url, f"{stem}.csv"))
        if urls:
            return sorted(set(urls))
    except (UnicodeDecodeError, HTTPError, URLError, TimeoutError) as exc:
        print(f"Inventory lookup failed ({exc}); trying the official directory listing.")

    html = fetch_bytes(collection_url).decode("utf-8", errors="replace")
    names = re.findall(r'href="([^"]+\.csv)"', html, flags=re.IGNORECASE)
    urls = [urljoin(collection_url, name) for name in names if "plibs_z903_" in name.lower()]
    if not urls:
        raise RuntimeError(f"No Z-903 spectral CSV files discovered at {collection_url}")
    return sorted(set(urls))


def is_readable_csv(path: Path) -> bool:
    """Reject HTML error pages and require at least one numeric data row."""
    with path.open("rb") as handle:
        prefix = handle.read(512).lstrip().lower()
    if not prefix or prefix.startswith((b"<!doctype html", b"<html", b"<?xml")):
        return False

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        rows = csv.reader(handle)
        for row in rows:
            numeric_values = 0
            for value in row:
                try:
                    float(value.strip())
                except ValueError:
                    continue
                numeric_values += 1
            if numeric_values >= 2:
                return True
    return False


def is_valid_xlsx(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == b"PK\x03\x04" and path.stat().st_size > 4096


def download_file(url: str, destination: Path, validator, retries: int, timeout: int) -> str:
    """Download to a part file, resuming when the server supports byte ranges."""
    if destination.exists() and validator(destination):
        return "skipped"
    part_path = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        existing_size = part_path.stat().st_size if part_path.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if existing_size:
            headers["Range"] = f"bytes={existing_size}-"
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                append = existing_size > 0 and response.status == 206
                if existing_size and not append:
                    existing_size = 0
                mode = "ab" if append else "wb"
                with part_path.open(mode) as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            if validator(part_path):
                part_path.replace(destination)
                return "downloaded"
            raise ValueError("downloaded content failed validation")
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            if attempt == retries:
                print(f"Failed {url}: {exc}", file=sys.stderr)
                return "failed"
            time.sleep(min(2**attempt, 10))
    return "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--metadata-name", default=DEFAULT_METADATA)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/") + "/"
    spectral_dir = ROOT / "data" / "raw" / "pds_z903"
    metadata_path = ROOT / "data" / "metadata" / args.metadata_name
    urls = discover_spectral_urls(base_url, args.collection)

    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    total_size = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_file,
                url,
                spectral_dir / Path(url).name,
                is_readable_csv,
                args.retries,
                args.timeout,
            ): url
            for url in urls
        }
        for future in as_completed(futures):
            result = future.result()
            counts[result] += 1
            if result in {"downloaded", "skipped"}:
                total_size += (spectral_dir / Path(futures[future]).name).stat().st_size

    metadata_url = urljoin(base_url, f"document/{args.metadata_name}")
    metadata_result = download_file(
        metadata_url, metadata_path, is_valid_xlsx, args.retries, args.timeout
    )
    if metadata_result in {"downloaded", "skipped"}:
        total_size += metadata_path.stat().st_size
    else:
        raise RuntimeError(f"Could not download valid metadata workbook: {metadata_url}")

    print(f"Spectral files discovered: {len(urls)}")
    print(f"Spectral files successfully downloaded: {counts['downloaded']}")
    print(f"Spectral files skipped: {counts['skipped']}")
    print(f"Spectral files failed: {counts['failed']}")
    print(f"Total downloaded size: {total_size / (1024**2):.2f} MiB")
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()