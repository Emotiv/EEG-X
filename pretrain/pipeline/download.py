"""Fetch author Kalunga or Crowdsourced data and verify published file digests."""

import argparse
import hashlib
import json
import re
import urllib.request
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pretrain.metadata import atomic_json
from pretrain.channel_identity import sha256_file


OSF_SOURCE = "https://api.osf.io/v2/nodes/9bvgh/files/osfstorage/61f9c8b9b63ee30b5ff36341/?page[size]=100&view_only=70744f62157c46d5bd731480db1873df"


def crowdsourced(output, cache=None):
    """Fetch the paper's author deposit, with optional verified local reuse."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    files, url = [], OSF_SOURCE
    while url:
        if not url.startswith("https://api.osf.io/"):
            raise ValueError("Unexpected author manifest host")
        with urllib.request.urlopen(url, timeout=60) as response:
            page = json.load(response)
        files.extend(page["data"])
        url = page["links"].get("next")
    selected = [
        item
        for item in files
        if item["attributes"]["name"].startswith("Alpha Supression_EPOC")
        or item["attributes"]["name"] == "Readme.txt"
    ]
    if not selected:
        raise ValueError("Author Crowdsourced manifest is empty")

    def fetch(item):
        name = item["attributes"]["name"]
        if Path(name).name != name or item["attributes"]["kind"] != "file":
            raise ValueError("Unexpected author source filename")
        checksum = item["attributes"]["extra"]["hashes"]["sha256"]
        path = root / name
        if path.exists() and sha256_file(path) == checksum:
            return
        partial = path.with_suffix(path.suffix + ".partial")
        cached = Path(cache) / name if cache else None
        if cached is not None and cached.is_file() and sha256_file(cached) == checksum:
            shutil.copyfile(cached, partial)
        else:
            with (
                urllib.request.urlopen(item["links"]["download"], timeout=60) as response,
                partial.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle)
        if partial.stat().st_size != item["attributes"]["size"] or sha256_file(partial) != checksum:
            raise ValueError("Crowdsourced file differs from the author digest")
        partial.replace(path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, selected))
    atomic_json(
        root / "manifest.json", {"record": "9bvgh", "source": OSF_SOURCE, "files": selected}
    )
    return {"status": "verified", "files": len(selected), "record": "9bvgh"}


def digest(path):
    value = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return "md5:" + value.hexdigest()


def kalunga(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen("https://zenodo.org/api/records/2392979", timeout=60) as response:
        metadata = json.load(response)
    if metadata["id"] != 2392979:
        raise ValueError("Author deposit identity changed")
    files = metadata["files"]
    if not files or any(
        not re.fullmatch(r"subject(0[1-9]|1[0-2])_run[1-5]_raw\.fif", f["key"]) for f in files
    ):
        raise ValueError("Unexpected author file manifest")

    def fetch(item):
        path = root / item["key"]
        if (
            path.exists()
            and path.stat().st_size == item["size"]
            and digest(path) == item["checksum"]
        ):
            return
        partial = path.with_suffix(".partial")
        url = f"https://zenodo.org/records/2392979/files/{item['key']}?download=1"
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as handle:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(block)
        if partial.stat().st_size != item["size"] or digest(partial) != item["checksum"]:
            raise ValueError("Downloaded file differs from author digest")
        partial.replace(path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, files))
    atomic_json(root / "record.json", metadata)
    return {"status": "verified", "files": len(files), "record": metadata["id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=["kalunga", "crowdsourced"], default="kalunga")
    parser.add_argument(
        "--cache", help="Existing author files, reused only after digest verification"
    )
    args = parser.parse_args()
    result = (
        kalunga(args.output) if args.task == "kalunga" else crowdsourced(args.output, args.cache)
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
