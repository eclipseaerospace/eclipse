#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# tools/fetch_terrain.py — fetch archived terrain products and verify them.
#
# Standard library only, and no import of eclipse. A verifier that shares code
# with the thing it verifies proves that the code agrees with itself; this one
# reads the manifest, fetches, and hashes, and would catch a republished product
# that the library would happily load.
#
# Products are never committed. This is what stands in for committing them.

from __future__ import annotations

import argparse
import hashlib
import sys
import tomllib
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY_ROOT / "data" / "terrain" / "manifest.toml"
CHUNK = 1 << 20


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            sha.update(block)
    return sha.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and verify archived terrain products."
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check what is present without fetching anything",
    )
    arguments = parser.parse_args(argv)

    manifest = tomllib.loads(arguments.manifest.read_text(encoding="utf-8"))
    directory = arguments.manifest.parent
    failures = 0

    for product in manifest["product"]:
        identifier = product["id"]
        if arguments.only and identifier not in arguments.only:
            continue
        path = directory / f"{identifier}.tif"

        if not path.exists():
            if arguments.verify_only:
                print(f"  {identifier:34s} absent")
                continue
            print(f"  {identifier:34s} fetching {product['bytes'] / 1e6:.0f} MB ...")
            with urllib.request.urlopen(product["url"]) as response:
                path.write_bytes(response.read())

        size = path.stat().st_size
        found = digest(path)
        if size != product["bytes"]:
            print(f"  {identifier:34s} SIZE MISMATCH {size} != {product['bytes']}")
            failures += 1
        elif found != product["sha256"]:
            print(f"  {identifier:34s} CHECKSUM MISMATCH")
            print(f"    manifest {product['sha256']}")
            print(f"    on disk  {found}")
            failures += 1
        else:
            print(f"  {identifier:34s} ok  {size / 1e6:.0f} MB")

    if failures:
        print(
            f"\n{failures} product(s) do not match the manifest. An archive can "
            "republish under the same name, and a study whose numbers moved for "
            "that reason would look like a physics result."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
