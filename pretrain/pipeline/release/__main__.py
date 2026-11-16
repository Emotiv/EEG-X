"""Publish a new fixed-split release from an existing prepared HDF5 corpus."""

import argparse
import json

from .build import build_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    manifest = build_release(args.source, args.output, args.protocol)
    print(json.dumps({"status": "ready", "datasets": len(manifest["datasets"])}))


if __name__ == "__main__":
    main()
