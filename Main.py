"""Compatibility entrypoint for the reproducible EEG-X pipeline."""

import sys

from pretrain.pipeline.__main__ import main


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "paper":
        del sys.argv[1]
    main()
