"""Bind a training process to its owning wrapper before executing the entrypoint."""

import ctypes
import os
import signal
import sys


def main():
    parent, entrypoint, *arguments = sys.argv[1:]
    library = ctypes.CDLL(None, use_errno=True)
    # Linux sends SIGTERM when the wrapper dies, including SIGKILL or an OOM kill.
    if library.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    # Cover parent death between fork and setting PR_SET_PDEATHSIG.
    if os.getppid() != int(parent):
        raise RuntimeError('Training owner exited before the child was ready')
    # The wrapper admits execution only after its PID receipt is durable.
    if sys.stdin.buffer.read(1) != b'G':
        raise RuntimeError('Training owner did not admit the child')
    os.execv(sys.executable, [sys.executable, entrypoint, *arguments])


if __name__ == '__main__':
    main()
