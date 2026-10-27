"""Linux process handles work even when Python was built without pidfd wrappers."""

import ctypes
import os
from pathlib import Path
import platform
import signal


def process_start(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def open_process(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("This launcher requires Linux process handles")
    # Both supported Linux architectures assign syscall 434 to pidfd_open.
    library = ctypes.CDLL(None, use_errno=True)
    library.syscall.restype = ctypes.c_long
    result = library.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return int(result)


def open_matching_process(pid, birth):
    if birth is None or process_start(pid) != birth:
        return None
    try:
        handle = open_process(pid)
    except ProcessLookupError:
        return None
    if process_start(pid) != birth:
        os.close(handle)
        return None
    return handle


def signal_process(handle, signum):
    try:
        if hasattr(signal, "pidfd_send_signal"):
            signal.pidfd_send_signal(handle, signum)
            return
        library = ctypes.CDLL(None, use_errno=True)
        library.syscall.restype = ctypes.c_long
        result = library.syscall(
            ctypes.c_long(424),
            ctypes.c_int(handle),
            ctypes.c_int(signum),
            ctypes.c_void_p(),
            ctypes.c_uint(0),
        )
        if result < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
    except ProcessLookupError:
        pass
