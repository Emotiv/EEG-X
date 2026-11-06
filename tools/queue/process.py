"""Own training children before dispatch and stop them when their owner disappears."""

import ctypes
import os
import platform
import signal
import subprocess


def protect_parent(parent):
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot establish parent-death protection')
    if os.getppid() != parent:
        raise RuntimeError('Queue owner exited before GPU setup')


def open_handle(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    if platform.system() != 'Linux' or platform.machine() not in {'x86_64', 'aarch64'}:
        raise RuntimeError('The GPU queue requires Linux process handles')
    library = ctypes.CDLL(None, use_errno=True)
    library.syscall.restype = ctypes.c_long
    descriptor = library.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), 'Cannot open training process handle')
    return int(descriptor)


def start(command, **kwargs):
    process = subprocess.Popen(command, **kwargs)
    try:
        descriptor = open_handle(process.pid)
    except BaseException:
        process.terminate()
        process.wait()
        raise
    return process, descriptor
