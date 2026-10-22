"""Inspect legacy NumPy pickle dictionaries without loading their signal arrays.

Large byte payloads stay in the source file. Only a small set of NumPy
reconstruction operations is accepted; arbitrary pickle globals are rejected.
"""

from __future__ import annotations

import pickle
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ByteSpan:
    offset: int
    length: int


class ArrayView:
    def __setstate__(self, state):
        _, self.shape, self.dtype, self.fortran, self.payload = state

    def read(self, source):
        if isinstance(self.payload, ByteSpan):
            if self.dtype.hasobject:
                raise ValueError("Object arrays cannot use a raw byte span")
            expected = int(np.prod(self.shape)) * self.dtype.itemsize
            if expected != self.payload.length:
                raise ValueError("Array shape does not match stored bytes")
            return np.memmap(
                source,
                mode="r",
                dtype=self.dtype,
                offset=self.payload.offset,
                shape=self.shape,
                order="F" if self.fortran else "C",
            )
        if isinstance(self.payload, (bytes, bytearray)):
            return np.frombuffer(self.payload, dtype=self.dtype).reshape(
                self.shape, order="F" if self.fortran else "C"
            )
        return np.asarray(self.payload, dtype=self.dtype).reshape(self.shape)


def _reconstruct(*_args):
    return ArrayView()


def _frombuffer(payload, dtype, shape, order):
    value = ArrayView()
    value.__setstate__((1, shape, dtype, order == "F", payload))
    return value


class _MetadataUnpickler(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, handle):
        super().__init__(handle)
        self.handle = handle

    def find_class(self, module, name):
        if module in {"numpy", "numpy.core.multiarray", "numpy._core.multiarray"}:
            if name == "_reconstruct":
                return _reconstruct
            if name == "ndarray":
                return ArrayView
            if name == "dtype":
                return np.dtype
            if name == "scalar":
                return lambda dtype, data: np.frombuffer(data, dtype=dtype)[0]
        if module in {"numpy.core.numeric", "numpy._core.numeric"} and name == "_frombuffer":
            return _frombuffer
        raise pickle.UnpicklingError(f"Unsupported metadata operation: {module}.{name}")

    def payload(self, length):
        if length < 4096:
            return self.read(length)
        frame = self._unframer.current_frame
        if frame is not None:
            position = frame.tell()
            remaining = len(frame.getbuffer()) - position
            if remaining >= length:
                offset = self.handle.tell() - remaining
                frame.seek(length, 1)
                return ByteSpan(offset, length)
            if remaining:
                raise pickle.UnpicklingError("Array payload crosses a pickle frame")
            self._unframer.current_frame = None
        offset = self.handle.tell()
        if offset + length > self.handle.seek(0, 2):
            raise pickle.UnpicklingError("Truncated array payload")
        self.handle.seek(offset + length)
        return ByteSpan(offset, length)

    def load_binbytes(self):
        length = struct.unpack("<I", self.read(4))[0]
        self.append(self.payload(length))

    def load_binbytes8(self):
        length = struct.unpack("<Q", self.read(8))[0]
        self.append(self.payload(length))

    def load_bytearray8(self):
        length = struct.unpack("<Q", self.read(8))[0]
        self.append(self.payload(length))

    dispatch[pickle.BINBYTES[0]] = load_binbytes
    dispatch[pickle.BINBYTES8[0]] = load_binbytes8
    dispatch[pickle.BYTEARRAY8[0]] = load_bytearray8


def inspect_artifact(path):
    """Read raw pickle dictionaries or scalar NumPy dictionary containers."""
    path = Path(path)
    with path.open("rb") as handle:
        magic = handle.read(6)
        handle.seek(0)
        numpy_container = magic == b"\x93NUMPY"
        if numpy_container:
            version = np.lib.format.read_magic(handle)
            readers = {
                (1, 0): np.lib.format.read_array_header_1_0,
                (2, 0): np.lib.format.read_array_header_2_0,
            }
            if version not in readers:
                raise ValueError("Unsupported NumPy container version")
            shape, _, dtype = readers[version](handle)
            if shape != () or not dtype.hasobject:
                raise ValueError("Expected a scalar NumPy dictionary container")
        value = _MetadataUnpickler(handle).load()
        if numpy_container:
            if not isinstance(value, ArrayView) or value.shape != () or len(value.payload) != 1:
                raise ValueError("Invalid NumPy dictionary container")
            value = value.payload[0]
    if not isinstance(value, dict):
        raise ValueError("Expected a dictionary artifact")
    return value


def summarize(path):
    values = inspect_artifact(path)
    output = {}
    for key, value in values.items():
        if isinstance(value, ArrayView):
            record = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if int(np.prod(value.shape)) <= 2_000_000 and not value.dtype.hasobject:
                array = value.read(path)
                if array.ndim <= 1:
                    classes, counts = np.unique(array, return_counts=True)
                    if len(classes) <= 64:
                        record["values"] = classes.tolist()
                        record["counts"] = counts.tolist()
            output[str(key)] = record
        elif isinstance(value, (str, int, float, bool, type(None))):
            output[str(key)] = value
        else:
            output[str(key)] = {"type": type(value).__name__}
    return output
