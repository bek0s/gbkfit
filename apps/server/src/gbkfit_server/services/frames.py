"""
Frames: messages of a header (JSON) and binary data (e.g. arrays),
between the server and the workers of its sessions, and to its clients.
A frame is the length of its header (4 bytes, big-endian), its header
(UTF-8), and its data: as many bytes as its header says ('nbytes').
"""

import asyncio
import json
import math
import struct
from collections.abc import Sequence
from typing import Any, BinaryIO

import numpy as np


__all__ = [
    'array_frame',
    'arrays',
    'decode',
    'encode',
    'read',
    'read_async'
]


_LENGTH = struct.Struct('>I')


def encode(header: dict[str, Any], data: bytes = b'') -> bytes:
    """Return a frame of a header and its data."""
    text = json.dumps(header | dict(nbytes=len(data))).encode()
    return _LENGTH.pack(len(text)) + text + data


def array_frame(
        header: dict[str, Any],
        arrays: Sequence[tuple[dict[str, Any], np.ndarray]]
) -> bytes:
    """
    Return a frame of arrays (float32, C order), each with its own header
    (e.g. its name): the header of the frame lists them ('arrays'), with
    their shapes and the offsets of their bytes in the data.
    """
    entries, chunks, offset = [], [], 0
    for info, array in arrays:
        array = np.ascontiguousarray(array, dtype=np.float32)
        entries.append(info | dict(
            shape=list(array.shape), dtype='float32', offset=offset,
            nbytes=array.nbytes))
        chunks.append(array.tobytes())
        offset += array.nbytes
    return encode(header | dict(arrays=entries), b''.join(chunks))


def decode(frame: bytes) -> tuple[dict[str, Any], bytes]:
    """Return the header and the data of a frame."""
    length = _LENGTH.unpack_from(frame)[0]
    header = json.loads(frame[_LENGTH.size:_LENGTH.size + length])
    return header, frame[_LENGTH.size + length:]


def arrays(header: dict[str, Any], data: bytes) -> list[np.ndarray]:
    """Return the arrays of a frame of arrays (see array_frame)."""
    return [
        np.frombuffer(
            data, np.float32, math.prod(info['shape']), info['offset'])
        .reshape(info['shape'])
        for info in header['arrays']]


def read(stream: BinaryIO) -> tuple[dict[str, Any], bytes] | None:
    """Read a frame from a stream; None at its end."""
    prefix = stream.read(_LENGTH.size)
    if len(prefix) < _LENGTH.size:
        return None
    header = json.loads(stream.read(_LENGTH.unpack(prefix)[0]))
    return header, stream.read(header['nbytes'])


async def read_async(
        stream: asyncio.StreamReader
) -> tuple[dict[str, Any], bytes]:
    """
    Read a frame from a stream.

    Raises
    ------
    asyncio.IncompleteReadError
        If the stream ends first.
    """
    prefix = await stream.readexactly(_LENGTH.size)
    header = json.loads(await stream.readexactly(_LENGTH.unpack(prefix)[0]))
    return header, await stream.readexactly(header['nbytes'])
