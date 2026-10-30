// Frames of arrays from the server (see its services/frames.py): the
// length of a header (4 bytes, big-endian), the header (JSON), and the
// values of its arrays (little-endian float32), each listed in the header
// with its shape and the offset of its bytes.

export interface FrameArray {
  // Its own entries of the header (e.g. its name), and its shape (numpy
  // order: rows last, e.g. [ny, nx])
  info: Record<string, unknown>;
  shape: number[];
  values: Float32Array;
}

export interface Frame {
  header: Record<string, unknown>;
  arrays: FrameArray[];
}

interface Entry {
  shape: number[];
  offset: number;
  nbytes: number;
}

export function decodeFrame(buffer: ArrayBuffer): Frame {
  const length = new DataView(buffer).getUint32(0, false);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, length))) as Record<string, unknown>;
  const start = 4 + length;
  const entries = (header.arrays ?? []) as (Entry & Record<string, unknown>)[];
  const arrays = entries.map(({ shape, offset, nbytes, ...info }) => ({
    info,
    shape,
    // (copied: the offsets of float32 need not be aligned in the buffer)
    values: new Float32Array(buffer.slice(start + offset, start + offset + nbytes)),
  }));
  return { header, arrays };
}
