// Frames: the binary messages of the server (arrays of files, results of
// sessions): the length of their header (4 bytes, big-endian), their
// header (JSON), and their data. The arrays of a frame are little-endian
// float32, in C order, at the offsets its header gives.

export interface FrameArray {
  // Its header (e.g. its observation and data item), with its shape
  info: Record<string, unknown> & { shape: number[] };
  data: Float32Array;
}

export interface Frame {
  header: Record<string, unknown>;
  arrays: FrameArray[];
}

type ArrayEntry = FrameArray['info'] & { offset: number; nbytes: number };

export function decode(buffer: ArrayBuffer): Frame {
  const length = new DataView(buffer).getUint32(0, false);
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 4, length)),
  ) as Record<string, unknown>;
  const start = 4 + length;
  const entries = (header.arrays ?? []) as ArrayEntry[];
  const arrays = entries.map((entry) => {
    const bytes = new DataView(buffer, start + entry.offset, entry.nbytes);
    const values = new Float32Array(entry.nbytes / 4);
    for (let i = 0; i < values.length; i++) values[i] = bytes.getFloat32(4 * i, true);
    return { info: entry, data: values };
  });
  return { header, arrays };
}
