import { describe, expect, it } from 'vitest';
import { decode } from './frames';

// A frame as the server writes it (see services/frames.py)
function encode(header: object, arrays: number[][]): ArrayBuffer {
  let offset = 0;
  const entries = arrays.map((values) => {
    const entry = { shape: [values.length], offset, nbytes: 4 * values.length };
    offset += entry.nbytes;
    return entry;
  });
  const text = new TextEncoder().encode(
    JSON.stringify({ ...header, arrays: entries, nbytes: offset }),
  );
  const buffer = new ArrayBuffer(4 + text.length + offset);
  const view = new DataView(buffer);
  view.setUint32(0, text.length, false);
  new Uint8Array(buffer, 4, text.length).set(text);
  arrays.flat().forEach((value, i) => view.setFloat32(4 + text.length + 4 * i, value, true));
  return buffer;
}

describe('decode', () => {
  it('reads the header and the arrays of a frame', () => {
    const frame = decode(encode({ type: 'result' }, [[1, 2, 3], [4.5]]));
    expect(frame.header.type).toBe('result');
    expect(frame.arrays.map((a) => Array.from(a.data))).toEqual([[1, 2, 3], [4.5]]);
    expect(frame.arrays[1].info.shape).toEqual([1]);
  });
});
