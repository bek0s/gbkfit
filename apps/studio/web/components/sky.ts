// The pixels of a map on the sky, as the model measures positions there
// (arcsec, from a reference pixel, as xpos and ypos): their size and that
// pixel (x, y; 0-based); and boxes of the sky (x0, x1, y0, y1, arcsec) in
// the pixels of a map, and back. A region of the sky is so on every map
// of it, whatever its pixels (e.g. the data, the model through an
// observation, the galaxy as it is).
import type { Region } from './MapView';

export interface SkyFrame {
  step: [number, number];
  rpix: [number, number];
}

// The frame of an array of the server (its step and reference pixel), if
// it has coordinates on the sky
export function frameOf(step: number[] | null | undefined, rpix: number[] | null | undefined): SkyFrame | null {
  if (!step || !rpix || step.length < 2 || rpix.length < 2 || !step[0] || !step[1]) return null;
  return { step: [step[0], step[1]], rpix: [rpix[0], rpix[1]] };
}

// (the edges of a pixel, half a pixel about its centre)
const sorted = (a: number, b: number): [number, number] => (a <= b ? [a, b] : [b, a]);

// The box of the sky of the pixels of a region
export function toSky([x0, x1, y0, y1]: Region, { step: [sx, sy], rpix: [rx, ry] }: SkyFrame): [number, number, number, number] {
  const [a, b] = sorted((x0 - 0.5 - rx) * sx, (x1 + 0.5 - rx) * sx);
  const [c, d] = sorted((y0 - 0.5 - ry) * sy, (y1 + 0.5 - ry) * sy);
  return [a, b, c, d];
}

// The pixels of a map (nx by ny) whose centres are in a box of the sky;
// none if none are
export function toPixels(box: number[], { step: [sx, sy], rpix: [rx, ry] }: SkyFrame, [nx, ny]: [number, number]): Region | null {
  const [a, b] = sorted(rx + box[0] / sx, rx + box[1] / sx);
  const [c, d] = sorted(ry + box[2] / sy, ry + box[3] / sy);
  const x0 = Math.max(0, Math.ceil(a - 1e-9));
  const x1 = Math.min(nx - 1, Math.floor(b + 1e-9));
  const y0 = Math.max(0, Math.ceil(c - 1e-9));
  const y1 = Math.min(ny - 1, Math.floor(d + 1e-9));
  return x0 <= x1 && y0 <= y1 ? [x0, x1, y0, y1] : null;
}
