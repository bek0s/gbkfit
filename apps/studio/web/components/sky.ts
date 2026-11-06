// The pixels of a map on the sky, as the model measures positions there
// (arcsec, from a reference pixel, as xpos and ypos): their size and that
// pixel (x, y; 0-based); a pixel (x, y) is at (x - rpix) * step.

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
