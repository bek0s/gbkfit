// The contour of a map at a level (e.g. half its peak: its FWHM), by
// marching squares: through the squares between the centres of its
// pixels, each crossing of a side of one where the level falls between
// its ends, linearly. As the path of an svg in the pixels of the map,
// as MapView draws them: x to the right, y down from its top row.
export function contourPath(values: number[][], level: number): string {
  const ny = values.length;
  const nx = values[0]?.length ?? 0;
  const at = (i: number, j: number) => values[j]?.[i] ?? 0;
  // (a point between the centres of two pixels, where the level falls)
  const cross = (i0: number, j0: number, i1: number, j1: number) => {
    const [v0, v1] = [at(i0, j0), at(i1, j1)];
    const t = (level - v0) / (v1 - v0);
    return [i0 + t * (i1 - i0) + 0.5, ny - 0.5 - (j0 + t * (j1 - j0))];
  };
  const parts: string[] = [];
  const segment = (a: number[], b: number[]) => parts.push(`M${a[0].toFixed(3)} ${a[1].toFixed(3)}L${b[0].toFixed(3)} ${b[1].toFixed(3)}`);
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      // (the corners of the square, around it, and its sides crossed)
      const corners: [number, number][] = [
        [i, j],
        [i + 1, j],
        [i + 1, j + 1],
        [i, j + 1],
      ];
      const crossings: number[][] = [];
      for (let k = 0; k < 4; k++) {
        const [p, q] = [corners[k], corners[(k + 1) % 4]];
        if (at(...p) >= level !== at(...q) >= level) crossings.push(cross(...p, ...q));
      }
      if (crossings.length === 2) segment(crossings[0], crossings[1]);
      else if (crossings.length === 4) {
        // (a saddle: its centre, the mean of its corners, says whether its
        // first and third corners join through it, the contour then cutting
        // off the other two, or the other two join)
        const centre = (at(i, j) + at(i + 1, j) + at(i + 1, j + 1) + at(i, j + 1)) / 4;
        const joined = centre >= level === at(i, j) >= level;
        segment(crossings[0], crossings[joined ? 1 : 3]);
        segment(crossings[2], crossings[joined ? 3 : 1]);
      }
    }
  }
  return parts.join('');
}

// The width of a profile at a level (e.g. half its peak: its FWHM), where
// it falls below the level on either side of its peak, linearly between
// the centres of its pixels: from and to, in pixels (that of a pixel at its
// centre), and the level, in the units it is drawn in (e.g. a logarithm)
export type Width = { level: number; from: number; to: number };

export function widthAt(values: (number | null)[], level: number): [number, number] | null {
  const at = (i: number) => values[i] ?? -Infinity;
  let top = 0;
  values.forEach((_, i) => at(i) > at(top) && (top = i));
  if (!(at(top) >= level)) return null;
  // (the crossing between a pixel at the level or above and its neighbour
  // below it)
  const cross = (i: number, next: number) => (at(next) === -Infinity ? i : i + ((at(i) - level) / (at(i) - at(next))) * (next - i));
  let [from, to] = [top, top];
  while (from > 0 && at(from - 1) >= level) from--;
  while (to < values.length - 1 && at(to + 1) >= level) to++;
  return [from > 0 ? cross(from, from - 1) : from, to < values.length - 1 ? cross(to, to + 1) : to];
}
