// The colours of the maps: ramps of the tokens of the page (see
// styles.css), sequential (low to high), or diverging about the middle.

const SEQUENTIAL = ['--seq0', '--seq1', '--seq2', '--seq3'];
const DIVERGING = ['--map-neg2', '--map-neg', '--map-mid', '--map-pos', '--map-pos2'];

// The colour (r, g, b) of a token of the page (a hex colour, of 6 digits or
// of 3, as the build shortens some: #fff)
export function token(name: string): number[] {
  let hex = getComputedStyle(document.documentElement).getPropertyValue(name).trim().replace('#', '');
  if (hex.length === 3) hex = [...hex].map((digit) => digit + digit).join('');
  return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
}

// The colours of a ramp, from low to high
export function rampStops(diverging: boolean): number[][] {
  return (diverging ? DIVERGING : SEQUENTIAL).map(token);
}

// The colour at a fraction (0 to 1) of a ramp
export function rampColour(stops: number[][], fraction: number): number[] {
  const t = Math.min(1, Math.max(0, fraction)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t));
  const u = t - i;
  return stops[i].map((low, k) => Math.round(low + (stops[i + 1][k] - low) * u));
}

// A ramp as a CSS gradient, in a direction (e.g. 'to top')
export function rampGradient(diverging: boolean, direction: string): string {
  return `linear-gradient(${direction}, ${rampStops(diverging)
    .map((colour) => `rgb(${colour.join(', ')})`)
    .join(', ')})`;
}
