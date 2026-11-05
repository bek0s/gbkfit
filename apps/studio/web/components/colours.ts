// The colours of the maps: ramps of the tokens of the page (see
// styles.css), sequential (low to high), or diverging about the middle;
// they change with the theme and scheme of the page (see useScheme). The
// sequential ramp can be another colour map, and inverted, as chosen (see
// useColourMap); the diverging one stays, its middle meaning a value (e.g.
// 0, or the median of velocities).
import { useState, useSyncExternalStore } from 'react';
import { colourMaps, STEPS } from '../palette';
import { preference, setPreference, usePreference } from '../shell/preferences';

const SEQUENTIAL = Array.from({ length: STEPS }, (_, i) => `--seq${i}`);
const DIVERGING = Array.from({ length: STEPS }, (_, i) => `--div${i}`);

// The theme and scheme of the page (the scheme chosen in the bar, or the
// system's), which mark its root (data-theme, data-mantine-color-scheme),
// followed as they change: the tokens of the page change with them, and so
// must what is drawn with them
const MARKS = ['data-theme', 'data-mantine-color-scheme'];

export function useScheme(): string {
  return useSyncExternalStore(
    (onChange) => {
      const observer = new MutationObserver(onChange);
      observer.observe(document.documentElement, { attributes: true, attributeFilter: MARKS });
      return () => observer.disconnect();
    },
    () => MARKS.map((mark) => document.documentElement.getAttribute(mark)).join(' '),
  );
}

// The colour (r, g, b) of a token of the page (a hex colour, of 6 digits or
// of 3, as the build shortens some: #fff)
export function token(name: string): number[] {
  let hex = getComputedStyle(document.documentElement).getPropertyValue(name).trim().replace('#', '');
  if (hex.length === 3) hex = [...hex].map((digit) => digit + digit).join('');
  return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
}

// The colour maps that can be chosen for the sequential ramp: that of
// the theme, or one of the palette
export type ColourMap = string;

export const COLOUR_MAPS: { name: ColourMap; label: string }[] = [
  { name: 'studio', label: 'Of the theme' },
  ...Object.entries(colourMaps).map(([name, { title }]) => ({ name, label: title })),
];

export type ColourChoice = { map: ColourMap; inverted: boolean };

// The colour map chosen, one for all the maps of the Studio, a preference
// of the user (see shell/preferences.ts): that of the theme, not inverted,
// if none
const UNCHOSEN: ColourChoice = { map: 'studio', inverted: false };
const isChoice = (value: unknown): value is ColourChoice =>
  typeof value === 'object' && value !== null && typeof (value as ColourChoice).map === 'string' && typeof (value as ColourChoice).inverted === 'boolean';

export function setColourMap(next: ColourChoice): void {
  setPreference('colour-map', next);
}

// The colour map chosen, followed as it changes: what draws with the ramps
// must draw again with it
export function useColourMap(): ColourChoice {
  return usePreference('colour-map', isChoice, UNCHOSEN);
}

// (a colour of the palette, #rrggbb, as r, g, b)
const rgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

// The colours of a colour map, from low to high, or from high to low
function mapStops({ map, inverted }: ColourChoice): number[][] {
  // (that of the theme, also for a map no longer known)
  const stops = map in colourMaps ? colourMaps[map].colours.map(rgb) : SEQUENTIAL.map(token);
  return inverted ? stops.reverse() : stops;
}

// The colours of a ramp, from low to high
export function rampStops(diverging: boolean): number[][] {
  return diverging ? DIVERGING.map(token) : mapStops(preference('colour-map', isChoice, UNCHOSEN));
}

// A colour map as a CSS gradient, to the right (e.g. its swatch in a menu)
export function mapGradient(map: ColourChoice): string {
  return `linear-gradient(to right, ${mapStops(map)
    .map((colour) => `rgb(${colour.join(', ')})`)
    .join(', ')})`;
}

// The colour at a fraction (0 to 1) of a ramp
export function rampColour(stops: number[][], fraction: number): number[] {
  const t = Math.min(1, Math.max(0, fraction)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t));
  const u = t - i;
  return stops[i].map((low, k) => Math.round(low + (stops[i + 1][k] - low) * u));
}

// A stretch of a ramp over the range of a map, as QFitsView sets it from
// where the pointer is on the map (see stretchAt in MapView): the fraction
// of the range at the middle of the ramp (less is brighter), and the
// fraction of the range the ramp spans about it (less is more contrast; 0
// makes it two colours), the values beyond it taking the colours of its
// ends
export type Stretch = { middle: number; width: number };

export const UNSTRETCHED: Stretch = { middle: 0.5, width: 1 };

export const isStretched = ({ middle, width }: Stretch) => middle !== UNSTRETCHED.middle || width !== UNSTRETCHED.width;

// The stretches of the colours of the maps of a viewer, one of each of
// what it shows (e.g. a channel of the data, or a moment), kept as it
// shows another: that of what it shows, and how it is set
export function useStretches(key: string): [Stretch, (stretch: Stretch) => void] {
  const [stretches, setStretches] = useState<Record<string, Stretch>>({});
  return [stretches[key] ?? UNSTRETCHED, (stretch) => setStretches((all) => ({ ...all, [key]: stretch }))];
}

// The fraction of a ramp at a fraction of the range, as stretched (a
// width of 0: the low end below the middle, the high end above it)
export function stretched(fraction: number, { middle, width }: Stretch): number {
  return width > 0 ? (fraction - middle) / width + 0.5 : fraction < middle ? 0 : 1;
}

// A ramp as a CSS gradient, in a direction (e.g. 'to top'), as stretched
// (sampled finely enough that its ends, where it clips, are sharp)
export function rampGradient(diverging: boolean, direction: string, stretch: Stretch = UNSTRETCHED): string {
  const stops = rampStops(diverging);
  const colours = isStretched(stretch)
    ? Array.from({ length: 65 }, (_, k) => `rgb(${rampColour(stops, stretched(k / 64, stretch)).join(', ')}) ${((k / 64) * 100).toFixed(2)}%`)
    : stops.map((colour) => `rgb(${colour.join(', ')})`);
  return `linear-gradient(${direction}, ${colours.join(', ')})`;
}
