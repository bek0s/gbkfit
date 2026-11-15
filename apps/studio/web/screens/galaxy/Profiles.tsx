// The profiles of a galaxy along its radius, as one view draws them all
// (see Views): the speeds of its components along each direction they
// move in (the circular speeds of its mass under them, dashed), their
// dispersion (along each direction, of one that is not isotropic), their
// density (each relative to its peak, on a logarithmic axis: densities of
// different units, e.g. of gas and of starlight, side by side), and their
// orientation (their position angle and inclination). Each curve is in
// the colour of its component, and its dash says its direction. The
// library gives them (see its GalaxyFunctions): of a disk along its
// radius, of a cone from its apex, of a spheroid from its axis.
import type { GalaxyRender } from '../../api/queries';
import { CurvePlot, type Curve } from './Curves';
import { componentName, componentsOf, type Options } from './options';

export type ProfileQuantity = 'speed' | 'dispersion' | 'density' | 'orientation';

export const PROFILE_QUANTITIES: { value: ProfileQuantity; label: string }[] = [
  { value: 'speed', label: 'Speed' },
  { value: 'dispersion', label: 'Dispersion' },
  { value: 'density', label: 'Density' },
  { value: 'orientation', label: 'Orientation' },
];

// The directions of speeds and dispersions, in words and dashes
const DIRECTIONS: Record<string, { word: string; dash?: string }> = {
  tan: { word: 'tangential' },
  rad: { word: 'radial', dash: '5 3' },
  ver: { word: 'vertical', dash: '2 3' },
  away: { word: 'away from the plane', dash: '7 2 1 2' },
  out: { word: 'outward', dash: '1 3' },
  los: { word: 'along the line of sight', dash: '8 3' },
};

// (a curve's name: its family, its component's suffix and its direction,
// e.g. 'speed_gas:rad'; of one component without a name, 'speed:rad')
function parse(name: string, family: string): { component: string; direction: string | null } | null {
  if (name !== family && !name.startsWith(`${family}_`) && !name.startsWith(`${family}:`)) return null;
  const [head, direction] = name.split(':');
  return { component: head === family ? '' : head.slice(family.length + 1), direction: direction ?? null };
}

export function Profiles({
  render,
  quantity,
  options,
  colours,
  rings,
}: {
  render: GalaxyRender | undefined;
  quantity: ProfileQuantity;
  options: Options;
  // The colour of each component, by name, and the rings to mark
  colours: Map<string, string>;
  rings: number[] | null;
}) {
  const components = componentsOf(options);
  // (one component without a name: its curves are named after none)
  const nameOf = (suffix: string) => suffix || (components.length ? componentName(components[0], 0) : 'component');
  const colourOf = (suffix: string) => colours.get(nameOf(suffix)) ?? 'fg2';
  const curves: Curve[] = [];
  const all = render?.curves ?? {};
  let unit = 'km/s';
  if (quantity === 'speed' || quantity === 'dispersion') {
    for (const name of Object.keys(all)) {
      const parsed = parse(name, quantity);
      if (parsed === null) continue;
      // (a dispersion along directions: the isotropic one is not given)
      const direction = parsed.direction ?? (quantity === 'dispersion' ? null : 'tan');
      const word = direction ? DIRECTIONS[direction]?.word ?? direction : 'isotropic';
      curves.push({ name, label: `${nameOf(parsed.component)} · ${word}`, colour: colourOf(parsed.component), values: all[name], dash: direction ? DIRECTIONS[direction]?.dash : undefined });
    }
    if (quantity === 'speed') {
      if (all.vcirc) curves.push({ name: 'vcirc', label: 'circular speed of the mass', colour: 'muted', values: all.vcirc, width: 1.4 });
      for (const name of Object.keys(all).filter((n) => n.startsWith('vcirc_'))) {
        curves.push({ name, label: `mass · ${name.slice(6)}`, colour: 'faint', values: all[name], dash: '2 3', width: 1.2 });
      }
    }
  } else if (quantity === 'density') {
    // (each relative to its peak)
    unit = 'relative to its peak';
    for (const name of Object.keys(all)) {
      const parsed = parse(name, 'density');
      if (parsed === null) continue;
      const peak = Math.max(0, ...all[name].filter((v): v is number => v !== null && Number.isFinite(v)));
      if (peak <= 0) continue;
      curves.push({ name, label: nameOf(parsed.component), colour: colourOf(parsed.component), values: all[name].map((v) => (v === null ? null : v / peak)) });
    }
  } else {
    // (the position angle, and the inclination dashed, of each component)
    unit = '°';
    for (const name of Object.keys(all)) {
      const parsed = parse(name, 'orientation');
      if (parsed === null) continue;
      const incl = parsed.direction === 'incl';
      curves.push({ name, label: `${nameOf(parsed.component)} · ${incl ? 'inclination' : 'position angle'}`, colour: colourOf(parsed.component), values: all[name], dash: incl ? '5 3' : undefined });
    }
  }
  const empty: Record<ProfileQuantity, string> = {
    speed: 'No speeds: they are drawn of the components that move by traits or by the Jeans equations.',
    dispersion: 'No dispersion: it is drawn of the components that move by traits or by the Jeans equations.',
    density: 'No densities: no component has any.',
    orientation: 'No orientation: the components have no position angle or inclination.',
  };
  return <CurvePlot curves={curves} radius={all.radius ?? []} rings={rings} unit={unit} empty={render ? empty[quantity] : 'Rendering…'} log={quantity === 'density'} />;
}
