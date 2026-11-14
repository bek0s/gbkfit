// The profiles of a galaxy along its radius, as one view draws them all
// (see Views): the speeds of its disks along each direction they move in
// (the circular speeds of its mass under them, dashed), their dispersion
// (along each direction, of one that is not isotropic), their surface
// density, and the orientation of their rings (their position angle and
// inclination at each ring). Each curve is in the colour of its component,
// and its dash says its direction. The library gives them for disks (see
// its GalaxyFunctions): cones and spheroids have none yet.
import type { GalaxyRender } from '../../api/queries';
import type { Param, Properties } from './params';
import { settingOf } from './params';
import { CurvePlot, type Curve } from './Curves';
import { componentName, componentsOf, ringRadii, type Options } from './options';

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
// e.g. 'speed_gas:rad'; of one disk without a name, 'speed:rad')
function parse(name: string, family: string): { component: string; direction: string | null } | null {
  if (name !== family && !name.startsWith(`${family}_`) && !name.startsWith(`${family}:`)) return null;
  const [head, direction] = name.split(':');
  return { component: head === family ? '' : head.slice(family.length + 1), direction: direction ?? null };
}

export function Profiles({
  render,
  quantity,
  options,
  params,
  properties,
  colours,
  rings,
}: {
  render: GalaxyRender | undefined;
  quantity: ProfileQuantity;
  options: Options;
  params: Param[];
  properties: Properties;
  // The colour of each component, by name, and the rings to mark
  colours: Map<string, string>;
  rings: number[] | null;
}) {
  const components = componentsOf(options);
  // (one disk without a name: its curves are named after none)
  const nameOf = (suffix: string) => suffix || (components.length ? componentName(components[0], 0) : 'disk');
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
    unit = 'in the unit of each';
    for (const name of Object.keys(all)) {
      const parsed = parse(name, 'density');
      if (parsed === null) continue;
      curves.push({ name, label: nameOf(parsed.component), colour: colourOf(parsed.component), values: all[name] });
    }
  } else {
    // (the position angle and inclination of each component, at its rings:
    // one value of each, or one per ring of a tilted disk)
    unit = '°';
    components.forEach((component, index) => {
      const radii = ringRadii(component.rnodes);
      for (const [key, dash] of [
        ['posa', undefined],
        ['incl', '5 3'],
      ] as const) {
        const param = params.find((p) => p.owner.length === 2 && p.owner[0] === 'components' && p.owner[1] === index && (p.name === key || p.name.endsWith(`_${key}`)));
        if (!param) continue;
        const setting = settingOf(properties, param);
        if (setting.kind !== 'fixed' && setting.kind !== 'free') continue;
        const values = setting.values.length ? setting.values : param.default === null ? [] : [param.default];
        if (values.length === 0) continue;
        const along = radii.length > 1 ? radii : [0, typeof component.rmax === 'number' ? component.rmax : 1];
        const each = along.map((_, i) => values[Math.min(i, values.length - 1)]);
        const name = componentName(component, index);
        curves.push({ name: `${name}/${key}`, label: `${name} · ${key === 'posa' ? 'position angle' : 'inclination'}`, colour: colours.get(name) ?? 'fg2', values: each, radius: along, dash });
      }
    });
  }
  const empty: Record<ProfileQuantity, string> = {
    speed: 'No speeds: they are drawn of disks with velocity traits.',
    dispersion: 'No dispersion: it is drawn of disks with dispersion traits.',
    density: 'No densities: they are drawn of disks.',
    orientation: 'No orientation: the components have no position angle or inclination.',
  };
  return <CurvePlot curves={curves} radius={all.radius ?? []} rings={rings} unit={unit} empty={render ? empty[quantity] : 'Rendering…'} />;
}
