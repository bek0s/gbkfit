// What a galaxy is made of, as the screen builds it: the galaxy as a whole
// (its distance, mass and gravity), and its components (disks, cones,
// spheroids, points), each of its parts (its geometry, density, motions,
// light, ...). A part is an option of its component (or of the galaxy):
// a list of traits of a family (e.g. the velocity traits, 'vpt'), or one
// option of a type of a family (e.g. the emission). Each part owns the
// parameters its option makes (see Param.owner).
import { asList, getAt, isObject, type Options, type Path } from './options';

export type ComponentType = 'thin_disk' | 'thick_disk' | 'cone' | 'spheroid' | 'point' | 'particles';

export interface Part {
  key: string;
  title: string;
  // What it is, in a few words (under its title)
  hint?: string;
  // Its option, in its component (or in the galaxy)
  option: Path;
  // A list of items of a family of the schema (e.g. 'vpt')...
  family?: string;
  // ... or one option of a type of a family (e.g. 'emission')
  typed?: string;
  // Whether it can be added and removed (else always there)
  optional?: boolean;
  // What it starts with, added
  start?: unknown;
  // Why it cannot be added to a component (or the galaxy) as it is, if not
  needs?: (owner: Options, galaxy: Options) => string | null;
  // The height traits paired with its items (one each; e.g. the height
  // traits of the density), of a thick disk or a cone
  heights?: { option: Path; family: string; title: string; nullable: boolean };
}

// The types of the disks: thin (its light in its plane) or thick (its
// density has a height, a trait for each density trait)
export const DISK_TYPES = ['thin_disk', 'thick_disk'];

export function isDisk(type: unknown): boolean {
  return DISK_TYPES.includes(String(type));
}

export function isThick(component: Options): boolean {
  return component.type === 'thick_disk';
}

// The height a thick disk gives each of its density traits at first
export const HEIGHT_START: Options = { type: 'sech2', zmax: 6 };

// The options only a thick disk has: of the disk, and of its density and
// kinematics (a thick disk made thin loses them)
export const THICK_OPTIONS: Path[] = [['opacity'], ['zptraits'], ['clouds'], ['density', 'htraits'], ['kinematics', 'vhtraits'], ['kinematics', 'dhtraits'], ['kinematics', 'equilibrium']];

// The parts of a thin or a thick disk: those of a thick one with their
// heights; on a thin one, warps and opacity are shown, but need it thick
function diskParts(thick: boolean): Part[] {
  const needsThick = thick ? undefined : () => 'Needs a thick disk: make it thick in its density.';
  const height = (option: Path, family: string, nullable: boolean) => (thick ? { option, family, title: 'Height', nullable } : undefined);
  return [
    { key: 'geometry', title: 'Geometry', hint: 'Its rings, centre and orientation', option: [] },
    {
      key: 'density',
      title: 'Density',
      hint: thick ? 'How its light is spread over its plane, and its height' : 'How its light is spread over its plane',
      option: ['density', 'ptraits'],
      family: 'rpt',
      heights: height(['density', 'htraits'], 'rht', false),
    },
    {
      key: 'motions',
      title: 'Motions',
      hint: 'Its velocities: rotation, inflow, outflow, ...',
      option: ['kinematics', 'vptraits'],
      family: 'vpt',
      heights: height(['kinematics', 'vhtraits'], 'vht', true),
    },
    {
      key: 'dispersion',
      title: 'Dispersion',
      hint: 'The spread of its velocities',
      option: ['kinematics', 'dptraits'],
      family: 'dpt',
      heights: height(['kinematics', 'dhtraits'], 'dht', true),
    },
    { key: 'warps', title: 'Warps', hint: 'Its plane bent up and down', option: ['zptraits'], family: 'zpt', optional: true, start: [{ type: 'nw_harmonic', order: 1, zmax: 6 }], needs: needsThick },
    { key: 'selections', title: 'Selections', hint: 'Only some of it: a range of radii or azimuths', option: ['sptraits'], family: 'spt', optional: true, start: [{ type: 'rrange' }] },
    { key: 'light', title: 'Light', hint: 'What it emits: lines, or light in bands', option: ['emission'], typed: 'emission' },
    { key: 'opacity', title: 'Opacity', hint: 'Dust, or the absorption of its own lines', option: ['opacity'], typed: 'opacity', optional: true, start: { type: 'dust' }, needs: needsThick },
  ];
}

const CONE: Part[] = [
  { key: 'geometry', title: 'Geometry', hint: 'Its shells, apex and axis', option: [] },
  {
    key: 'density',
    title: 'Density',
    hint: 'How its light is spread over its walls',
    option: ['density', 'ptraits'],
    family: 'rpt',
    heights: { option: ['density', 'htraits'], family: 'rht', title: 'Across its walls', nullable: false },
  },
  { key: 'motions', title: 'Motions', hint: 'Its velocities: out of its apex, or about its axis', option: ['kinematics', 'vptraits'], family: 'vpt' },
  { key: 'dispersion', title: 'Dispersion', hint: 'The spread of its velocities', option: ['kinematics', 'dptraits'], family: 'dpt' },
  { key: 'selections', title: 'Selections', hint: 'Only some of it: a range of radii or azimuths', option: ['sptraits'], family: 'spt', optional: true, start: [{ type: 'rrange' }] },
  { key: 'light', title: 'Light', hint: 'What it emits: lines, or light in bands', option: ['emission'], typed: 'emission' },
  { key: 'opacity', title: 'Opacity', hint: 'Dust in its walls', option: ['opacity'], typed: 'opacity', optional: true, start: { type: 'dust' } },
];

const SPHEROID: Part[] = [
  { key: 'geometry', title: 'Geometry', hint: 'Its centre, orientation and extent', option: [] },
  { key: 'profile', title: 'Profile', hint: 'How its light falls with radius', option: ['profile'], typed: 'light_profile' },
  { key: 'shape', title: 'Shape', hint: 'Round, or flattened', option: ['shape'], typed: 'spheroid_shape', optional: true, start: { type: 'flattened' } },
  { key: 'motions', title: 'Motions', hint: 'Its stars: in equilibrium (Jeans), on orbits, or as traits', option: ['kinematics'], typed: 'spheroid_kinematics', optional: true, start: { type: 'jeans' } },
  { key: 'light', title: 'Light', hint: 'Its light in bands, or its stars', option: ['emission'], typed: 'emission' },
];

const POINT: Part[] = [
  { key: 'geometry', title: 'Position', hint: 'Where it is', option: [] },
  { key: 'light', title: 'Light', hint: 'What it emits', option: ['emission'], typed: 'emission' },
];

export const PARTS: Record<string, Part[]> = {
  thin_disk: diskParts(false),
  thick_disk: diskParts(true),
  cone: CONE,
  spheroid: SPHEROID,
  point: POINT,
  particles: POINT,
};

export const GALAXY_PARTS: Part[] = [
  {
    key: 'distance',
    title: 'Distance',
    hint: 'How far it is: as a parameter, or of its redshift',
    option: ['distance'],
    typed: 'distance',
    optional: true,
    start: { type: 'direct' },
  },
  {
    key: 'mass',
    title: 'Mass',
    hint: 'What its gravity comes from: halos, disks, bulges, ...',
    option: ['mass'],
    family: 'mass_component',
    optional: true,
    start: [],
  },
  {
    key: 'gravity',
    title: 'Gravity',
    hint: 'How its mass pulls: Newton, MOND, ...',
    option: ['gravity'],
    typed: 'gravity',
    optional: true,
    start: { type: 'newton' },
    needs: (galaxy) => (asList(galaxy.mass).length ? null : 'Needs mass: gravity is that of the mass.'),
  },
];

// The parts of a component (of its type)
export function partsOf(component: Options | undefined): Part[] {
  return PARTS[String(component?.type ?? '')] ?? [];
}

// The order of the types of components, to add one (others after them)
export const COMPONENT_ORDER = ['thin_disk', 'thick_disk', 'cone', 'spheroid', 'point', 'particles'];

// The name a new component of a type is given (made unique): a disk is a
// disk, thin or thick
export function nameBase(type: string): string {
  return isDisk(type) ? 'disk' : type;
}

// A type of component in words (e.g. 'thick disk')
export function typeLabel(type: unknown): string {
  return String(type ?? '').replaceAll('_', ' ');
}

// What each type of component starts with: the least that shows (light,
// and of a disk or cone of lines, its motions)
const DISK_START: Options = {
  rnodes: { max: 16, sep: 2 },
  density: { ptraits: { type: 'exponential' } },
  kinematics: { type: 'traits', vptraits: { type: 'arctan', direction: 'tan' }, dptraits: { type: 'uniform' } },
  emission: { type: 'lines' },
};

export const COMPONENT_STARTS: Record<string, Options> = {
  thin_disk: DISK_START,
  thick_disk: { ...DISK_START, density: { ptraits: { type: 'exponential' }, htraits: HEIGHT_START } },
  cone: {
    rnodes: { max: 6, sep: 1 },
    density: { ptraits: { type: 'uniform' }, htraits: { type: 'gauss', zmax: 2 } },
    kinematics: { type: 'traits', vptraits: { type: 'uniform', direction: 'out' }, dptraits: { type: 'uniform' } },
    emission: { type: 'lines' },
  },
  spheroid: { rmax: 10, profile: { type: 'sersic' }, emission: { type: 'band' } },
  point: { emission: { type: 'band' } },
};

// What the numbers an item requires start at, by name, where the schema
// gives none (e.g. the zmax of a height trait); others at their least
export const NUMBER_STARTS: Record<string, number> = { zmax: 6, order: 2, nblobs: 1, ngauss: 3 };

// What some new items start with, by family and type, before the options
// they require: a halo of an NFW profile, a mass disk of an exponential
// surface, thin; self-absorption the same everywhere
export const ITEM_STARTS: Record<string, Record<string, Options>> = {
  opacity: { line: { coefficients: { type: 'uniform' } } },
  mass_component: {
    spheroid: { profile: { type: 'nfw' } },
    disk: { surface: { type: 'exponential' }, vertical: { type: 'thin' } },
  },
};

// The labels of types where their names say too little
export const TYPE_LABELS: Record<string, Record<string, string>> = {
  opacity: { dust: 'dust', line: 'self-absorption (of its lines)' },
  emission: { lines: 'emission lines', band: 'light in bands', stars: 'stars (their spectra)' },
};

// The options of items the screen sets in their heads, not among their
// other options (see Inspector); of mass components, their names too
export const HEAD_OPTIONS = new Set(['type', 'direction', 'component']);
export const MASS_HEAD_OPTIONS = new Set([...HEAD_OPTIONS, 'name']);

// The galaxy as a whole, or a component, as the owner of a part
export function ownerOf(galaxy: Options, component: number | null): Options {
  if (component === null) return galaxy;
  const owner = getAt(galaxy, ['components', component]);
  return isObject(owner) ? owner : {};
}
