// The test galaxy of the picture of an observation: an ordinary galaxy of
// the library (a thin disk of exponential light, arctan rotation and a
// uniform dispersion, seen in a line, or in a band by an observation that
// sees bands), which the Studio fills with a few numbers, its knobs: its
// inclination, position angle, size (the scale length of its light) and
// rotation speed. It is placed at the centre of the field of the
// observation (as the library places positions, from the reference pixel
// of its grid: see the server's /observations/sampling), and its systemic
// velocity at the centre of its channels.
import type { Options } from '../../components/SchemaForm';

export interface TestKnobs {
  inclination: number;
  angle: number;
  size: number;
  rotation: number;
}

// The name of its disk, which names its parameters (e.g. disk_incl)
const DISK = 'disk';

// Its rings reach this many scale lengths, a ring each quarter of one;
// its dispersion (km/s), and the turnover radius of its rotation, in
// scale lengths
const REACH = 4;
const RINGS_PER_SIZE = 4;
const DISPERSION = 10;
const TURNOVER = 0.5;

// The knobs of a test galaxy as the picture opens, for a field of a width
// (arcsec): a twelfth of it as its size, round (its rings then reach a
// third of the field from its centre)
export function startKnobs(field: number | null): TestKnobs {
  return { inclination: 60, angle: 40, size: roundSize((field ?? 400) / 12), rotation: 150 };
}

// The spans of the knobs, of a field of a width (arcsec)
export function knobSpans(field: number | null): Record<keyof TestKnobs, [number, number]> {
  const width = field ?? 400;
  return { inclination: [0, 90], angle: [0, 360], size: [roundSize(width / 60), roundSize(width / 2)], rotation: [0, 400] };
}

// The galaxy of the knobs: its options and the properties of its
// parameters, at a centre (x, y; arcsec) and a systemic velocity (km/s);
// seen in lines (an observation of channels), or in a band (an image)
export function testGalaxy(knobs: TestKnobs, centre: [number, number], vsys: number, lines: boolean): { options: Options; properties: Options } {
  const options: Options = {
    components: [
      {
        type: 'thin_disk',
        name: DISK,
        loose: false,
        tilted: false,
        rnodes: { max: REACH * knobs.size, sep: knobs.size / RINGS_PER_SIZE },
        rptraits: { type: 'exponential' },
        vptraits: { type: 'arctan', direction: 'tan' },
        dptraits: { type: 'uniform' },
        emission: { type: lines ? 'lines' : 'band' },
      },
    ],
  };
  const p = (key: string) => `${DISK}_${key}`;
  const properties: Options = {
    [p('xpos')]: centre[0],
    [p('ypos')]: centre[1],
    [p('vsys')]: vsys,
    [p('posa')]: knobs.angle,
    [p('incl')]: knobs.inclination,
    [p('rpt_a')]: 1,
    [p('rpt_s')]: knobs.size,
    [p('vpt_rt')]: TURNOVER * knobs.size,
    [p('vpt_vt')]: knobs.rotation,
    [p('dpt_a')]: DISPERSION,
  };
  return { options, properties };
}

// A size rounded to 2 significant digits
function roundSize(size: number): number {
  return Number(size.toPrecision(2));
}
