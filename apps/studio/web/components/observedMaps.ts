// The maps of what an observation sees of a model (see the server's
// /views/observed and /views/setup-observed): one map of its model or of
// its data, the same for every kind of observation on a grid of pixels
// (its intensity, velocity or dispersion: the moment maps of moment maps,
// the moments of a cube measured as from data, or the light of an image).
import { mapUnit, SNR } from './CubeViewer';
import { useChannel, useMoments, type useObserved } from '../api/queries';

// What the cells show: a moment of the light (an image shows its light
// whatever is chosen: it has no other map)
export type Shown = 'intensity' | 'velocity' | 'dispersion';

export const SHOWN: { value: Shown; label: string; colours: Colours }[] = [
  { value: 'intensity', label: 'Intensity', colours: 'sequential' },
  { value: 'velocity', label: 'Velocity', colours: 'median' },
  { value: 'dispersion', label: 'Dispersion', colours: 'sequential' },
];

export type Colours = 'sequential' | 'median' | 'zero';

// The item of each map in moment maps, and the moment of a cube
const MOMENT_ITEMS: Record<Shown, string> = { intensity: 'moment0', velocity: 'moment1', dispersion: 'moment2' };
const CUBE_MOMENTS = { intensity: 'flux', velocity: 'velocity', dispersion: 'dispersion' } as const;

export type Observed = NonNullable<ReturnType<typeof useObserved>['data']>;

// A map of a galaxy observed: of its model or of its data, of the item
// shown (moment maps), or the moment of its cube, measured as from data
// (of the data, in the mask of their signal; see CubeViewer); an image is
// its light alone
export function useObservedMap(observed: Observed | undefined, kind: 'model' | 'data', show: Shown) {
  const arrays = observed?.arrays.filter((array) => array.kind === kind) ?? [];
  const cube = observed?.type === 'pixel_spectra';
  const item = cube ? arrays[0] : (arrays.find((array) => array.item === MOMENT_ITEMS[show]) ?? arrays.find((array) => array.item === 'brightness'));
  const plane = useChannel(!cube && item ? item.path : null, 0, 0);
  const moments = useMoments(item?.path ?? '', 0, kind === 'data' ? SNR : 0, cube && item !== undefined);
  const values = cube ? moments.data?.find((array) => array.info.name === CUBE_MOMENTS[show])?.values : plane.data?.plane.values;
  const unit = cube ? mapUnit(CUBE_MOMENTS[show], item?.unit || null) : (item?.unit ?? '');
  const colours: Colours = item?.item === 'brightness' ? 'sequential' : SHOWN.find((s) => s.value === show)!.colours;
  return { item, values, unit, colours, error: (cube ? moments.error : plane.error) ?? null };
}

