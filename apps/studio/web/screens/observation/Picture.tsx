// The picture of an observation, as the lens screen's: a galaxy as this
// observation sees it (the main view: its model on the pixels of the
// observation, through its instrument; no data), and the same galaxy
// before the observation (the view beside it, as the galaxy is, of no
// instrument), over one field: that of the observation, or a zoom of it
// about its centre. The galaxy is a test galaxy (see testGalaxy) whose
// knobs are under its view, or any galaxy of the project (the Galaxy
// picker); a drag in its view moves it, in both views (only for the
// picture, not saved; as the source of the lens screen). Its bar shows the light, the velocity or the dispersion, the
// PSF to scale and the pixels or not, and zooms the field. A click on a
// pixel of a cube shows its spectrum under the main view, with the LSF to
// scale. Under the views, the checks: what the library notes as it makes
// the observation, the width of its PSF in pixels and of its LSF in
// channels, its mock noise, and whether the galaxy is in its field and
// its channels. The picture keeps its last state while the next is made,
// or while the observation cannot be drawn (a caption says why), so that
// nothing moves or flashes.
import { useDebouncedValue } from '@mantine/hooks';
import { useEffect, useMemo, useState } from 'react';
import { useGalaxyRender, useMaps, useMoments, useObserved, useOptionalPiece, usePiece, useProject, useSpectrum, type ObservationSampling, type Observed } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import type { Options } from '../../components/SchemaForm';
import { DropMenu } from '../../components/ui/overlays';
import { useLast } from '../../components/useLast';
import { snap } from '../../components/values';
import { problemText } from '../galaxy/Model';
import { roundText, zoomed } from '../lens/Picture';
import { isObject, short, type Ellipse } from './model';
import { Chips, Knob, MapBar, mapRange, psfPixels, SkyPlane, SpectrumStrip, type Chip, type SkyMap } from './Planes';
import { knobSpans, startKnobs, testGalaxy, type TestKnobs } from './testGalaxy';

type Quantity = 'intensity' | 'velocity' | 'dispersion';

const QUANTITIES: [Quantity, string][] = [
  ['intensity', 'Intensity'],
  ['velocity', 'Velocity'],
  ['dispersion', 'Dispersion'],
];

// The kinds of observations the picture draws (on a grid of pixels), and
// the names of the maps of each quantity: of the moments of a cube (see
// the server's /views/moments), of the items of moment maps, of an image
const DRAWN = ['pixel_spectra', 'pixel_moments', 'pixel_brightness'];
const MOMENT_MAPS: Record<Quantity, string> = { intensity: 'flux', velocity: 'velocity', dispersion: 'dispersion' };
const ITEM_MAPS: Record<Quantity, string> = { intensity: 'moment0', velocity: 'moment1', dispersion: 'moment2' };

// The pixels across the view of the galaxy before the observation
const GALAXY_PIXELS = 200;

// The light below this fraction of the brightest pixel has no velocity
// nor dispersion shown (the light of a model has no edge: e.g. that of an
// exponential disk reaches beyond its last ring)
const FAINT = 0.01;

// (as the edits and the knobs settle, ms; a render that takes longer
// than SLOW_MS is said to be under way)
const SETTLE_MS = 150;
const SLOW_MS = 1000;

export function ObservationPicture({
  project,
  options,
  canDraw,
  kindTitle,
  sampling,
  psf,
  lsf,
  lsfFwhm,
  noise,
  galaxies,
}: {
  project: string;
  // The observation as edited, and whether it can be drawn (none of its
  // problems is an error; null: not known yet, as its problems come), and
  // the title of its kind
  options: Options;
  canDraw: boolean | null;
  kindTitle: string;
  sampling: ObservationSampling | undefined;
  // Its PSF, and its LSF on channels of its own width (km/s), as the
  // server draws them, and the FWHM of its LSF (km/s)
  psf: Ellipse | null;
  lsf: { values: number[]; step: number } | null;
  lsfFwhm: number | null;
  // The noise of its mock data in a few words (none: none)
  noise: string | null;
  // The galaxies of the project
  galaxies: string[];
}) {
  const type = String(options.type ?? '');
  const drawn = DRAWN.includes(type);
  // (the last field and channels known: kept while the observation has a
  // problem, so that the picture keeps its field)
  const field = useLast(sampling?.field ?? undefined) ?? null;
  const channels = useLast(sampling?.channels ?? undefined) ?? null;
  const fieldWidth = field ? Math.max(field.size[0], field.size[1]) : null;
  const [quantity, setQuantity] = useState<Quantity>('velocity');
  const [showPsf, setShowPsf] = useState(true);
  const [showPixels, setShowPixels] = useState(true);
  const [picked, setPicked] = useState<string | null>(null);
  const [knobs, setKnobs] = useState<TestKnobs | null>(null);
  const [zoom, setZoom] = useState<number | null>(null);
  const [pixel, setPixel] = useState<[number, number] | null>(null);
  // (how far the galaxy is dragged from where it is: from the centre of
  // the field, the test galaxy; from the centre its parameters give, a
  // galaxy of the project; x, y, arcsec)
  const [shift, setShift] = useState<[number, number] | null>(null);
  const [dx, dy] = shift ?? [0, 0];
  // (the knobs as the picture opens, of the field, once it is known)
  useEffect(() => {
    if (knobs === null && fieldWidth !== null) setKnobs(startKnobs(fieldWidth));
  }, [knobs, fieldWidth]);
  const width = zoom ?? fieldWidth ?? 100;
  // (a quantity an image has not: its light)
  const shownQuantity: Quantity = type === 'pixel_brightness' ? 'intensity' : quantity;

  // The galaxy: the test galaxy, at the centre of the field, its systemic
  // velocity at the centre of the channels; or a galaxy of the project,
  // and its parameters as its params block gives them
  const info = useProject(project);
  const paramsNames = info.data?.pieces.params;
  const piece = usePiece(picked ? joinPath(project, `galaxies/${picked}.yaml`) : null);
  const params = useOptionalPiece(joinPath(project, `params/${picked ?? ''}.yaml`), picked !== null && paramsNames ? paramsNames.includes(picked) : undefined);
  const galaxy = useMemo(() => {
    if (picked === null) {
      if (!field || !knobs) return null;
      const vsys = channels ? (channels.first + channels.last) / 2 : 0;
      return { ...testGalaxy(knobs, [field.centre[0] + dx, field.centre[1] + dy], vsys, type !== 'pixel_brightness'), label: 'test' };
    }
    if (!isObject(piece.data) || params.isPending) return null;
    const properties = isObject(params.data) && isObject(params.data.properties) ? params.data.properties : {};
    return { options: piece.data, properties, label: picked };
  }, [picked, field, knobs, channels, type, piece.data, params.data, params.isPending, dx, dy]);

  // What the views are of, as they settle: the galaxy, the observation,
  // and how far a galaxy of the project is dragged (the server moves it)
  const [askedKey] = useDebouncedValue(JSON.stringify({ galaxy, options, canDraw, moved: picked === null ? null : shift }), SETTLE_MS);
  const asked = useMemo(() => JSON.parse(askedKey) as { galaxy: typeof galaxy; options: Options; canDraw: boolean | null; moved: [number, number] | null }, [askedKey]);
  const observed = useObserved(project, asked.galaxy && asked.canDraw === true && DRAWN.includes(String(asked.options.type)) ? asked.galaxy.options : undefined, asked.galaxy?.properties ?? {}, asked.options, null, asked.galaxy?.label, null, asked.moved);
  // (whether it could be drawn, as last known)
  const couldDraw = useLast(canDraw ?? undefined) ?? true;
  // (the last galaxy observed as it is, with what it is, for its checks,
  // and how far it was moved)
  const current = observed.data && !observed.isPlaceholderData ? { ...observed.data, of: asked.galaxy?.label === 'test' ? 'the test galaxy' : (asked.galaxy?.label ?? ''), label: asked.galaxy?.label, moved: asked.moved } : undefined;
  const lastObserved = useLast(current);
  const maps = useObservedMaps(observed.data, shownQuantity);
  const mainMap = useLast(maps.map);
  const cube = maps.cube;

  // The galaxy before the observation, over the same field as the main
  // view: the test galaxy, moved itself; a galaxy of the project as its
  // parameters place it, the field moved the other way
  const back = picked === null ? [0, 0] : (asked.moved ?? [0, 0]);
  const renderField = field ? { width, centre: [field.centre[0] - back[0], field.centre[1] - back[1]] as [number, number] } : null;
  // (where the galaxy is without the drag, and with it, which a drag in
  // its view sets: the test galaxy at the centre of the field; a galaxy
  // of the project where its parameters place it, once known)
  const placed = lastObserved?.label === picked && lastObserved?.placement?.centre ? lastObserved.placement.centre : null;
  const home: [number, number] | null = picked === null ? (field ? [field.centre[0], field.centre[1]] : null) : placed ? [placed[0] - (lastObserved?.moved?.[0] ?? 0), placed[1] - (lastObserved?.moved?.[1] ?? 0)] : null;
  const galaxyCentre: [number, number] | null = home ? [home[0] + dx, home[1] + dy] : null;
  const drag = (at: [number, number]) => home && setShift([at[0] - home[0], at[1] - home[1]]);
  const render = useGalaxyRender(project, asked.galaxy && renderField ? asked.galaxy.options : undefined, asked.galaxy?.properties ?? {}, 'sky', GALAXY_PIXELS, renderField);
  const galaxyMap = useLast(useMemo(() => galaxyMapOf(render.data, shownQuantity), [render.data, shownQuantity]));
  // (the velocities and dispersions of both views in the colours of the
  // main one, to compare them, of a galaxy in the field)
  const inField = lastObserved?.placement?.in_field === true;
  const sharedRange = useMemo(() => (drawn && inField ? mapRange(mainMap) : null), [drawn, inField, mainMap]);

  // The spectrum of the pixel picked (by default, the middle one)
  const nx = cube?.shape[0] ?? 0;
  const ny = cube?.shape[1] ?? 0;
  const at: [number, number] | null = cube ? (pixel && pixel[0] < nx && pixel[1] < ny ? pixel : [Math.floor((nx - 1) / 2), Math.floor((ny - 1) / 2)]) : null;
  const spectrum = useSpectrum(cube?.path ?? '', 0, at, null, null, cube !== undefined && at !== null);
  const strip = useLast(spectrum.data && spectrum.data.spectral ? { values: spectrum.data.values, velocities: spectrum.data.spectral, pixel: at } : undefined);
  const spectral = type === 'pixel_spectra';

  // Why the main view is not the galaxy through the observation as it is:
  // of a kind not drawn, a problem, or a render that takes long (not one
  // of a moment, which would flash)
  const slow = useLater(observed.isFetching || maps.pending, SLOW_MS);
  const note: { text: string; bad: boolean } | null = !drawn
    ? { text: `Not drawn yet: ${kindTitle.toLowerCase()}`, bad: false }
    : !couldDraw
      ? { text: mainMap ? 'The last picture: the observation has a problem (see the head)' : 'The observation has a problem (see the head)', bad: true }
      : observed.isError
        ? { text: problemText(observed.error.message), bad: true }
        : slow
          ? { text: mainMap ? 'The last picture: making the next…' : 'Making the picture…', bad: false }
          : null;

  // The checks
  const step = lastObserved?.arrays[0]?.step[0] ?? null;
  const chips: Chip[] = [];
  if (type !== 'profile') {
    chips.push(psf ? { kind: 'note', text: step ? `PSF ${psfPixels(psf, step)} pixels across` : `PSF ${short(psf.major)}″`, title: 'The FWHM of the PSF in pixels' } : { kind: 'note', text: 'no PSF' });
  }
  if (channels) chips.push(lsfFwhm !== null ? { kind: 'note', text: `LSF ${short(lsfFwhm / Math.abs(channels.step), 2)} channels`, title: 'The FWHM of the LSF in channels' } : { kind: 'note', text: 'no LSF' });
  const placement = drawn ? lastObserved?.placement : null;
  if (placement && lastObserved) chips.push(placementChip(placement, lastObserved.of, channels));
  chips.push(noise ? { kind: 'note', text: `mock noise: ${noise}` } : { kind: 'note', text: 'no mock noise' });
  for (const text of lastObserved?.notes ?? []) chips.push({ kind: 'warn', text: problemText(text) });

  // (a galaxy picked, where it is, undragged)
  const pick = (name: string | null) => {
    setPicked(name);
    setShift(null);
  };
  const spans = knobSpans(fieldWidth);
  const setKnob = (key: keyof TestKnobs, t: number) => {
    const [lo, hi] = spans[key];
    setKnobs((k) => (k ? { ...k, [key]: snap(lo + t * (hi - lo), [lo, hi]) } : k));
  };
  const knobAt = (key: keyof TestKnobs) => (knobs ? (knobs[key] - spans[key][0]) / (spans[key][1] - spans[key][0]) : 0);
  const unitOf = (map: SkyMap | undefined) => map?.unit ?? '';
  return (
    <div className="ob-view lp-view ob-picture">
      <div className="lp-tools">
        <span className="lp-grp lp-grp-even" role="tablist" aria-label="What the views show">
          {QUANTITIES.map(([key, title]) => (
            <button key={key} type="button" role="tab" aria-selected={shownQuantity === key} disabled={type === 'pixel_brightness' && key !== 'intensity'} onClick={() => setQuantity(key)}>
              {title}
            </button>
          ))}
        </span>
        <span className="lp-grp lp-grp-even">
          <button type="button" aria-pressed={showPsf} title="The PSF to scale" onClick={() => setShowPsf(!showPsf)}>
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <ellipse className="ob-ic-psf" cx="8" cy="8" rx="4" ry="5.5" />
            </svg>
            PSF
          </button>
          <button type="button" aria-pressed={showPixels} title="The pixels as they are, and their lines" onClick={() => setShowPixels(!showPixels)}>
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M2.5 2.5h11v11h-11zM6.2 2.5v11M9.8 2.5v11M2.5 6.2h11M2.5 9.8h11" />
            </svg>
            Pixels
          </button>
        </span>
        <span className="ob-pick">
          Galaxy
          <DropMenu
            label="The galaxy of the picture"
            width={280}
            entries={[
              { label: 'Test galaxy', description: "A disk at the field's centre, of a few knobs", current: picked === null, onClick: () => pick(null) },
              ...galaxies.map((name) => ({ label: name, description: 'A galaxy of the project', current: picked === name, onClick: () => pick(name) })),
            ]}
            target={
              <button type="button" className="sh-num gm-choice ob-pick-box">
                <span>{picked ?? 'test galaxy'}</span>
              </button>
            }
          />
        </span>
        <span className="lp-field">
          Field
          <span className="lp-grp">
            <button type="button" className="lp-grp-icon" title="Zoom out" aria-label="Zoom out" onClick={() => setZoom(zoomed(width, true))}>
              <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 8h8" /></svg>
            </button>
            <span className="lp-grp-val">{roundText(width)}″</span>
            <button type="button" className="lp-grp-icon" title="Zoom in" aria-label="Zoom in" onClick={() => setZoom(zoomed(width, false))}>
              <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 8h8M8 4v8" /></svg>
            </button>
            <button type="button" className="lp-grp-icon" title="The field of the observation" aria-label="Fit the field of the observation" onClick={() => setZoom(null)}>
              <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10" /></svg>
            </button>
          </span>
        </span>
      </div>
      <div className="ob-planes" data-spectral={spectral || undefined}>
        <div className="ob-main">
          <figure className="lp-plane-box ob-main-box">
            <figcaption>
              As observed
              {note ? (
                <span className="lp-key" data-bad={note.bad || undefined} title={note.text}>
                  {note.text}
                </span>
              ) : (
                <span className="lp-key">
                  <span className="ob-key" data-off={!showPsf || !psf || undefined}>
                    <i className="ob-key-psf" /> PSF
                  </span>
                  <span className="ob-key" data-off={!showPixels || undefined}>
                    <i className="ob-key-pixels" /> pixels
                  </span>
                </span>
              )}
            </figcaption>
            <SkyPlane
              map={drawn ? mainMap : undefined}
              width={width}
              psf={showPsf ? psf : null}
              pixels={showPixels}
              picked={spectral ? strip?.pixel ?? at : null}
              onPick={spectral ? setPixel : undefined}
              readout={(p, v) => `pixel ${p[0]}, ${p[1]} · ${significantText(v)} ${unitOf(mainMap)}`}
            />
            <MapBar map={drawn ? mainMap : undefined} />
          </figure>
          {spectral && (
            <figure className="lp-plane-box ob-spectrum-box">
              <figcaption>
                {strip?.pixel ? `Spectrum of pixel ${strip.pixel[0]}, ${strip.pixel[1]}` : 'Spectrum of a pixel'}
                <span className="lp-key">
                  <span className="ob-key">
                    <i className="ob-key-spectrum" /> as observed
                  </span>
                  <span className="ob-key" data-off={!lsf || undefined}>
                    <i className="ob-key-lsf" /> LSF
                  </span>
                </span>
              </figcaption>
              <div className="ob-strip-box">
                <SpectrumStrip spectrum={strip} lsf={lsf} />
              </div>
              <div className="ob-bar ob-axis-line">
                <span>{channels ? short(Math.min(channels.first, channels.last), 4) : ' '}</span>
                <i />
                <span>{channels ? `${short(Math.max(channels.first, channels.last), 4)} km/s · ${channels.size} channels of ${short(Math.abs(channels.step))}` : ''}</span>
              </div>
            </figure>
          )}
        </div>
        <figure className="lp-plane-box ob-galaxy-box">
          <figcaption>
            The galaxy
            <span className="lp-key">before the observation</span>
          </figcaption>
          <SkyPlane
            map={galaxyMap}
            range={shownQuantity === 'intensity' ? null : sharedRange}
            width={width}
            psf={null}
            pixels={false}
            origin={field ? [field.centre[0], field.centre[1]] : undefined}
            centre={galaxyCentre}
            onCentre={home ? drag : undefined}
          />
          {/* (the knobs of the test galaxy; of a galaxy of the project, a
              note in their place, which keeps their room) */}
          <div className="ob-knobs" data-off={picked !== null || undefined}>
            <Knob label="Inclination" at={knobAt('inclination')} onAt={(t) => setKnob('inclination', t)} value={knobs ? `${short(knobs.inclination)}°` : ''} />
            <Knob label="Position angle" at={knobAt('angle')} onAt={(t) => setKnob('angle', t)} value={knobs ? `${short(knobs.angle)}°` : ''} />
            <Knob label="Size" at={knobAt('size')} onAt={(t) => setKnob('size', t)} value={knobs ? `${short(knobs.size)}″` : ''} />
            <Knob label="Rotation" at={knobAt('rotation')} onAt={(t) => setKnob('rotation', t)} value={knobs ? `${short(knobs.rotation)} km/s` : ''} />
            {picked !== null && (
              <p className="sh-note ob-knobs-note">
                {picked}: at the values of its parameters{paramsNames?.includes(picked) ? ` (params/${picked})` : ', their defaults'}{shift ? ', moved' : ''}.
              </p>
            )}
          </div>
        </figure>
        <div className="ob-checks">
          <Chips chips={chips} />
        </div>
      </div>
    </div>
  );
}

// The maps of a galaxy as an observation sees it, of a quantity (see
// SkyMap): of the moments of its cube, or of its moment maps, or of its
// image; and its cube, if it has one
function useObservedMaps(observed: Observed | undefined, quantity: Quantity): { map: SkyMap | undefined; cube: Observed['arrays'][number] | undefined; pending: boolean } {
  const models = observed?.arrays.filter((a) => a.kind === 'model') ?? [];
  const type = observed?.type;
  const cube = type === 'pixel_spectra' ? models.find((a) => a.item === 'spectra') : undefined;
  const moments = useMoments(cube?.path ?? '', 0, 0, cube !== undefined);
  const item = type === 'pixel_moments' ? models.find((a) => a.item === ITEM_MAPS[quantity]) : type === 'pixel_brightness' && quantity === 'intensity' ? models.find((a) => a.item === 'brightness') : undefined;
  const light = type === 'pixel_moments' ? models.find((a) => a.item === 'moment0') : undefined;
  const [itemValues, lightValues] = useMaps([...(item ? [{ path: item.path, hdu: 0 }] : []), ...(light && light !== item ? [{ path: light.path, hdu: 0 }] : [])]);
  const pending = moments.isFetching;
  return useMemo(() => {
    const source = cube ?? item;
    if (!source) return { map: undefined, cube, pending };
    const [nx, ny] = source.shape;
    let values: Float32Array | undefined;
    let intensity: Float32Array | undefined;
    if (cube) {
      const named = (name: string) => moments.data?.find((a) => a.info.name === name)?.values;
      values = named(MOMENT_MAPS[quantity]);
      intensity = named('flux');
    } else {
      values = itemValues;
      intensity = light === item ? itemValues : lightValues;
    }
    if (!values) return { map: undefined, cube, pending };
    const unit = quantity === 'intensity' ? (cube ? `${cube.unit} km/s`.trim() : (item?.unit ?? '')) : 'km/s';
    return { map: { values: quantity === 'intensity' ? values : faintOut(values, intensity), nx, ny, step: source.step[0], centre: [(nx - 1) / 2, (ny - 1) / 2], diverging: quantity === 'velocity', unit }, cube, pending };
  }, [cube, item, light, moments.data, itemValues, lightValues, quantity, pending]);
}

// The values of a map where the light is not faint (see FAINT); NaN
// elsewhere
function faintOut(values: Float32Array, light: Float32Array | undefined): Float32Array {
  if (!light) return values;
  let peak = 0;
  for (const v of light) if (Number.isFinite(v) && v > peak) peak = v;
  return values.map((v, i) => (Number.isFinite(light[i]) && light[i] > FAINT * peak ? v : NaN));
}

// The map of a quantity of a render of a galaxy (see SkyMap), over the
// field it was rendered on
function galaxyMapOf(render: ReturnType<typeof useGalaxyRender>['data'], quantity: Quantity): SkyMap | undefined {
  if (!render) return undefined;
  const name = { intensity: 'light', velocity: 'velocity', dispersion: 'dispersion' }[quantity];
  const values = render.maps[name];
  if (!values) return undefined;
  const n = render.size;
  const shown = quantity === 'intensity' ? values : faintOut(values, render.maps.light);
  return { values: shown, nx: n, ny: n, step: render.step, centre: [(n - 1) / 2, (n - 1) / 2], diverging: quantity === 'velocity', unit: quantity === 'intensity' ? '' : 'km/s' };
}

// Whether the galaxy is in the field and the channels, a check
function placementChip(placement: NonNullable<Observed['placement']>, label: string, channels: ObservationSampling['channels']): Chip {
  const { in_field: inField, in_channels: inChannels, velocity } = placement;
  const range = channels ? `${short(Math.min(channels.first, channels.last), 4)} … ${short(Math.max(channels.first, channels.last), 4)}` : '';
  if (inField === false) return { kind: 'warn', text: `${label}: its centre is outside the field` };
  if (inChannels === false && velocity !== null) return { kind: 'warn', text: `${label}: vsys ${short(velocity, 4)} km/s, outside ${range}` };
  if (inField && inChannels) return { kind: 'ok', text: 'the galaxy is inside the field and the channels' };
  if (inField) return { kind: 'ok', text: 'the galaxy is inside the field' };
  return { kind: 'note', text: 'where the galaxy is: unknown' };
}

// Whether a state has held for a time (ms): e.g. a render under way for
// long enough to say so
function useLater(on: boolean, ms: number): boolean {
  const [late, setLate] = useState(false);
  useEffect(() => {
    if (!on) {
      setLate(false);
      return;
    }
    const timer = setTimeout(() => setLate(true), ms);
    return () => clearTimeout(timer);
  }, [on, ms]);
  return on && late;
}

function significantText(value: number): string {
  return String(Number(value.toPrecision(4)));
}
