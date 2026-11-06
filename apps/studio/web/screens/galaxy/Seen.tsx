// The galaxy as an observation of the project sees it ("Seen by", in the
// viewer of the galaxy; see the server's /views/observed): its model on the
// grid of the observation, through its instrument and its line of sight,
// at the values of its parameters as they are set (as the edits settle,
// not as saved); in the viewer of its kind of data: a cube (its channels,
// its spectra, and the moments measured from it, as from data), or maps
// (an image, or moment maps, measured as the observation measures them).
// What the library notes as it sets them up shows over it, and why the
// observation cannot see the galaxy, if it cannot (e.g. a thick disk
// along a flat line of sight). Under it, the curves of the galaxy.
import { useDebouncedValue } from '@mantine/hooks';
import { useMemo, useState } from 'react';
import { useGalaxyRender, useObserved, usePiece } from '../../api/queries';
import { CubeViewer } from '../../components/CubeViewer';
import { joinPath } from '../../components/FilePicker';
import { MapsViewer, type MapItem } from '../../components/MapsViewer';
import type { Region } from '../../components/MapView';
import { frameOf } from '../../components/sky';
import { Curves, type CurvesFocus } from './Curves';
import type { Options } from './options';

// (the kinds of observables shown, on a grid of pixels: a cube, an image,
// moment maps)
const SHOWN = ['pixel_spectra', 'pixel_brightness', 'pixel_moments'];

// (as the edits settle, ms)
const SETTLE_MS = 300;

// (the height of the curves under it, px)
const CURVES_HEIGHT = 170;

export function SeenView({
  project,
  galaxy,
  properties,
  observation,
  curves,
  components,
}: {
  project: string;
  galaxy: Options;
  properties: Record<string, unknown>;
  observation: string;
  curves: CurvesFocus;
  components: { name: string; colour: string }[];
}) {
  // (a key of text, so that the same galaxy, settled, is not asked again)
  const [settledKey] = useDebouncedValue(JSON.stringify([galaxy, properties]), SETTLE_MS);
  const [settledGalaxy, settledProperties] = useMemo(() => JSON.parse(settledKey) as [Options, Record<string, unknown>], [settledKey]);
  const seen = usePiece(joinPath(project, `observations/${observation}.yaml`));
  const observed = useObserved(project, settledGalaxy, settledProperties, observation, seen.data);
  // (the curves: of the galaxy as it is, whatever sees it)
  const render = useGalaxyRender(project, settledGalaxy, settledProperties, 'sky', 64);
  return (
    <div className="gx-seen">
      <div className="gx-seen-view">
        {observed.data && observed.data.notes.length > 0 && (
          <ul className="obs-notes" aria-label="Notes of the observation">
            {observed.data.notes.map((note, i) => (
              <li key={i}>{note}</li>
            ))}
          </ul>
        )}
        {observed.error && (
          <div className="obs-problem">
            <p>
              <b>{observation}</b> cannot see this galaxy:
            </p>
            <p className="error">{observed.error.message}</p>
          </div>
        )}
        {!observed.error && !observed.data && <div className="gx-map-wait gx-seen-wait" />}
        {!observed.error && observed.data && <Shown key={observation} observed={observed.data} />}
      </div>
      {render.data && <Curves render={render.data} focus={curves} components={components} height={CURVES_HEIGHT} />}
    </div>
  );
}

type Observed = NonNullable<ReturnType<typeof useObserved>['data']>;

// The model as observed, in the viewer of its kind
function Shown({ observed }: { observed: Observed }) {
  const [selection, setSelection] = useState<Region | null>(null);
  const [band, setBand] = useState<[number, number] | null>(null);
  const models = observed.arrays.filter((array) => array.kind === 'model');
  if (!SHOWN.includes(observed.observable) || models.length === 0) {
    return <p className="note">The Studio does not show this kind of observation ({observed.observable.replaceAll('_', ' ')}) yet.</p>;
  }
  if (observed.observable === 'pixel_spectra') {
    const [model] = models;
    return (
      <CubeViewer
        key={model.path}
        layers={[{ key: 'data', title: 'Model', path: model.path, hdu: 0, unit: model.unit || null }]}
        shape={model.shape}
        selection={selection}
        band={band !== null && band[1] < model.shape[2] ? band : null}
        onSelection={setSelection}
        onBand={setBand}
        noiseless
        frame={frameOf(model.step, model.rpix)}
      />
    );
  }
  const maps: MapItem[] = models.map((model) => ({
    key: model.item,
    title: itemTitle(model.item),
    colours: model.item === 'moment1' ? 'median' : /^(moment[3-9]|h[34])$/.test(model.item) ? 'zero' : 'sequential',
    layers: [{ key: 'data', title: 'Model', path: model.path, hdu: 0, unit: model.unit || null }],
  }));
  const [nx, ny] = models[0].shape;
  return (
    <MapsViewer
      key={maps.map((map) => map.layers[0].path).join()}
      maps={maps}
      shape={[nx, ny]}
      selection={selection}
      onSelection={setSelection}
      frame={frameOf(models[0].step, models[0].rpix)}
    />
  );
}

// (what an item of an observation is, to people)
function itemTitle(item: string): string {
  const named: Record<string, string> = { brightness: 'Brightness', moment0: 'Intensity', moment1: 'Velocity', moment2: 'Dispersion' };
  return named[item] ?? (/^moment\d$/.test(item) ? `Moment ${item.slice(6)}` : item);
}
