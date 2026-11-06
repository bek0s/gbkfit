// Observe: a galaxy of the project as an observation of the project sees
// it (see the server's /views/observed): its model on the grid of the
// observation, through its instrument and its line of sight, at the
// values of its parameters (its block of parameters, as saved; the
// defaults for the others), in the viewer of its kind of data: a cube
// (its channels, its spectra, and the moments measured from it, as from
// data), or maps (an image, or moment maps, measured as the observation
// measures them). The galaxy and the observation are chosen in the head
// (remembered for the project). What the library notes as it sets them up
// shows over the viewer, and why the observation cannot see the galaxy,
// if it cannot (e.g. a thick disk along a flat line of sight).
import { useState } from 'react';
import { useObserved, usePiece, useProject } from '../../api/queries';
import { CubeViewer } from '../../components/CubeViewer';
import { joinPath } from '../../components/FilePicker';
import type { MapItem } from '../../components/MapsViewer';
import { MapsViewer } from '../../components/MapsViewer';
import type { Region } from '../../components/MapView';
import { Button } from '../../components/ui/basics';
import { Choice } from '../../components/ui/inputs';
import { recall, remember, useRemembered } from '../../shell/remember';
import { useStudio } from '../../studio';

// What opens Observe with a galaxy, or an observation, chosen (e.g. from
// their screens), the other as it was
export function useObserve() {
  const { project, go } = useStudio();
  return (choice: { galaxy?: string; observation?: string }) => {
    if (project === null) return;
    const key = `observe:${project}`;
    remember(key, { ...recall(key, { galaxy: null, observation: null }), ...choice });
    go({ kind: 'observe' });
  };
}

// (the kinds of observables shown, on a grid of pixels: a cube, an image,
// moment maps)
const SHOWN = ['pixel_spectra', 'pixel_brightness', 'pixel_moments'];

export function ObserveScreen() {
  const { project, go } = useStudio();
  const info = useProject(project);
  const galaxies = info.data?.pieces.galaxies ?? [];
  const observations = info.data?.pieces.observations ?? [];
  const [chosen, setChosen] = useRemembered<{ galaxy: string | null; observation: string | null }>(`observe:${project}`, { galaxy: null, observation: null });
  // (one no longer there, the first)
  const galaxy = chosen.galaxy !== null && galaxies.includes(chosen.galaxy) ? chosen.galaxy : (galaxies[0] ?? null);
  const observation = chosen.observation !== null && observations.includes(chosen.observation) ? chosen.observation : (observations[0] ?? null);
  const at = (folder: string, name: string | null) => (project !== null && name !== null ? joinPath(project, `${folder}/${name}.yaml`) : null);
  const galaxyPiece = usePiece(at('galaxies', galaxy));
  // (its parameters: its block of the same name, if there is one)
  const paramsPiece = usePiece(at('params', galaxy));
  const seen = usePiece(at('observations', observation));
  const params = paramsPiece.data as { properties?: Record<string, unknown> } | undefined;
  const properties = paramsPiece.isError ? {} : (params?.properties ?? {});
  const ready = galaxyPiece.data !== undefined && (paramsPiece.isError || paramsPiece.data !== undefined);
  const observed = useObserved(project ?? '', ready ? (galaxyPiece.data as Record<string, unknown>) : undefined, properties, observation, seen.data);
  if (project === null) return null;
  return (
    <section className="screen ds-screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">Observe</div>
          <h1>
            {galaxy ?? 'No galaxy'} <span className="obs-through">through</span> {observation ?? 'no observation'}
          </h1>
        </div>
        <div className="head-actions obs-choices">
          <label className="obs-choice">
            <span>Galaxy</span>
            <Choice items={galaxies.map((name) => ({ value: name, label: name }))} value={galaxy ?? ''} onChange={(name) => setChosen({ galaxy: name, observation })} label="The galaxy observed" />
          </label>
          <label className="obs-choice">
            <span>Observation</span>
            <Choice items={observations.map((name) => ({ value: name, label: name }))} value={observation ?? ''} onChange={(name) => setChosen({ galaxy, observation: name })} label="The observation that sees it" />
          </label>
        </div>
      </div>
      {galaxy === null || observation === null ? (
        <div className="obs-empty">
          <p className="ds-empty-title">{galaxy === null ? 'No galaxy to observe yet' : 'No observation to see it through yet'}</p>
          <p>{galaxy === null ? 'Build a galaxy, in the tree under Model, to see it as an observation sees it.' : 'Make an observation, in the tree under Model: its grid, instrument and line of sight.'}</p>
          <Button kind="primary" onClick={() => go({ kind: 'home' })}>
            Go to the project
          </Button>
        </div>
      ) : (
        <div className="ds-view obs-view">
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
                <b>{observation}</b> cannot see <b>{galaxy}</b>:
              </p>
              <p className="error">{observed.error.message}</p>
              <div className="obs-problem-actions">
                <Button onClick={() => go({ kind: 'block', folder: 'galaxies', name: galaxy })}>Open the galaxy</Button>
                <Button onClick={() => go({ kind: 'block', folder: 'observations', name: observation })}>Open the observation</Button>
              </div>
            </div>
          )}
          {!observed.error && !observed.data && <div className="obs-wait">Observing {galaxy}…</div>}
          {!observed.error && observed.data && <Shown key={`${galaxy}:${observation}`} observed={observed.data} />}
        </div>
      )}
    </section>
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
  return <MapsViewer key={maps.map((map) => map.layers[0].path).join()} maps={maps} shape={[nx, ny]} selection={selection} onSelection={setSelection} />;
}

// (what an item of an observation is, to people)
function itemTitle(item: string): string {
  const named: Record<string, string> = { brightness: 'Brightness', moment0: 'Intensity', moment1: 'Velocity', moment2: 'Dispersion' };
  return named[item] ?? (/^moment\d$/.test(item) ? `Moment ${item.slice(6)}` : item);
}
