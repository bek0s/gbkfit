// The next run of a fit, before it runs: the checks of its recipe (its
// problems as the library finds them, and what to know: its parameters,
// those free without a range, an observation without data, no fitter),
// and the model at its starts against the data, through each of its
// observations (see the server's /views/setup-observed): the data, the
// model and the residuals side by side, in the same colours, so that a
// mistake costs seconds, not a run.
import { useMemo, useState } from 'react';
import { useSetupObserved } from '../../api/queries';
import { MapPanel, robustRange } from '../../components/MapPanel';
import { SHOWN, useObservedMap, type Shown } from '../../components/observedMaps';
import { frameOf } from '../../components/sky';
import { Segments } from '../../components/ui/inputs';
import { isObject } from '../galaxy/options';
import type { SetupParam } from './Recipe';
import { itemsOf, type Options } from './recipe';

interface Problem {
  severity: string;
  path: (string | number)[];
  message: string;
}

export function StartView({ path, setup, params, problems, pending, failed }: { path: string; setup: Options | undefined; params: SetupParam[]; problems: Problem[]; pending: boolean; failed: string | null }) {
  const errors = problems.filter((p) => p.severity === 'error');
  const ready = setup !== undefined && !pending && failed === null && errors.length === 0 && params.length > 0;
  return (
    <div className="fx-picture">
      <div className="fx-scroll">
        <div className="fx-view-grid">
          <Checks setup={setup} params={params} problems={problems} pending={pending} failed={failed} />
          {ready ? <AtTheStart path={path} setup={setup} /> : <div className="fx-card fx-wait-card">The model at its starts shows once the recipe has no error.</div>}
        </div>
      </div>
    </div>
  );
}

function Checks({ setup, params, problems, pending, failed }: { setup: Options | undefined; params: SetupParam[]; problems: Problem[]; pending: boolean; failed: string | null }) {
  const lines: { sign: 'ok' | 'warn' | 'error' | 'info'; text: string }[] = [];
  if (failed) lines.push({ sign: 'error', text: failed });
  for (const p of problems) lines.push({ sign: p.severity === 'error' ? 'error' : 'warn', text: `${plain(p.message)}${p.path.length ? ` (${p.path.join(' › ')})` : ''}` });
  if (setup && !pending && !failed && !problems.some((p) => p.severity === 'error') && params.length > 0) {
    const free = params.filter((p) => isObject(p.properties[p.name]));
    const tied = params.filter((p) => typeof p.properties[p.name] === 'string').length;
    lines.unshift({ sign: 'ok', text: `The library sets the fit up: ${free.length} free parameters, ${params.length - free.length - tied} fixed${tied ? `, ${tied} tied` : ''}.` });
    const loose = free.filter((p) => {
      const property = p.properties[p.name] as Options;
      return typeof property.min !== 'number' || typeof property.max !== 'number';
    });
    if (loose.length) lines.push({ sign: 'info', text: `Free without a range: ${loose.map((p) => p.name).join(', ')}. Samplers need one (least squares does not).` });
  }
  if (setup && setup.fitter === undefined) lines.push({ sign: 'warn', text: 'No fitter yet: choose one to run the fit.' });
  if (setup && itemsOf(setup, 'galaxies').length === 0) lines.push({ sign: 'warn', text: 'No galaxy yet: add one in the recipe.' });
  if (setup && itemsOf(setup, 'observations').length === 0) lines.push({ sign: 'warn', text: 'No observation yet: add one in the recipe.' });
  const signs = { ok: '✓', warn: '!', error: '✕', info: '·' };
  return (
    <div className="fx-card fx-checks">
      <div className="fx-card-head">
        <h2>Before it runs</h2>
        <span className="fx-faint fx-small">{pending ? 'checking…' : 'checked as the recipe changes'}</span>
      </div>
      <ul>
        {lines.map((line, i) => (
          <li key={i} data-sign={line.sign}>
            <span className="fx-sign">{signs[line.sign]}</span>
            <span>{line.text}</span>
          </li>
        ))}
        {lines.length === 0 && <li data-sign="info">{pending ? 'Checking the recipe…' : 'Nothing to check yet.'}</li>}
      </ul>
    </div>
  );
}

// The model at its starts through one observation of the fit: the data,
// the model and the residuals, in the colours of the data (of the model,
// without data), the residuals about zero
function AtTheStart({ path, setup }: { path: string; setup: Options }) {
  const observations = itemsOf(setup, 'observations');
  const [chosen, setChosen] = useState(0);
  const [show, setShow] = useState<Shown>('intensity');
  const index = Math.min(chosen, observations.length - 1);
  const observed = useSetupObserved(path, setup, index);
  const model = useObservedMap(observed.data, 'model', show);
  const measured = useObservedMap(observed.data, 'data', show);
  const residual = useMemo(() => {
    if (!model.values || !measured.values || model.values.length !== measured.values.length) return undefined;
    return measured.values.map((d, i) => d - model.values![i]);
  }, [model.values, measured.values]);
  const range = measured.values ? robustRange(measured.values, measured.colours) : model.values ? robustRange(model.values, model.colours) : ([0, 1] as [number, number]);
  const half = residual ? Math.max(...robustRange(residual, 'zero').map(Math.abs)) : 1;
  const item = model.item;
  const [nx, ny] = item?.shape ?? [1, 1];
  const frame = item ? frameOf(item.step, item.rpix) : null;
  const labels = observations.map((o, i) => o.piece ?? o.name ?? `observation ${i + 1}`);
  const panel = (values: Float32Array | undefined, title: string, r: [number, number], diverging: boolean, unit: string, key: string) => (
    <div>
      <span className="fx-map-title">{title}</span>
      <MapPanel values={values ?? null} shape={[ny, nx]} range={r} diverging={diverging} unit={unit} stretchKey={key} frame={frame} label={title} note={observed.error ? null : <div className="gx-map-wait" />} />
    </div>
  );
  return (
    <figure className="fx-card fx-triptych">
      <figcaption>
        <span>At the start</span>
        <span className="fx-caption-note">the model at the starts of the parameters</span>
        {observations.length > 1 && <Segments label="The observation" value={String(index)} onChange={(v) => setChosen(Number(v))} items={labels.map((label, i) => ({ value: String(i), label }))} />}
        <Segments label="The map" value={show} onChange={setShow} items={SHOWN.map(({ value, label }) => ({ value, label }))} />
      </figcaption>
      {observed.error ? (
        <p className="error fx-start-error">{observed.error.message}</p>
      ) : (
        <div className="fx-three" data-count={measured.item ? 3 : 1}>
          {measured.item && panel(measured.values, 'Data', range, measured.colours !== 'sequential', measured.unit, 'fx-start-data')}
          {panel(model.values, 'Model', range, model.colours !== 'sequential', model.unit, 'fx-start-data')}
          {measured.item && panel(residual, 'Residuals', [-half, half], true, measured.unit, 'fx-start-residual')}
        </div>
      )}
      {observed.data && observed.data.notes.length > 0 && (
        <ul className="fx-notes">
          {observed.data.notes.map((note, i) => (
            <li key={i}>{note}</li>
          ))}
        </ul>
      )}
    </figure>
  );
}

// (a message of the library without the class of its error)
export function plain(message: string): string {
  return message.replace(/^[A-Za-z]+Error: /, '');
}
