// The result of a run of a fit (see the library's FitResult, fit/result.json
// of the run): whether its fitter ended well, its merits (χ² reduced, χ²,
// AIC, BIC, the values fitted, the evaluations and the time), its free
// parameters at the best fit with their errors and where each sits in its
// range (one at a bound is flagged), which can become the starts of the
// fit, and the data, the model and the residuals at the best fit through
// each observation, as the run wrote them.
import { useMemo, useState } from 'react';
import { useChannel, useFileText, useGalaxyParameters, useGalaxyRender, useMoments, usePiece } from '../../api/queries';
import { CurvePlot, massCurves } from '../galaxy/Curves';
import { mapUnit, SNR } from '../../components/CubeViewer';
import { MapPanel, robustRange } from '../../components/MapPanel';
import { SHOWN, type Colours, type Shown } from '../../components/observedMaps';
import { Button } from '../../components/ui/basics';
import { Segments } from '../../components/ui/inputs';
import { isObject } from '../galaxy/options';
import { PosteriorCard } from './Posterior';

interface Product {
  kind: string;
  path: string;
  ids?: Record<string, string>;
}

interface Solution {
  free: Record<string, number>;
  elements: Record<string, number>;
  chi2: number | null;
  nresiduals: number | null;
  nfitted: number | null;
  reduced_chi2: number | null;
  aic: number | null;
  bic: number | null;
  covariance?: { std?: Record<string, number> } | null;
}

interface FitResult {
  params: { properties?: Record<string, unknown> };
  units: Record<string, string>;
  nevals: number;
  time: number;
  success: boolean;
  message: string;
  evidence: unknown;
  solutions: Solution[];
}

export function ResultView({ project, run, products, onStarts }: { project: string; run: string; products: Product[]; onStarts: (values: Record<string, number>) => void }) {
  const file = products.find((p) => p.kind === 'fit_result');
  const text = useFileText(file ? `${run}/${file.path}` : null);
  const result = useMemo(() => (text.data ? (JSON.parse(text.data) as FitResult) : undefined), [text.data]);
  if (!file) return <p className="ob-text">The run has no result: it did not end with one.</p>;
  if (text.error) return <p className="error">{text.error.message}</p>;
  if (!result) return <p className="ob-text">Reading the result…</p>;
  const best = result.solutions[0];
  const posterior = products.find((p) => p.kind === 'posterior');
  return (
    <div className="fx-view-grid">
      {!result.success && (
        <div className="fx-alert">
          <span className="fx-sign">!</span>
          <span>The fitter ended without converging: {result.message}</span>
        </div>
      )}
      {best && <Merits result={result} best={best} />}
      <Maps run={run} products={products} />
      {best && (
        <div className="fx-two-cards">
          <BestParams result={result} best={best} onStarts={onStarts} />
          <CurvesCard project={project} run={run} best={best} />
        </div>
      )}
      {best && posterior && <PosteriorCard path={`${run}/${posterior.path}`} free={Object.keys(best.free)} />}
    </div>
  );
}

function Merits({ result, best }: { result: FitResult; best: Solution }) {
  const cells: [string, string][] = [
    ['χ² reduced', fixed(best.reduced_chi2, 4)],
    ['χ²', fixed(best.chi2, 1)],
    ['AIC', fixed(best.aic, 1)],
    ['BIC', fixed(best.bic, 1)],
    ['Values', best.nresiduals === null ? '—' : grouped(best.nresiduals)],
    ['Free', best.nfitted === null ? '—' : String(best.nfitted)],
    ['Models', grouped(result.nevals)],
    ['Time', duration(result.time)],
  ];
  return (
    <div className="fx-card fx-merits">
      {cells.map(([label, value]) => (
        <div key={label}>
          <span className="fx-overline">{label}</span>
          <b>{value}</b>
        </div>
      ))}
    </div>
  );
}

// The curves of the (first) galaxy of the run at the best fit: its rotation
// and its circular speeds, of the galaxy as the run fitted it (its
// configuration as it ran)
function CurvesCard({ project, run, best }: { project: string; run: string; best: Solution }) {
  const config = usePiece(`${run}/config.yaml`).data;
  const listed = isObject(config) ? config.galaxies : undefined;
  const galaxy = Array.isArray(listed) ? listed[0] : listed;
  const options = isObject(galaxy) ? galaxy : undefined;
  const prefix = options && typeof options.name === 'string' ? `${options.name}_` : '';
  const known = new Set(((useGalaxyParameters(project, options).data?.parameters ?? []) as { name: string }[]).map((p) => p.name));
  const properties = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(best.elements)
          .filter(([key]) => key.startsWith(prefix))
          .map(([key, value]) => [key.slice(prefix.length), value] as const)
          .filter(([key]) => known.has(key.split('[', 1)[0])),
      ),
    [best.elements, prefix, known],
  );
  const render = useGalaxyRender(project, known.size ? options : undefined, properties, 'sky', 64);
  const curves = render.data ? massCurves(render.data, new Map(), new Map()) : [];
  return (
    <div className="fx-card">
      <div className="fx-card-head">
        <h2>Curves</h2>
        <span className="fx-faint fx-small">{prefix ? `of ${prefix.slice(0, -1)} ` : ''}at the best fit</span>
      </div>
      <div className="fx-curves">
        {render.error ? (
          <p className="error fx-pad">{render.error.message}</p>
        ) : render.data ? (
          <CurvePlot curves={curves} radius={render.data.curves.radius ?? []} unit="km/s" empty="No curves: the galaxy has no rotation or mass." />
        ) : (
          <div className="gx-map-wait" />
        )}
      </div>
    </div>
  );
}

// The free parameters at the best fit, with their errors and where each
// sits in its range
function BestParams({ result, best, onStarts }: { result: FitResult; best: Solution; onStarts: (values: Record<string, number>) => void }) {
  const properties = result.params.properties ?? {};
  const std = best.covariance?.std ?? {};
  const names = Object.keys(best.free);
  const fixedNames = Object.keys(best.elements).filter((name) => !(name in best.free));
  return (
    <div className="fx-card">
      <div className="fx-card-head">
        <h2>Parameters</h2>
        <span className="fx-faint fx-small">at the best fit</span>
        <span className="fx-right">
          <Button kind="small" title="Writes these values as the starts of the free parameters of the fit (its own values; not saved until Save)" onClick={() => onStarts(best.free)}>
            Use as starts
          </Button>
        </span>
      </div>
      <table className="fx-best">
        <thead>
          <tr>
            <th>Parameter</th>
            <th className="fx-num">Value</th>
            <th className="fx-num">Error</th>
            <th>In its range</th>
          </tr>
        </thead>
        <tbody>
          {names.map((name) => {
            const property = properties[name];
            const [min, max] = isObject(property) ? [property.min, property.max] : [undefined, undefined];
            const value = best.free[name];
            const span = typeof min === 'number' && typeof max === 'number' && max > min ? (value - min) / (max - min) : null;
            const bound = span !== null && (span < 1e-3 ? 'min' : span > 1 - 1e-3 ? 'max' : null);
            return (
              <tr key={name} data-bound={bound || undefined}>
                <td className="fx-name">{name}</td>
                <td className="fx-num">
                  {significant(value)} <span className="fx-faint">{unitText(result.units[name])}</span>
                </td>
                <td className="fx-num">{std[name] !== undefined && Number.isFinite(std[name]) ? `± ${significant(std[name], 2)}` : '—'}</td>
                <td>
                  {span !== null ? (
                    <span className="fx-range" title={`${min} to ${max}`}>
                      <i style={{ left: `${Math.min(100, Math.max(0, span * 100))}%` }} />
                      {bound && <em>at its {bound}</em>}
                    </span>
                  ) : (
                    <span className="fx-faint fx-small">no range</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {fixedNames.length > 0 && (
        <p className="fx-fixed-list">
          Fixed: {fixedNames.map((name) => `${name} = ${significant(best.elements[name])}`).join(' · ')}
        </p>
      )}
    </div>
  );
}

// The data, the model and the residuals at the best fit, through each
// observation of the run (its products: of each kind, by observation and
// item)
function Maps({ run, products }: { run: string; products: Product[] }) {
  const observations = [...new Set(products.filter((p) => p.kind === 'model' && p.ids?.observation).map((p) => p.ids!.observation))];
  const [chosen, setChosen] = useState(0);
  const [show, setShow] = useState<Shown>('intensity');
  const observation = observations[Math.min(chosen, observations.length - 1)];
  const of = (kind: string) => products.filter((p) => p.kind === kind && p.ids?.observation === observation);
  const model = useRunMap(run, of('model'), 'model', show);
  const measured = useRunMap(run, of('data'), 'data', show);
  const residual = useMemo(() => {
    if (!model.values || !measured.values || model.values.length !== measured.values.length) return undefined;
    return measured.values.map((d, i) => d - model.values![i]);
  }, [model.values, measured.values]);
  if (observations.length === 0) return null;
  const range = measured.values ? robustRange(measured.values, measured.colours) : model.values ? robustRange(model.values, model.colours) : ([0, 1] as [number, number]);
  const half = residual ? Math.max(...robustRange(residual, 'zero').map(Math.abs)) : 1;
  const shape = model.shape ?? measured.shape ?? [1, 1];
  const panel = (values: Float32Array | undefined, title: string, r: [number, number], diverging: boolean, unit: string, key: string) => (
    <div>
      <span className="fx-map-title">{title}</span>
      <MapPanel values={values ?? null} shape={shape} range={r} diverging={diverging} unit={unit} stretchKey={key} label={title} note={<div className="gx-map-wait" />} />
    </div>
  );
  return (
    <figure className="fx-card fx-triptych">
      <figcaption>
        <span>Data, model and residuals</span>
        <span className="fx-caption-note">at the best fit</span>
        {observations.length > 1 && <Segments label="The observation" value={String(observations.indexOf(observation))} onChange={(v) => setChosen(Number(v))} items={observations.map((o, i) => ({ value: String(i), label: o }))} />}
        <Segments label="The map" value={show} onChange={setShow} items={SHOWN.map(({ value, label }) => ({ value, label }))} />
      </figcaption>
      <div className="fx-three" data-count={measured.item ? 3 : 1}>
        {measured.item && panel(measured.values, 'Data', range, measured.colours !== 'sequential', measured.unit, 'fx-result-data')}
        {panel(model.values, 'Model', range, model.colours !== 'sequential', model.unit, 'fx-result-data')}
        {measured.item && panel(residual, 'Residuals', [-half, half], true, measured.unit, 'fx-result-residual')}
      </div>
    </figure>
  );
}

// A map of a product of a run: the moment shown of a cube (measured as from
// data; of the data, in the mask of their signal), or the moment map shown
// (or the image), with its shape (ny, nx)
function useRunMap(run: string, products: Product[], kind: 'model' | 'data', show: Shown) {
  const items: Record<Shown, string> = { intensity: 'moment0', velocity: 'moment1', dispersion: 'moment2' };
  const moments = { intensity: 'flux', velocity: 'velocity', dispersion: 'dispersion' } as const;
  const cubeProduct = products.find((p) => p.ids?.item === 'spectra');
  const mapProduct = products.find((p) => p.ids?.item === items[show]) ?? products.find((p) => p.ids?.item === 'brightness');
  const product = cubeProduct ?? mapProduct;
  const path = product ? `${run}/${product.path}` : null;
  const plane = useChannel(!cubeProduct && path ? path : null, 0, 0);
  const cube = useMoments(path ?? '', 0, kind === 'data' ? SNR : 0, cubeProduct !== undefined && path !== null);
  const array = cubeProduct ? cube.data?.find((a) => a.info.name === moments[show]) : plane.data?.plane;
  const colours: Colours = product?.ids?.item === 'brightness' ? 'sequential' : SHOWN.find((s) => s.value === show)!.colours;
  const unit = cubeProduct ? mapUnit(moments[show], null) : '';
  return { item: product, values: array?.values, shape: array ? ([array.shape[0], array.shape[1]] as [number, number]) : null, colours, unit };
}

function fixed(value: number | null, digits: number): string {
  return value === null || !Number.isFinite(value) ? '—' : value.toFixed(digits);
}

function grouped(value: number): string {
  return Math.round(value).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

function duration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
  return `${Math.floor(seconds / 3600)} h ${Math.round((seconds % 3600) / 60)} min`;
}

function significant(value: number, digits = 4): string {
  return Number.isFinite(value) ? String(Number(value.toPrecision(digits))) : String(value);
}

function unitText(unit: string | undefined): string {
  if (!unit) return '';
  return unit === 'deg' ? '°' : unit === 'arcsec' ? '″' : unit === 'km s-1' ? 'km/s' : unit;
}
