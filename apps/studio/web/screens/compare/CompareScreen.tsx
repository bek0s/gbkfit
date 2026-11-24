// The screen of a comparison: galaxies of the project seen by observations
// of it, side by side, in a grid (a comparison of the Studio, which the
// library does not know: comparisons/<name>, its galaxies, its
// observations, the datasets they are paired with, by position, as a setup
// pairs them, and how it shows them; saved as it changes). Each cell is a
// galaxy as an observation sees it (see the server's /views/observed), at
// the values of its parameters as saved (its params block of the same
// name): one map, the same in every cell (its intensity, velocity or
// dispersion; the moments of a cube measured as from data), or what is
// left of the data by it (the residual: data less model). The data of each
// observation paired with them are a row (or a column) of their own. The
// maps of one observation share their colours, so that they read alike
// (or each has its own, as chosen, e.g. of models far fainter than the
// data): those of its data, if it has them, or else those that hold all
// its models; the residuals, colours about zero that hold them all.
import { useQuery } from '@tanstack/react-query';
import { Fragment, useCallback, useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { api, data } from '../../api/client';
import { keys, useGalaxyParameters, useObserved, usePiece, useProject, useUpdateBlock } from '../../api/queries';
import { Choice } from '../../components/ui/inputs';
import { joinPath } from '../../components/FilePicker';
import { MapPanel, robustRange } from '../../components/MapPanel';
import { SHOWN, useObservedMap, type Colours, type Shown } from '../../components/observedMaps';
import { frameOf } from '../../components/sky';
import { Dock, DockSection } from '../../components/ui/dock';
import { Segments } from '../../components/ui/inputs';
import { SheetLayout } from '../../components/ui/sheet';
import { useBlockHead } from '../../shell/BlockMenu';
import { SaveControls, ScreenHead } from '../../shell/ScreenHead';
import { useStudio } from '../../studio';
import { isObject } from '../galaxy/options';
import { forRender, type Param, type Properties } from '../galaxy/params';

// The sizes of the cells (their height, px)
const SIZES = { s: 260, m: 340, l: 460 } as const;
type Size = keyof typeof SIZES;

// A comparison, as its file holds it
interface Comparison {
  galaxies: string[];
  observations: string[];
  // (the dataset of the project each observation is paired with, at its
  // place; null: none)
  datasets: (string | null)[];
  // (the rows: the galaxies, an observation a column; or the observations)
  rows: 'galaxies' | 'observations';
  show: Shown;
  residuals: boolean;
  // (the colours of the maps of an observation: shared, or each its own)
  colours: 'shared' | 'own';
  size: Size;
}

function comparisonOf(content: unknown): Comparison {
  const options = isObject(content) ? content : {};
  const names = (value: unknown) => (Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : []);
  const observations = names(options.observations);
  const datasets = Array.isArray(options.datasets) ? options.datasets : [];
  return {
    galaxies: names(options.galaxies),
    observations,
    datasets: observations.map((_, i) => (typeof datasets[i] === 'string' ? (datasets[i] as string) : null)),
    rows: options.rows === 'observations' ? 'observations' : 'galaxies',
    show: SHOWN.some((s) => s.value === options.show) ? (options.show as Shown) : 'intensity',
    residuals: options.residuals === true,
    colours: options.colours === 'own' ? 'own' : 'shared',
    size: typeof options.size === 'string' && options.size in SIZES ? (options.size as Size) : 'm',
  };
}

export function CompareScreen({ name }: { name: string }) {
  const { project } = useStudio();
  const root = project ?? '.';
  const head = useBlockHead(root, 'comparisons', name);
  const path = joinPath(root, `comparisons/${name}.yaml`);
  const piece = usePiece(path);
  const update = useUpdateBlock();
  const pieces = useProject(project).data?.pieces;
  // (as it changes, it is shown at once and saved)
  const [comparison, setComparison] = useState<Comparison | null>(null);
  useEffect(() => {
    if (piece.data !== undefined && comparison === null) setComparison(comparisonOf(piece.data));
  }, [piece.data, comparison]);
  const change = (next: Partial<Comparison>) => {
    if (comparison === null) return;
    const changed = { ...comparison, ...next };
    setComparison(changed);
    // (no datasets: none paired)
    const { datasets, ...rest } = changed;
    update.mutate({ path, content: datasets.some((dataset) => dataset !== null) ? changed : rest });
  };
  if (project === null) return null;
  const count = comparison ? `${comparison.galaxies.length} × ${comparison.observations.length}` : '';
  return (
    <section className="screen cmp-screen">
      <ScreenHead eyebrow={`Comparison${count && ` · ${count}`}`} title={head.title}>
        <SaveControls auto saving={{ dirty: false, saving: update.isPending, saveError: update.error, save: () => undefined, revert: () => undefined }} />
        {head.menu}
      </ScreenHead>
      {piece.error && <p className="error screen-error">{piece.error.message}</p>}
      {comparison === null ? (
        <p className="note screen-error">Reading the comparison…</p>
      ) : (
        <SheetLayout
          className="ob-body"
          width={360}
          sheet={
            <Dock label="The comparison">
              <DockSection id="cmp-what" title="Compare">
                <div className="ob-form">
                  <Row label="Galaxies">
                    <Toggles all={pieces?.galaxies ?? []} chosen={comparison.galaxies} what="galaxy" onChange={(galaxies) => change({ galaxies })} />
                  </Row>
                  <Row label="Observations">
                    <Toggles
                      all={pieces?.observations ?? []}
                      chosen={comparison.observations}
                      what="observation"
                      // (the data of each kept at its place; one added, none)
                      onChange={(observations) => change({ observations, datasets: observations.map((o) => comparison.datasets[comparison.observations.indexOf(o)] ?? null) })}
                    />
                  </Row>
                  {comparison.observations.map((o, i) => (
                    <Row key={o} label={`Data of ${o}`}>
                      <Choice
                        label={`The data ${o} is paired with`}
                        value={comparison.datasets[i] ?? ''}
                        onChange={(dataset) => change({ datasets: comparison.datasets.map((each, j) => (j === i ? dataset || null : each)) })}
                        items={[
                          { value: '', label: 'None: models alone' },
                          // (one it names that the project has not, as it is)
                          ...[...new Set([...(comparison.datasets[i] ? [comparison.datasets[i]] : []), ...(pieces?.datasets ?? [])])].map((name) => ({ value: name, label: name })),
                        ]}
                      />
                    </Row>
                  ))}
                  <p className="ob-text">Each galaxy at the values of its parameters as saved, seen through each observation: paired with data, on their grid.</p>
                </div>
              </DockSection>
              <DockSection id="cmp-show" title="Show">
                <div className="ob-form">
                  <Row label="Map">
                    <Segments items={SHOWN.map(({ value, label }) => ({ value, label }))} value={comparison.show} onChange={(show) => change({ show })} label="The map each cell shows" />
                  </Row>
                  <Row label="Of">
                    <Segments
                      items={[
                        { value: 'model', label: 'Model' },
                        { value: 'residual', label: 'Residual', title: 'The data less the model, of the observations paired with data' },
                      ]}
                      value={comparison.residuals ? 'residual' : 'model'}
                      onChange={(of) => change({ residuals: of === 'residual' })}
                      label="What each cell of a galaxy shows"
                    />
                  </Row>
                  <Row label="Colours">
                    <Segments
                      items={[
                        { value: 'shared', label: 'Shared', title: 'The maps of an observation in the same colours: those of its data, or of all its models' },
                        { value: 'own', label: 'Each its own', title: 'Each map in the colours of its own values' },
                      ]}
                      value={comparison.colours}
                      onChange={(colours) => change({ colours })}
                      label="The colours of the maps of an observation"
                    />
                  </Row>
                  <Row label="Rows">
                    <Segments
                      items={[
                        { value: 'galaxies', label: 'Galaxies' },
                        { value: 'observations', label: 'Observations' },
                      ]}
                      value={comparison.rows}
                      onChange={(rows) => change({ rows })}
                      label="What the rows of the grid are"
                    />
                  </Row>
                  <Row label="Size">
                    <Segments
                      items={[
                        { value: 's', label: 'S' },
                        { value: 'm', label: 'M' },
                        { value: 'l', label: 'L' },
                      ]}
                      value={comparison.size}
                      onChange={(size) => change({ size })}
                      label="The size of the cells"
                    />
                  </Row>
                </div>
              </DockSection>
            </Dock>
          }
          picture={<Grid project={project} comparison={comparison} />}
        />
      )}
    </section>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="ob-row cmp-row">
      <span className="ob-label">{label}</span>
      <div>{children}</div>
    </div>
  );
}

// The blocks of a folder, each a chip that is on as it is in the
// comparison (in its order: one turned on goes last); those it names
// that the project has not, missing, to take out
function Toggles({ all, chosen, what, onChange }: { all: string[]; chosen: string[]; what: string; onChange: (names: string[]) => void }) {
  const missing = chosen.filter((name) => !all.includes(name));
  if (all.length === 0 && missing.length === 0) return <span className="ob-text">The project has no {what} yet.</span>;
  return (
    <span className="cmp-toggles">
      {all.map((name) => {
        const on = chosen.includes(name);
        return (
          <button key={name} className="cmp-toggle" aria-pressed={on} title={on ? `Take ${name} out` : `Compare ${name}`} onClick={() => onChange(on ? chosen.filter((n) => n !== name) : [...chosen, name])}>
            {name}
          </button>
        );
      })}
      {missing.map((name) => (
        <button key={name} className="cmp-toggle cmp-missing" title={`The project has no ${what} ${name}: take it out`} onClick={() => onChange(chosen.filter((n) => n !== name))}>
          {name} ×
        </button>
      ))}
    </span>
  );
}

// The ranges of the colours of the maps of the grid, as each cell reports
// its own (by `${observation}|${galaxy}|${layer}`; the galaxy '' of the
// data): those each observation shares (see the head)
type Ranges = Record<string, [number, number]>;

function sharedRange(ranges: Ranges, observation: string, galaxies: string[], residuals: boolean): [number, number] | null {
  if (residuals) {
    const ends = galaxies.flatMap((g) => ranges[`${observation}|${g}|residual`] ?? []).map(Math.abs);
    return ends.length ? [-Math.max(...ends), Math.max(...ends)] : null;
  }
  const own = ranges[`${observation}||data`];
  if (own) return own;
  const all = galaxies.map((g) => ranges[`${observation}|${g}|model`]).filter((r): r is [number, number] => r !== undefined);
  return all.length ? [Math.min(...all.map((r) => r[0])), Math.max(...all.map((r) => r[1]))] : null;
}

// The grid: a head of the columns, a row of the data (of the observations
// that have them), and a row of each galaxy; or, its rows the
// observations, a column of the data and one of each galaxy
function Grid({ project, comparison }: { project: string; comparison: Comparison }) {
  const { galaxies, observations, datasets, rows, show, residuals, colours, size } = comparison;
  const [ranges, setRanges] = useState<Ranges>({});
  const report = useCallback((key: string, range: [number, number]) => setRanges((all) => (all[key]?.[0] === range[0] && all[key]?.[1] === range[1] ? all : { ...all, [key]: range })), []);
  // (the dataset of each observation, by its name)
  const dataOf = (observation: string) => datasets[observations.indexOf(observation)] ?? null;
  const withData = observations.filter((o) => dataOf(o) !== null);
  if (galaxies.length === 0 || observations.length === 0) {
    return (
      <div className="cmp-picture cmp-empty">
        <p>Choose the galaxies and the observations to compare, in the sheet.</p>
      </div>
    );
  }
  const cell = (galaxy: string, observation: string) => (
    <Cell
      key={`${galaxy}|${observation}`}
      project={project}
      galaxy={galaxy}
      observation={observation}
      dataset={dataOf(observation)}
      show={show}
      residual={residuals && withData.includes(observation)}
      range={colours === 'own' ? null : sharedRange(ranges, observation, galaxies, residuals && withData.includes(observation))}
      onRange={report}
    />
  );
  const dataCell = (observation: string) =>
    withData.includes(observation) ? (
      <Cell key={`|${observation}`} project={project} galaxy={galaxies[0]} observation={observation} dataset={dataOf(observation)} show={show} ofData range={colours === 'own' ? null : sharedRange(ranges, observation, galaxies, false)} onRange={report} />
    ) : (
      <div key={`|${observation}`} className="cmp-cell cmp-none">
        <span>No data</span>
      </div>
    );
  const style = { '--cmp-cell': `${SIZES[size]}px` } as CSSProperties;
  if (rows === 'galaxies') {
    return (
      <div className="cmp-picture" style={style}>
        <div className="cmp-grid" style={{ gridTemplateColumns: `auto repeat(${observations.length}, minmax(240px, 1fr))` }}>
          <span />
          {observations.map((o) => (
            <Head key={o} name={o} note={residuals && withData.includes(o) ? 'residuals' : undefined} />
          ))}
          {withData.length > 0 && (
            <>
              <Side name="Data" data />
              {observations.map(dataCell)}
            </>
          )}
          {galaxies.map((g) => (
            <Fragment key={g}>
              <Side name={g} />
              {observations.map((o) => cell(g, o))}
            </Fragment>
          ))}
        </div>
      </div>
    );
  }
  return (
    <div className="cmp-picture" style={style}>
      <div className="cmp-grid" style={{ gridTemplateColumns: `auto repeat(${galaxies.length + (withData.length > 0 ? 1 : 0)}, minmax(240px, 1fr))` }}>
        <span />
        {withData.length > 0 && <Head name="Data" data />}
        {galaxies.map((g) => (
          <Head key={g} name={g} />
        ))}
        {observations.map((o) => (
          <Fragment key={o}>
            <Side name={o} note={residuals && withData.includes(o) ? 'residuals' : undefined} />
            {withData.length > 0 && dataCell(o)}
            {galaxies.map((g) => cell(g, o))}
          </Fragment>
        ))}
      </div>
    </div>
  );
}

function Head({ name, note, data: ofData }: { name: string; note?: string; data?: boolean }) {
  return (
    <div className="cmp-head" data-data={ofData || undefined}>
      <b>{name}</b>
      {note && <span>{note}</span>}
    </div>
  );
}

function Side({ name, note, data: ofData }: { name: string; note?: string; data?: boolean }) {
  return (
    <div className="cmp-side" data-data={ofData || undefined}>
      <b>{name}</b>
      {note && <span>{note}</span>}
    </div>
  );
}

// A galaxy as saved: its options, and the properties of its parameters
// (its params block of the same name, none if it has not one) as a render
// takes them
function useSavedGalaxy(project: string, name: string) {
  const galaxy = usePiece(joinPath(project, `galaxies/${name}.yaml`));
  const paramsPath = joinPath(project, `params/${name}.yaml`);
  // (a params block that is not there: none, not asked for again)
  const params = useQuery({
    queryKey: keys.piece(paramsPath),
    retry: false,
    queryFn: () => data(api.GET('/api/v1/pieces/{path}', { params: { path: { path: paramsPath } } })),
  });
  const options = isObject(galaxy.data) ? galaxy.data : undefined;
  const described = useGalaxyParameters(project, options);
  const saved = isObject(params.data) && isObject(params.data.properties) ? (params.data.properties as Properties) : null;
  const properties = useMemo(
    () => (described.data && (saved !== null || params.isError) ? forRender(saved ?? {}, described.data.parameters as unknown as Param[]) : undefined),
    [described.data, saved, params.isError],
  );
  return { options: properties === undefined ? undefined : options, properties: properties ?? {}, error: galaxy.error ?? described.error };
}

// A cell: a galaxy as an observation sees it, paired with its dataset if
// any (its model, or the residual of the data), or the data, in the
// colours the observation shares (until they are known, its own)
function Cell({
  project,
  galaxy,
  observation,
  dataset,
  show,
  residual = false,
  ofData = false,
  range,
  onRange,
}: {
  project: string;
  galaxy: string;
  observation: string;
  dataset: string | null;
  show: Shown;
  residual?: boolean;
  ofData?: boolean;
  range: [number, number] | null;
  onRange: (key: string, range: [number, number]) => void;
}) {
  const saved = useSavedGalaxy(project, galaxy);
  const seen = usePiece(joinPath(project, `observations/${observation}.yaml`));
  const datasetPiece = usePiece(dataset === null ? null : joinPath(project, `datasets/${dataset}.yaml`));
  const observed = useObserved(project, saved.options, saved.properties, observation, [seen.data, datasetPiece.data], galaxy, dataset);
  const model = useObservedMap(observed.data, 'model', show);
  const measured = useObservedMap(observed.data, 'data', show);
  const values = useMemo(() => {
    if (ofData) return measured.values;
    if (!residual) return model.values;
    if (!model.values || !measured.values || model.values.length !== measured.values.length) return undefined;
    return measured.values.map((d, i) => d - model.values![i]);
  }, [ofData, residual, model.values, measured.values]);
  const layer = ofData ? 'data' : residual ? 'residual' : 'model';
  const colours: Colours = residual ? 'zero' : model.colours;
  const own = useMemo(() => (values ? robustRange(values, colours) : null), [values, colours]);
  const key = `${observation}|${ofData ? '' : galaxy}|${layer}`;
  useEffect(() => {
    if (own) onRange(key, own);
  }, [key, own, onRange]);
  const item = (ofData ? measured : model).item;
  const error = saved.error ?? observed.error ?? model.error ?? measured.error;
  const note = error ? (
    <p className="error cmp-error">{ofData ? error.message : `${observation} cannot see ${galaxy}: ${error.message}`}</p>
  ) : observed.data && !item ? (
    <p className="note">The Studio does not show this kind of observation ({observed.data.type.replaceAll('_', ' ')}) here yet.</p>
  ) : (
    <div className="gx-map-wait" />
  );
  const [nx, ny] = item?.shape ?? [1, 1];
  return (
    <div className="cmp-cell">
      <MapPanel
        values={values ?? null}
        shape={[ny, nx]}
        range={range ?? own ?? [0, 1]}
        diverging={colours !== 'sequential'}
        unit={(ofData ? measured : model).unit}
        stretchKey={`${observation}/${show}/${layer === 'residual' ? 'residual' : 'map'}`}
        frame={item ? frameOf(item.step, item.rpix) : null}
        label={ofData ? `The data of ${observation}` : `${galaxy} seen by ${observation}${residual ? ': the residual' : ''}`}
        note={note}
      />
    </div>
  );
}
