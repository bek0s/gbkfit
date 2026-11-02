// A response of an instrument, a block of the project: a PSF (how it
// spreads the light of a point on the sky) or an LSF (along the spectrum).
// Its options are a form of the schema of the library (see SchemaForm),
// numbers with sliders, and it is drawn as they change, by the server: at
// its own scale, or on pixels (channels) of a size given or of a dataset of
// the project, linear or logarithmic (its wings), with what it amounts to
// (its widths along its axes, its beam, its resolving power) and the
// blocks that name it. Its edits are written to its file when saved (also
// Ctrl+S).
import { useDebouncedValue } from '@mantine/hooks';
import { useEffect, useState } from 'react';
import { useDescription, useLsfView, usePiece, useProject, usePsfView, useReferences, useSchema, useUpdateBlock } from '../api/queries';
import { ColourBar } from '../components/ColourBar';
import { joinPath } from '../components/FilePicker';
import { significant } from '../components/format';
import { PanelsIcon } from '../components/icons';
import { CutPlot } from '../components/CutPlot';
import { MapView } from '../components/MapView';
import { SchemaForm, typeTitle, type Defs, type Options } from '../components/SchemaForm';
import { SpectrumPlot } from '../components/SpectrumPlot';
import { Button, IconButton, NavItem } from '../components/ui/basics';
import { Choice, NumberField, Segments } from '../components/ui/inputs';
import { Panel } from '../components/ui/panel';
import { SidePanes, type PanesControl } from '../components/ui/panes';
import { BlockMenu } from '../shell/BlockMenu';
import { kindOf } from '../shell/blocks';
import { useStudio } from '../studio';
import { useUnsaved } from './editing';

export type Family = 'psf' | 'lsf';

// What the members of each family start with, when chosen (their options
// that must be given): a PSF of a seeing of about 3.5", an LSF of about
// 35 km/s
export const STARTS: Record<string, Record<string, Options>> = {
  psf: {
    gauss: { sigma: 1.5 },
    gauss_beam: { bmaj: 3.5, bmin: 3.5, bpa: 0 },
    moffat: { alpha: 2, beta: 2.5 },
    ggauss: { alpha: 2, beta: 2 },
    point: {},
    sum: { psfs: [{ type: 'gauss', sigma: 1 }, { type: 'gauss', sigma: 3 }], weights: [0.8, 0.2] },
    convolution: { psfs: [{ type: 'gauss', sigma: 1 }, { type: 'moffat', alpha: 2, beta: 2.5 }] },
  },
  lsf: {
    gauss: { sigma: 15 },
    moffat: { alpha: 15, beta: 2.5 },
    ggauss: { alpha: 15, beta: 2 },
    lorentz: { gamma: 10 },
    hanning: { width: 10 },
    point: {},
    sum: { lsfs: [{ type: 'gauss', sigma: 10 }, { type: 'gauss', sigma: 30 }], weights: [0.8, 0.2] },
    convolution: { lsfs: [{ type: 'gauss', sigma: 10 }, { type: 'hanning', width: 10 }] },
  },
};

const UNITS: Record<Family, { step: string; what: string; of: string }> = {
  psf: { step: 'arcsec', what: 'The size of the pixels', of: 'pixels' },
  lsf: { step: 'km/s', what: 'The width of the channels', of: 'channels' },
};

// The speed of light (km/s), of the resolving power of an LSF (c / FWHM)
const LIGHT = 299792.458;

// What it is drawn on: its own scale, a size typed, or those of the data
// of a dataset of the project
type Drawn = { source: 'own' } | { source: 'typed'; step: string } | { source: 'dataset'; name: string };

export function ResponseScreen({ family, name }: { family: Family; name: string }) {
  const { project } = useStudio();
  const folder = `${family}s`;
  const path = project === null ? null : joinPath(project, `${folder}/${name}.yaml`);
  const piece = usePiece(path);
  const schema = useSchema();
  const update = useUpdateBlock();
  // The options as edited (null: as saved)
  const [draft, setDraft] = useState<Options | null>(null);
  const [drawn, setDrawn] = useState<Drawn>({ source: 'own' });
  // (the size of the pixels or channels of the dataset chosen, once read)
  const [ofData, setOfData] = useState<number | null>(null);
  const saved = piece.data as Options | undefined;
  const options = draft ?? saved;
  const dirty = draft !== null && JSON.stringify(draft) !== JSON.stringify(saved);
  useUnsaved(dirty);
  const save = () => draft && path && update.mutate({ path, content: draft }, { onSuccess: () => setDraft(null) });
  // (Ctrl+S, or Cmd+S, saves)
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 's') {
        event.preventDefault();
        if (dirty && !update.isPending) save();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });
  if (project === null || path === null) return null;
  const defs = (schema.data?.$defs ?? {}) as Defs;
  const step =
    drawn.source === 'typed' ? (Number(drawn.step) > 0 ? Number(drawn.step) : null) : drawn.source === 'dataset' ? ofData : null;
  return (
    <section className="screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">
            {kindOf(folder)}
            {options && typeof options.type === 'string' ? ` · ${typeTitle(options.type)}` : ''}
          </div>
          <h1>{name}</h1>
        </div>
        <div className="head-actions">
          {dirty && <span className="why">Not saved</span>}
          {update.error && <span className="error">{update.error.message}</span>}
          <Button disabled={!dirty} onClick={() => setDraft(null)}>
            Revert
          </Button>
          <Button kind="primary" title="Save (Ctrl+S)" disabled={!dirty || update.isPending} onClick={save}>
            {update.isPending ? 'Saving…' : 'Save'}
          </Button>
          <BlockMenu project={project} folder={folder} name={name} />
        </div>
      </div>
      {piece.error && <p className="error screen-error">{piece.error.message}</p>}
      {schema.error && <p className="error screen-error">{schema.error.message}</p>}
      {options !== undefined && (
        <div className="screen-body dataset">
          <SidePanes
            width={440}
            main={(control) => <Preview family={family} project={project} options={options} step={step} control={control} />}
            side={
              <div className="col">
                <Panel id={`${family}-options`} title="Options">
                  {schema.data ? (
                    <SchemaForm defs={defs} def={family} project={project} value={options} onChange={setDraft} starts={STARTS} sliders />
                  ) : (
                    <p className="note">Reading the options of the library…</p>
                  )}
                </Panel>
                <DrawnOn family={family} project={project} drawn={drawn} onDrawn={setDrawn} ofData={ofData} onOfData={setOfData} />
                <NamedBy path={path} />
              </div>
            }
          />
        </div>
      )}
    </section>
  );
}

// What it is drawn on: its own scale (to see its shape), a size typed, or
// those of a dataset of the project (to see it as the model draws it)
function DrawnOn({
  family,
  project,
  drawn,
  onDrawn,
  ofData,
  onOfData,
}: {
  family: Family;
  project: string;
  drawn: Drawn;
  onDrawn: (drawn: Drawn) => void;
  ofData: number | null;
  onOfData: (step: number | null) => void;
}) {
  const datasets = useProject(project).data?.pieces.datasets ?? [];
  const units = UNITS[family];
  const value = drawn.source === 'dataset' ? `dataset:${drawn.name}` : drawn.source;
  return (
    <Panel id={`${family}-drawn`} title="Drawn on">
      <div className="schema-row">
        <span className="schema-label">{units.of}</span>
        <div className="schema-field">
          <Choice
            items={[
              { value: 'own', label: 'its own scale', description: 'Fine, to see its shape' },
              { value: 'typed', label: 'of a size…', description: `${units.what}, typed` },
              ...datasets.map((d) => ({ value: `dataset:${d}`, label: `those of ${d}`, description: 'To see it as the model draws it on these data' })),
            ]}
            value={value}
            label={`What the ${family.toUpperCase()} is drawn on`}
            onChange={(next) => {
              onOfData(null);
              onDrawn(next.startsWith('dataset:') ? { source: 'dataset', name: next.slice(8) } : next === 'typed' ? { source: 'typed', step: '' } : { source: 'own' });
            }}
          />
        </div>
      </div>
      {drawn.source === 'typed' && (
        <div className="schema-row">
          <span className="schema-label">size</span>
          <div className="schema-field">
            <NumberField value={drawn.step} onChange={(step) => onDrawn({ source: 'typed', step })} min={0} placeholder="e.g. 0.2" label={units.what} />
            <span className="schema-unit">{units.step}</span>
          </div>
        </div>
      )}
      {drawn.source === 'dataset' && <DatasetStep key={drawn.name} family={family} project={project} name={drawn.name} step={ofData} onStep={onOfData} />}
    </Panel>
  );
}

// The size of the pixels (PSF) or the width of the channels (LSF) of the
// data of a dataset of the project, from the header of its file (of its
// first map, of moment maps)
function DatasetStep({
  family,
  project,
  name,
  step,
  onStep,
}: {
  family: Family;
  project: string;
  name: string;
  step: number | null;
  onStep: (step: number | null) => void;
}) {
  const dataset = usePiece(joinPath(project, `datasets/${name}.yaml`)).data as Options | undefined;
  const item = (dataset?.type === 'pixel_moments' ? (dataset.moment0 ?? dataset.moment1) : dataset) as Options | undefined;
  const data = item?.data as string | { file?: string; hdu?: string | number } | undefined;
  const file = typeof data === 'string' ? data : data?.file;
  const hdu = typeof data === 'object' ? data.hdu : undefined;
  const arrays = useDescription(file ? joinPath(project, file) : null);
  const array = (arrays.data ?? []).find((a) => a.name === hdu || a.index === hdu) ?? arrays.data?.[0];
  const found = array?.step ? (family === 'psf' ? Math.abs(array.step[0]) : array.step.length === 3 ? Math.abs(array.step[2]) : null) : null;
  useEffect(() => onStep(found), [found]);
  if (arrays.isPending) return <p className="note">Reading its header…</p>;
  if (step === null) {
    return (
      <p className="note">
        {family === 'lsf' && array && kept(array).length < 3
          ? `${name} has no spectral axis: an LSF is drawn on the channels of a cube.`
          : `The header of ${name} does not give its ${UNITS[family].of}${array?.ambiguous_velocity ? ' (say whether its velocities are radio or optical)' : ''}.`}
      </p>
    );
  }
  return (
    <p className="note">
      {UNITS[family].of} of {significant(step, 4)}
      {family === 'psf' ? '″' : ' km/s'}, from the header of {name}.
    </p>
  );
}

function kept(array: { shape: number[] }): number[] {
  return array.shape.filter((n) => n > 1);
}

// The blocks that name it (e.g. observations, in their instruments)
function NamedBy({ path }: { path: string }) {
  const { go } = useStudio();
  const references = useReferences(path).data ?? [];
  return (
    <Panel id="named-by" title="Named by">
      {references.length === 0 ? (
        <p className="note">No block names it yet: an observation names it in its instrument (psf: or lsf: its name).</p>
      ) : (
        references.map((r) => (
          <NavItem
            key={`${r.folder}/${r.name}`}
            mono
            label={r.name}
            right={<small>{kindOf(r.folder)}</small>}
            onClick={() => r.folder !== 'setups' && go({ kind: 'block', folder: r.folder, name: r.name })}
          />
        ))
      )}
    </Panel>
  );
}

type Scale = 'linear' | 'log';

// The values shown on a scale: as they are, or the logarithm of their
// fraction of the peak, down to a floor (where they are too, below it or
// 0, e.g. beyond the extent it is drawn out to)
const FLOOR = -5;

function scaled(values: number[], peak: number, scale: Scale): number[] {
  if (scale === 'linear' || !(peak > 0)) return values;
  return values.map((v) => (v > 0 ? Math.max(FLOOR, Math.log10(v / peak)) : FLOOR));
}

// The response drawn, of its options (as they settle while they change),
// and what it amounts to
function Preview({
  family,
  project,
  options,
  step,
  control,
}: {
  family: Family;
  project: string;
  options: Options;
  step: number | null;
  control: PanesControl;
}) {
  const [settled] = useDebouncedValue(options, 120);
  const [scale, setScale] = useState<Scale>('linear');
  return (
    <div className="card viewer response">
      <div className="card-head viewer-head">
        <h2>{family === 'psf' ? 'The PSF' : 'The LSF'}</h2>
        <div className="response-tools">
          <Segments
            items={[
              { value: 'linear', label: 'linear' },
              { value: 'log', label: 'log' },
            ]}
            value={scale}
            label="The scale of its values"
            onChange={setScale}
          />
          <IconButton label={control.folded ? 'Show the panels at the side' : 'Hide the panels at the side'} pressed={!control.folded} onClick={control.toggle}>
            <PanelsIcon />
          </IconButton>
        </div>
      </div>
      {family === 'psf' ? (
        <PsfPreview project={project} options={settled} step={step} scale={scale} />
      ) : (
        <LsfPreview project={project} options={settled} step={step} scale={scale} />
      )}
    </div>
  );
}

function PsfPreview({ project, options, step, scale }: { project: string; options: Options; step: number | null; scale: Scale }) {
  const view = usePsfView(project, options, step === null ? null : [step, step]);
  // The pixel the cuts go through: under the pointer, or else that clicked
  // (none: the centre)
  const [hover, setHover] = useState<[number, number] | null>(null);
  const [pinned, setPinned] = useState<[number, number] | null>(null);
  const data = view.data;
  if (data === undefined) return <Waiting error={view.error} />;
  const ny = data.values.length;
  const nx = data.values[0]?.length ?? 0;
  const flat = data.values.flat();
  const peak = Math.max(...flat);
  const log = scale === 'log';
  const values = Float32Array.from(scaled(flat, peak, scale));
  const range: [number, number] = log ? [FLOOR, 0] : [0, peak || 1];
  const cx = Math.floor(nx / 2);
  const cy = Math.floor(ny / 2);
  // (shown where it is above a thousandth of its peak, or a hundred
  // thousandth on a log scale: it is drawn out to its faint wings)
  const floor = peak * (log ? 1e-5 : 1e-3);
  const reach = (cut: number[], c: number) => Math.max(1, ...cut.map((v, i) => (v > floor ? Math.abs(i - c) : 0)));
  const h = Math.max(Math.min(cx, reach(data.values[cy] ?? [], cx) + 1), Math.min(cy, reach(data.values.map((row) => row[cx]), cy) + 1));
  const window: [number, number, number, number] = [Math.max(0, cx - h), Math.min(nx - 1, cx + h), Math.max(0, cy - h), Math.min(ny - 1, cy + h)];
  const inside = (p: [number, number] | null) => p !== null && p[0] < nx && p[1] < ny;
  const [x, y] = inside(hover) ? hover! : inside(pinned) ? pinned! : [cx, cy];
  const rowX = scaled(data.values[y] ?? [], peak, scale);
  const colY = scaled(
    data.values.map((row) => row[x]),
    peak,
    scale,
  );
  // (the offset of a pixel from the centre, arcsec)
  const offset = (i: number, c: number, d: number) => `${i === c ? '' : i > c ? '+' : '−'}${significant(Math.abs((i - c) * d), 3)}″`;
  // (none from a server older than its measure: that of its beam)
  const axes = (data.axes ?? (data.fwhm === null ? null : [data.fwhm, data.fwhm, null])) as [number, number, number | null] | null;
  return (
    <div className="response-body">
      {view.error && <p className="error">{view.error.message}</p>}
      <div className="psf-grid">
        <div className="psf-map">
          <MapView
            values={values}
            shape={[ny, nx]}
            range={range}
            window={nx > 1 ? window : null}
            drag="pan"
            mark={pinned ?? undefined}
            onHover={setHover}
            onPick={(px, py) => setPinned(pinned && pinned[0] === px && pinned[1] === py ? null : [px, py])}
            label="The PSF"
          />
        </div>
        <div className="psf-ycut">
          <CutPlot values={colY} window={[window[2], window[3]]} limits={range} at={y} vertical label="The cut along y" />
        </div>
        <div className="psf-bar">
          <ColourBar range={range} diverging={false} unit={log ? 'log₁₀ of the peak' : 'per pixel'} />
        </div>
        <div className="psf-xcut">
          <CutPlot values={rowX} window={[window[0], window[1]]} limits={range} at={x} label="The cut along x" />
        </div>
        <p className="note psf-cut-note">
          Cuts through x {offset(x, cx, data.step[0])}, y {offset(y, cy, data.step[1])}
          {inside(hover) ? '' : pinned ? ' (clicked; click it again for the centre)' : ' (the centre: move over the map, or click a pixel)'}
          {' · '}
          {significant(Math.abs((window[0] - cx - 0.5) * data.step[0]), 3)}″ to each side
        </p>
      </div>
      <dl className="facts">
        <dt>FWHM</dt>
        <dd>
          {axes === null
            ? 'none: a point'
            : axes[2] === null
              ? `${significant(axes[0], 3)}″ (round)`
              : `${significant(axes[0], 3)}″ × ${significant(axes[1], 3)}″, the major axis at ${significant(axes[2], 3)}°`}
        </dd>
        <dt>beam</dt>
        <dd>
          {data.beam_area === null
            ? 'none'
            : `${significant(data.beam_area, 4)} arcsec², as of a round Gaussian of FWHM ${significant(data.fwhm!, 3)}″`}
        </dd>
        <dt>drawn on</dt>
        <dd>
          {nx} × {ny} pixels of {significant(data.step[0], 3)}″
        </dd>
      </dl>
    </div>
  );
}

function LsfPreview({ project, options, step, scale }: { project: string; options: Options; step: number | null; scale: Scale }) {
  const view = useLsfView(project, options, step);
  const data = view.data;
  if (data === undefined) return <Waiting error={view.error} />;
  const n = data.values.length;
  const centre = Math.floor(n / 2);
  const peak = Math.max(...data.values);
  const log = scale === 'log';
  return (
    <div className="response-body">
      {view.error && <p className="error">{view.error.message}</p>}
      <div className="response-profiles">
        <SpectrumPlot
          values={scaled(data.values, peak, scale)}
          spectral={data.values.map((_, i) => (i - centre) * data.step)}
          unit="km/s"
          valueUnit={log ? 'log₁₀ of the peak' : null}
          channel={centre}
          limits={log ? [FLOOR, 0] : null}
          onPick={() => {}}
        />
      </div>
      <dl className="facts">
        <dt>FWHM</dt>
        <dd>{data.fwhm === null ? 'narrower than a channel' : `${significant(data.fwhm, 3)} km/s`}</dd>
        {data.fwhm !== null && (
          <>
            <dt>resolution</dt>
            <dd title="The resolving power of a spectrograph of this LSF: λ/Δλ = c/FWHM">
              R = {Math.round(LIGHT / data.fwhm).toLocaleString('en')} · σ of a Gaussian of its FWHM: {significant(data.fwhm / (2 * Math.sqrt(2 * Math.log(2))), 3)} km/s
            </dd>
          </>
        )}
        <dt>drawn on</dt>
        <dd>
          {n} channels of {significant(data.step, 3)} km/s
        </dd>
      </dl>
    </div>
  );
}

function Waiting({ error }: { error: Error | null }) {
  return <div className="response-body">{error ? <p className="error">{error.message}</p> : <p className="note">Drawing…</p>}</div>;
}
