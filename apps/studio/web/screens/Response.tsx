// A response of an instrument, a block of the project: a PSF (how it
// spreads the light of a point on the sky) or an LSF (along the spectrum).
// Its options are a form of the schema of the library (see SchemaForm),
// and it is drawn as they change, by the server: at its own scale, or on
// pixels (channels) of a size given, e.g. those of the data, with what it
// amounts to (its widths, and the beam of a PSF) and the blocks that name
// it. Its edits are written to its file when saved.
import { useDebouncedValue } from '@mantine/hooks';
import { useState } from 'react';
import { useLsfView, usePiece, usePsfView, useReferences, useSchema, useUpdateBlock } from '../api/queries';
import { ColourBar } from '../components/ColourBar';
import { joinPath } from '../components/FilePicker';
import { significant } from '../components/format';
import { PanelsIcon } from '../components/icons';
import { MapView } from '../components/MapView';
import { SchemaForm, typeTitle, type Defs, type Options } from '../components/SchemaForm';
import { SpectrumPlot } from '../components/SpectrumPlot';
import { Button, IconButton, NavItem } from '../components/ui/basics';
import { NumberField } from '../components/ui/inputs';
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

const UNITS: Record<Family, { step: string; across: string; what: string }> = {
  psf: { step: 'arcsec', across: 'pixels', what: 'The size of the pixels' },
  lsf: { step: 'km/s', across: 'channels', what: 'The width of the channels' },
};

export function ResponseScreen({ family, name }: { family: Family; name: string }) {
  const { project } = useStudio();
  const folder = `${family}s`;
  const path = project === null ? null : joinPath(project, `${folder}/${name}.yaml`);
  const piece = usePiece(path);
  const schema = useSchema();
  const update = useUpdateBlock();
  // The options as edited (null: as saved)
  const [draft, setDraft] = useState<Options | null>(null);
  const [step, setStep] = useState('');
  const saved = piece.data as Options | undefined;
  const options = draft ?? saved;
  const dirty = draft !== null && JSON.stringify(draft) !== JSON.stringify(saved);
  useUnsaved(dirty);
  if (project === null || path === null) return null;
  const defs = (schema.data?.$defs ?? {}) as Defs;
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
          <Button
            kind="primary"
            disabled={!dirty || update.isPending}
            onClick={() => draft && update.mutate({ path, content: draft }, { onSuccess: () => setDraft(null) })}
          >
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
            main={(control) => <Preview family={family} project={project} options={options} step={step} control={control} />}
            side={
              <div className="col">
                <Panel id={`${family}-options`} title="Options">
                  {schema.data ? (
                    <SchemaForm defs={defs} def={family} project={project} value={options} onChange={setDraft} starts={STARTS} />
                  ) : (
                    <p className="note">Reading the options of the library…</p>
                  )}
                </Panel>
                <Panel id={`${family}-drawn`} title="Drawn on">
                  <div className="schema-row" title={`${UNITS[family].what} it is drawn on: blank, its own scale; e.g. those of the data`}>
                    <span className="schema-label">{family === 'psf' ? 'pixels of' : 'channels of'}</span>
                    <div className="schema-field">
                      <NumberField value={step} onChange={setStep} min={0} placeholder="its own scale" label={UNITS[family].what} />
                      <span className="schema-unit">{UNITS[family].step}</span>
                    </div>
                  </div>
                  <p className="note">Blank: its own scale. Give those of the data to see it as the model draws it.</p>
                </Panel>
                <NamedBy path={path} />
              </div>
            }
          />
        </div>
      )}
    </section>
  );
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

// The response drawn, of its options (as they settle while typed), and
// what it amounts to
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
  step: string;
  control: PanesControl;
}) {
  const [settled] = useDebouncedValue(options, 250);
  const [settledStep] = useDebouncedValue(step, 400);
  const size = Number(settledStep) > 0 ? Number(settledStep) : null;
  const toggle = (
    <IconButton label={control.folded ? 'Show the panels at the side' : 'Hide the panels at the side'} pressed={!control.folded} onClick={control.toggle}>
      <PanelsIcon />
    </IconButton>
  );
  return (
    <div className="card viewer response">
      <div className="card-head viewer-head">
        <h2>{family === 'psf' ? 'The PSF' : 'The LSF'}</h2>
        {toggle}
      </div>
      {family === 'psf' ? (
        <PsfPreview project={project} options={settled} step={size} />
      ) : (
        <LsfPreview project={project} options={settled} step={size} />
      )}
    </div>
  );
}

function PsfPreview({ project, options, step }: { project: string; options: Options; step: number | null }) {
  const view = usePsfView(project, options, step === null ? null : [step, step]);
  const data = view.data;
  if (data === undefined) return <Waiting error={view.error} />;
  const ny = data.values.length;
  const nx = data.values[0]?.length ?? 0;
  const values = Float32Array.from(data.values.flat());
  const peak = Math.max(...values);
  // (the cuts through its centre, along x and along y)
  const cx = Math.floor(nx / 2);
  const cy = Math.floor(ny / 2);
  const along = (n: number, d: number) => Array.from({ length: n }, (_, i) => (i - Math.floor(n / 2)) * d);
  // (shown where it is above a thousandth of its peak: it is drawn out to
  // its faint wings)
  const rowX = data.values[cy] ?? [];
  const colY = data.values.map((row) => row[cx]);
  const reach = (cut: number[], c: number) => Math.max(1, ...cut.map((v, i) => (v > peak / 1000 ? Math.abs(i - c) : 0)));
  const hx = Math.min(cx, reach(rowX, cx) + 1);
  const hy = Math.min(cy, reach(colY, cy) + 1);
  const h = Math.max(hx, hy);
  const window: [number, number, number, number] = [Math.max(0, cx - h), Math.min(nx - 1, cx + h), Math.max(0, cy - h), Math.min(ny - 1, cy + h)];
  return (
    <div className="response-body">
      {view.error && <p className="error">{view.error.message}</p>}
      <div className="response-map">
        <MapView values={values} shape={[ny, nx]} range={[0, peak || 1]} window={nx > 1 ? window : null} label="The PSF" />
        <ColourBar range={[0, peak || 1]} diverging={false} unit="per pixel" />
      </div>
      <div className="response-profiles">
        <div className="eyebrow">Through its centre, along x</div>
        <SpectrumPlot values={rowX} spectral={along(nx, data.step[0])} unit="arcsec" channel={cx} window={[window[0], window[1]]} onPick={() => {}} />
        <div className="eyebrow">along y</div>
        <SpectrumPlot values={colY} spectral={along(ny, data.step[1])} unit="arcsec" channel={cy} window={[window[2], window[3]]} onPick={() => {}} />
      </div>
      <dl className="facts">
        <dt>FWHM</dt>
        <dd>{data.fwhm === null ? 'none: a point' : `${significant(data.fwhm, 3)}″ (of a circular Gaussian of its beam)`}</dd>
        <dt>beam</dt>
        <dd>{data.beam_area === null ? 'none' : `${significant(data.beam_area, 4)} arcsec²`}</dd>
        <dt>drawn on</dt>
        <dd>
          {nx} × {ny} pixels of {significant(data.step[0], 3)}″
        </dd>
      </dl>
    </div>
  );
}

function LsfPreview({ project, options, step }: { project: string; options: Options; step: number | null }) {
  const view = useLsfView(project, options, step);
  const data = view.data;
  if (data === undefined) return <Waiting error={view.error} />;
  const n = data.values.length;
  const centre = Math.floor(n / 2);
  return (
    <div className="response-body">
      {view.error && <p className="error">{view.error.message}</p>}
      <div className="response-profiles">
        <SpectrumPlot
          values={data.values}
          spectral={data.values.map((_, i) => (i - centre) * data.step)}
          unit="km/s"
          channel={centre}
          onPick={() => {}}
        />
      </div>
      <dl className="facts">
        <dt>FWHM</dt>
        <dd>{data.fwhm === null ? 'narrower than a channel' : `${significant(data.fwhm, 3)} km/s`}</dd>
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
