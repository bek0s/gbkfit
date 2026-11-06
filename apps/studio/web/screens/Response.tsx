// A response of an instrument, a block of the project: a PSF (how it
// spreads the light of a point on the sky) or an LSF (along the spectrum).
// Its options are a form of the schema of the library (see SchemaForm),
// numbers with sliders, and it is drawn as they change, by the server: at
// its own scale, or on pixels (channels) of a size given or of a dataset of
// the project, as a fraction of its peak, linear or logarithmic (its
// wings), with what it amounts to (its widths along its axes, the solid
// angle of its beam, its resolving power) and the blocks that name it. Its edits are written to its file when saved (also
// Ctrl+S).
import { Combobox, Menu, UnstyledButton } from '@mantine/core';
import { useDebouncedValue } from '@mantine/hooks';
import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { useDescription, useLsfView, usePiece, useProject, usePsfView, useReferences, useSchema, useUpdateBlock } from '../api/queries';
import { ColourBar } from '../components/ColourBar';
import { UNSTRETCHED, type Stretch } from '../components/colours';
import { joinPath } from '../components/FilePicker';
import { significant } from '../components/format';
import { CutPlot } from '../components/CutPlot';
import { contourPath, widthAt, type Width } from '../components/contour';
import { LinkIcon, UnlinkIcon } from '../components/icons';
import { MapView } from '../components/MapView';
import { SchemaForm, titleOf, type Defs, type Options } from '../components/SchemaForm';
import { SpectrumPlot } from '../components/SpectrumPlot';
import { Button, NavItem } from '../components/ui/basics';
import { Segments } from '../components/ui/inputs';
import { Dock, DockSection } from '../components/ui/dock';
import { useBlockHead } from '../shell/BlockMenu';
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

// The size of the pixels it is drawn on, along x and y (arcsec), of a PSF;
// the width of its channels (km/s), of an LSF (none: its own scale)
type Step = number[] | null;

// (of a PSF, its x and y; of an LSF, its width)
const pairOf = (step: Step): [number, number] | null => (step === null ? null : [step[0], step[1] ?? step[0]]);
const widthOf = (step: Step): number | null => (step === null ? null : step[0]);

// A size as typed, of its x and y (of an LSF, its width is x): none if
// they are not sizes
function typedStep(x: string, y: string, family: Family): Step {
  const [a, b] = [Number(x), Number(y)];
  if (!(a > 0)) return null;
  if (family === 'lsf') return [a];
  return b > 0 ? [a, b] : null;
}

// The speed of light (km/s), of the resolving power of an LSF (c / FWHM)
const LIGHT = 299792.458;

// What it is drawn on: its own scale, a size typed, or the pixels of an
// observation or of the data of a dataset of the project
type Drawn = { source: 'own' } | { source: 'typed'; x: string; y: string } | { source: 'dataset' | 'observation'; name: string };

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
  const [ofData, setOfData] = useState<Step>(null);
  const saved = piece.data as Options | undefined;
  const options = draft ?? saved;
  // (drawn as they settle while they change)
  const [settled] = useDebouncedValue(options, 120);
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
  const step = drawn.source === 'typed' ? typedStep(drawn.x, drawn.y, family) : drawn.source === 'own' ? null : ofData;
  // (the size of the pixels or channels it is drawn on, as the server drew
  // it (of its own scale, the server's): the query of the preview)
  const psfView = usePsfView(project ?? '', settled ?? {}, pairOf(step), family === 'psf' && project !== null && settled !== undefined);
  const lsfView = useLsfView(project ?? '', settled ?? {}, widthOf(step), family === 'lsf' && project !== null && settled !== undefined);
  const used: Step = family === 'psf' ? (psfView.data?.step ?? null) : lsfView.data ? [lsfView.data.step] : null;
  const head = useBlockHead(project ?? '', folder, name);
  if (project === null || path === null) return null;
  const defs = (schema.data?.$defs ?? {}) as Defs;
  return (
    <section className="screen response-screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">
            {kindOf(folder)}
            {options && typeof options.type === 'string' ? ` · ${titleOf(defs, family, options.type)}` : ''}
          </div>
          {head.title}
        </div>
        <div className="head-actions">
          {dirty && <span className="why">Not saved</span>}
          {update.error && <span className="error">{update.error.message}</span>}
          <Button disabled={!dirty} onClick={() => setDraft(null)}>
            Revert
          </Button>
          <Button kind="primary" title="Save (Ctrl+S)" disabled={!dirty || update.isPending} onClick={save}>
            Save
          </Button>
          {head.menu}
        </div>
      </div>
      {piece.error && <p className="error screen-error">{piece.error.message}</p>}
      {schema.error && <p className="error screen-error">{schema.error.message}</p>}
      {options !== undefined && (
        <div className="rs-body">
          <div className="rs-view">
          <Preview
            family={family}
            project={project}
            options={settled ?? options}
            step={step}
            pixels={<DrawnOn family={family} project={project} path={path} drawn={drawn} onDrawn={setDrawn} used={used} />}
            problem={
              drawn.source === 'dataset' ? (
                <DatasetStep key={drawn.name} family={family} project={project} name={drawn.name} step={ofData} onStep={setOfData} />
              ) : drawn.source === 'observation' ? (
                <ObservationStep key={drawn.name} family={family} project={project} name={drawn.name} step={ofData} onStep={setOfData} />
              ) : null
            }
          />
          </div>
          <Dock label={`The ${family.toUpperCase()}`}>
            <DockSection id={`${family}-options`} title="Options">
              {schema.data ? (
                <SchemaForm defs={defs} def={family} project={project} value={options} onChange={setDraft} starts={STARTS} sliders />
              ) : (
                <p className="note">Reading the options of the library…</p>
              )}
            </DockSection>
            {family === 'psf' ? (
              <PsfProperties project={project} options={settled ?? options} step={step} />
            ) : (
              <LsfProperties project={project} options={settled ?? options} step={step} />
            )}
            <NamedBy path={path} />
          </Dock>
        </div>
      )}
    </section>
  );
}

// What it is drawn on, a control of the view (not an option of the block,
// which has no pixels of its own: an observation draws it on those of its
// data): one box of a fixed width, which shows the size of the pixels (x
// and y, linked unless they differ) or of the channels it is drawn on and
// where it comes from, and takes one typed; its menu chooses fine ones (to
// see its shape) or those of an observation (first those that use it) or
// of a dataset of the project (to see it as the model draws it), each in a
// submenu. Nothing appears or moves as it changes.
function DrawnOn({
  family,
  project,
  path,
  drawn,
  onDrawn,
  used,
}: {
  family: Family;
  project: string;
  path: string;
  drawn: Drawn;
  onDrawn: (drawn: Drawn) => void;
  // (the size it is drawn on, as drawn; none before)
  used: Step;
}) {
  const [opened, setOpened] = useState(false);
  // (x and y kept one, but where those chosen differ)
  const [linked, setLinked] = useState(true);
  const pieces = useProject(project).data?.pieces;
  const references = useReferences(path).data ?? [];
  const users = new Set(references.filter((r) => r.folder === 'observations').map((r) => r.name));
  const observations = pieces?.observations ?? [];
  const units = UNITS[family];
  const psf = family === 'psf';
  const [x, y] =
    drawn.source === 'typed' ? [drawn.x, drawn.y] : used === null ? ['', ''] : [significant(used[0], 3), significant(used[1] ?? used[0], 3)];
  useEffect(() => {
    if (drawn.source !== 'typed' && used !== null && used.length === 2 && used[0] !== used[1]) setLinked(false);
  }, [used?.join()]);
  const source = drawn.source === 'own' ? 'Fine' : drawn.source === 'typed' ? '' : drawn.name;
  const choose = (next: Drawn) => {
    onDrawn(next);
    setOpened(false);
  };
  // (the items of a submenu: those of the blocks given, the one in use
  // marked; none, said)
  const items = (kind: 'dataset' | 'observation', names: string[], note?: (name: string) => string | undefined) =>
    names.length === 0 ? (
      <Menu.Item disabled>None in the project</Menu.Item>
    ) : (
      names.map((name) => (
        <Menu.Item
          key={name}
          onClick={() => choose({ source: kind, name })}
          data-current={(drawn.source === kind && drawn.name === name) || undefined}
          rightSection={note?.(name) && <small className="response-pixels-note">{note(name)}</small>}
        >
          {name}
        </Menu.Item>
      ))
    );
  const unit = psf ? '″' : 'km/s';
  return (
    <Menu opened={opened} onChange={(next) => !next && setOpened(false)} position="bottom-end" offset={4} width="target" classNames={{ dropdown: 'ui-menu' }}>
      <Menu.Target>
        <div className="response-pixels" data-family={family}>
          <span className="response-pixels-label">{psf ? 'Pixel size' : 'Channel width'}</span>
          <SizeField value={x} unit={unit} label={psf ? 'The size of the pixels along x' : units.what} onChange={(next) => onDrawn({ source: 'typed', x: next, y: linked || !psf ? next : y })} />
          {psf && (
            <>
              <UnstyledButton
                className="response-pixels-link"
                aria-label={linked ? 'Let x and y differ' : 'Keep x and y one'}
                title={linked ? 'x and y are one: click to let them differ' : 'x and y differ: click to keep them one'}
                data-linked={linked || undefined}
                onClick={() => {
                  if (!linked && x !== y) onDrawn({ source: 'typed', x, y: x });
                  setLinked(!linked);
                }}
              >
                {linked ? <LinkIcon /> : <UnlinkIcon />}
              </UnstyledButton>
              <SizeField value={y} unit={unit} label="The size of the pixels along y" onChange={(next) => onDrawn({ source: 'typed', x: linked ? next : x, y: next })} />
            </>
          )}
          <span className="response-pixels-source" title={source}>
            {source}
          </span>
          <UnstyledButton className="response-pixels-open" aria-label={`Choose the ${units.of} it is drawn on`} onClick={() => setOpened((o) => !o)}>
            <Combobox.Chevron />
          </UnstyledButton>
        </div>
      </Menu.Target>
      <Menu.Dropdown>
        <Menu.Item onClick={() => choose({ source: 'own' })} data-current={drawn.source === 'own' || undefined} rightSection={<small className="response-pixels-note">To see its shape</small>}>
          Fine
        </Menu.Item>
        <Menu.Sub>
          <Menu.Sub.Target>
            <Menu.Sub.Item data-current={drawn.source === 'observation' || undefined}>From an observation</Menu.Sub.Item>
          </Menu.Sub.Target>
          <Menu.Sub.Dropdown className="ui-menu ui-menu-scroll">
            {items('observation', [...observations.filter((o) => users.has(o)), ...observations.filter((o) => !users.has(o))], (o) => (users.has(o) ? 'Uses it' : undefined))}
          </Menu.Sub.Dropdown>
        </Menu.Sub>
        <Menu.Sub>
          <Menu.Sub.Target>
            <Menu.Sub.Item data-current={drawn.source === 'dataset' || undefined}>From a dataset</Menu.Sub.Item>
          </Menu.Sub.Target>
          <Menu.Sub.Dropdown className="ui-menu ui-menu-scroll">{items('dataset', pieces?.datasets ?? [])}</Menu.Sub.Dropdown>
        </Menu.Sub>
        <Menu.Divider />
        <Menu.Label className="response-pixels-hint">Or type a size in the box</Menu.Label>
      </Menu.Dropdown>
    </Menu>
  );
}

// A size in the box of what it is drawn on, with its unit: typed, or
// stepped by the wheel over it or the arrow keys in it, each step about 5%
// of it (as those of the sliders of widths), to 3 significant digits
function SizeField({ value, unit, label, onChange }: { value: string; unit: string; label: string; onChange: (value: string) => void }) {
  const cell = useRef<HTMLLabelElement>(null);
  // (the value and how it changes as they are now, for the listener of the
  // wheel, added once)
  const latest = useRef({ value, onChange });
  latest.current = { value, onChange };
  const step = (by: number) => {
    const now = Number(latest.current.value);
    if (now > 0) latest.current.onChange(String(Number((now * 10 ** (0.02 * by)).toPrecision(3))));
  };
  // (the wheel moves the value, not the page: a listener that is not
  // passive, which React's is)
  useEffect(() => {
    const element = cell.current;
    if (!element) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      step(event.deltaY < 0 ? 1 : -1);
    };
    element.addEventListener('wheel', onWheel, { passive: false });
    return () => element.removeEventListener('wheel', onWheel);
  }, []);
  return (
    <label ref={cell} className="response-pixels-cell" title={`${label}: type it, or turn the wheel over it`}>
      <input
        className="response-pixels-field"
        value={value}
        onChange={(event) => onChange(event.currentTarget.value)}
        onKeyDown={(event) => {
          if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
          event.preventDefault();
          step(event.key === 'ArrowUp' ? 1 : -1);
        }}
        inputMode="decimal"
        aria-label={label}
      />
      <span className="response-pixels-suffix">{unit}</span>
    </label>
  );
}

// The pixels (PSF) or channels (LSF) of an observation of the project:
// those of its dataset, if it names one (they are those the model is drawn
// on), or else those its observable gives
function ObservationStep({
  family,
  project,
  name,
  step,
  onStep,
}: {
  family: Family;
  project: string;
  name: string;
  step: Step;
  onStep: (step: Step) => void;
}) {
  const observation = usePiece(joinPath(project, `observations/${name}.yaml`));
  const options = observation.data as Options | undefined;
  const data = typeof options?.data === 'string' ? options.data : null;
  const observable = (options?.observable ?? {}) as Options;
  const given = family === 'psf' ? observable.step : (observable.spectral as Options | undefined)?.step;
  // (a PSF of a number, of x and y; or of its x and y)
  const own: Step =
    typeof given === 'number'
      ? family === 'psf'
        ? [Math.abs(given), Math.abs(given)]
        : [Math.abs(given)]
      : family === 'psf' && Array.isArray(given) && given.length === 2 && given.every((n) => typeof n === 'number')
        ? (given as number[]).map(Math.abs)
        : null;
  useEffect(() => {
    if (data === null) onStep(own);
  }, [data, own?.join()]);
  if (data !== null) return <DatasetStep family={family} project={project} name={data} step={step} onStep={onStep} />;
  if (observation.isPending) return <p className="note response-problem">Reading {name}…</p>;
  if (own === null) return <p className="note response-problem">{name} has no data and its observable gives no {UNITS[family].of}.</p>;
  return null;
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
  step: Step;
  onStep: (step: Step) => void;
}) {
  const dataset = usePiece(joinPath(project, `datasets/${name}.yaml`)).data as Options | undefined;
  const item = (dataset?.type === 'pixel_moments' ? (dataset.moment0 ?? dataset.moment1) : dataset) as Options | undefined;
  const data = item?.data as string | { file?: string; hdu?: string | number } | undefined;
  const file = typeof data === 'string' ? data : data?.file;
  const hdu = typeof data === 'object' ? data.hdu : undefined;
  const arrays = useDescription(file ? joinPath(project, file) : null);
  const array = (arrays.data ?? []).find((a) => a.name === hdu || a.index === hdu) ?? arrays.data?.[0];
  // (of a PSF, its x and y; of an LSF, the width of its channels)
  const found: Step = array?.step ? (family === 'psf' ? [Math.abs(array.step[0]), Math.abs(array.step[1])] : array.step.length === 3 ? [Math.abs(array.step[2])] : null) : null;
  useEffect(() => onStep(found), [found?.join()]);
  if (arrays.isPending) return <p className="note response-problem">Reading the header of {name}…</p>;
  if (step === null) {
    return (
      <p className="note response-problem">
        {family === 'lsf' && array && kept(array).length < 3
          ? `${name} has no spectral axis: an LSF is drawn on the channels of a cube.`
          : `The header of ${name} does not give its ${UNITS[family].of}${array?.ambiguous_velocity ? ' (say whether its velocities are radio or optical)' : ''}.`}
      </p>
    );
  }
  return null;
}

function kept(array: { shape: number[] }): number[] {
  return array.shape.filter((n) => n > 1);
}

// The blocks that name it (e.g. observations, in their instruments)
function NamedBy({ path }: { path: string }) {
  const { go } = useStudio();
  const references = useReferences(path).data ?? [];
  return (
    <DockSection id="named-by" title="Used by">
      {references.length === 0 ? (
        <p className="note">No observation uses it yet.</p>
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
    </DockSection>
  );
}

type Scale = 'linear' | 'log';

// The values shown on a scale: their fraction of the peak (the library
// makes a response sum to 1, to keep the flux it spreads, which tells
// little of its shape: a fraction of its peak reads at a glance, its half
// at its FWHM, as a radio beam, of a peak of 1, does), or the logarithm of
// it, down to a floor (where they are too, below it or 0, e.g. beyond the
// extent it is drawn out to)
const FLOOR = -5;

function scaled(values: number[], peak: number, scale: Scale): number[] {
  if (!(peak > 0)) return values;
  if (scale === 'linear') return values.map((v) => v / peak);
  return values.map((v) => (v > 0 ? Math.max(FLOOR, Math.log10(v / peak)) : FLOOR));
}

// The response drawn, of its options
function Preview({
  family,
  project,
  options,
  step,
  pixels,
  problem,
}: {
  family: Family;
  project: string;
  options: Options;
  step: Step;
  // (what it is drawn on, chosen; and why that is not known, if it is not)
  pixels: ReactNode;
  problem: ReactNode;
}) {
  const [scale, setScale] = useState<Scale>('linear');
  // (its controls, in a row over the figure, as wide as it, so that they
  // stay by it however wide the window is)
  const tools = (
    <div className="response-tools">
      {pixels}
      <div className="response-tools-end">
        <Segments
          items={[
            { value: 'linear', label: 'Linear' },
            { value: 'log', label: 'Log' },
          ]}
          value={scale}
          label="The scale of its values"
          onChange={setScale}
        />
      </div>
    </div>
  );
  return (
    <div className="viewer response">
      {family === 'psf' ? (
        <PsfPreview project={project} options={options} step={step} scale={scale} tools={tools} problem={problem} />
      ) : (
        <LsfPreview project={project} options={options} step={step} scale={scale} tools={tools} problem={problem} />
      )}
    </div>
  );
}

// What a preview of each family is given: its options, what it is drawn
// on, its scale, and its controls and why its pixels are not known, if
// they are not, to lay out
type PreviewProps = { project: string; options: Options; step: Step; scale: Scale; tools: ReactNode; problem: ReactNode };

function PsfPreview({ project, options, step, scale, tools, problem }: PreviewProps) {
  const view = usePsfView(project, options, pairOf(step));
  // The pixel the cuts go through: under the pointer, or else that clicked
  // (none: the centre), which is kept as its place on the sky (its offset
  // from the centre, arcsec), so that it stays there on other pixels
  const [hover, setHover] = useState<[number, number] | null>(null);
  const [pin, setPin] = useState<[number, number] | null>(null);
  // (the stretch of the colours, by the right button on the map)
  const [stretch, setStretch] = useState<Stretch>(UNSTRETCHED);
  // (Esc lets go of the pixel clicked)
  useEffect(() => {
    if (pin === null) return;
    const onKey = (event: KeyboardEvent) => event.key === 'Escape' && setPin(null);
    // (on the document: window is the window of the map here)
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [pin]);
  const data = view.data;
  if (data === undefined) return <Waiting error={view.error} tools={tools} problem={problem} />;
  const ny = data.values.length;
  const nx = data.values[0]?.length ?? 0;
  const flat = data.values.flat();
  // (not Math.max(...flat): a large map would overflow the stack)
  const peak = flat.reduce((a, b) => Math.max(a, b), -Infinity);
  const log = scale === 'log';
  const values = Float32Array.from(scaled(flat, peak, scale));
  const range: [number, number] = log ? [FLOOR, 0] : [0, 1];
  const cx = Math.floor(nx / 2);
  const cy = Math.floor(ny / 2);
  // (shown, a square about its centre, out to its farthest pixel in any
  // direction (e.g. the tips of a thin one, tilted) above a thousandth of
  // its peak, or a hundred thousandth on a log scale: its faint wings)
  const floor = peak * (log ? 1e-5 : 1e-3);
  // (a square on the sky, of as many pixels along x and along y as their
  // sizes make it, for pixels that are not square)
  const [dx, dy] = [Math.abs(data.step[0]), Math.abs(data.step[1])];
  let reach = Math.max(dx, dy);
  data.values.forEach((row, j) => row.forEach((v, i) => v > floor && (reach = Math.max(reach, Math.abs(i - cx) * dx, Math.abs(j - cy) * dy))));
  const [hx, hy] = [Math.min(cx, Math.ceil(reach / dx) + 1), Math.min(cy, Math.ceil(reach / dy) + 1)];
  const window: [number, number, number, number] = [Math.max(0, cx - hx), Math.min(nx - 1, cx + hx), Math.max(0, cy - hy), Math.min(ny - 1, cy + hy)];
  // (the map as wide and as tall as the window is on the sky)
  const aspect = ((window[1] - window[0] + 1) * dx) / ((window[3] - window[2] + 1) * dy);
  const inside = (p: [number, number] | null) => p !== null && p[0] < nx && p[1] < ny;
  // (the pixel of the place clicked, on these pixels: the nearest)
  const clamp = (i: number, n: number) => Math.min(n - 1, Math.max(0, i));
  const pinned: [number, number] | null = pin === null ? null : [clamp(cx + Math.round(pin[0] / dx), nx), clamp(cy + Math.round(pin[1] / dy), ny)];
  // (the cuts: through the peak always, through the pixel clicked, which
  // stays, and through that under the pointer, which follows it; the
  // readout is of the latter, or else of the former)
  const shownHover = inside(hover) ? hover : null;
  const [x, y] = shownHover ?? pinned ?? [cx, cy];
  const rowOf = (j: number) => scaled(data.values[j] ?? [], peak, scale);
  const columnOf = (i: number) =>
    scaled(
      data.values.map((row) => row[i]),
      peak,
      scale,
    );
  const through = (p: [number, number] | null, kind: string) => (p === null ? [] : [{ p, kind }]);
  const cuts = [{ p: [cx, cy] as [number, number], kind: 'peak' }, ...through(pinned, 'pinned'), ...through(shownHover, 'hover')];
  const linesX = cuts.map(({ p, kind }) => ({ values: rowOf(p[1]), kind }));
  const linesY = cuts.map(({ p, kind }) => ({ values: columnOf(p[0]), kind }));
  const marksX = cuts.filter(({ kind }) => kind !== 'peak').map(({ p, kind }) => ({ at: p[0], kind }));
  const marksY = cuts.filter(({ kind }) => kind !== 'peak').map(({ p, kind }) => ({ at: p[1], kind }));
  // (the offset of a pixel from the centre, arcsec, and its value, in the
  // units of the colour bar: each to a fixed number of decimals, so that
  // they keep still as the pointer moves; a 0 has a blank for its sign)
  const offset = (i: number, c: number, d: number) => {
    const v = (i - c) * d;
    const decimals = Math.max(0, 1 - Math.floor(Math.log10(Math.abs(d)) + 1e-9));
    return `${v > 0 ? '+' : v < 0 ? '−' : '\u2007'}${Math.abs(v).toFixed(decimals)}″`;
  };
  const fraction = peak > 0 ? (data.values[y]?.[x] ?? 0) / peak : 0;
  // (the contour at half the peak, on the map, and the width at that level
  // of the cuts through the peak, in their units: its FWHM)
  const half = contourPath(data.values, peak / 2);
  const level = log ? Math.log10(0.5) : 0.5;
  const widthOn = (values: number[]): Width | null => {
    const at = widthAt(values, peak / 2);
    return at && { level, from: at[0], to: at[1] };
  };
  const shown = log ? (fraction > 10 ** FLOOR ? Math.log10(fraction).toFixed(2).replace('-', '−') : `<${FLOOR}`.replace('-', '−')) : fraction.toFixed(3);
  return (
    <div className="response-body">
      {view.error && <p className="error">{view.error.message}</p>}
      {problem}
      <div className="psf-grid" style={{ '--aspect': aspect } as CSSProperties}>
        <div className="psf-tools">{tools}</div>
        <dl className="psf-at" data-idle={(!inside(hover) && pinned === null) || undefined} title="The pixel under the pointer, or else the one clicked (click it again, or Esc, to let it go)">
          <dt>x</dt>
          <dd>{offset(x, cx, data.step[0])}</dd>
          <dt>y</dt>
          <dd>{offset(y, cy, data.step[1])}</dd>
          <dt>Value</dt>
          <dd>{shown}</dd>
        </dl>
        <div className="psf-xcut">
          <CutPlot lines={linesX} window={[window[0], window[1]]} limits={range} marks={marksX} width={widthOn(data.values[cy] ?? [])} label="The cuts along x" />
        </div>
        <div className="psf-map">
          <MapView
            values={values}
            shape={[ny, nx]}
            range={range}
            window={nx > 1 ? window : null}
            mark={pinned ?? undefined}
            contour={half}
            stretch={stretch}
            onStretch={setStretch}
            onHover={setHover}
            onPick={(px, py) => setPin(pinned && pinned[0] === px && pinned[1] === py ? null : [(px - cx) * dx, (py - cy) * dy])}
            label="The PSF"
          />
          {half && <FwhmKey title="The contour at half the peak, and the widths at half the peak of the cuts through it: its FWHM" />}
        </div>
        <div className="psf-ycut">
          <CutPlot lines={linesY} window={[window[2], window[3]]} limits={range} marks={marksY} width={widthOn(data.values.map((row) => row[cx]))} vertical label="The cuts along y" />
        </div>
        <div className="psf-bar">
          <ColourBar
            range={range}
            diverging={false}
            horizontal
            unit={log ? 'Log₁₀ of the fraction of the peak' : 'Fraction of the peak'}
            stretch={stretch}
            onReset={() => setStretch(UNSTRETCHED)}
          />
        </div>
      </div>
    </div>
  );
}

// The key of the FWHM drawn on a preview, in a corner of it
function FwhmKey({ title }: { title: string }) {
  return (
    <div className="fwhm-key" title={title}>
      <span className="fwhm-key-line" />
      FWHM
    </div>
  );
}

// The resolution of a PSF: its widths along its axes, and the area of its
// beam (what each is, on its label)
function PsfProperties({ project, options, step }: { project: string; options: Options; step: Step }) {
  const data = usePsfView(project, options, pairOf(step)).data;
  // (none from a server older than its measure: that of its beam)
  const axes = data === undefined ? null : ((data.axes ?? (data.fwhm === null ? null : [data.fwhm, data.fwhm, null])) as [number, number, number | null] | null);
  return (
    <DockSection id="psf-resolution" title="Resolution">
      {data === undefined ? (
        <p className="note">Measuring…</p>
      ) : (
        <dl className="facts">
            <dt title="The full width at half maximum, along the major and the minor axes">FWHM</dt>
            <dd>
              {axes === null ? (
                <span className="word">None (a point)</span>
              ) : axes[2] === null ? (
                <>
                  {significant(axes[0], 3)}″<span className="word">, round</span>
                </>
              ) : (
                `${significant(axes[0], 3)}″ × ${significant(axes[1], 3)}″`
              )}
            </dd>
            {axes && axes[2] !== null && (
              <>
                <dt title="The position angle of the major axis">Major axis</dt>
                <dd>
                  {significant(axes[2], 3)}° <span className="word">(east of north)</span>
                </dd>
              </>
            )}
            <dt title="The area under the PSF when its peak is 1 (its solid angle): it converts data in Jy/beam, as radio data are, to flux per arcsec²">Beam area</dt>
            <dd>{data.beam_area === null ? 'None' : `${significant(data.beam_area, 4)} arcsec²`}</dd>
          </dl>
      )}
    </DockSection>
  );
}

function LsfPreview({ project, options, step, scale, tools, problem }: PreviewProps) {
  const view = useLsfView(project, options, widthOf(step));
  const data = view.data;
  if (data === undefined) return <Waiting error={view.error} tools={tools} problem={problem} />;
  const n = data.values.length;
  const centre = Math.floor(n / 2);
  const peak = Math.max(...data.values);
  const log = scale === 'log';
  // (its width at half its peak: its FWHM)
  const width = widthAt(data.values, peak / 2);
  return (
    <div className="response-body">
      {view.error && <p className="error">{view.error.message}</p>}
      {problem}
      <div className="response-profiles">
        {tools}
        <SpectrumPlot
          values={scaled(data.values, peak, scale)}
          spectral={data.values.map((_, i) => (i - centre) * data.step)}
          unit="km/s"
          valueUnit={log ? 'Log₁₀ of the fraction of the peak' : 'Fraction of the peak'}
          channel={centre}
          limits={log ? [FLOOR, 0] : null}
          zero={!log}
          width={width && { level: log ? Math.log10(0.5) : 0.5, from: width[0], to: width[1] }}
          onPick={() => {}}
        >
          {width && <FwhmKey title="The width at half the peak: its FWHM" />}
        </SpectrumPlot>
      </div>
    </div>
  );
}

// The resolution of an LSF: its width, and the resolving power of a
// spectrograph of it (what each is, on its label)
function LsfProperties({ project, options, step }: { project: string; options: Options; step: Step }) {
  const data = useLsfView(project, options, widthOf(step)).data;
  return (
    <DockSection id="lsf-resolution" title="Resolution">
      {data === undefined ? (
        <p className="note">Measuring…</p>
      ) : (
        <dl className="facts">
          <dt title="The full width at half maximum">FWHM</dt>
          <dd>{data.fwhm === null ? <span className="word">Narrower than a channel</span> : `${significant(data.fwhm, 3)} km/s`}</dd>
          {data.fwhm !== null && (
            <>
              <dt title="The σ of a Gaussian of the same FWHM">σ</dt>
              <dd>{significant(data.fwhm / (2 * Math.sqrt(2 * Math.log(2))), 3)} km/s</dd>
              <dt title="The resolving power of a spectrograph of this LSF: λ/Δλ = c/FWHM">R</dt>
              <dd>{Math.round(LIGHT / data.fwhm).toLocaleString('en')}</dd>
            </>
          )}
        </dl>
      )}
    </DockSection>
  );
}

function Waiting({ error, tools, problem }: { error: Error | null; tools: ReactNode; problem: ReactNode }) {
  return (
    <div className="response-body">
      {problem}
      {tools}
      {error ? <p className="error">{error.message}</p> : <p className="note">Drawing…</p>}
    </div>
  );
}
