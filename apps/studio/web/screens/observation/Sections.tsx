// The sections of the dock of an observation on the sky: its data, its
// observable, its instrument, its line of sight, the light it sees, and
// its other options. The common choices are made by hand; the options of
// each choice are a form of the schema (see SchemaForm). How its data are
// compared with its model, its likelihood, is not of the observation but
// of the fits of it (the section likelihoods of a setup).
import { useState, type ReactNode } from 'react';
import { membersOf, propertiesOf, SchemaForm, typeOf, type Defs, type Options, type Schema } from '../../components/SchemaForm';
import { Button } from '../../components/ui/basics';
import { Choice, NumberField, Segments } from '../../components/ui/inputs';
import { dataHaveChannels, gridOptions, isObject, isSpectral, objectAt, observableMember, withKey, type Ellipse } from './model';

interface Common {
  defs: Defs;
  project: string;
  options: Options;
  onChange: (options: Options) => void;
  blocks: Record<string, string[]>;
  openBlock: (folder: string, name: string) => void;
}

// ---------------------------------------------------------------------------
// The data

export function DataFields({ defs, options, onChange, blocks, openBlock, datasetType, titleOf }: Common & { datasetType: string | undefined; titleOf: (type: unknown) => string }) {
  const datasets = blocks.datasets ?? [];
  const value = typeof options.data === 'string' ? options.data : options.data ? '(in place)' : '';
  return (
    <div className="ob-form">
      <Row label="Dataset">
        <Choice
          items={[
            { value: '', label: 'None: a model' },
            ...datasets.map((name) => ({ value: name, label: name })),
            ...(value === '(in place)' ? [{ value, label: 'Given in place (in the file)', disabled: true }] : []),
          ]}
          value={value}
          label="The dataset of the observation"
          onChange={(name) => {
            // (the data give the grid: the observable's own goes)
            const observable = objectAt(options, 'observable');
            const pruned = name ? Object.fromEntries(Object.entries(observable).filter(([key]) => !gridOptions(defs, observable).includes(key))) : observable;
            onChange({ ...withKey(options, 'data', name || undefined), observable: pruned });
          }}
        />
      </Row>
      {typeof options.data === 'string' && datasets.includes(options.data) && (
        <p className="ob-text">
          {datasetType ? `${titleOf(datasetType)}. ` : ''}
          <button className="ob-link" onClick={() => openBlock('datasets', options.data as string)}>
            Open the dataset →
          </button>
        </p>
      )}
      {options.data == null && <p className="ob-text">Without data, the observation makes a model on a grid of its own (see Observable), or mock data.</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// The observable

// The options of the observable set by hand: its grid on the sky and its
// channels
const GRID_KEYS = ['size', 'step', 'spectral'];

export function ObservableFields({ defs, project, options, onChange, blocks, datasetType, titleOf }: Common & { datasetType: string | undefined; titleOf: (type: unknown) => string }) {
  const observable = objectAt(options, 'observable');
  const withData = options.data != null;
  // (the channels of data of a spectral axis are those of the model; moment
  // maps have none, and the model needs its own)
  const fromData = withData && dataHaveChannels(datasetType);
  const extra = withData ? gridOptions(defs, observable) : [];
  const types = membersOf(defs.observable ?? {}, defs)
    .map((m) => typeOf(m))
    .filter((type, i, all): type is string => type !== undefined && all.indexOf(type) === i);
  const member = observableMember(defs, observable.type, withData);
  const properties = member ? propertiesOf(member) : {};
  const set = (next: Options) => onChange({ ...options, observable: next });
  const setKey = (key: string, value: unknown) => set(withKey(observable, key, value));
  const spectral = objectAt(observable, 'spectral');
  const setSpectral = (key: string, value: unknown) => setKey('spectral', withKey(spectral, key, value));
  const others = Object.fromEntries(Object.entries(properties).filter(([key]) => key !== 'type' && !GRID_KEYS.includes(key)));
  const size = pair(observable.size);
  const step = pair(observable.step);
  return (
    <div className="ob-form">
      <Row label="Type">
        <Choice
          items={types.map((type) => ({ value: type, label: titleOf(type) }))}
          value={String(observable.type ?? '')}
          label="The type of the observable"
          onChange={(type) => set({ type, ...pick(observable, GRID_KEYS) })}
        />
      </Row>
      {withData && datasetType !== undefined && datasetType !== observable.type && (
        <div className="ob-warn">
          The data are {titleOf(datasetType)}.{' '}
          <Button kind="small" onClick={() => set({ type: datasetType })}>
            Make it {titleOf(datasetType)}
          </Button>
        </div>
      )}
      {extra.length > 0 && (
        <div className="ob-warn">
          The data give the grid, but the observable has its own {extra.join(', ')}.{' '}
          <Button kind="small" onClick={() => set(Object.fromEntries(Object.entries(observable).filter(([key]) => !extra.includes(key))))}>
            Remove them
          </Button>
        </div>
      )}
      {'size' in properties && (
        <>
          <Row label="Pixels" unit="px">
            <span className="ob-pair">
              <NumberOption value={size[0]} placeholder="nx" label="The number of pixels along x" onChange={(nx) => setKey('size', joined(nx, size[1]))} />
              <span>×</span>
              <NumberOption value={size[1]} placeholder="ny" label="The number of pixels along y" onChange={(ny) => setKey('size', joined(size[0], ny))} />
            </span>
          </Row>
          <Row label="Pixel size" unit="″">
            <span className="ob-pair">
              <NumberOption value={step[0]} placeholder="1" label="The size of a pixel along x" onChange={(sx) => setKey('step', joined(sx, step[1]))} />
              <span>×</span>
              <NumberOption value={step[1]} placeholder="1" label="The size of a pixel along y" onChange={(sy) => setKey('step', joined(step[0], sy))} />
            </span>
          </Row>
        </>
      )}
      {'spectral' in properties && isSpectral(observable.type) && (
        <>
          <div className="ob-sub">{withData ? 'Channels of the model' : 'Channels'}</div>
          <Row label="Channels">
            <NumberOption value={spectral.size} placeholder={fromData ? 'from the data' : 'needed'} label="The number of channels" onChange={(value) => setSpectral('size', value)} />
          </Row>
          <Row label="Width" unit="km/s">
            <NumberOption value={spectral.step} placeholder={fromData ? 'from the data' : 'needed'} label="The width of a channel" onChange={(value) => setSpectral('step', value)} />
          </Row>
          <Row label="Centre" unit="km/s">
            <NumberOption value={spectral.rval} placeholder="0" label="The velocity of the middle channel" onChange={(value) => setSpectral('rval', value)} />
          </Row>
          <Row label="Velocities">
            <Segments
              items={[
                { value: 'radio', label: 'Radio' },
                { value: 'optical', label: 'Optical' },
                { value: 'relativistic', label: 'Relativistic' },
              ]}
              value={typeof spectral.convention === 'string' ? (spectral.convention as 'radio' | 'optical' | 'relativistic') : null}
              label="The convention of the velocities"
              onChange={(convention) => setSpectral('convention', convention)}
            />
          </Row>
        </>
      )}
      {Object.keys(others).length > 0 && (
        <More title="More of the observable" count={Object.keys(pick(observable, Object.keys(others))).length}>
          <SchemaForm
            defs={defs}
            node={{ type: 'object', properties: others, required: [] }}
            project={project}
            value={pick(observable, Object.keys(others))}
            onChange={(next) => set({ ...pick(observable, ['type', ...GRID_KEYS]), ...next })}
            blocks={blocks}
          />
        </More>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// The instrument

export function InstrumentFields({
  defs,
  project,
  options,
  onChange,
  blocks,
  openBlock,
  beam,
  psf,
  titleOf,
}: Common & {
  // The beam of the header of the data (none: none), and the PSF as the
  // server measures it
  beam: Ellipse | null;
  psf: Ellipse | null;
  titleOf: (family: string, type: unknown) => string;
}) {
  const instrument = objectAt(options, 'instrument');
  const set = (key: string, value: unknown) => {
    const next = withKey(instrument, key, value);
    onChange(withKey(options, 'instrument', Object.keys(next).length ? next : undefined));
  };
  const beamPsf = beam && { type: 'gauss', bmaj: round(beam.major), bmin: round(beam.minor), bpa: round(beam.angle) };
  // (a PSF within 5% of the beam is that of the beam)
  const psfIsBeam = beam !== null && psf !== null && Math.abs(psf.major / beam.major - 1) < 0.05 && Math.abs(psf.minor / beam.minor - 1) < 0.05;
  return (
    <div className="ob-form">
      <Response family="psf" folder="psfs" title="PSF" value={instrument.psf} onChange={(value) => set('psf', value)} defs={defs} project={project} blocks={blocks} openBlock={openBlock} titleOf={titleOf} />
      {beamPsf && !psfIsBeam && (
        <div className="ob-hint">
          <span>
            The data have a beam of {round(beam.major)}″ × {round(beam.minor)}″ at {round(beam.angle)}°.
          </span>
          <Button kind="small" onClick={() => set('psf', beamPsf)}>
            Use it as the PSF
          </Button>
        </div>
      )}
      {isSpectral(objectAt(options, 'observable').type) && (
        <Response family="lsf" folder="lsfs" title="LSF" value={instrument.lsf} onChange={(value) => set('lsf', value)} defs={defs} project={project} blocks={blocks} openBlock={openBlock} titleOf={titleOf} />
      )}
      <More title="Primary beam" count={instrument.primary_beam == null ? 0 : 1}>
        <SchemaForm
          defs={defs}
          node={{ type: 'object', properties: { primary_beam: propertiesOf(defs.Instrument ?? {}).primary_beam ?? {} }, required: [] }}
          project={project}
          value={pick(instrument, ['primary_beam'])}
          onChange={(next) => set('primary_beam', next.primary_beam)}
          blocks={blocks}
        />
      </More>
    </div>
  );
}

// A PSF or an LSF: none, one of the project (by name), or given in place
// (of a type, with its options)
function Response({
  family,
  folder,
  title,
  value,
  onChange,
  defs,
  project,
  blocks,
  openBlock,
  titleOf,
}: {
  family: 'psf' | 'lsf';
  folder: string;
  title: string;
  value: unknown;
  onChange: (value: unknown) => void;
  defs: Defs;
  project: string;
  blocks: Record<string, string[]>;
  openBlock: (folder: string, name: string) => void;
  titleOf: (family: string, type: unknown) => string;
}) {
  const named = blocks[folder] ?? [];
  const members = membersOf(defs[family] ?? {}, defs).filter((m, i, all) => all.findIndex((o) => typeOf(o) === typeOf(m)) === i);
  const types = members.map((m) => typeOf(m)).filter((type): type is string => type !== undefined && type !== 'point');
  const chosen = value == null ? 'none' : typeof value === 'string' ? `block:${value}` : `type:${String((value as Options).type)}`;
  const member = isObject(value) ? membersOf(defs[family] ?? {}, defs).find((m) => typeOf(m) === value.type && formFits(m, value)) : undefined;
  const properties = member ? Object.fromEntries(Object.entries(propertiesOf(member)).filter(([key]) => key !== 'type')) : {};
  return (
    <div className="ob-response">
      <Row label={title}>
        <Choice
          items={[
            { value: 'none', label: family === 'psf' ? 'None: a perfect instrument' : 'None: a perfect spectrograph' },
            ...named.map((name) => ({ value: `block:${name}`, label: name, description: `the ${title} ${name} of the project` })),
            ...types.map((type) => ({ value: `type:${type}`, label: titleOf(family, type), description: 'given here' })),
          ]}
          value={chosen}
          label={`The ${title} of the instrument`}
          onChange={(next) => {
            if (next === 'none') onChange(undefined);
            else if (next.startsWith('block:')) onChange(next.slice(6));
            else onChange({ type: next.slice(5) });
          }}
        />
      </Row>
      {typeof value === 'string' && named.includes(value) && (
        <p className="ob-text">
          <button className="ob-link" onClick={() => openBlock(folder, value)}>
            Open the {title} {value} →
          </button>
        </p>
      )}
      {isObject(value) && Object.keys(properties).length > 0 && (
        <div className="ob-nested">
          <SchemaForm
            defs={defs}
            node={{ type: 'object', properties, required: (member?.required as string[] | undefined)?.filter((key) => key !== 'type') ?? [] }}
            project={project}
            value={Object.fromEntries(Object.entries(value).filter(([key]) => key !== 'type'))}
            onChange={(next) => onChange({ type: value.type, ...next })}
            blocks={blocks}
          />
        </div>
      )}
    </div>
  );
}

// Whether options fit a form of a member (one of a type has several, e.g.
// a beam by its axes or by a file): its required options are there
function formFits(member: Schema, value: Options): boolean {
  const required = (member.required as string[] | undefined) ?? [];
  return required.every((key) => key === 'type' || key in value) || Object.keys(value).length <= 1;
}

// ---------------------------------------------------------------------------
// The line of sight

export function LineOfSightFields({ defs, project, options, onChange, blocks, titleOf }: Common & { titleOf: (family: string, type: unknown) => string }) {
  const sight = objectAt(options, 'line_of_sight');
  // (none chosen: the observation needs one; the library has no default)
  const type = typeof sight.type === 'string' ? sight.type : null;
  const members = membersOf(defs.line_of_sight ?? {}, defs);
  const member = members.find((m) => typeOf(m) === type);
  const properties = member ? Object.fromEntries(Object.entries(propertiesOf(member)).filter(([key]) => key !== 'type')) : {};
  const set = (next: Options) => onChange({ ...options, line_of_sight: next });
  return (
    <div className="ob-form">
      <Segments
        items={members.map((m) => ({ value: typeOf(m) ?? '', label: titleOf('line_of_sight', typeOf(m)).replace(/ \(.*\)$/, '') }))}
        value={type}
        label="The line of sight"
        onChange={(next) => set({ type: next })}
      />
      {typeof member?.description === 'string' && <p className="ob-text">{firstSentence(member.description)}</p>}
      {Object.keys(properties).length > 0 && (
        <More title="Its depth" count={Object.keys(sight).length - 1}>
          <SchemaForm
            defs={defs}
            node={{ type: 'object', properties, required: [] }}
            project={project}
            value={Object.fromEntries(Object.entries(sight).filter(([key]) => key !== 'type'))}
            onChange={(next) => set({ type: type ?? 'flat', ...next })}
            blocks={blocks}
          />
        </More>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// The light it sees: of which galaxies, components and lines, and in which
// band (lists of names; all by default)

export function LightFields({ options, onChange }: Common) {
  const names = (key: string, label: string, placeholder: string, text: string) => {
    const value = options[key];
    const shown = Array.isArray(value) ? value.join(', ') : '';
    return (
      <Row label={label}>
        <NamesField
          value={shown}
          placeholder={placeholder}
          label={text}
          onChange={(typed) => {
            const list = typed
              .split(',')
              .map((name) => name.trim())
              .filter(Boolean);
            onChange(withKey(options, key, list.length ? list : undefined));
          }}
        />
      </Row>
    );
  };
  const band = options.band;
  return (
    <div className="ob-form">
      {names('galaxies', 'Galaxies', 'all of the setup', 'The galaxies it sees, by name (needed when the setup has several)')}
      {names('components', 'Components', 'all that emit', 'The components whose light it sees, by name')}
      {names('lines', 'Lines', 'all', 'The emission lines it sees, by name (e.g. Halpha)')}
      <Row label="Band">
        {band == null || typeof band === 'string' ? (
          <NamesField value={band ?? ''} placeholder="none" label="The band of an image, by its name in the catalogue" onChange={(typed) => onChange(withKey(options, 'band', typed.trim() || undefined))} />
        ) : (
          <span className="ob-text">Given in place (in the file)</span>
        )}
      </Row>
      <p className="ob-text">Names are separated by commas; blank, the observation sees all of them.</p>
    </div>
  );
}

// A text of names as typed, kept as the user types (the list is made from
// it as the field is left)
function NamesField({ value, placeholder, label, onChange }: { value: string; placeholder: string; label: string; onChange: (text: string) => void }) {
  return <input key={value} className="ob-names" defaultValue={value} placeholder={placeholder} aria-label={label} title={label} onBlur={(event) => onChange(event.target.value)} />;
}

// ---------------------------------------------------------------------------
// Its other options (by the schema), and the offsets of its frame

const OFFSETS = [
  { key: 'position', title: 'Position', text: 'A shift of its frame on the sky, fitted' },
  { key: 'velocity', title: 'Velocity', text: 'A shift of its velocities, fitted' },
];

export function OptionFields({ defs, project, options, onChange, blocks, member, keys }: Common & { member: Schema; keys: string[] }) {
  const formKeys = keys.filter((key) => key !== 'offsets');
  const properties = Object.fromEntries(Object.entries(propertiesOf(member)).filter(([key]) => formKeys.includes(key)));
  const offsets = objectAt(options, 'offsets');
  return (
    <div className="ob-form ob-schema">
      {keys.includes('offsets') && (
        <Row label="Fit offsets">
          <span className="ob-checks-row">
            {OFFSETS.map(({ key, title, text }) => (
              <label key={key} title={text}>
                <input
                  type="checkbox"
                  checked={offsets[key] === true}
                  onChange={(event) => {
                    const next = withKey(offsets, key, event.target.checked || undefined);
                    onChange(withKey(options, 'offsets', Object.keys(next).length ? next : undefined));
                  }}
                />
                {title}
              </label>
            ))}
          </span>
        </Row>
      )}
      <SchemaForm
        defs={defs}
        node={{ type: 'object', properties, required: [] }}
        project={project}
        value={pick(options, formKeys)}
        onChange={(next) => onChange({ ...Object.fromEntries(Object.entries(options).filter(([key]) => !formKeys.includes(key))), ...next })}
        blocks={blocks}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// The pieces of the forms

function Row({ label, unit, children }: { label: string; unit?: string; children: ReactNode }) {
  return (
    <div className="ob-row">
      <span className="ob-label">{label}</span>
      <div className="ob-field">{children}</div>
      {unit !== undefined && <span className="ob-unit">{unit}</span>}
    </div>
  );
}

// A form folded under its title, which says how many of its options are
// set
function More({ title, count, children }: { title: string; count: number; children: ReactNode }) {
  return (
    <details className="ob-more">
      <summary>
        {title}
        {count > 0 && <small>{count} set</small>}
      </summary>
      <div className="ob-more-body">{children}</div>
    </details>
  );
}

function pick(options: Options, keys: string[]): Options {
  return Object.fromEntries(Object.entries(options).filter(([key]) => keys.includes(key)));
}

// A number of an option: the field keeps the text as it is typed (e.g.
// '-', on the way to a negative number), and the option takes it once it
// is a number (blank: none); left, the field shows the option
function NumberOption({ value, onChange, placeholder, label }: { value: unknown; onChange: (value: number | undefined) => void; placeholder: string; label: string }) {
  const [typing, setTyping] = useState<string | null>(null);
  return (
    <span className="ob-number" onBlur={() => setTyping(null)}>
      <NumberField
        value={typing ?? (typeof value === 'number' ? String(value) : '')}
        placeholder={placeholder}
        label={label}
        onChange={(text) => {
          setTyping(text);
          if (text.trim() === '' || number(text) !== undefined) onChange(number(text));
        }}
      />
    </span>
  );
}

// A pair of values of an option (a value for both, or a list of two)
function pair(value: unknown): [unknown, unknown] {
  return Array.isArray(value) ? [value[0], value[1]] : [value, value];
}

// The pair of two values as an option (a list of two; one given: both)
function joined(a: unknown, b: unknown): number[] | undefined {
  const [x, y] = [typeof a === 'number' ? a : undefined, typeof b === 'number' ? b : undefined];
  if (x === undefined && y === undefined) return undefined;
  return [x ?? y!, y ?? x!];
}

function number(text: string): number | undefined {
  const value = text.trim() === '' ? NaN : Number(text);
  return Number.isFinite(value) ? value : undefined;
}

function round(value: number): number {
  return Number(value.toPrecision(4));
}

function firstSentence(text: string): string {
  const end = text.indexOf('. ');
  return end < 0 ? text : text.slice(0, end + 1);
}
