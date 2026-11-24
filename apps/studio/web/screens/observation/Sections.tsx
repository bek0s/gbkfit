// The sections of the dock of an observation on the sky, in the order of
// the path of the light: what it sees (galaxies, components, lines or a
// band), its instrument (primary beam, PSF, LSF), what it records (its kind
// of data and its own sampling), the noise of its mock data, and how
// accurately and fast its model is made. The common choices are made by
// hand; the options of each choice are a form of the schema (see
// SchemaForm), titled by the library. The data it is fitted to, and how
// they are compared with its model (its likelihood), are not of the
// observation but of the fits of it (the sections datasets and
// likelihoods of a setup).
import { useState, type ReactNode } from 'react';
import { useLines } from '../../api/queries';
import { membersOf, propertiesOf, SchemaForm, typeOf, type Defs, type Options, type Schema } from '../../components/SchemaForm';
import { Choice, NumberField, Segments } from '../../components/ui/inputs';
import { DropMenu } from '../../components/ui/overlays';
import { ACCURACY_KEYS, isObject, isRecorded, isSpectral, objectAt, observationMember, skyKeys, withKey, type Ellipse } from './model';

interface Common {
  defs: Defs;
  project: string;
  options: Options;
  onChange: (options: Options) => void;
  blocks: Record<string, string[]>;
  openBlock: (folder: string, name: string) => void;
}

// ---------------------------------------------------------------------------
// How accurately and fast its model is made

export function AccuracyFields(props: Common & { member: Schema }) {
  return <OptionFields {...props} keys={ACCURACY_KEYS} />;
}

// ---------------------------------------------------------------------------
// The kind of data: the type of the observation, and the options of its kind

// The options of the kind set by hand: its grid on the sky and its channels
const GRID_KEYS = ['size', 'step', 'spectral'];

export function RecordsFields({ defs, project, options, onChange, blocks, titleOf }: Common & { titleOf: (type: unknown) => string }) {
  const sky = skyKeys(defs);
  const types = membersOf(defs.observation ?? {}, defs)
    .map((m) => typeOf(m))
    .filter((type, i, all): type is string => type !== undefined && type !== 'profile' && all.indexOf(type) === i);
  const member = observationMember(defs, options.type);
  // (the options of its kind: those of its type that every observation on
  // the sky does not have)
  const properties = Object.fromEntries(Object.entries(member ? propertiesOf(member) : {}).filter(([key]) => !sky.includes(key)));
  const setKey = (key: string, value: unknown) => onChange(withKey(options, key, value));
  // Another type: the options of every observation on the sky, and the
  // grid and the channels, stay
  const setType = (type: string) => {
    const kept = [...sky, ...GRID_KEYS];
    onChange({ ...Object.fromEntries(Object.entries(options).filter(([key]) => kept.includes(key))), type });
  };
  const spectral = objectAt(options, 'spectral');
  const setSpectral = (key: string, value: unknown) => setKey('spectral', withKey(spectral, key, value));
  const others = Object.fromEntries(Object.entries(properties).filter(([key]) => key !== 'type' && !GRID_KEYS.includes(key)));
  const size = pair(options.size);
  const step = pair(options.step);
  return (
    <div className="ob-form">
      <Row label="Type">
        <Choice items={types.map((type) => ({ value: type, label: titleOf(type) }))} value={String(options.type ?? '')} label="The kind of data of the observation" onChange={setType} />
      </Row>
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
      {'spectral' in properties && isSpectral(options.type) && (
        <>
          <div className="ob-sub">Channels</div>
          <Row label="Channels">
            <NumberOption value={spectral.size} placeholder="needed" label="The number of channels" onChange={(value) => setSpectral('size', value)} />
          </Row>
          <Row label="Width" unit="km/s">
            <NumberOption value={spectral.step} placeholder="needed" label="The width of a channel" onChange={(value) => setSpectral('step', value)} />
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
      {member && 'unit' in propertiesOf(member) && <OptionFields defs={defs} project={project} options={options} onChange={onChange} blocks={blocks} openBlock={() => {}} member={member} keys={['unit']} />}
      {Object.keys(others).length > 0 && (
        <More title={`More of ${titleOf(options.type).toLowerCase()}`} count={Object.keys(pick(options, Object.keys(others))).length}>
          <SchemaForm
            defs={defs}
            node={{ type: 'object', properties: others, required: [] }}
            project={project}
            value={pick(options, Object.keys(others))}
            onChange={(next) => onChange({ ...Object.fromEntries(Object.entries(options).filter(([key]) => !(key in others))), ...next })}
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
  psf,
  lsfWidth,
  channel,
  titleOf,
}: Common & {
  // The PSF as the server measures it, and the FWHM of the LSF and the
  // width of a channel
  psf: Ellipse | null;
  lsfWidth: number | null;
  channel: number | null;
  titleOf: (family: string, type: unknown) => string;
}) {
  const instrument = objectAt(options, 'instrument');
  const set = (key: string, value: unknown) => {
    const next = withKey(instrument, key, value);
    onChange(withKey(options, 'instrument', Object.keys(next).length ? next : undefined));
  };
  return (
    <div className="ob-form">
      <Response family="primary_beam" folder="" title="Primary beam" none="None: no attenuation" value={instrument.primary_beam} onChange={(value) => set('primary_beam', value)} defs={defs} project={project} blocks={blocks} openBlock={openBlock} titleOf={titleOf} />
      <Response family="psf" folder="psfs" title="PSF" none="None: a perfect instrument" value={instrument.psf} onChange={(value) => set('psf', value)} defs={defs} project={project} blocks={blocks} openBlock={openBlock} titleOf={titleOf} />
      {psf && (
        <p className="ob-measure">
          FWHM {round(psf.major)}″{Math.abs(psf.minor - psf.major) > 1e-6 * psf.major ? ` × ${round(psf.minor)}″ at ${round(psf.angle)}°` : ''}
        </p>
      )}
      {isSpectral(options.type) && (
        <>
          <Response family="lsf" folder="lsfs" title="LSF" none="None: a perfect spectrograph" value={instrument.lsf} onChange={(value) => set('lsf', value)} defs={defs} project={project} blocks={blocks} openBlock={openBlock} titleOf={titleOf} />
          {lsfWidth !== null && instrument.lsf != null && (
            <p className="ob-measure">
              FWHM {round(lsfWidth)} km/s{channel ? ` · ${round(lsfWidth / channel)} channels` : ''}
            </p>
          )}
        </>
      )}
    </div>
  );
}

// A PSF, an LSF or a noise: none, one of the project (by name), or given in
// place (of a type, with its options)
export function Response({
  family,
  folder,
  title,
  none,
  value,
  onChange,
  defs,
  project,
  blocks,
  openBlock,
  titleOf,
}: {
  family: string;
  folder: string;
  title: string;
  // What none is, in words (e.g. 'None: a perfect instrument')
  none: string;
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
            { value: 'none', label: none },
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
// What it sees: which galaxies (by their names in the setup) and
// components, and which lines (of a spectral observation) or band (of an
// image); all of them by default

export function SeesFields({ options, onChange }: Common) {
  const catalogue = (useLines().data ?? []) as { name: string; rest: string }[];
  const spectral = isSpectral(options.type);
  const names = (key: string) => (Array.isArray(options[key]) ? (options[key] as unknown[]).map(String) : []);
  const setNames = (key: string, list: string[]) => onChange(withKey(options, key, list.length ? list : undefined));
  const band = options.band;
  return (
    <div className="ob-form">
      <Row label="Galaxies">
        <Chips names={names('galaxies')} none="all of the setup" label="A galaxy it sees, by its name in the setup" onChange={(list) => setNames('galaxies', list)} />
      </Row>
      <Row label="Components">
        <Chips names={names('components')} none="all that emit" label="A component whose light it sees, by name" onChange={(list) => setNames('components', list)} />
      </Row>
      {spectral ? (
        <>
          <Row label="Lines">
            <Chips names={names('lines')} none="all" label="A line it sees" choices={catalogue.map((l) => ({ name: l.name, description: l.rest }))} onChange={(list) => setNames('lines', list)} />
          </Row>
          <Row label="Continuum">
            <Segments
              items={[
                { value: 'off', label: 'Off' },
                { value: 'on', label: 'On' },
              ]}
              value={options.continuum === true ? 'on' : 'off'}
              label="Whether it sees the velocities of the continuum"
              onChange={(value) => onChange(withKey(options, 'continuum', value === 'on' || undefined))}
            />
          </Row>
        </>
      ) : (
        <Row label="Band">
          {band == null || typeof band === 'string' ? (
            <NamesField value={band ?? ''} placeholder="none" label="The band of an image, by its name in the catalogue" onChange={(typed) => onChange(withKey(options, 'band', typed.trim() || undefined))} />
          ) : (
            <span className="ob-text">Given in place (in the file)</span>
          )}
        </Row>
      )}
    </div>
  );
}

// Names as chips, each removable, and what adds one: typed (Enter), or
// chosen from given choices (e.g. the lines of the catalogue); none: its
// placeholder, what none means
function Chips({ names, none, label, choices, onChange }: { names: string[]; none: string; label: string; choices?: { name: string; description: string }[]; onChange: (names: string[]) => void }) {
  const [text, setText] = useState('');
  const add = (name: string) => {
    const clean = name.trim();
    if (clean && !names.includes(clean)) onChange([...names, clean]);
    setText('');
  };
  return (
    <span className="ob-chips">
      {names.map((name) => (
        <span key={name} className="ob-chip">
          {name}
          <button aria-label={`Remove ${name}`} onClick={() => onChange(names.filter((n) => n !== name))}>
            ×
          </button>
        </span>
      ))}
      {choices ? (
        <DropMenu
          label={label}
          width={260}
          entries={choices.filter((c) => !names.includes(c.name)).map((c) => ({ label: c.name, description: c.description, onClick: () => add(c.name) }))}
          target={<button className="ob-chip-add">{names.length ? '+ add' : `${none} · + add`}</button>}
        />
      ) : (
        <input
          className="ob-chip-input"
          value={text}
          placeholder={names.length ? '+ name' : none}
          aria-label={label}
          onChange={(event) => setText(event.currentTarget.value)}
          onKeyDown={(event) => event.key === 'Enter' && add(text)}
          onBlur={() => add(text)}
        />
      )}
    </span>
  );
}

// A text of names as typed, kept as the user types (the list is made from
// it as the field is left)
function NamesField({ value, placeholder, label, onChange }: { value: string; placeholder: string; label: string; onChange: (text: string) => void }) {
  return <input key={value} className="ob-names" defaultValue={value} placeholder={placeholder} aria-label={label} title={label} onBlur={(event) => onChange(event.target.value)} />;
}

// ---------------------------------------------------------------------------
// The noise of its mock data: none, a noise of the project, or one in place;
// and, without one, what its mock data take

export function NoiseFields({ defs, project, options, onChange, blocks, openBlock, titleOf }: Common & { titleOf: (family: string, type: unknown) => string }) {
  let note: string;
  if (options.noise != null) note = 'Its mock data: the model plus this noise (a setup with mock draws them); fits of data use their own errors.';
  else if (isRecorded(options.type)) note = 'Without one, its mock data take the errors of the data a setup pairs it with, as Gaussian noise; without data, they need a noise here.';
  else note = 'Its data are measured from what its instrument records (e.g. moments of a cube), to which the noise is added: its mock data need a noise here.';
  return (
    <div className="ob-form">
      <Response family="noise" folder="noise" title="Kind" none="None" value={options.noise} onChange={(value) => onChange(withKey(options, 'noise', value))} defs={defs} project={project} blocks={blocks} openBlock={openBlock} titleOf={titleOf} />
      <p className="ob-text">{note}</p>
    </div>
  );
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
