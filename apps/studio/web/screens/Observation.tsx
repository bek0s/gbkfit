// Drafts of the screen of an observation (see shell/drafts): how an
// instrument sees the galaxies (the line of sight, the instrument: its
// PSF and LSF, often blocks of the project; the observable: what the
// data measure, on what grid) and the data it is compared with (a dataset
// of the project) by a likelihood, in two layouts to compare. Their forms
// are of the schema of the library (see SchemaForm).
import { useState } from 'react';
import { usePiece } from '../api/queries';
import { joinPath } from '../components/FilePicker';
import { membersOf, propertiesOf, SchemaForm, typeOf, type Options, type Schema } from '../components/SchemaForm';
import { Button } from '../components/ui/basics';
import { Panel } from '../components/ui/panel';
import { useStudio } from '../studio';
import type { LayoutProps } from './Galaxy';

// The stages of an observation of the sky, from the galaxy to the data,
// and what each is
const STAGES: { key: string; title: string; text: string }[] = [
  { key: 'galaxies', title: 'Galaxies', text: 'Those it sees (all of the setup by default)' },
  { key: 'line_of_sight', title: 'Line of sight', text: 'The plane of the sky, or a depth' },
  { key: 'instrument', title: 'Instrument', text: 'Its PSF, LSF and primary beam' },
  { key: 'observable', title: 'Observable', text: 'What the data measure, on what grid' },
  { key: 'data', title: 'Data', text: 'The dataset it is compared with' },
  { key: 'likelihood', title: 'Likelihood', text: 'How the model is compared with the data' },
];

// A value at a glance: a block named, a type, or what it has
function glance(value: unknown): string {
  if (value === undefined || value === null) return '—';
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return value.map(glance).join(', ') || '—';
  if (typeof value === 'object') {
    const options = value as Options;
    if (typeof options.type === 'string') return options.type;
    return (
      Object.entries(options)
        .map(([key, v]) => `${key}: ${glance(v)}`)
        .join(' · ') || '—'
    );
  }
  return String(value);
}

// What a stage of an observation is, at a glance (and its defaults)
function stageGlance(key: string, options: Options): string {
  const value = options[key];
  if (key === 'galaxies') return value ? glance(value) : 'all of the setup';
  if (key === 'instrument') {
    const instrument = (value ?? {}) as Options;
    const parts = ['psf', 'lsf', 'primary_beam'].filter((part) => instrument[part] != null).map((part) => `${part}: ${glance(instrument[part])}`);
    return parts.join(' · ') || 'a perfect one';
  }
  if (key === 'observable') {
    const observable = (value ?? {}) as Options;
    const grid = Array.isArray(observable.size) ? ` · ${(observable.size as number[]).join(' × ')}` : options.data ? ' · the grid of the data' : '';
    return `${glance(observable.type)}${grid}`;
  }
  if (key === 'data') return value == null ? 'none: a model, or mock data' : typeof value === 'string' ? `the dataset ${value}` : `in place · ${glance(value)}`;
  if (key === 'likelihood') return value == null ? (options.data ? 'Gaussian (the default)' : 'none (no data)') : glance(value);
  return glance(value);
}

// The member of the schema of the observation of its type
function observationMember(edited: LayoutProps['edited'], options: Options): Schema | undefined {
  return edited.defs && membersOf(edited.defs.observation ?? {}, edited.defs).find((m) => typeOf(m) === options.type);
}

// A: the stages as a row of cards, the stage chosen edited under it
export function ObservationPipeline({ project, edited }: LayoutProps) {
  const [chosen, setChosen] = useState('observable');
  const { go } = useStudio();
  const defs = edited.defs;
  const options = edited.options;
  if (!defs || !options) return <p className="note screen-error">Reading the observation…</p>;
  const member = observationMember(edited, options);
  const properties = member ? propertiesOf(member) : {};
  const stages = STAGES.filter(({ key }) => key in properties);
  const node = properties[chosen];
  return (
    <div className="screen-body draft-pipeline">
      {member === undefined ? (
        <div className="card">
          <div className="card-body">
            <p className="note">An observation of type {String(options.type)}: edit it as a whole.</p>
            <SchemaForm defs={defs} def="observation" project={project} value={options} onChange={edited.set} blocks={edited.blocks} openBlock={(folder, name) => go({ kind: 'block', folder, name })} />
          </div>
        </div>
      ) : (
        <>
          <div className="pipeline">
            {stages.map(({ key, title, text }, i) => (
              <div key={key} className="pipeline-step">
                {i > 0 && <span className="pipeline-arrow">→</span>}
                <button type="button" className={chosen === key ? 'card pipeline-card active' : 'card pipeline-card'} onClick={() => setChosen(key)} title={text}>
                  <span className="eyebrow">{title}</span>
                  <span className="pipeline-glance">{stageGlance(key, options)}</span>
                </button>
              </div>
            ))}
          </div>
          <div className="pipeline-detail">
            <div className="card">
              <h2 className="card-head">{STAGES.find((s) => s.key === chosen)?.title ?? chosen}</h2>
              <div className="card-body">
                <p className="note">{STAGES.find((s) => s.key === chosen)?.text}</p>
                {node && (
                  <SchemaForm
                    defs={defs}
                    node={{ type: 'object', properties: { [chosen]: node }, required: [] }}
                    project={project}
                    value={chosen in options ? { [chosen]: options[chosen] } : {}}
                    onChange={(next) => {
                      const { [chosen]: _, ...rest } = options;
                      edited.set(next && chosen in next ? { ...rest, [chosen]: next[chosen] } : rest);
                    }}
                    blocks={edited.blocks}
                    openBlock={(folder, name) => go({ kind: 'block', folder, name })}
                  />
                )}
              </div>
            </div>
            <Panel id="observation-other" title="Its other options">
              <SchemaForm
                defs={defs}
                node={{
                  ...member,
                  properties: Object.fromEntries(Object.entries(properties).filter(([key]) => key !== 'type' && !STAGES.some((s) => s.key === key))),
                  required: [],
                }}
                project={project}
                value={Object.fromEntries(Object.entries(options).filter(([key]) => key !== 'type' && !STAGES.some((s) => s.key === key)))}
                onChange={(next) => edited.set({ ...Object.fromEntries(Object.entries(options).filter(([key]) => key === 'type' || STAGES.some((s) => s.key === key))), ...next })}
                blocks={edited.blocks}
              />
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}

// B: all the options in one form, beside the checks of the observation
// against its data
export function ObservationForm({ project, edited }: LayoutProps) {
  const { go } = useStudio();
  const defs = edited.defs;
  const options = edited.options;
  if (!defs || !options) return <p className="note screen-error">Reading the observation…</p>;
  return (
    <div className="screen-body draft-form">
      <div className="card form-main">
        <h2 className="card-head">The observation</h2>
        <div className="card-body">
          <SchemaForm
            defs={defs}
            def="observation"
            project={project}
            value={options}
            onChange={edited.set}
            blocks={edited.blocks}
            openBlock={(folder, name) => go({ kind: 'block', folder, name })}
          />
        </div>
      </div>
      <div className="col">
        <Checks project={project} options={options} datasets={edited.blocks.datasets ?? []} />
        <Panel id="observation-footprint" title="Footprint">
          <div className="draft-placeholder">
            <p>The grid of the observable on the sky, over the map of its data (and the PSF to scale): to come.</p>
          </div>
        </Panel>
      </div>
    </div>
  );
}

// The checks of an observation against the dataset it names: that its
// observable is of the type of the dataset, that the dataset is there,
// and what an instrument needs
function Checks({ project, options, datasets }: { project: string; options: Options; datasets: string[] }) {
  const { go } = useStudio();
  const named = typeof options.data === 'string' ? options.data : null;
  const dataset = usePiece(named === null || !datasets.includes(named) ? null : joinPath(project, `datasets/${named}.yaml`)).data as Options | undefined;
  const observable = (options.observable ?? {}) as Options;
  const instrument = (options.instrument ?? {}) as Options;
  const spectral = ['pixel_spectra', 'pixel_moments', 'slit_spectra', 'slitless_spectra', 'region_spectra', 'region_moments'].includes(String(observable.type));
  const checks: { ok: boolean | null; text: string }[] = [];
  if (options.data == null) checks.push({ ok: null, text: 'No data: it makes a model, or mock data.' });
  else if (named !== null && !datasets.includes(named)) checks.push({ ok: false, text: `The project has no dataset ${named}.` });
  else if (dataset) {
    checks.push(
      dataset.type === observable.type
        ? { ok: true, text: `The observable and the dataset are both ${String(observable.type)}.` }
        : { ok: false, text: `The observable is ${String(observable.type)}; the dataset ${named} is ${String(dataset.type)}.` },
    );
    if (Array.isArray(observable.size)) checks.push({ ok: null, text: 'The observable gives its grid; with data, the grid of the data is used.' });
  }
  checks.push(instrument.psf != null ? { ok: true, text: `PSF: ${glance(instrument.psf)}.` } : { ok: null, text: 'No PSF: a perfect instrument on the sky.' });
  if (spectral) checks.push(instrument.lsf != null ? { ok: true, text: `LSF: ${glance(instrument.lsf)}.` } : { ok: null, text: 'No LSF: a perfect spectrograph.' });
  return (
    <Panel id="observation-checks" title="Checks">
      <ul className="checks">
        {checks.map((check, i) => (
          <li key={i} data-ok={check.ok === null ? 'note' : String(check.ok)}>
            {check.ok === null ? '·' : check.ok ? '✓' : '✗'} {check.text}
          </li>
        ))}
      </ul>
      {named !== null && datasets.includes(named) && (
        <Button kind="small" onClick={() => go({ kind: 'block', folder: 'datasets', name: named })}>
          Open the dataset {named} →
        </Button>
      )}
      <p className="note">To come: its pixels and channels against those of the data, and its PSF against the beam of their header.</p>
    </Panel>
  );
}
