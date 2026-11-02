// Drafts of the screen of a galaxy (see shell/drafts): its components
// (disks, spheroids, cones, points, particles), each of parts (density,
// motions, light, ...), and the options of the galaxy as a whole
// (distance, mass, gravity, ...), in three layouts to compare. Their
// forms are of the schema of the library (see SchemaForm); the model
// itself (its maps and profiles) is to come, from a live session of a
// setup that sees the galaxy.
import { useState } from 'react';
import { membersOf, propertiesOf, resolve, SchemaForm, typeOf, type Defs, type Options, type Schema } from '../components/SchemaForm';
import { Button, Close, NavItem } from '../components/ui/basics';
import { Segments } from '../components/ui/inputs';
import { DropMenu } from '../components/ui/overlays';
import { Panel } from '../components/ui/panel';
import { useStudio } from '../studio';
import type { EditedBlock } from './editing';

export interface LayoutProps {
  project: string;
  name: string;
  edited: EditedBlock;
}

// What a new component of each type starts with (that of disks: as the
// template of galaxies)
const COMPONENT_STARTS: Record<string, Options> = {
  disk: {
    loose: false,
    tilted: false,
    rnodes: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
    density: { ptraits: { type: 'exponential' } },
    kinematics: { type: 'traits', vptraits: { type: 'arctan', direction: 'tan' }, dptraits: { type: 'uniform' } },
    emission: { type: 'lines' },
  },
};

const TITLES: Record<string, string> = {
  density: 'Density',
  kinematics: 'Motions',
  emission: 'Light',
  opacity: 'Opacity',
  zptraits: 'Vertical',
  sptraits: 'Selection',
  clouds: 'Clouds',
  profile: 'Profile',
  shape: 'Shape',
  geometry: 'Geometry',
};

// The components of a galaxy (one, or a list of them)
function componentsOf(options: Options): Options[] {
  const components = options.components;
  if (Array.isArray(components)) return components as Options[];
  return components && typeof components === 'object' ? [components as Options] : [];
}

function withComponents(options: Options, components: Options[]): Options {
  return { ...options, components };
}

// The member of the schema of a type of component
function componentMember(defs: Defs, type: unknown): Schema | undefined {
  return membersOf(defs.component ?? {}, defs).find((m) => typeOf(m) === type);
}

// Whether an option of a component is a part of it (options of their own,
// e.g. its density), rather than a value (e.g. its rings)
function isPart(node: Schema, defs: Defs): boolean {
  const branches = ((node.anyOf as Schema[] | undefined) ?? [node]).map((b) => resolve(b, defs)).filter((b) => b.type !== 'null');
  return branches.some((b) => b.oneOf !== undefined || (b.type === 'object' && b.properties !== undefined));
}

// The parts of a component, and its options that are values (with its
// type), by the schema of its type
function splitComponent(defs: Defs, member: Schema): { parts: [string, Schema][]; values: Schema } {
  const properties = Object.entries(propertiesOf(member));
  const parts = properties.filter(([key, node]) => key !== 'type' && isPart(node, defs));
  const values = { ...member, properties: Object.fromEntries(properties.filter(([key]) => !parts.some(([p]) => p === key))) };
  return { parts, values };
}

// What a part is at a glance: its type (or those of a list), or its
// first traits
function summary(value: unknown): string {
  if (value === undefined || value === null) return '—';
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return value.map(summary).join(' + ');
  if (typeof value === 'object') {
    const options = value as Options;
    if (typeof options.type === 'string') return options.type;
    const traits = Object.entries(options).find(([key]) => key.endsWith('traits'));
    if (traits) return summary(traits[1]);
    return Object.keys(options).join(', ') || '—';
  }
  return String(value);
}

// The rings of a component, at a glance
function rings(component: Options): string {
  const nodes = component.rnodes;
  if (Array.isArray(nodes) && nodes.length > 0) return `${nodes.length} rings to ${nodes[nodes.length - 1]}″`;
  if (typeof component.rnmax === 'number') return `to ${component.rnmax}″${component.rnsep ? ` every ${component.rnsep}″` : ''}`;
  return '—';
}

function componentLabel(component: Options, i: number): string {
  const type = typeof component.type === 'string' ? component.type : 'component';
  return typeof component.name === 'string' ? `${component.name} · ${type}` : `${type} ${i + 1}`;
}

// The menu that adds a component of a type
function AddComponent({ defs, onAdd }: { defs: Defs; onAdd: (component: Options) => void }) {
  const types = membersOf(defs.component ?? {}, defs).map((m) => typeOf(m)!);
  return (
    <DropMenu
      label="Add a component"
      entries={[...new Set(types)].map((type) => ({ label: type, onClick: () => onAdd({ type, ...(COMPONENT_STARTS[type] ?? {}) }) }))}
      target={<Button kind="small">+ Add a component</Button>}
    />
  );
}

// The options of the galaxy as a whole (all but its components)
function GalaxyOptions({ project, edited, defs }: { project: string; edited: EditedBlock; defs: Defs }) {
  const options = edited.options ?? {};
  const galaxy = defs.Galaxy ?? {};
  const { components: _, ...rest } = propertiesOf(galaxy);
  const { components, ...values } = options;
  return (
    <SchemaForm
      defs={defs}
      node={{ ...galaxy, properties: rest, required: [] }}
      project={project}
      value={values}
      onChange={(next) => edited.set({ ...(next ?? {}), ...(components === undefined ? {} : { components }) })}
    />
  );
}

// The placeholder of the model, until it is drawn live
function ModelCard({ compact = false }: { compact?: boolean }) {
  return (
    <div className={compact ? 'card draft-model compact' : 'card draft-model'}>
      <h2 className="card-head">The model</h2>
      <div className="card-body">
        <div className="draft-placeholder">
          <p>
            Its maps (intensity, velocity, dispersion) and radial profiles, drawn live as it changes: from a live session (the
            server's /sessions) of a setup that sees it through an observation, at the values of its parameters.
          </p>
        </div>
      </div>
    </div>
  );
}

// A: the outline of the galaxy (the galaxy, its components and their
// parts), the part chosen edited beside it, the model at the right
export function GalaxyOutline({ project, edited }: LayoutProps) {
  const [chosen, setChosen] = useState<string>('galaxy');
  const defs = edited.defs;
  const options = edited.options;
  if (!defs || !options) return <p className="note screen-error">Reading the galaxy…</p>;
  const components = componentsOf(options);
  const setComponent = (i: number, component: Options | undefined) =>
    edited.set(withComponents(options, component === undefined ? components.filter((_, j) => j !== i) : components.map((c, j) => (j === i ? component : c))));
  // (the part chosen: the galaxy, a component c<i>, or a part of one
  // c<i>.<key>)
  const [, index, key] = chosen.match(/^c(\d+)(?:\.(\w+))?$/) ?? [];
  const component = index === undefined ? undefined : components[Number(index)];
  const member = component && componentMember(defs, component.type);
  let inspector;
  if (component === undefined || member === undefined) {
    inspector = (
      <>
        <h2 className="card-head">The galaxy as a whole</h2>
        <div className="card-body">
          <p className="note">Its distance, redshift, mass and gravity, and the geometry its components share.</p>
          <GalaxyOptions project={project} edited={edited} defs={defs} />
        </div>
      </>
    );
  } else {
    const i = Number(index);
    const { values } = splitComponent(defs, member);
    const part = key ? propertiesOf(member)[key] : undefined;
    inspector = (
      <>
        <h2 className="card-head">
          {componentLabel(component, i)}
          {key ? ` · ${TITLES[key] ?? key}` : ''}
        </h2>
        <div className="card-body">
          {part ? (
            <SchemaForm defs={defs} node={part} project={project} value={component[key!] as Options} onChange={(next) => setComponent(i, { ...component, [key!]: next })} />
          ) : (
            <SchemaForm defs={defs} node={values} project={project} value={component} onChange={(next) => setComponent(i, { ...next, type: component.type })} />
          )}
        </div>
      </>
    );
  }
  return (
    <div className="screen-body draft-outline">
      <nav className="card outline" aria-label="The outline of the galaxy">
        <NavItem label="Galaxy" active={chosen === 'galaxy'} onClick={() => setChosen('galaxy')} />
        <div className="outline-label">Components</div>
        {components.map((c, i) => {
          const m = componentMember(defs, c.type);
          const parts = m ? splitComponent(defs, m).parts : [];
          return (
            <div key={i} className="outline-group">
              <div className="outline-row">
                <NavItem mono label={componentLabel(c, i)} active={chosen === `c${i}`} onClick={() => setChosen(`c${i}`)} />
                <Close
                  label={`Remove ${componentLabel(c, i)}`}
                  onClick={() => {
                    setChosen('galaxy');
                    setComponent(i, undefined);
                  }}
                />
              </div>
              {parts.map(([part]) => (
                <div key={part} className="outline-part">
                  <NavItem
                    label={TITLES[part] ?? part}
                    right={<small>{summary(c[part])}</small>}
                    active={chosen === `c${i}.${part}`}
                    onClick={() => setChosen(`c${i}.${part}`)}
                  />
                </div>
              ))}
            </div>
          );
        })}
        <div className="outline-add">
          <AddComponent
            defs={defs}
            onAdd={(added) => {
              edited.set(withComponents(options, [...components, added]));
              setChosen(`c${components.length}`);
            }}
          />
        </div>
      </nav>
      <div className="card inspector">{inspector}</div>
      <div className="col">
        <ModelCard compact />
        <Panel id="galaxy-parameters" title="Its parameters">
          <p className="note">
            The parameters it has (e.g. xpos, ypos, vsys, posa, incl, and those of the traits of its components), with their values in
            the blocks of parameters of the project: to come.
          </p>
        </Panel>
      </div>
    </div>
  );
}

// B: a card of each component, what it is at a glance, opened to edit;
// the galaxy as a whole and the profiles at the side
export function GalaxyCards({ project, edited }: LayoutProps) {
  const [open, setOpen] = useState<number | null>(null);
  const defs = edited.defs;
  const options = edited.options;
  if (!defs || !options) return <p className="note screen-error">Reading the galaxy…</p>;
  const components = componentsOf(options);
  const setComponent = (i: number, component: Options | undefined) =>
    edited.set(withComponents(options, component === undefined ? components.filter((_, j) => j !== i) : components.map((c, j) => (j === i ? component : c))));
  return (
    <div className="screen-body draft-cards">
      <div className="cards-main">
        <div className="component-grid">
          {components.map((c, i) => {
            const kinematics = (c.kinematics ?? {}) as Options;
            const density = (c.density ?? {}) as Options;
            return (
              <div key={i} className={open === i ? 'card component-card open' : 'card component-card'}>
                <div className="card-head component-head">
                  <h2>{componentLabel(c, i)}</h2>
                  <div className="component-actions">
                    <Button kind="small" onClick={() => setOpen(open === i ? null : i)}>
                      {open === i ? 'Done' : 'Edit'}
                    </Button>
                    <Close label={`Remove ${componentLabel(c, i)}`} onClick={() => setComponent(i, undefined)} />
                  </div>
                </div>
                <div className="card-body">
                  {open === i ? (
                    <SchemaForm defs={defs} def="component" project={project} value={c} onChange={(next) => setComponent(i, next)} />
                  ) : (
                    <dl className="facts">
                      <dt>density</dt>
                      <dd>{summary(density.ptraits ?? c.profile ?? c.density)}</dd>
                      <dt>vertical</dt>
                      <dd>{summary(c.zptraits ?? density.zptraits ?? c.shape)}</dd>
                      <dt>rotation</dt>
                      <dd>{summary(kinematics.vptraits ?? c.kinematics)}</dd>
                      <dt>dispersion</dt>
                      <dd>{summary(kinematics.dptraits)}</dd>
                      <dt>light</dt>
                      <dd>{summary(c.emission)}</dd>
                      <dt>rings</dt>
                      <dd>{rings(c)}</dd>
                      <dt>warps</dt>
                      <dd>{[c.loose && 'loose (centre, vsys)', c.tilted && 'tilted (posa, incl)'].filter(Boolean).join(', ') || 'none'}</dd>
                    </dl>
                  )}
                </div>
              </div>
            );
          })}
          <div className="card component-card add">
            <AddComponent
              defs={defs}
              onAdd={(added) => {
                edited.set(withComponents(options, [...components, added]));
                setOpen(components.length);
              }}
            />
          </div>
        </div>
      </div>
      <div className="col">
        <Panel id="galaxy-whole" title="The galaxy as a whole">
          <GalaxyOptions project={project} edited={edited} defs={defs} />
        </Panel>
        <Panel id="galaxy-profiles" title="Radial profiles">
          <div className="draft-placeholder">
            <p>The rotation curve, the dispersion and the surface brightness of each component against radius, at the values of the parameters: to come.</p>
          </div>
        </Panel>
      </div>
    </div>
  );
}

const VIEWS = ['intensity', 'velocity', 'dispersion', 'profiles', 'edge-on'] as const;

// C: the model takes the screen, with the components and the parameters
// in a drawer under it
export function GalaxyCanvas({ project, edited }: LayoutProps) {
  const [view, setView] = useState<(typeof VIEWS)[number]>('velocity');
  const [chosen, setChosen] = useState<number | null>(null);
  const { go } = useStudio();
  const defs = edited.defs;
  const options = edited.options;
  if (!defs || !options) return <p className="note screen-error">Reading the galaxy…</p>;
  const components = componentsOf(options);
  return (
    <div className="screen-body draft-canvas">
      <div className="card canvas-stage">
        <div className="card-head viewer-head">
          <Segments items={VIEWS.map((v) => ({ value: v, label: v }))} value={view} label="What the model shows" onChange={setView} />
          <Button kind="small" onClick={() => go({ kind: 'observe' })}>
            Observe it →
          </Button>
        </div>
        <div className="draft-placeholder canvas-placeholder">
          <svg viewBox="-100 -60 200 120" aria-hidden="true">
            <ellipse cx="0" cy="0" rx="80" ry="34" transform="rotate(-25)" className="canvas-disk" />
            <ellipse cx="0" cy="0" rx="40" ry="17" transform="rotate(-25)" className="canvas-disk inner" />
          </svg>
          <p>
            The {view} of the model, drawn live (a live session of a setup with this galaxy): a drag rotates it (posa), the wheel tilts it
            (incl), and the sliders of the drawer move its parameters.
          </p>
        </div>
      </div>
      <div className="card canvas-drawer">
        <div className="drawer-chips">
          {components.map((c, i) => (
            <Button key={i} kind={chosen === i ? 'primary' : 'plain'} onClick={() => setChosen(chosen === i ? null : i)}>
              {componentLabel(c, i)}
            </Button>
          ))}
          <AddComponent defs={defs} onAdd={(added) => edited.set(withComponents(options, [...components, added]))} />
        </div>
        {chosen !== null && components[chosen] ? (
          <SchemaForm
            defs={defs}
            def="component"
            project={project}
            value={components[chosen]}
            onChange={(next) => edited.set(withComponents(options, components.map((c, j) => (j === chosen ? next : c))))}
          />
        ) : (
          <div className="drawer-sliders">
            {['xpos', 'ypos', 'vsys', 'posa', 'incl'].map((p) => (
              <div key={p} className="drawer-slider">
                <span className="schema-label">{p}</span>
                <div className="slider-placeholder" />
              </div>
            ))}
            <p className="note">The sliders of its parameters (of the blocks of parameters of the project): to come. Choose a component to edit it.</p>
          </div>
        )}
      </div>
    </div>
  );
}
