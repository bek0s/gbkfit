// The screen of a galaxy: its sheet, the model (the galaxy as a whole, its
// components and its mass, see Model), beside its picture, the views (its
// maps or its cube, and its curves, see Viewer). Its options and the
// values of its parameters (the params block of the same name) are edited
// as one draft, with undo and redo, and saved together (see draft.ts).
//
// Keys: Ctrl+Z undoes, Ctrl+Shift+Z (or Ctrl+Y) redoes, Ctrl+S saves.
import { useDebouncedValue } from '@mantine/hooks';
import { useMemo, useState } from 'react';
import { RenderLabel, useProject } from '../../api/queries';
import { SheetLayout } from '../../components/ui/sheet';
import { useBlockHead } from '../../shell/BlockMenu';
import { SaveControls, ScreenHead } from '../../shell/ScreenHead';
import { useRemembered } from '../../shell/remember';
import { useStudio } from '../../studio';
import { useUnsaved } from '../editing';
import { useGalaxyDraft } from './draft';
import { kpcPerArcsecOf, Model } from './Model';
import { Params } from './Params';
import { useComponentTypes } from '../../api/queries';
import type { ComponentTypes } from './types';
import { asList, componentName, componentsOf, ringRadii, type Options } from './options';
import { forRender } from './params';
import { Views } from './Views';

// The colours of the components, in turn (tokens): the first colour of a
// theme is of the hue of its accent, which marks what is chosen, free or
// to act on; it comes last, so that a component of that colour is rare
export const COLOURS = ['cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat1'];

// The component a part of the mass follows, if any (e.g. of a traced disk;
// of a disk of the mass, by its surface or its heights)
function followedOf(item: Options): string | null {
  for (const where of [item, item.surface, item.vertical]) {
    if (typeof where === 'object' && where !== null && typeof (where as Options).component === 'string') return (where as Options).component as string;
  }
  return null;
}

// The colour of each part of the mass: that of the component it follows,
// else the colours after those of the components, in turn
function massColoursOf(mass: Options[], components: Options[], colours: string[]): string[] {
  let next = components.length;
  return mass.map((item) => {
    const at = components.findIndex((c, i) => componentName(c, i) === followedOf(item));
    return at >= 0 ? colours[at] : COLOURS[next++ % COLOURS.length];
  });
}

// The field of the view: this times the largest radius of the galaxy, as
// the server renders it (its half-width, arcsec)
const MARGIN = 1.15;

export function GalaxyScreen({ name }: { name: string }) {
  const { project } = useStudio();
  // (what sees it, remembered for it; the observations of the project)
  const [seenBy, setSeenBy] = useRemembered<string | null>(`seen-by:${project}/${name}`, null);
  const observations = useProject(project).data?.pieces.observations ?? [];
  const draft = useGalaxyDraft(project ?? '.', name);
  const head = useBlockHead(project ?? '', 'galaxies', name);
  useUnsaved(draft.dirty);
  // (the pieces of the sheet unfolded, by their keys (see Model), in the
  // order they were, remembered for the galaxy (none yet: its first
  // component); the one selected)
  const [unfolded, setUnfolded] = useRemembered<string[] | null>(`galaxy-pieces:${project}/${name}`, null);
  const [selected, setSelected] = useState<string | null>(null);
  const [isolated, setIsolated] = useState<string | null>(null);
  // (the tab of the sheet: the model, or the list of the parameters)
  const [tab, setTab] = useRemembered<'model' | 'params'>('galaxy-sheet-tab', 'model');
  const types = useComponentTypes().data as ComponentTypes | undefined;
  const options = useMemo(() => draft.state?.options ?? {}, [draft.state]);
  const components = componentsOf(options);
  const colours = components.map((_, i) => COLOURS[i % COLOURS.length]);
  const mass = asList(options.mass);
  const massColours = massColoursOf(mass, components, colours);

  // (the render follows the state the library has described, as it settles)
  const [rendered] = useDebouncedValue(draft.consistent, 60);
  const renderedComponents = rendered ? componentsOf(rendered.options) : [];
  // (a galaxy is something to show with components or mass: of mass alone,
  // its curves, as a rotation curve is fitted)
  const something = (galaxy: Options) => componentsOf(galaxy).length > 0 || asList(galaxy.mass).length > 0;
  const renderProperties = useMemo(() => (rendered ? forRender(rendered.properties, draft.params) : {}), [rendered, draft.params]);
  const names = components.map((c, i) => componentName(c, i));
  const open = unfolded ?? names.slice(0, 1);

  if (project === null) return null;

  // The field of the view (half its width, arcsec), for the reach of the
  // sliders of lengths and positions
  const radii = components.flatMap((c) => (typeof c.rmax === 'number' ? [c.rmax] : ringRadii(c.rnodes).slice(-1)));
  const field = Math.max(...radii, 0) * MARGIN || 20;

  // (the rings of the component unfolded last, marked on the profiles)
  const last = open.filter((key) => names.includes(key)).at(-1);
  const rings = last === undefined ? null : ringRadii(components[names.indexOf(last)].rnodes);

  return (
    // (its renders are of it, in the Activity drawer)
    <RenderLabel.Provider value={name}>
    <section className="screen gx-screen">
      <ScreenHead up="galaxies" title={head.title}>
        <SaveControls saving={draft} />
        {head.menu}
      </ScreenHead>
      {draft.error && <p className="error screen-error">{draft.error.message}</p>}
      {!draft.loaded ? (
        <div className="gx-loading" aria-busy="true">
          <div className="gx-map-wait" />
        </div>
      ) : (
        <SheetLayout
          className="gx-body"
          width={400}
          sheet={
            <section className="gm-sheet" aria-label="The galaxy">
              <div className="gm-tabs" role="tablist">
                <button role="tab" aria-selected={tab === 'model'} onClick={() => setTab('model')}>
                  Model
                </button>
                <button role="tab" aria-selected={tab === 'params'} onClick={() => setTab('params')}>
                  Parameters <span>{draft.params.length}</span>
                </button>
              </div>
              <div className="gm-scroll">
                {tab === 'model' ? (
                  <Model
                    draft={draft}
                    project={project}
                    galaxy={name}
                    field={field}
                    colours={colours}
                    massColours={massColours}
                    open={open}
                    onOpen={(change) => setUnfolded((o) => change(o ?? names.slice(0, 1)))}
                    selected={selected}
                    onSelect={setSelected}
                    isolated={isolated}
                    onIsolate={setIsolated}
                  />
                ) : (
                  <Params draft={draft} field={field} colours={colours} types={types} defs={draft.defs} />
                )}
              </div>
            </section>
          }
          picture={
            <Views
              project={project}
              seenBy={seenBy}
              onSeenBy={setSeenBy}
              observations={observations}
              galaxy={rendered && something(options) && something(rendered.options) ? rendered.options : undefined}
              why={!something(options) ? 'empty' : draft.problems.length > 0 ? 'problems' : 'waiting'}
              properties={renderProperties}
              components={renderedComponents.map((c, i) => ({ name: componentName(c, i), colour: COLOURS[i % COLOURS.length] }))}
              mass={mass.map((m, i) => ({ name: String(m.name ?? ''), colour: massColours[i] }))}
              isolated={isolated}
              options={options}
              params={draft.params}
              kpcPerArcsec={kpcPerArcsecOf(draft)}
              rings={rings}
            />
          }
        />
      )}
    </section>
    </RenderLabel.Provider>
  );
}

