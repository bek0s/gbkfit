// The screen of a galaxy: its sheet, the model (the galaxy as a whole, its
// components and its mass, see Model), beside its picture, the views (its
// maps or its cube, and its curves, see Viewer). Its options and the
// values of its parameters (the params block of the same name) are edited
// as one draft, with undo and redo, and saved together (see draft.ts).
//
// Keys: Ctrl+Z undoes, Ctrl+Shift+Z (or Ctrl+Y) redoes, Ctrl+S saves.
import { useDebouncedValue } from '@mantine/hooks';
import { useEffect, useMemo, useState } from 'react';
import { useProject } from '../../api/queries';
import { Button } from '../../components/ui/basics';
import { SheetLayout } from '../../components/ui/sheet';
import { useBlockHead } from '../../shell/BlockMenu';
import { BlockLinks } from '../../shell/BlockLinks';
import { useRemembered } from '../../shell/remember';
import { useStudio } from '../../studio';
import { useUnsaved } from '../editing';
import { useGalaxyDraft } from './draft';
import { kpcPerArcsecOf, Model, type Open } from './Model';
import { Params } from './Params';
import { useComponentTypes } from '../../api/queries';
import type { ComponentTypes } from './types';
import { componentName, componentsOf, ringRadii } from './options';
import { forRender } from './params';
import { Views } from './Views';

// The colours of the components, in turn (tokens)
const COLOURS = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6'];

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
  const [open, setOpen] = useState<Open>({ kind: 'component', index: 0 });
  const [isolated, setIsolated] = useState<string | null>(null);
  // (the tab of the sheet: the model, or the list of the parameters)
  const [tab, setTab] = useRemembered<'model' | 'params'>('galaxy-sheet-tab', 'model');
  const types = useComponentTypes().data as ComponentTypes | undefined;
  const options = useMemo(() => draft.state?.options ?? {}, [draft.state]);
  const components = componentsOf(options);
  const colours = components.map((_, i) => COLOURS[i % COLOURS.length]);

  // (the render follows the state the library has described, as it settles)
  const [rendered] = useDebouncedValue(draft.consistent, 60);
  const renderedComponents = rendered ? componentsOf(rendered.options) : [];
  const renderProperties = useMemo(() => (rendered ? forRender(rendered.properties, draft.params) : {}), [rendered, draft.params]);

  // (keys: undo, redo and save, but in a field of text, whose own undo it is)
  const { undo, redo, save } = draft;
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey)) return;
      const key = event.key.toLowerCase();
      if (key === 's') {
        event.preventDefault();
        void save();
        return;
      }
      const typing =
        event.target instanceof HTMLElement &&
        event.target.closest('input:not([type=radio]):not([type=checkbox]):not([type=range]):not([type=button]), textarea, [contenteditable]:not([contenteditable=false])');
      if (typing) return;
      if (key === 'z' && !event.shiftKey) {
        event.preventDefault();
        undo();
      } else if ((key === 'z' && event.shiftKey) || key === 'y') {
        event.preventDefault();
        redo();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [undo, redo, save]);

  if (project === null) return null;

  // The field of the view (half its width, arcsec), for the reach of the
  // sliders of lengths and positions
  const radii = components.flatMap((c) => (typeof c.rmax === 'number' ? [c.rmax] : ringRadii(c.rnodes).slice(-1)));
  const field = Math.max(...radii, 0) * MARGIN || 20;

  // (the rings of the component open, marked on the profiles)
  const openComponent = open?.kind === 'component' && open.index < components.length ? open.index : null;
  const rings = openComponent === null ? null : ringRadii(components[openComponent]?.rnodes);

  return (
    <section className="screen gx-screen">
      <div className="screen-head gx-head">
        <div>
          <div className="eyebrow">Galaxy</div>
          {head.title}
        </div>
        <BlockLinks folder="galaxies" name={name} />
        <div className="head-actions">
          <div className="gx-history" role="group" aria-label="History">
            <button onClick={draft.undo} disabled={!draft.canUndo} title="Undo (Ctrl+Z)" aria-label="Undo">
              <svg className="icon" viewBox="0 0 24 24">
                <path d="M9 14 4 9l5-5" />
                <path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" />
              </svg>
            </button>
            <button onClick={draft.redo} disabled={!draft.canRedo} title="Redo (Ctrl+Shift+Z)" aria-label="Redo">
              <svg className="icon" viewBox="0 0 24 24">
                <path d="m15 14 5-5-5-5" />
                <path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H13" />
              </svg>
            </button>
          </div>
          <span className="gx-status" data-dirty={draft.dirty || undefined}>
            {draft.saving ? 'Saving…' : draft.dirty ? 'Not saved' : 'Saved'}
          </span>
          {draft.saveError && <span className="error">{draft.saveError.message}</span>}
          <Button disabled={!draft.dirty} onClick={draft.revert}>
            Revert
          </Button>
          <Button kind="primary" disabled={!draft.dirty || draft.saving} onClick={() => void draft.save()} title="Save (Ctrl+S)">
            Save
          </Button>
          {head.menu}
        </div>
      </div>
      {draft.error && <p className="error screen-error">{draft.error.message}</p>}
      {!draft.loaded ? (
        <div className="gx-loading" aria-busy="true">
          <div className="gx-map-wait" />
        </div>
      ) : (
        <SheetLayout
          className="gx-body"
          width={480}
          sheet={
            <section className="gm-sheet" aria-label="The galaxy">
              <div className="gm-tabs" role="tablist">
                <button role="tab" aria-selected={tab === 'model'} onClick={() => setTab('model')}>
                  Model
                </button>
                <button role="tab" aria-selected={tab === 'params'} onClick={() => setTab('params')}>
                  Parameters <span>{draft.params.length}</span>
                </button>
                <span className="gm-tabs-hint">{tab === 'model' ? 'every component, closed until you open it' : 'every parameter, by what owns it'}</span>
              </div>
              <div className="gm-scroll">
                {tab === 'model' ? (
                  <Model draft={draft} project={project} field={field} colours={colours} open={open} onOpen={setOpen} isolated={isolated} onIsolate={setIsolated} />
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
              galaxy={components.length && renderedComponents.length ? rendered?.options : undefined}
              why={components.length === 0 ? 'empty' : draft.problems.length > 0 ? 'problems' : 'waiting'}
              properties={renderProperties}
              components={renderedComponents.map((c, i) => ({ name: componentName(c, i), colour: COLOURS[i % COLOURS.length] }))}
              isolated={isolated}
              options={options}
              kpcPerArcsec={kpcPerArcsecOf(draft)}
              rings={rings}
            />
          }
        />
      )}
    </section>
  );
}

