// The screen of a galaxy, built as it is seen: its outline (the galaxy as
// a whole, and its components, each of its parts), its view (its maps or
// its cube, and its curves) and the inspector of the part chosen (its
// types, its parameters and its options). Its options and the values of
// its parameters (the params block of the same name) are edited as one
// draft, with undo and redo, and saved together (see draft.ts).
//
// Keys: Ctrl+Z undoes, Ctrl+Shift+Z (or Ctrl+Y) redoes, Ctrl+S saves.
import { useDebouncedValue } from '@mantine/hooks';
import { useEffect, useMemo, useState } from 'react';
import { useProject } from '../../api/queries';
import { Button } from '../../components/ui/basics';
import { useBlockHead } from '../../shell/BlockMenu';
import { useRemembered } from '../../shell/remember';
import { useStudio } from '../../studio';
import { BlockLinks } from '../../shell/BlockLinks';
import { useUnsaved } from '../editing';
import { COMPONENT_STARTS, GALAXY_PARTS, isSpheroid, nameBase, partsOf, withThickness, type Part } from './catalogue';
import { useGalaxyDraft } from './draft';
import { Inspector, inPlace, removedFrom } from './Inspector';
import { componentName, componentsOf, freshName, getAt, ringRadii, setAt } from './options';
import { Outline, type Selected } from './Outline';
import { forRender, ownedBy } from './params';
import { startOf } from './schema';
import { Viewer, type Orientation, type Quantity } from './Viewer';

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
  const [selected, setSelected] = useState<Selected | null>(null);
  const [orientation, setOrientation] = useState<Orientation>('sky');
  const [quantity, setQuantity] = useState<Quantity>('light');
  const [isolated, setIsolated] = useState<string | null>(null);
  const options = useMemo(() => draft.state?.options ?? {}, [draft.state]);
  const components = componentsOf(options);
  const colours = components.map((_, i) => COLOURS[i % COLOURS.length]);

  // (the render follows the state the library has described, as it settles)
  const [rendered] = useDebouncedValue(draft.consistent, 60);
  const renderedComponents = rendered ? componentsOf(rendered.options) : [];
  const renderProperties = useMemo(() => (rendered ? forRender(rendered.properties, draft.params) : {}), [rendered, draft.params]);

  // (the part chosen: of a component that is there, else none; at first,
  // the geometry of the first component)
  const valid = selected !== null && (selected.component === null || selected.component < components.length);
  const shown: Selected | null = valid ? selected : components.length ? { component: 0, part: 'geometry' } : null;
  const part = shown === null ? undefined : (shown.component === null ? GALAXY_PARTS : partsOf(components[shown.component])).find((p) => p.key === shown.part);
  const partThere = part !== undefined && (shown?.component === null ? getAt(options, part.option) !== undefined : part.key === 'geometry' || getAt(components[shown!.component!], part.option) !== undefined);

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
  const radii = components.flatMap((c) => (isSpheroid(c.type) ? [Number(c.rmax) || 0] : ringRadii(c.rnodes).slice(-1)));
  const field = Math.max(...radii, 0) * MARGIN || 20;

  const addPart = (component: number | null, p: Part) => {
    draft.edit((s) => {
      const base = component === null ? [] : ['components', component];
      let next = s.options;
      // (mass comes with a distance, which it needs)
      if (p.key === 'mass' && next.distance === undefined) next = setAt(next, ['distance'], { type: 'parameter' });
      if (p.key === 'mass') {
        next = setAt(next, ['mass'], [startOf(draft.defs, 'mass_component', 'halo', [], { name: 'halo' })]);
      } else next = setAt(next, [...base, ...p.option], p.start ?? {});
      return { ...s, options: next };
    });
    setSelected({ component, part: p.key });
  };

  const addComponent = (type: string) => {
    const index = components.length;
    draft.edit(
      (s) => {
        const list = componentsOf(s.options);
        // (several components are all named: one without a name is named
        // after its type, and its parameters with it)
        const named = list.map((c, i) => (typeof c.name === 'string' && c.name ? c : { ...c, name: freshName(list.map((o) => o.name), c.type ? nameBase(String(c.type)) : `component${i + 1}`) }));
        const added = { type, name: freshName(named.map((c) => c.name), nameBase(type)), ...COMPONENT_STARTS[type] };
        return { ...s, options: { ...s.options, components: [...named, added] } };
      },
      { moved: inPlace },
    );
    setSelected({ component: index, part: 'geometry' });
  };

  const renameComponent = (index: number, next: string) => {
    if (components.some((c, i) => i !== index && c.name === next)) return;
    const old = componentName(components[index], index);
    draft.edit((s) => ({ ...s, options: setAt(s.options, ['components', index, 'name'], next) }), { moved: inPlace });
    if (isolated === old) setIsolated(next);
  };

  const duplicateComponent = (index: number) => {
    const original = components[index];
    const oldName = typeof original.name === 'string' ? original.name : null;
    const copyName = freshName(
      components.map((c) => c.name),
      `${oldName ?? String(original.type)}_copy`.replace(/^_/, ''),
    );
    draft.edit(
      (s) => {
        const list = componentsOf(s.options);
        // (its values too: the parameters of a named component are
        // named after it)
        const properties = { ...s.properties };
        if (oldName) {
          for (const p of ownedBy(draft.params, ['components', index])) {
            if (p.name.startsWith(`${oldName}_`) && p.name in s.properties) properties[`${copyName}${p.name.slice(oldName.length)}`] = s.properties[p.name];
          }
        }
        return { options: { ...s.options, components: [...list.slice(0, index + 1), { ...structuredClone(original), name: copyName }, ...list.slice(index + 1)] }, properties };
      },
      // (those after it, a place on)
      { moved: (owner) => (owner[0] === 'components' && typeof owner[1] === 'number' && owner[1] > index ? ['components', owner[1] + 1, ...owner.slice(2)] : owner) },
    );
    setSelected({ component: index + 1, part: shown?.part ?? 'geometry' });
  };

  // (a disk made thin or thick: its density shown, where it changes)
  const changeThickness = (index: number, thick: boolean) => {
    draft.edit((s) => ({ ...s, options: setAt(s.options, ['components', index], withThickness(componentsOf(s.options)[index], thick)) }));
    setSelected({ component: index, part: 'density' });
  };

  const removeComponent = (index: number) => {
    const removed = componentName(components[index], index);
    draft.edit((s) => ({ ...s, options: setAt(s.options, ['components', index], undefined) }), { moved: removedFrom(['components'], index) });
    if (isolated === removed) setIsolated(null);
    if (shown?.component !== null && shown?.component !== undefined) {
      if (shown.component === index) setSelected(null);
      else if (shown.component > index) setSelected({ ...shown, component: shown.component - 1 });
    }
  };

  const focusComponent = shown?.component ?? null;
  const curves = {
    mode: shown?.component === null && (shown.part === 'mass' || shown.part === 'gravity' || shown.part === 'distance') ? ('mass' as const) : ('motions' as const),
    rings: focusComponent === null ? null : ringRadii(components[focusComponent]?.rnodes),
  };

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
        <div className="gx-body gx-loading" aria-busy="true">
          <div className="gx-outline" />
          <div className="gx-view">
            <div className="gx-map-wait" />
          </div>
          <aside className="gx-side" />
        </div>
      ) : (
      <div className="gx-body">
        <Outline
          options={options}
          defs={draft.defs}
          problems={draft.problems}
          colours={colours}
          selected={shown}
          isolated={isolated}
          onSelect={setSelected}
          onAddPart={addPart}
          onAddComponent={addComponent}
          onRenameComponent={renameComponent}
          onDuplicateComponent={duplicateComponent}
          onThickness={changeThickness}
          onRemoveComponent={removeComponent}
          onIsolate={setIsolated}
        />
        <Viewer
          project={project}
          seenBy={seenBy}
          onSeenBy={setSeenBy}
          observations={observations}
          galaxy={components.length && renderedComponents.length ? rendered?.options : undefined}
          properties={renderProperties}
          orientation={orientation}
          onOrientation={setOrientation}
          quantity={quantity}
          onQuantity={setQuantity}
          components={renderedComponents.map((c, i) => ({ name: componentName(c, i), colour: COLOURS[i % COLOURS.length] }))}
          isolated={isolated}
          onIsolated={setIsolated}
          curves={curves}
        />
        <aside className="gx-side">
          {part && partThere && shown ? (
            <Inspector
              key={`${shown.component}/${shown.part}`}
              draft={draft}
              options={options}
              defs={draft.defs}
              params={draft.params}
              project={project}
              field={field}
              component={shown.component}
              part={part}
              problems={draft.problems}
              onClose={() => setSelected(null)}
            />
          ) : (
            <div className="gx-side-empty">
              <p>{components.length ? 'Choose a part in the outline to set it.' : 'Add a component in the outline to start the galaxy.'}</p>
            </div>
          )}
          {draft.problems.length > 0 && <OtherProblems problems={draft.problems} />}
          {draft.describeError && (
            <div className="gx-problems">
              <p>{draft.describeError.message}</p>
            </div>
          )}
        </aside>
      </div>
      )}
    </section>
  );
}

// The problems of the galaxy that are not of a part (e.g. of it as a whole)
function OtherProblems({ problems }: { problems: { path: (string | number)[]; message: string }[] }) {
  const general = problems.filter((p) => p.path.length === 0);
  if (general.length === 0) return null;
  return (
    <div className="gx-problems">
      {general.map((p, i) => (
        <p key={i}>{p.message}</p>
      ))}
    </div>
  );
}
