// The screen of an observation in one of the drafts of its layout (see
// shell/drafts): its head names the block (one of its folder,
// on the screen of a draft), switches between the drafts, chooses one for
// the blocks of the folder, and saves; its edits stay as the draft
// changes.
import type { ComponentType } from 'react';
import { useProject } from '../api/queries';
import { Button } from '../components/ui/basics';
import { Choice, Segments } from '../components/ui/inputs';
import { useBlockHead } from '../shell/BlockMenu';
import { kindOf } from '../shell/blocks';
import { draft, draftsOf, type Draft } from '../shell/drafts';
import { useRemembered } from '../shell/remember';
import { useStudio } from '../studio';
import { SaveActions, useEditedBlock } from './editing';
import { ObservationForm, ObservationPipeline, type LayoutProps } from './Observation';

const LAYOUTS: Record<string, ComponentType<LayoutProps>> = {
  'observation-pipeline': ObservationPipeline,
  'observation-form': ObservationForm,
};

// The draft chosen for the blocks of a folder (the first by default), and
// how another is chosen
export function useLayout(folder: string) {
  const [id, setId] = useRemembered(`layout:${folder}`, draftsOf(folder)[0]?.id ?? '');
  return [draft(id) ?? draftsOf(folder)[0], setId] as const;
}

// A block of the project in the draft chosen for its folder
export function BlockInLayout({ folder, name }: { folder: string; name: string }) {
  const [chosen] = useLayout(folder);
  return <DraftView key={`${folder}/${name}`} shown={chosen} name={name} fixed />;
}

// The screen of a draft (from the tree), of a block of its folder (none:
// the first)
export function DraftScreen({ id, name }: { id: string; name: string | null }) {
  const { project } = useStudio();
  const shown = draft(id);
  const names = useProject(project).data?.pieces[shown?.folder ?? ''] ?? [];
  if (!shown) return <p className="error screen-error">No draft {id}.</p>;
  const block = name ?? names[0] ?? null;
  if (block === null) {
    return (
      <section className="screen">
        <div className="screen-head">
          <div>
            <div className="eyebrow">
              Draft {shown.letter} · {kindOf(shown.folder)}
            </div>
            <h1>{shown.title}</h1>
          </div>
        </div>
        <div className="screen-body">
          <p className="note">{shown.text}</p>
          <p className="note">The project has no {kindOf(shown.folder).toLowerCase()} yet: add one (+ in the tree) to see it in this draft.</p>
        </div>
      </section>
    );
  }
  return <DraftView key={`${shown.folder}/${block}`} shown={shown} name={block} />;
}

function DraftView({ shown, name, fixed = false }: { shown: Draft; name: string; fixed?: boolean }) {
  const { project, go } = useStudio();
  const [chosen, choose] = useLayout(shown.folder);
  const names = useProject(project).data?.pieces[shown.folder] ?? [];
  const edited = useEditedBlock(project ?? '.', shown.folder, name);
  const head = useBlockHead(project ?? '', shown.folder, name);
  const drafts = draftsOf(shown.folder);
  const Layout = LAYOUTS[shown.id];
  if (project === null) return null;
  // (another draft: on the screen of a block, it is chosen for its folder;
  // on that of a draft, the screen of the other)
  const switchTo = (id: string) => (fixed ? choose(id) : go({ kind: 'draft', id, name }));
  return (
    <section className="screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">
            {kindOf(shown.folder)} · draft {shown.letter}: {shown.title}
          </div>
          {fixed ? (
            head.title
          ) : (
            <div className="draft-block">
              <Choice
                items={names.map((n) => ({ value: n, label: n }))}
                value={name}
                label={`The ${kindOf(shown.folder).toLowerCase()} shown`}
                onChange={(next) => go({ kind: 'draft', id: shown.id, name: next })}
              />
            </div>
          )}
        </div>
        <div className="head-actions">
          <Segments
            items={drafts.map((d) => ({ value: d.id, label: `${d.letter} · ${d.title}` }))}
            value={shown.id}
            label="The draft of the layout"
            onChange={switchTo}
          />
          {!fixed && chosen?.id !== shown.id && (
            <Button kind="small" title={`Open the ${shown.folder} of the project in this draft`} onClick={() => choose(shown.id)}>
              Use for {shown.folder}
            </Button>
          )}
          <SaveActions edited={edited} />
          {head.menu}
        </div>
      </div>
      {edited.error && <p className="error screen-error">{edited.error.message}</p>}
      <p className="note draft-text">{shown.text}</p>
      <Layout project={project} name={name} edited={edited} />
    </section>
  );
}
