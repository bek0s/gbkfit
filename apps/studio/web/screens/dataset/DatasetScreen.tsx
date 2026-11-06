// The screens of a dataset: a new one, and one of the project (to look at,
// or to edit). Each is a view of its data (as large as the screen allows)
// and a dock at its side: its files, its spectral axis, the edits that
// prep makes (Prepare), what GBKFIT reads from its header, and, for one of
// the project, how it was made.
//
// A new dataset (or one edited) is added once nothing it needs is
// missing; the head says what is, and leads to it. Without edits, the
// dataset reads its files as they are; with edits, of copies that prep
// writes into the project (the originals stay). Prepare can preview the
// copies first, to look at them beside the originals; adding keeps the
// preview, if it is of the edits as they are.
import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useCreateBlock, useDeleteData, useDescriptions, useKeepJob, usePiece, usePreparation, useProject, useUpdateBlock } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import { Button } from '../../components/ui/basics';
import { Dock, DockSection, NeedsNote as DockNeeds } from '../../components/ui/dock';
import { Segments } from '../../components/ui/inputs';
import { useBlockHead } from '../../shell/BlockMenu';
import { useStudio } from '../../studio';
import { useUnsaved } from '../editing';
import { Files } from './Files';
import { requestKey, useDraftState, usePrepJob, useResolved, type DraftState } from './hooks';
import { MadeFrom, originalsOf, preparedFilesOf } from './MadeFrom';
import {
  dataFiles,
  datasetOptions,
  draftOfDataset,
  draftOfPrepared,
  draftOfPreparation,
  EMPTY_ITEM,
  hasEdits,
  kept,
  KINDS,
  kindOfType,
  needsOf,
  needsPrep,
  newDraft,
  NO_EDITS,
  NO_SPECTRAL,
  presentSlots,
  preparedOptions,
  prepRequest,
  suggestName,
  type Draft,
  type Kind,
  type Need,
  type Place,
  type Prepared,
  type Preparation,
  type PrepRequest,
  type Resolved,
} from './model';
import { PrepareForm } from './Prepare';
import { hasSpectralChoices, SpectralFields } from './Spectral';
import { DatasetViewer, type Selection } from './Viewer';

// The element of the page where the user gives what a place needs
const PLACES: Record<Place, string> = {
  name: 'ds-name',
  files: 'ds-files',
  spectral: 'ds-spectral',
  prepare: 'ds-prepare',
};

const NO_SELECTION: Selection = { region: null, band: null };

const NOTHING: Draft = { items: {}, spectral: NO_SPECTRAL, edits: NO_EDITS };

// ---------------------------------------------------------------------------
// A new dataset

export function NewDataset({ kind }: { kind: Kind }) {
  const { project, go, setUnsaved } = useStudio();
  const root = project ?? '.';
  const taken = useProject(project).data?.pieces.datasets ?? [];
  const editor = useEditor(root, kind, newDraft(kind));
  const create = useCreateBlock(root);
  // (the name follows the file of the data until the user types one)
  const [name, setName] = useState('');
  const [named, setNamed] = useState(false);
  const firstFile = presentSlots(kind, editor.state.draft)
    .map((slot) => editor.state.draft.items[slot.key].files.data?.file)
    .find((file) => file !== undefined);
  useEffect(() => {
    // (not of the names taken, which change as this one is added)
    if (!named) setName(firstFile ? suggestName(firstFile, taken) : '');
  }, [firstFile, named]);
  const trimmed = name.trim();
  const commit = useCommit(root, editor, async (content) => {
    await create.mutateAsync({ folder: 'datasets', name: trimmed, content });
    setUnsaved(false);
    go({ kind: 'block', folder: 'datasets', name: trimmed });
  });
  const nameNeed: Need[] =
    trimmed === '' ? [{ place: 'name', text: 'Name the dataset' }] : taken.includes(trimmed) ? [{ place: 'name', text: 'The project has a dataset of that name' }] : [];
  const needs = [...editor.needs, ...nameNeed];
  useUnsaved(firstFile !== undefined || named);
  if (project === null) return null;
  return (
    <section className="screen ds-screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">New dataset · {KINDS[kind].title}</div>
          <input
            id={PLACES.name}
            className="name-input"
            value={name}
            placeholder="Name of the dataset"
            aria-label="Name of the dataset"
            onChange={(event) => {
              setName(event.target.value);
              setNamed(true);
            }}
          />
        </div>
        <div className="head-actions">
          {commit.phase ? <span className="note">{commit.phase}</span> : <NeedsNote needs={needs} />}
          <Button onClick={() => go({ kind: 'home' })}>Cancel</Button>
          <Button kind="primary" disabled={needs.length > 0 || commit.busy} title={needs.map((need) => need.text).join('\n') || undefined} onClick={commit.commit}>
            Add to the project
          </Button>
        </div>
      </div>
      {commit.error && <pre className="error screen-error">{commit.error}</pre>}
      <EditorBody project={root} editor={editor} />
    </section>
  );
}

// ---------------------------------------------------------------------------
// A dataset of the project

export function Dataset({ name }: { name: string }) {
  const { project } = useStudio();
  const root = project ?? '.';
  const piece = usePiece(project === null ? null : joinPath(root, `datasets/${name}.yaml`));
  const options = piece.data as Record<string, unknown> | undefined;
  const kind = kindOfType(options?.type);
  const head = useBlockHead(root, 'datasets', name);
  const draft = useMemo(() => (kind && options ? draftOfDataset(kind, options, root) : NOTHING), [kind, options, root]);
  const resolved = useResolved(kind ?? 'cube', draft);
  const firstData = kind && options ? Object.values(dataFiles(kind, options, root))[0] : undefined;
  const preparation = usePreparation(firstData ?? null);
  const record = (preparation.data ?? null) as Preparation | null;
  const originals = record ? originalsOf(record) : [];
  const describeOriginals = useDescriptions(originals);
  const [editing, setEditing] = useState<Draft | null>(null);
  const [selection, setSelection] = useState(NO_SELECTION);
  if (project === null) return null;
  if (editing && kind && options) {
    return <EditDataset name={name} kind={kind} dataset={options} record={record} initial={editing} onDone={() => setEditing(null)} />;
  }
  // (edited from the originals and the edits that made it, if its files
  // were prepared and the originals are there; or else from its files)
  const startEditing = () => {
    if (!kind || !options) return;
    const canRedo = record !== null && originals.every((path) => !describeOriginals(path).error);
    setEditing(canRedo ? draftOfPreparation(kind, options, record, root) : draft);
  };
  return (
    <section className="screen ds-screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">Dataset{kind ? ` · ${KINDS[kind].title}` : ''}</div>
          {head.title}
        </div>
        <div className="head-actions">
          {kind && <Button onClick={startEditing}>Edit</Button>}
          {head.menu}
        </div>
      </div>
      {piece.error && <p className="error screen-error">{piece.error.message}</p>}
      {options && !kind && <p className="note screen-error">Datasets of the type {String(options.type)} have no screen yet.</p>}
      {kind && options && (
        <div className="ds-body">
          <div className="ds-view">
            <DatasetViewer resolved={resolved} edits={null} selection={selection} onSelection={setSelection} />
          </div>
          <Dock label="The dataset">
            <DockSection id="ds-made" title="Made from">
              {preparation.isLoading ? <p className="note">…</p> : <MadeFrom project={root} kind={kind} dataset={options} record={record} />}
            </DockSection>
            <DockSection id={PLACES.files} title="Files">
              <Files project={root} resolved={resolved} />
            </DockSection>
            {hasSpectralChoices(resolved) && (
              <DockSection id={PLACES.spectral} title="Spectral axis">
                <SpectralFields resolved={resolved} />
              </DockSection>
            )}
          </Dock>
        </div>
      )}
    </section>
  );
}

// A dataset of the project being edited: saved over it; the copies that
// prep made before and no longer used go
function EditDataset({
  name,
  kind,
  dataset,
  record,
  initial,
  onDone,
}: {
  name: string;
  kind: Kind;
  dataset: Record<string, unknown>;
  record: Preparation | null;
  initial: Draft;
  onDone: () => void;
}) {
  const { project } = useStudio();
  const root = project ?? '.';
  const path = joinPath(root, `datasets/${name}.yaml`);
  const editor = useEditor(root, kind, initial);
  const update = useUpdateBlock();
  const remove = useDeleteData();
  const client = useQueryClient();
  const commit = useCommit(root, editor, async (content) => {
    await update.mutateAsync({ path, content });
    // (the copies of the last preparation that the dataset no longer reads;
    // the server keeps any that the project still names)
    if (record) {
      const kept = new Set(Object.values(draftOfDataset(kind, content, root).items).flatMap((item) => Object.values(item.files).map((file) => file.file)));
      for (const file of preparedFilesOf(record)) if (!kept.has(file)) await remove.mutateAsync(file).catch(() => undefined);
    }
    void client.invalidateQueries({ queryKey: ['preparation'] });
    onDone();
  });
  const dirty = JSON.stringify(editor.state.draft) !== JSON.stringify(initial);
  useUnsaved(dirty);
  const needs = editor.needs;
  return (
    <section className="screen ds-screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">Dataset · {KINDS[kind].title} · editing</div>
          <h1>{name}</h1>
        </div>
        <div className="head-actions">
          {commit.phase ? <span className="note">{commit.phase}</span> : <NeedsNote needs={needs} />}
          <Button onClick={onDone}>Cancel</Button>
          <Button kind="primary" disabled={needs.length > 0 || commit.busy || !dirty} title={needs.map((need) => need.text).join('\n') || undefined} onClick={commit.commit}>
            Save
          </Button>
        </div>
      </div>
      {commit.error && <pre className="error screen-error">{commit.error}</pre>}
      <EditorBody project={root} editor={editor} dataset={dataset} />
    </section>
  );
}

// ---------------------------------------------------------------------------
// The editor of a draft (of a new dataset, or of one edited)

type PreviewStatus = 'none' | 'running' | 'ready' | 'stale' | 'failed' | 'gone';

interface Editor {
  kind: Kind;
  state: DraftState;
  resolved: Resolved;
  needs: Need[];
  // The job of prep of the draft (null: it needs none, or it needs
  // something first)
  request: PrepRequest | null;
  preview: ReturnType<typeof usePrepJob>;
  previewStatus: PreviewStatus;
  // The files of the preview, resolved (null: none to look at)
  prepared: Resolved | null;
}

function useEditor(project: string, kind: Kind, initial: Draft): Editor {
  const state = useDraftState(initial);
  const resolved = useResolved(kind, state.draft);
  const needs = needsOf(resolved);
  const request = needs.length === 0 && needsPrep(resolved) ? prepRequest(resolved, project) : null;
  const preview = usePrepJob();
  const files = preview.job?.state === 'complete' && preview.job.preview ? (preview.job.files as Prepared | null) : null;
  const prepared = useResolved(kind, files ? draftOfPrepared(kind, state.draft, files) : NOTHING);
  const previewStatus: PreviewStatus = preview.running
    ? 'running'
    : preview.error
      ? 'failed'
      : preview.job?.state === 'complete' && preview.job.preview && !files
        ? 'gone'
        : files
          ? request && preview.key === requestKey(request)
            ? 'ready'
            : 'stale'
          : 'none';
  return { kind, state, resolved, needs, request, preview, previewStatus, prepared: files ? prepared : null };
}

// Adds (or saves) the dataset of an editor: of its files as they are, or
// of the copies of its preview (kept), or of copies that prep writes now
function useCommit(project: string, editor: Editor, write: (content: Record<string, unknown>) => Promise<void>) {
  const keep = useKeepJob();
  const run = usePrepJob();
  const [writing, setWriting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const finished = useRef<string | null>(null);
  const finish = async (prepared: Prepared | null) => {
    setWriting(true);
    setError(null);
    try {
      await write(prepared ? preparedOptions(editor.resolved, prepared, project) : datasetOptions(editor.resolved, project));
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setWriting(false);
    }
  };
  // (the files prepared, the dataset is written; once for each job)
  useEffect(() => {
    const job = run.job;
    if (job?.state === 'complete' && job.files && finished.current !== job.id) {
      finished.current = job.id;
      void finish(job.files as Prepared);
    }
  });
  const commit = () => {
    setError(null);
    if (editor.request === null) return void finish(null);
    const { job } = editor.preview;
    if (editor.previewStatus === 'ready' && job) {
      keep
        .mutateAsync(job.id)
        .then((kept) => (kept.files ? finish(kept.files as Prepared) : setError('The preview has no files to keep')))
        .catch((failure: Error) => setError(failure.message));
      return;
    }
    run.start(editor.request, false);
  };
  const preparing = run.running || keep.isPending;
  return {
    commit,
    busy: preparing || writing,
    phase: preparing ? 'Preparing the files…' : writing ? 'Saving…' : null,
    error: error ?? run.error ?? keep.error?.message ?? null,
  };
}

function EditorBody({ project, editor, dataset }: { project: string; editor: Editor; dataset?: Record<string, unknown> }) {
  const { kind, state, resolved, needs, prepared, previewStatus } = editor;
  const { draft } = state;
  const [showing, setShowing] = useState<'original' | 'prepared'>('original');
  // (a selection of each view: their pixels differ, e.g. cropped; the crop
  // is set from that of the original)
  const [selections, setSelections] = useState({ original: NO_SELECTION, prepared: NO_SELECTION });
  const selection = selections.original;
  // (a preview shows as it is ready)
  const previewId = editor.preview.job?.id;
  const hasPreview = prepared !== null;
  useEffect(() => {
    if (hasPreview) setShowing('prepared');
  }, [previewId, hasPreview]);
  const places = new Set(needs.map((need) => need.place));
  const first = firstArray(resolved);
  const view = showing === 'prepared' && prepared ? 'prepared' : 'original';
  const shown = view === 'prepared' && prepared ? prepared : resolved;
  const toggle = prepared && (
    <Segments
      items={[
        { value: 'original', label: 'Original' },
        { value: 'prepared', label: previewStatus === 'stale' ? 'Prepared (earlier edits)' : 'Prepared' },
      ]}
      value={showing}
      label="The files shown"
      onChange={setShowing}
    />
  );
  const keys = presentSlots(kind, draft).map((slot) => slot.key);
  const cropToSelection =
    selection.region === null && selection.band === null
      ? undefined
      : () => {
          let edits = draft.edits;
          if (selection.region) {
            const [x0, x1, y0, y1] = selection.region;
            edits = { ...edits, x: [String(x0), String(x1)], y: [String(y0), String(y1)] };
          }
          if (selection.band && kind === 'cube') edits = { ...edits, channels: [String(selection.band[0]), String(selection.band[1])] };
          state.setEdits(edits);
        };
  return (
    <div className="ds-body">
      <div className="ds-view">
        <DatasetViewer
          key={view}
          resolved={shown}
          edits={view === 'original' ? draft.edits : null}
          selection={selections[view]}
          onSelection={(next) => setSelections((all) => ({ ...all, [view]: next }))}
          actions={toggle}
        />
      </div>
      <Dock label="The dataset">
        <DockSection id={PLACES.files} title="Files" needed={places.has('files')}>
          <Files
            project={project}
            resolved={resolved}
            onItem={state.setItem}
            onAdd={kind === 'moments' ? (key) => state.addItem(key, EMPTY_ITEM) : undefined}
            onRemove={kind === 'moments' ? state.removeItem : undefined}
          />
        </DockSection>
        {hasSpectralChoices(resolved) && (
          <DockSection id={PLACES.spectral} title="Spectral axis" needed={places.has('spectral')}>
            <SpectralFields resolved={resolved} onChange={state.setSpectral} />
          </DockSection>
        )}
        {first && kept(first).length === KINDS[kind].ndim && (
          <DockSection
            id={PLACES.prepare}
            title="Prepare"
            note={hasEdits(draft.edits, keys) ? 'makes edited copies' : 'optional'}
            needed={places.has('prepare')}
            actions={
              hasEdits(draft.edits, keys) && (
                <Button kind="small" onClick={() => state.setEdits(NO_EDITS)}>
                  Clear
                </Button>
              )
            }
          >
            <PrepareForm resolved={resolved} onEdits={state.setEdits} onCropToSelection={cropToSelection} />
            <PreviewBar editor={editor} onShow={() => setShowing('prepared')} />
          </DockSection>
        )}
        {dataset && <p className="ds-dock-note">Saving prepares new copies from the originals; the copies of the last preparation that are no longer used go.</p>}
      </Dock>
    </div>
  );
}

// The foot of Prepare: what the edits do to the files, and their preview
function PreviewBar({ editor, onShow }: { editor: Editor; onShow: () => void }) {
  const { request, previewStatus, preview } = editor;
  const keys = presentSlots(editor.kind, editor.state.draft).map((slot) => slot.key);
  if (!hasEdits(editor.state.draft.edits, keys)) {
    return <p className="ds-preview-note">Without edits, the dataset reads the files as they are.</p>;
  }
  const status: Record<PreviewStatus, string> = {
    none: 'Adding writes edited copies into the project; the originals stay.',
    running: 'Preparing a preview…',
    ready: 'The preview is of these edits: adding keeps it.',
    stale: 'The edits changed since the preview.',
    failed: 'The preview failed:',
    gone: 'The preview was removed (newer ones were made).',
  };
  return (
    <div className="ds-preview">
      <p className="ds-preview-note">{status[previewStatus]}</p>
      {previewStatus === 'failed' && preview.error && <pre className="error">{preview.error}</pre>}
      <div className="ds-preview-actions">
        {previewStatus === 'ready' ? (
          <Button kind="small" onClick={onShow}>
            Show the preview
          </Button>
        ) : (
          <Button
            kind="small"
            disabled={request === null || previewStatus === 'running'}
            title={request === null ? 'The dataset needs something first (see the head of the screen)' : 'Prepare the copies, to look at them before adding'}
            onClick={() => request && preview.start(request, true)}
          >
            {previewStatus === 'stale' || previewStatus === 'gone' ? 'Preview again' : 'Preview'}
          </Button>
        )}
      </div>
    </div>
  );
}

// What the dataset needs, leading to where it is given
function NeedsNote({ needs }: { needs: Need[] }) {
  return <DockNeeds needs={needs.map((need) => ({ text: need.text, at: PLACES[need.place] }))} />;
}

// The data of the first item of a draft, once described
function firstArray({ kind, draft, arrays }: Resolved) {
  return presentSlots(kind, draft)
    .map((slot) => arrays[slot.key]?.found.data)
    .find((data) => data !== undefined);
}
