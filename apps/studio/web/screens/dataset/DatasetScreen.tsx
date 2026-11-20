// The screens of a dataset: a new one, and one of the project (to look at,
// or to edit). Each is a view of its data (as large as the screen allows)
// and a dock at its side: its files, its spectral axis, the edits that
// prep makes (Prepare), what GBKFIT reads from its header, and, for one of
// the project, how it was made.
//
// A new dataset (or one edited) is added once it has no error; the
// counter in the head counts the errors and warnings and leads to the
// field of each (see components/Problems). Without edits, the dataset
// reads its files as they are; with edits, of copies that prep writes into
// the project (the originals stay). Prepare can preview the copies first;
// adding keeps the preview, if it is of the edits as they are.
import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useCreateBlock, useDeleteData, useDescriptions, useKeepJob, usePiece, usePreparation, useProject, useUpdateBlock } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import { Button } from '../../components/ui/basics';
import { ProblemCounter, ProblemField, ProblemsProvider, type Problem } from '../../components/Problems';
import { Dock, DockSection } from '../../components/ui/dock';
import { Segments } from '../../components/ui/inputs';
import { useBlockHead } from '../../shell/BlockMenu';
import { useStudio } from '../../studio';
import { ScreenHead } from '../../shell/ScreenHead';
import { useUnsaved } from '../editing';
import { Files } from './Files';
import { dataKey, requestKey, useDraftState, usePrepJob, useResolved, type DraftState } from './hooks';
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
  isReading,
  kindOfType,
  needsPrep,
  newDraft,
  NO_EDITS,
  NO_SPECTRAL,
  presentSlots,
  preparedOptions,
  prepRequest,
  problemsOf,
  suggestName,
  type Draft,
  type Kind,
  type Prepared,
  type Preparation,
  type PrepRequest,
  type Resolved,
} from './model';
import { PrepareForm } from './Prepare';
import { CopyConfig } from './CopyConfig';
import { hasSpectralChoices, SpectralFields } from './Spectral';
import { DatasetViewer, type Selection } from './Viewer';
import { SheetLayout } from '../../components/ui/sheet';

const hasErrors = (problems: readonly Problem[]) => problems.some((problem) => problem.severity === 'error');

// A failure of the screen itself (e.g. writing the dataset), as a problem
// of no field
const failure = (message: string | null): Problem[] => (message ? [{ severity: 'error', path: [], message }] : []);

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
  const nameProblems: Problem[] =
    trimmed === '' ? [{ severity: 'error', path: ['name'], message: 'name missing' }] : taken.includes(trimmed) ? [{ severity: 'error', path: ['name'], message: `name '${trimmed}' taken` }] : [];
  const problems = [...editor.problems, ...nameProblems, ...failure(commit.error)];
  useUnsaved(firstFile !== undefined || named);
  if (project === null) return null;
  return (
    <ProblemsProvider problems={problems}>
      <section className="screen ds-screen">
        <ScreenHead
          up="data"
          tags={[`New ${KINDS[kind].title.toLowerCase()}`]}
          title={
            <ProblemField path={['name']} label="Name" className="ds-name-field">
              <input
                className="name-input"
                value={name}
                placeholder="Name of the dataset"
                aria-label="Name of the dataset"
                onChange={(event) => {
                  setName(event.target.value);
                  setNamed(true);
                }}
              />
            </ProblemField>
          }
        >
          {commit.phase && <span className="note">{commit.phase}</span>}
          <ProblemCounter />
          <CopyConfig options={editor.options} />
          <Button onClick={() => go({ kind: 'home' })}>Cancel</Button>
          <Button kind="primary" disabled={hasErrors(problems) || editor.reading || commit.busy} onClick={commit.commit}>
            Add to the project
          </Button>
        </ScreenHead>
        <EditorBody project={root} editor={editor} />
      </section>
    </ProblemsProvider>
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
    return <EditDataset name={name} kind={kind} record={record} initial={editing} onDone={() => setEditing(null)} />;
  }
  // (edited from the originals and the edits that made it, if its files
  // were prepared and the originals are there; or else from its files)
  const startEditing = () => {
    if (!kind || !options) return;
    const canRedo = record !== null && originals.every((path) => !describeOriginals(path).error);
    setEditing(canRedo ? draftOfPreparation(kind, options, record, root) : draft);
  };
  const problems = [
    ...(kind && options ? problemsOf(resolved) : []),
    ...failure(piece.error?.message ?? null),
    ...failure(options && !kind ? `type '${String(options.type)}' has no screen` : null),
  ];
  return (
    <ProblemsProvider problems={problems}>
    <section className="screen ds-screen">
      <ScreenHead up="data" title={head.title} tags={kind ? [KINDS[kind].title] : []}>
        <ProblemCounter />
        {options && <CopyConfig options={options} />}
        {kind && <Button onClick={startEditing}>Edit</Button>}
        {head.menu}
      </ScreenHead>
      {kind && options && (
        <SheetLayout
          className="ds-body"
          width={380}
          picture={
            <div className="ds-view">
              <DatasetViewer resolved={resolved} edits={null} selection={selection} onSelection={setSelection} />
            </div>
          }
          sheet={
            <Dock label="The dataset">
              <DockSection id="ds-made" title="Made from">
                {preparation.isLoading ? <p className="note">…</p> : <MadeFrom project={root} kind={kind} dataset={options} record={record} />}
              </DockSection>
              <DockSection id="ds-files" title="Files">
                <Files project={root} resolved={resolved} />
              </DockSection>
              {hasSpectralChoices(resolved) && (
                <DockSection id="ds-spectral" title="Spectral axis">
                  <SpectralFields resolved={resolved} />
                </DockSection>
              )}
            </Dock>
          }
        />
      )}
    </section>
    </ProblemsProvider>
  );
}

// A dataset of the project being edited: saved over it; the copies that
// prep made before and no longer used go
function EditDataset({
  name,
  kind,
  record,
  initial,
  onDone,
}: {
  name: string;
  kind: Kind;
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
  const problems = [...editor.problems, ...failure(commit.error)];
  return (
    <ProblemsProvider problems={problems}>
      <section className="screen ds-screen">
        <ScreenHead up="data" title={name} tags={[KINDS[kind].title, 'Editing']}>
          {commit.phase && <span className="note">{commit.phase}</span>}
          <ProblemCounter />
          <CopyConfig options={editor.options} />
          <Button onClick={onDone}>Cancel</Button>
          <Button kind="primary" disabled={hasErrors(problems) || editor.reading || commit.busy || !dirty} onClick={commit.commit}>
            Save
          </Button>
        </ScreenHead>
        <EditorBody project={root} editor={editor} />
      </section>
    </ProblemsProvider>
  );
}

// ---------------------------------------------------------------------------
// The editor of a draft (of a new dataset, or of one edited)

type PreviewStatus = 'none' | 'running' | 'ready' | 'stale' | 'failed' | 'gone';

interface Editor {
  kind: Kind;
  state: DraftState;
  resolved: Resolved;
  // The problems of the draft (see problemsOf), those of its preview with
  // them; whether it waits for a header; its options as a dataset
  problems: Problem[];
  reading: boolean;
  options: Record<string, unknown>;
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
  const own = problemsOf(resolved);
  const reading = isReading(resolved);
  const request = !hasErrors(own) && !reading && needsPrep(resolved) ? prepRequest(resolved, project) : null;
  const preview = usePrepJob(dataKey(state.draft.items));
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
  const previewProblems: Problem[] =
    previewStatus === 'failed' && preview.error ? preview.error.split('\n').filter((line) => line.trim()).map((message) => ({ severity: 'error', path: ['prepare'], message })) : [];
  const options = datasetOptions(resolved, project);
  return { kind, state, resolved, problems: [...own, ...previewProblems], reading, options, request, preview, previewStatus, prepared: files ? prepared : null };
}

// Adds (or saves) the dataset of an editor: of its files as they are, or
// of the copies of its preview (kept), or of copies that prep writes now
function useCommit(project: string, editor: Editor, write: (content: Record<string, unknown>) => Promise<void>) {
  const keep = useKeepJob();
  const run = usePrepJob(dataKey(editor.state.draft.items));
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

function EditorBody({ project, editor }: { project: string; editor: Editor }) {
  const { kind, state, resolved, prepared, previewStatus } = editor;
  const { draft } = state;
  const [showing, setShowing] = useState<'original' | 'prepared'>('original');
  // (a selection of each view, of the data it was made on: their pixels
  // differ, e.g. cropped; the crop is set from that of the original)
  const files = dataKey(draft.items);
  const [made, setSelections] = useState({ files, original: NO_SELECTION, prepared: NO_SELECTION });
  const selections = made.files === files ? made : { files, original: NO_SELECTION, prepared: NO_SELECTION };
  const selection = selections.original;
  // (a preview shows as it is ready)
  const previewId = editor.preview.job?.id;
  const hasPreview = prepared !== null;
  useEffect(() => {
    if (hasPreview) setShowing('prepared');
  }, [previewId, hasPreview]);
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
    <SheetLayout
      className="ds-body"
      width={380}
      picture={
        <div className="ds-view">
          <DatasetViewer
            key={view}
            resolved={shown}
            edits={view === 'original' ? draft.edits : null}
            selection={selections[view]}
            onSelection={(next) => setSelections({ ...selections, [view]: next })}
            actions={toggle}
          />
        </div>
      }
      sheet={
        <Dock label="The dataset">
          <DockSection id="ds-files" title="Files">
            <Files
              project={project}
              resolved={resolved}
              onItem={state.setItem}
              onAdd={kind === 'moments' ? (key) => state.addItem(key, EMPTY_ITEM) : undefined}
              onRemove={kind === 'moments' ? state.removeItem : undefined}
            />
          </DockSection>
          {hasSpectralChoices(resolved) && (
            <DockSection id="ds-spectral" title="Spectral axis">
              <SpectralFields resolved={resolved} onChange={state.setSpectral} />
            </DockSection>
          )}
          {first && kept(first).length === KINDS[kind].ndim && (
            <DockSection
              id="ds-prepare"
              title="Prepare"
              note={hasEdits(draft.edits, keys) ? 'edited copies' : 'optional'}
              actions={
                hasEdits(draft.edits, keys) && (
                  <Button kind="small" onClick={() => state.setEdits(NO_EDITS)}>
                    Clear
                  </Button>
                )
              }
            >
              <ProblemField path={['prepare']} label="Prepare">
                <PrepareForm resolved={resolved} onEdits={state.setEdits} onCropToSelection={cropToSelection} />
                <PreviewBar editor={editor} onShow={() => setShowing('prepared')} />
              </ProblemField>
            </DockSection>
          )}
        </Dock>
      }
    />
  );
}

// The foot of Prepare: what the edits do to the files, and their preview
function PreviewBar({ editor, onShow }: { editor: Editor; onShow: () => void }) {
  const { request, previewStatus, preview } = editor;
  const keys = presentSlots(editor.kind, editor.state.draft).map((slot) => slot.key);
  if (!hasEdits(editor.state.draft.edits, keys)) return null;
  const status: Record<PreviewStatus, string> = {
    none: '',
    running: 'Preparing…',
    ready: 'Preview of these edits',
    stale: 'Preview of earlier edits',
    failed: 'Preview failed',
    gone: 'Preview removed',
  };
  return (
    <div className="ds-preview">
      <span className="ds-preview-note">{status[previewStatus]}</span>
      <div className="ds-preview-actions">
        {previewStatus === 'ready' ? (
          <Button kind="small" onClick={onShow}>
            Show the preview
          </Button>
        ) : (
          <Button
            kind="small"
            disabled={request === null || previewStatus === 'running'}
            onClick={() => request && preview.start(request, true)}
          >
            {previewStatus === 'stale' || previewStatus === 'gone' ? 'Preview again' : 'Preview'}
          </Button>
        )}
      </div>
    </div>
  );
}

// The data of the first item of a draft, once described
function firstArray({ kind, draft, arrays }: Resolved) {
  return presentSlots(kind, draft)
    .map((slot) => arrays[slot.key]?.found.data)
    .find((data) => data !== undefined);
}
