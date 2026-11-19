// The screen of a fit: a setup of the project (setups/<name>), with its
// fitter. Its sheet is the recipe, what the next run fits (see Recipe):
// the galaxies, lenses and observations it puts together, its parameters
// (each with the layer it comes from; the edits go to the fit's own), and
// its fitter. Its picture is its runs (see Runs): a strip of them, oldest
// first, and the next run at its end (see Start: the checks of the
// recipe, and the model at its starts against the data). The recipe is
// edited as a draft, described by the library as it settles, and saved
// as a whole; Run saves it, and queues a run of it on the server.
//
// Keys: Ctrl+S saves.
import { useDebouncedValue } from '@mantine/hooks';
import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { usePiece, useSetupDescription, useSubmitJob, useUpdateBlock } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import { Button } from '../../components/ui/basics';
import { SheetLayout } from '../../components/ui/sheet';
import { BlockLinks } from '../../shell/BlockLinks';
import { useBlockHead } from '../../shell/BlockMenu';
import { useStudio } from '../../studio';
import { useUnsaved } from '../editing';
import { isObject } from '../galaxy/options';
import { RecipeSheet, type SetupParam } from './Recipe';
import { itemsOf, withOwnProperty, type Options } from './recipe';
import { RunCard, RunPage, useFitRuns } from './Runs';
import { StartView } from './Start';

// The draft of a setup: as saved, or as edited
function useSetupDraft(path: string) {
  const piece = usePiece(path);
  const update = useUpdateBlock();
  const saved = useMemo(() => (isObject(piece.data) ? piece.data : undefined), [piece.data]);
  const [edited, setEdited] = useState<Options | null>(null);
  const state = edited ?? saved;
  const dirty = edited !== null && JSON.stringify(edited) !== JSON.stringify(saved);
  const edit = useCallback((change: (setup: Options) => Options) => setEdited((current) => change(current ?? saved ?? {})), [saved]);
  const save = useCallback(async () => {
    if (!dirty || edited === null) return;
    await update.mutateAsync({ path, content: edited });
    setEdited(null);
  }, [dirty, edited, path, update]);
  return { state, saved, loaded: saved !== undefined, error: piece.error, dirty, edit, save, revert: () => setEdited(null), saving: update.isPending, saveError: update.error };
}

export type SetupDraft = ReturnType<typeof useSetupDraft>;

export function FitScreen({ name }: { name: string }) {
  const { project } = useStudio();
  const root = project ?? '.';
  const path = joinPath(root, `setups/${name}.yaml`);
  const draft = useSetupDraft(path);
  const head = useBlockHead(root, 'setups', name);
  useUnsaved(draft.dirty);
  // (described by the library as it settles)
  const [settled] = useDebouncedValue(draft.state, 300);
  const description = useSetupDescription(path, settled);
  const params = (description.data?.parameters ?? []) as unknown as SetupParam[];
  const problems = (description.data?.problems ?? []) as { severity: string; path: (string | number)[]; message: string }[];
  const { save } = draft;
  // (the run shown: by its key, or the id of its job; null: the next one)
  const [shown, setShown] = useState<string | null>(null);
  const submit = useSubmitJob();
  const errors = problems.some((p) => p.severity === 'error');
  const run = async () => {
    if (draft.dirty) await draft.save();
    const job = await submit.mutateAsync({ task: 'fit', options: { config: path }, preview: false });
    setShown(job.id);
  };
  // (the best values of a run as the starts of the free parameters: in the
  // fit's own layer, their ranges kept; the elements of vectors set apart
  // are left as they are)
  const startFrom = (values: Record<string, number>) =>
    draft.edit((s) =>
      Object.entries(values).reduce((next, [key, value]) => {
        if (key.includes('[')) return next;
        const property = params.find((p) => p.name === key)?.properties[key];
        return withOwnProperty(next, key, isObject(property) ? { ...property, value } : { value });
      }, s),
    );
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
        event.preventDefault();
        void save();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [save]);
  if (project === null) return null;
  const setup = draft.state;
  const observations = setup ? itemsOf(setup, 'observations').length : 0;
  const fitter = setup?.fitter;
  const fitterText = typeof fitter === 'string' ? fitter : isObject(fitter) && typeof fitter.type === 'string' ? fitter.type : 'no fitter';
  return (
    <section className="screen fx-screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">
            Fit · {observations} {observations === 1 ? 'observation' : 'observations'} · {fitterText}
          </div>
          {head.title}
        </div>
        <BlockLinks folder="setups" name={name} />
        <div className="head-actions">
          {draft.dirty && <span className="why">Not saved</span>}
          {draft.saveError && <span className="error">{draft.saveError.message}</span>}
          {submit.error && <span className="error">{submit.error.message}</span>}
          <Button disabled={!draft.dirty} onClick={draft.revert}>
            Revert
          </Button>
          <Button disabled={!draft.dirty || draft.saving} onClick={() => void draft.save()}>
            Save
          </Button>
          <Button
            kind="primary"
            disabled={!setup || setup.fitter === undefined || errors || description.isPending || submit.isPending}
            title={setup?.fitter === undefined ? 'The fit needs a fitter' : errors ? 'The recipe has an error (see Before it runs)' : 'Saves the recipe, and runs it on the server (it waits if the server is busy)'}
            onClick={() => void run()}
          >
            Run
          </Button>
          {head.menu}
        </div>
      </div>
      {draft.error && <p className="error screen-error">{draft.error.message}</p>}
      {!draft.loaded || !setup ? (
        <p className="note screen-error">Reading the fit…</p>
      ) : (
        <SheetLayout
          className="fx-body"
          width={480}
          sheet={<RecipeSheet project={project} path={path} setup={setup} draft={draft} params={params} problems={problems} described={description.isSuccess && !description.isPlaceholderData} />}
          picture={
            <FitPicture project={project} name={name} path={path} shown={shown} onShow={setShown} onStarts={startFrom}>
              <StartView path={path} setup={settled} params={params} problems={problems} pending={description.isPending} failed={description.error?.message ?? null} />
            </FitPicture>
          }
        />
      )}
    </section>
  );
}

// The picture of a fit: the strip of its runs and the next run, and the
// one shown (the next one: its checks and start, given)
function FitPicture({
  project,
  name,
  path,
  shown,
  onShow,
  onStarts,
  children,
}: {
  project: string;
  name: string;
  path: string;
  shown: string | null;
  onShow: (key: string | null) => void;
  onStarts: (values: Record<string, number>) => void;
  children: ReactNode;
}) {
  const entries = useFitRuns(project, name, path);
  const current = shown === null ? undefined : entries.find((entry) => entry.key === shown || entry.job?.id === shown);
  return (
    <div className="fx-picture">
      <ol className="fx-runs-strip" aria-label="The runs of the fit">
        {entries.map((entry) => (
          <RunCard key={entry.key} entry={entry} current={entry === current} onOpen={() => onShow(entry.key)} />
        ))}
        <li className="fx-strip-next">
          <button aria-current={current === undefined || undefined} onClick={() => onShow(null)}>
            <span className="fx-strip-top">
              <span className="fx-strip-title">{entries.length ? 'Next run' : 'First run'}</span>
            </span>
            <span className="fx-strip-glance">checks · the model at its start</span>
          </button>
        </li>
      </ol>
      {current ? <RunPage key={current.key} entry={current} onStarts={onStarts} /> : children}
    </div>
  );
}
