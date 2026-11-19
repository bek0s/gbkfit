// The runs of a fit: those in the folder of its runs (runs/<fit>/<n>,
// numbered; the runs of older projects, runs/<fit> and runs/<fit>_<k>,
// too), and its jobs that wait or run (a job that runs writes its run);
// as a strip, oldest first, with the next run at its end. A run shows
// what it is (its state, when it started and ran), what stops it while it
// runs, its log (followed while it runs), and its recipe as it ran.
import { useMemo, useState } from 'react';
import { useFileText, useJobs, useProject, useRun, useStopJob } from '../../api/queries';
import { Button } from '../../components/ui/basics';
import { Segments } from '../../components/ui/inputs';
import { LiveView } from './Live';
import { ResultView } from './Results';

type Job = NonNullable<ReturnType<typeof useJobs>['data']>[number];

export interface RunEntry {
  // (its number, or its label: e.g. 'queued', or the folder of an older run)
  key: string;
  title: string;
  run: string | null;
  job: Job | null;
}

// The runs of a fit of a project (its setup, by name, and its file
// relative to the workspace), oldest first
export function useFitRuns(project: string, name: string, path: string): RunEntry[] {
  const runs = useProject(project).data?.runs ?? [];
  const jobs = useJobs().data ?? [];
  return useMemo(() => {
    const folder = `${project}/runs/${name}`;
    const older = new RegExp(`^${escape(folder)}(_\\d+)?$`);
    const numbered = new RegExp(`^${escape(folder)}/(\\d+)$`);
    const mine = jobs.filter((job) => job.task === 'fit' && job.options.config === path);
    const entries: RunEntry[] = [];
    for (const run of runs) {
      const number = numbered.exec(run)?.[1];
      if (number === undefined && !older.test(run)) continue;
      entries.push({ key: run, title: number !== undefined ? `#${number}` : run.slice(project.length + 1), run, job: mine.findLast((job) => job.run === run) ?? null });
    }
    // (numbered runs in their order; the older ones first)
    entries.sort((a, b) => order(a.title) - order(b.title));
    for (const job of mine) {
      if (job.run !== null && runs.includes(job.run)) continue;
      if (job.state !== 'queued' && job.state !== 'running') continue;
      const number = job.run ? numbered.exec(job.run)?.[1] : undefined;
      entries.push({ key: job.id, title: number !== undefined ? `#${number}` : 'Waiting', run: job.run, job });
    }
    return entries;
  }, [runs, jobs, project, name, path]);
}

function order(title: string): number {
  return title.startsWith('#') ? Number(title.slice(1)) : -1;
}

function escape(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// The state of a run, as a word and a sign (never colour alone): of its
// job while it waits or runs, else of its manifest
export function runState(entry: RunEntry, status: string | undefined): { word: string; sign: string; state: string } {
  const state = entry.job && (entry.job.state === 'queued' || entry.job.state === 'running') ? entry.job.state : (status ?? entry.job?.state ?? 'unknown');
  const signs: Record<string, [string, string]> = {
    queued: ['Waiting', '◌'],
    running: ['Running', '●'],
    complete: ['Done', '✓'],
    stopped: ['Stopped', '■'],
    failed: ['Failed', '✕'],
    cancelled: ['Cancelled', '✕'],
  };
  const [word, sign] = signs[state] ?? [state, '·'];
  return { word, sign, state };
}

export function RunCard({ entry, current, onOpen }: { entry: RunEntry; current: boolean; onOpen: () => void }) {
  const manifest = useRun(entry.run).data;
  const { word, sign, state } = runState(entry, manifest?.status);
  return (
    <li>
      <button aria-current={current || undefined} onClick={onOpen}>
        <span className="fx-strip-top">
          <span className="fx-state" data-state={state}>
            <span className="fx-state-sign">{sign}</span>
          </span>
          <span className="fx-strip-title">{entry.title}</span>
        </span>
        <span className="fx-strip-glance">
          {word}
          {manifest ? ` · ${when(manifest.created)}` : entry.job ? ` · ${when(entry.job.created)}` : ''}
        </span>
      </button>
    </li>
  );
}

// A run: its state and times, Stop while it runs, and its result (see
// Results), its log, or its recipe as it ran
export function RunPage({ project, entry, onStarts }: { project: string; entry: RunEntry; onStarts: (values: Record<string, number>) => void }) {
  const manifest = useRun(entry.run).data;
  const stop = useStopJob();
  const { word, state } = runState(entry, manifest?.status);
  const running = state === 'running' || state === 'queued';
  const products = manifest?.products ?? [];
  const [chosen, setTab] = useState<'result' | 'log' | 'recipe' | null>(null);
  // (the result, once there is one; else the log)
  const tab = chosen ?? (products.some((p) => p.kind === 'fit_result') || running ? 'result' : 'log');
  const productPath = (kind: string) => {
    const product = products.find((p) => p.kind === kind);
    return product && entry.run ? `${entry.run}/${String(product.path)}` : null;
  };
  const logPath = productPath('log');
  const log = useFileText(tab === 'log' ? logPath : null, running ? 3000 : false, manifest?.updated);
  const recipe = useFileText(tab === 'recipe' ? productPath('config') : null);
  const shown = tab === 'log' ? log : recipe;
  return (
    <>
      <div className="fx-run-head">
        <div className="fx-run-what">
          <span className="fx-overline">Run {entry.title}</span>
          <span className="fx-run-title">
            {word}
            {manifest ? ` · started ${when(manifest.created)}` : ''}
            {manifest && !running ? ` · ended ${when(manifest.updated)}` : ''}
          </span>
        </div>
        <Segments
          label="What of the run"
          value={tab}
          onChange={setTab}
          items={[
            { value: 'result', label: running ? 'Live' : 'Result' },
            { value: 'log', label: 'Log' },
            { value: 'recipe', label: 'Its recipe' },
          ]}
        />
        <span className="fx-right">
          {running && entry.job && (
            <Button kind="small" disabled={stop.isPending} title="Ends the run: a fit keeps its best values so far" onClick={() => stop.mutate(entry.job!.id)}>
              Stop
            </Button>
          )}
          {entry.run && <span className="fx-faint fx-small fx-run-folder">{entry.run}</span>}
        </span>
      </div>
      <div className="fx-scroll">
        {entry.job?.state === 'failed' && entry.job.output && <pre className="fx-log fx-log-error">{entry.job.output}</pre>}
        {state === 'queued' ? (
          <p className="ob-text">It waits for a free place on the server; it runs as the runs before it end.</p>
        ) : tab === 'result' ? (
          running ? <LiveView path={productPath('progress')} version={manifest?.updated ?? ''} /> : entry.run && <ResultView project={project} run={entry.run} products={products as { kind: string; path: string; ids?: Record<string, string> }[]} onStarts={onStarts} />
        ) : shown.error ? (
          <p className="error">{shown.error.message}</p>
        ) : shown.data !== undefined ? (
          <pre className="fx-log">{shown.data || '(empty)'}</pre>
        ) : (tab === 'log' ? logPath : productPath('config')) === null ? (
          <p className="ob-text">{tab === 'log' ? 'The run has no log yet.' : 'The run has no recipe yet.'}</p>
        ) : (
          <p className="ob-text">Reading…</p>
        )}
      </div>
    </>
  );
}

// (a time of the server, as people read it: today's by its time)
export function when(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const today = new Date();
  const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return date.toDateString() === today.toDateString() ? `today ${time}` : `${date.toLocaleDateString([], { day: 'numeric', month: 'short' })} ${time}`;
}
