// The activity of the server, in one place: the pill in the bar counts
// what runs, and opens a drawer over the screen with the renders that run
// long (see the server's RenderPool), each with Cancel, and the jobs
// (preparations of data, fits), each with Stop, then those that ended
// today. Previews of preparations that were not kept are left out.
import { useState } from 'react';
import type { components } from '../api/schema';
import { useCancelRender, useJobs, useRenders, useStopJob } from '../api/queries';
import { useServerEvents } from '../api/events';
import { Button, Close } from '../components/ui/basics';

type Job = components['schemas']['JobInfo'];
type Render = components['schemas']['RenderInfo'];

export function ActivityPill() {
  const connected = useServerEvents();
  const renders = useRenders().data ?? [];
  const jobs = (useJobs().data ?? []).filter((job) => !job.preview);
  const [open, setOpen] = useState(false);
  const running = renders.length + jobs.filter((job) => job.state === 'running').length;
  const waiting = jobs.filter((job) => job.state === 'queued').length;
  const text = !connected
    ? 'not connected'
    : running || waiting
      ? [running && `${running} running`, waiting && `${waiting} waiting`].filter(Boolean).join(' · ')
      : 'nothing running';
  return (
    <>
      <button
        type="button"
        className="activity-pill"
        data-state={!connected ? 'off' : running ? 'run' : 'ok'}
        title="What the server is doing"
        onClick={() => setOpen(!open)}
      >
        <span className="dot" />
        {text}
      </button>
      {open && <ActivityDrawer renders={renders} jobs={jobs} onClose={() => setOpen(false)} />}
    </>
  );
}

function ActivityDrawer({ renders, jobs, onClose }: { renders: Render[]; jobs: Job[]; onClose: () => void }) {
  const cancel = useCancelRender();
  const stop = useStopJob();
  const active = jobs.filter((job) => job.state === 'running' || job.state === 'queued');
  const today = new Date().toDateString();
  const ended = jobs
    .filter((job) => job.ended !== null && new Date(job.ended).toDateString() === today)
    .slice(-8)
    .reverse();
  return (
    <aside className="activity" aria-label="Activity">
      <div className="activity-head">
        <b>Activity</b>
        <span>of the server</span>
        <Close label="Close the activity" onClick={onClose} />
      </div>
      <Group title="Renders" count={renders.length} empty="No render runs long.">
        {renders.map((render) => (
          <div key={render.id} className="activity-item">
            <div className="activity-title">
              <b>Render</b>
              <span>{render.what}</span>
              <Button kind="small" onClick={() => cancel.mutate(render.id)}>
                Cancel
              </Button>
            </div>
            <span className="activity-note">
              {render.stage} · {Math.round(render.elapsed)} s
            </span>
          </div>
        ))}
      </Group>
      <Group title="Jobs" count={active.length} empty="No job runs or waits.">
        {active.map((job) => (
          <div key={job.id} className="activity-item">
            <div className="activity-title">
              <b>{taskTitle(job.task)}</b>
              <span>{jobWhat(job)}</span>
              <Button kind="small" onClick={() => stop.mutate(job.id)}>
                {job.state === 'queued' ? 'Remove' : 'Stop'}
              </Button>
            </div>
            <span className="activity-note">{job.state === 'queued' ? 'waiting' : `running since ${time(job.started)}`}</span>
          </div>
        ))}
      </Group>
      {ended.length > 0 && (
        <Group title="Ended today" count={ended.length}>
          {ended.map((job) => (
            <div key={job.id} className="activity-item" data-state={job.state}>
              <div className="activity-title">
                <b>{taskTitle(job.task)}</b>
                <span>{jobWhat(job)}</span>
                <em>{job.state === 'complete' ? 'done' : job.state}</em>
              </div>
              <span className="activity-note">at {time(job.ended)}</span>
            </div>
          ))}
        </Group>
      )}
    </aside>
  );
}

function Group({ title, count, empty, children }: { title: string; count: number; empty?: string; children: React.ReactNode }) {
  return (
    <section className="activity-group">
      <h3>
        {title} · {count}
      </h3>
      {count === 0 && empty ? <p className="activity-note">{empty}</p> : children}
    </section>
  );
}

// The task of a job, for people (e.g. 'prep' is a preparation of data)
function taskTitle(task: string): string {
  return { prep: 'Prepare', fit: 'Fit', eval: 'Evaluate' }[task] ?? task;
}

// What a job works on: the configuration or the files it names
function jobWhat(job: Job): string {
  const config = job.options.config;
  if (typeof config === 'string') return config.replace(/\.yaml$/, '');
  const file = Object.values(job.options).find((value) => typeof value === 'string');
  return typeof file === 'string' ? file : '';
}

function time(iso: string | null): string {
  return iso === null ? '' : new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
