// The jobs: their state, the progress of fits (the best chi-square so far
// against the evaluations), and their runs. Running jobs can be stopped,
// as Ctrl-C does.
import { useMutation, useQuery } from '@tanstack/react-query';
import { api, data, type Job } from '../api/client';
import { useProgress, type ProgressReport } from '../api/events';
import { format } from '../components/ImageView';
import { useStudio } from '../studio';

export function JobsPanel() {
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: () => data(api.GET('/api/v1/jobs')) });
  return (
    <section className="panel">
      <div className="jobs">
        {jobs.error && <p className="error">{jobs.error.message}</p>}
        {jobs.data?.length === 0 && <p className="hint">No jobs yet: run a setup (Eval or Fit).</p>}
        {[...(jobs.data ?? [])].reverse().map((job) => <JobRow key={job.id} job={job} />)}
      </div>
    </section>
  );
}

function JobRow({ job }: { job: Job }) {
  const { openRun, openPanel, live } = useStudio();
  const reports = useProgress(job.id);
  const last = reports.at(-1);
  const stop = useMutation({
    mutationFn: () => data(api.POST('/api/v1/jobs/{id_}/stop', { params: { path: { id_: job.id } } })),
  });
  const config = String(job.options.config ?? job.options.output_dir ?? '');
  return (
    <div className="job" data-state={job.state}>
      <div className="job-head">
        <b>{job.task}</b>
        <span className="kind">{config}</span>
        <span className="badge">{job.state}</span>
        {job.task !== 'eval' && job.run && last && (
          <button
            className="ghost"
            onClick={() => {
              live.watch(`${job.run}/config.yaml`, job.id);
              openPanel('preview');
            }}
          >
            {live.watching === job.id ? 'Watching' : 'Watch'}
          </button>
        )}
        {(job.state === 'queued' || job.state === 'running') && (
          <button className="danger" onClick={() => stop.mutate()}>
            Stop
          </button>
        )}
      </div>
      {last && (
        <div className="progress">
          <Trace reports={reports} />
          <span className="num">
            {last.stage} · {last.nevals} evaluations · best χ² {last.best_chi2 === null ? '–' : format(last.best_chi2)} ·{' '}
            {last.elapsed.toFixed(0)} s
          </span>
        </div>
      )}
      {job.run && (
        <button className="link" onClick={() => openRun(job.run!)}>
          {job.run}
        </button>
      )}
      {job.state === 'failed' && <pre className="output">{job.output.slice(-2000)}</pre>}
    </div>
  );
}

// The best chi-square so far against the evaluations (log scale)
function Trace({ reports }: { reports: ProgressReport[] }) {
  const points = reports.filter((r) => r.best_chi2 !== null && r.best_chi2 > 0);
  if (points.length < 2) return null;
  const logs = points.map((r) => Math.log10(r.best_chi2!));
  const low = Math.min(...logs);
  const high = Math.max(...logs);
  const last = points.at(-1)!.nevals || 1;
  const line = points
    .map((r, i) => `${(100 * r.nevals) / last},${high > low ? 5 + (90 * (high - logs[i])) / (high - low) : 50}`)
    .join(' ');
  return (
    <svg className="trace" viewBox="0 0 100 100" preserveAspectRatio="none">
      <polyline points={line} />
    </svg>
  );
}
