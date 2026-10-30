// The tabs of the bar: the home of the project, and the tasks done with
// its blocks (observe, compare, fit), with a dot on Fit while a job of the
// project runs. Without a project, they are off.
import { useJobs } from '../api/queries';
import { useStudio } from '../studio';

const TABS = [
  { kind: 'home', title: 'Home' },
  { kind: 'observe', title: 'Observe' },
  { kind: 'compare', title: 'Compare' },
  { kind: 'fit', title: 'Fit' },
] as const;

export function Tabs() {
  const { project, screen, go } = useStudio();
  const jobs = useJobs();
  const running =
    project !== null &&
    (jobs.data ?? []).some((job) => job.state === 'running' && inProject(String(job.options.config ?? ''), project));
  return (
    <nav className="tabs" aria-label="Tasks">
      {TABS.map(({ kind, title }) => (
        <button
          key={kind}
          disabled={project === null}
          aria-current={project !== null && screen.kind === kind ? 'page' : undefined}
          onClick={() => go({ kind })}
        >
          {title}
          {kind === 'fit' && running && <span className="dot" data-state="run" title="A job of the project runs" />}
        </button>
      ))}
    </nav>
  );
}

// Whether a path of the workspace is in a project
function inProject(path: string, project: string): boolean {
  return project === '.' || path.startsWith(`${project}/`);
}
