// The tabs of the bar: the home of the project, and the tasks done with
// its blocks (compare, fit), with a dot on Fit while a job of the
// project runs. Without a project, they are off.
import { useJobs } from '../api/queries';
import { BarTabs } from '../components/ui/basics';
import { useStudio } from '../studio';

const TABS = [
  { kind: 'home', title: 'Home' },
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
    <BarTabs
      label="Tasks"
      tabs={TABS.map(({ kind, title }) => ({
        value: kind,
        label: title,
        disabled: project === null,
        mark: kind === 'fit' && running ? <span className="dot" data-state="run" title="A job of the project runs" /> : undefined,
      }))}
      value={project !== null && TABS.some(({ kind }) => kind === screen.kind) ? (screen.kind as (typeof TABS)[number]['kind']) : null}
      onChange={(kind) => go({ kind })}
    />
  );
}

// Whether a path of the workspace is in a project
function inProject(path: string, project: string): boolean {
  return project === '.' || path.startsWith(`${project}/`);
}
