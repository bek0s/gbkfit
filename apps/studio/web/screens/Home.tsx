// The home of the project: the three steps of the work (data, model, fit)
// with what the project has for each and the next thing to do as the main
// button, and the blocks opened last.
import { Button, NavItem } from '../components/ui/basics';
import type { ReactNode } from 'react';
import { useProject } from '../api/queries';
import { kindOf } from '../shell/sections';
import { address } from '../shell/navigation';
import { useStudio } from '../studio';

export function Home() {
  const { project, go, startAdding, visited } = useStudio();
  const info = useProject(project);
  const names = (folder: string) => info.data?.pieces[folder] ?? [];
  const count = (folder: string, one: string, many: string) => {
    const n = names(folder).length;
    return `${n} ${n === 1 ? one : many}`;
  };
  const links = (folder: string) =>
    names(folder).map((name) => (
      <Button key={name} kind="small" onClick={() => go({ kind: 'block', folder, name })}>
        {name}
      </Button>
    ));
  const runs = info.data?.runs.length ?? 0;
  const hasData = names('datasets').length > 0;
  const hasGalaxy = names('galaxies').length > 0;
  return (
    <section className="screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">Project</div>
          <h1>{project}</h1>
        </div>
      </div>
      <div className="screen-body home">
        <div className="steps">
          {info.error && <p className="error">{info.error.message}</p>}
          <Step
            number={1}
            title="Data"
            text="The observations of the galaxy, with their errors and masks, and the PSF and LSF of each instrument. Look at each one before anything else."
            have={
              hasData ? (
                <>
                  <div className="have">
                    {count('datasets', 'dataset', 'datasets')} · {count('psfs', 'PSF', 'PSFs')} · {count('lsfs', 'LSF', 'LSFs')}
                  </div>
                  <div className="links">{links('datasets')}</div>
                </>
              ) : (
                <p className="note">Nothing yet: start here.</p>
              )
            }
          >
            <Button kind={hasData ? 'plain' : 'primary'} onClick={() => startAdding('datasets')}>
              + Add data
            </Button>
          </Step>
          <Step
            number={2}
            title="Model"
            text="Galaxies: their structure and motions. Observations: how an instrument sees a galaxy, with or without data. Observe shows the galaxies through them, and makes mocks."
            have={
              hasGalaxy ? (
                <>
                  <div className="have">
                    {count('galaxies', 'galaxy', 'galaxies')} · {count('observations', 'observation', 'observations')}
                  </div>
                  <div className="links">{links('galaxies')}</div>
                </>
              ) : (
                <p className="note">No galaxy yet. You can also start here, to look at models without data.</p>
              )
            }
          >
            <Button kind={hasData && !hasGalaxy ? 'primary' : 'plain'} onClick={() => startAdding('galaxies')}>
              + New galaxy
            </Button>
          </Step>
          <Step
            number={3}
            title="Fit"
            text="Compare a galaxy with the data while you move its parameters, then fit it. Every fit is a run, kept with its results."
            have={
              <div className="have">
                {runs} {runs === 1 ? 'run' : 'runs'}
              </div>
            }
          >
            <Button onClick={() => go({ kind: 'compare' })}>Compare</Button>
            <Button onClick={() => go({ kind: 'fit' })}>Fit</Button>
          </Step>
        </div>
        <div className="card">
          <h2 className="card-head">Recent</h2>
          <div className="card-body recent">
            {visited.length === 0 && <p className="note">Nothing opened yet.</p>}
            {visited.map(
              (screen) =>
                screen.kind === 'block' && (
                  <NavItem key={address(screen)} mono label={screen.name} right={<small>{kindOf(screen.folder)}</small>} onClick={() => go(screen)} />
                ),
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function Step({
  number,
  title,
  text,
  have,
  children,
}: {
  number: number;
  title: string;
  text: string;
  // What the project has for the step
  have: ReactNode;
  // Its buttons
  children: ReactNode;
}) {
  return (
    <div className="card step">
      <div className="step-number">{number}</div>
      <div>
        <h2>{title}</h2>
        <p>{text}</p>
        {have}
      </div>
      <div className="step-actions">{children}</div>
    </div>
  );
}
