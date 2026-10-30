// The start, with no project open: make one, open one of the workspace,
// or one of those opened last.
import { useState } from 'react';
import { ProjectDialogs } from '../shell/ProjectMenu';
import { useStudio } from '../studio';

export function Welcome() {
  const { openProject, recent } = useStudio();
  const [dialog, setDialog] = useState<'new' | 'open' | null>(null);
  return (
    <section className="screen">
      <div className="welcome">
        <h1>GBKFIT Studio</h1>
        <p>Everything lives in a project: a folder of the workspace with its data, its galaxies and its fits.</p>
        <div className="welcome-actions">
          <button className="primary" onClick={() => setDialog('new')}>
            New project…
          </button>
          <button className="ghost" onClick={() => setDialog('open')}>
            Open a project…
          </button>
        </div>
        {recent.length > 0 && (
          <div className="recent">
            <div className="eyebrow">Recent</div>
            {recent.map((path) => (
              <button key={path} onClick={() => openProject(path)}>
                {path}
              </button>
            ))}
          </div>
        )}
      </div>
      <ProjectDialogs dialog={dialog} onClose={() => setDialog(null)} />
    </section>
  );
}
