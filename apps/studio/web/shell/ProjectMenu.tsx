// The project button at the top of the tree. Its menu makes a new
// project, opens one of the workspace, closes the open one, and lists the
// recent ones. Its dialogs serve the start (no project) too.
import { useEffect, useRef, useState } from 'react';
import { useCreateProject, useProjects } from '../api/queries';
import { Dialog } from '../components/Dialog';
import { useStudio } from '../studio';

export function ProjectMenu() {
  const { project, openProject, recent } = useStudio();
  const [open, setOpen] = useState(false);
  const [dialog, setDialog] = useState<'new' | 'open' | null>(null);
  const menu = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (event: MouseEvent) => {
      if (!menu.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);
  const choose = (action: () => void) => () => {
    setOpen(false);
    action();
  };
  return (
    <div className="project-menu" ref={menu}>
      <button className="project-button" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span>{project ?? 'No project'}</span>
        <i>▾</i>
      </button>
      {open && (
        <div className="menu" role="menu">
          <button role="menuitem" onClick={choose(() => setDialog('new'))}>
            New project…
          </button>
          <button role="menuitem" onClick={choose(() => setDialog('open'))}>
            Open a project…
          </button>
          {project !== null && (
            <button role="menuitem" onClick={choose(() => openProject(null))}>
              Close the project
            </button>
          )}
          {recent.length > 0 && <div className="menu-label">Recent</div>}
          {recent.map((path) => (
            <button key={path} role="menuitem" aria-current={path === project} onClick={choose(() => openProject(path))}>
              {path}
            </button>
          ))}
        </div>
      )}
      <ProjectDialogs dialog={dialog} onClose={() => setDialog(null)} />
    </div>
  );
}

// The dialog of a new project, or of opening one, if any
export function ProjectDialogs({ dialog, onClose }: { dialog: 'new' | 'open' | null; onClose: () => void }) {
  if (dialog === 'new') return <NewProjectDialog onClose={onClose} />;
  if (dialog === 'open') return <OpenProjectDialog onClose={onClose} />;
  return null;
}

// A new project: its folder in the workspace, made with the folders of its
// blocks; it opens once made
function NewProjectDialog({ onClose }: { onClose: () => void }) {
  const { openProject } = useStudio();
  const create = useCreateProject();
  const [path, setPath] = useState('');
  const folder = path.trim();
  return (
    <Dialog title="New project" onClose={onClose}>
      <form
        className="dialog-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (!folder) return;
          create.mutate(folder, {
            onSuccess: (made) => {
              onClose();
              openProject(made.path);
            },
          });
        }}
      >
        <label>
          Folder of the project, in the workspace
          <input autoFocus value={path} placeholder="e.g. ngc1313" onChange={(event) => setPath(event.target.value)} />
        </label>
        <p className="note">It holds the data, the galaxies and the fits of the study.</p>
        {create.error && <p className="error">{create.error.message}</p>}
        <div className="dialog-actions">
          <button type="button" className="ghost" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="primary" disabled={!folder || create.isPending}>
            Create
          </button>
        </div>
      </form>
    </Dialog>
  );
}

// Open a project of the workspace, found by its path
function OpenProjectDialog({ onClose }: { onClose: () => void }) {
  const { project, openProject } = useStudio();
  const projects = useProjects();
  const [find, setFind] = useState('');
  const query = find.trim().toLowerCase();
  const shown = (projects.data ?? []).filter((path) => path.toLowerCase().includes(query));
  return (
    <Dialog title="Open a project" onClose={onClose}>
      <input
        className="dialog-find"
        autoFocus
        value={find}
        placeholder="Find a project"
        aria-label="Find a project"
        onChange={(event) => setFind(event.target.value)}
      />
      <div className="dialog-list">
        {projects.error && <p className="error">{projects.error.message}</p>}
        {shown.map((path) => (
          <button
            key={path}
            aria-current={path === project}
            onClick={() => {
              onClose();
              openProject(path);
            }}
          >
            {path === '.' ? '(the workspace)' : path}
          </button>
        ))}
        {projects.data !== undefined && shown.length === 0 && (
          <p className="note">
            {projects.data.length === 0 ? 'The workspace has no projects yet.' : `No project called “${find.trim()}”.`}
          </p>
        )}
      </div>
    </Dialog>
  );
}
