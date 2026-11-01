// The project button at the top of the tree. Its menu makes a new
// project, opens one of the workspace, closes the open one, and lists the
// recent ones. Its dialogs serve the start (no project) too.
import { useState } from 'react';
import { useCreateProject, useProjects } from '../api/queries';
import { Button, NavItem, TextField } from '../components/ui/basics';
import { Dialog, DropMenu, type MenuEntry } from '../components/ui/overlays';
import { useStudio } from '../studio';

export function ProjectMenu() {
  const { project, openProject, recent } = useStudio();
  const [dialog, setDialog] = useState<'new' | 'open' | null>(null);
  const entries: MenuEntry[] = [
    { label: 'New project…', onClick: () => setDialog('new') },
    { label: 'Open a project…', onClick: () => setDialog('open') },
    ...(project !== null ? [{ label: 'Close the project', onClick: () => openProject(null) }] : []),
    ...(recent.length > 0 ? [{ section: 'Recent' }] : []),
    ...recent.map((path) => ({ label: path, onClick: () => openProject(path), current: path === project })),
  ];
  return (
    <div className="project-menu">
      <DropMenu
        label="The projects"
        entries={entries}
        target={
          <Button full right="▾" title="The projects">
            <span className="project-name">{project ?? 'No project'}</span>
          </Button>
        }
      />
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
        <TextField autoFocus mono showLabel value={path} placeholder="e.g. ngc1313" label="Folder of the project, in the workspace" onChange={setPath} />
        <p className="note">It holds the data, the galaxies and the fits of the study.</p>
        {create.error && <p className="error">{create.error.message}</p>}
        <div className="dialog-actions">
          <Button onClick={onClose}>Cancel</Button>
          <Button kind="primary" submit disabled={!folder || create.isPending}>
            Create
          </Button>
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
      <TextField search autoFocus value={find} placeholder="Find a project" label="Find a project" onChange={setFind} />
      <div className="dialog-list">
        {projects.error && <p className="error">{projects.error.message}</p>}
        {shown.map((path) => (
          <NavItem
            key={path}
            mono
            label={path === '.' ? '(the workspace)' : path}
            active={path === project}
            onClick={() => {
              onClose();
              openProject(path);
            }}
          />
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
