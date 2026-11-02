// The project button at the top of the tree. Its menu makes a new
// project, opens one of the workspace, renames, deletes or closes the open
// one, and lists the recent ones. Its dialogs serve the start (no project)
// too.
import { useState } from 'react';
import { useCreateProject, useDeleteProject, useProject, useProjects, useRenameProject } from '../api/queries';
import { Button, NavItem, TextField } from '../components/ui/basics';
import { Dialog, DropMenu, type MenuEntry } from '../components/ui/overlays';
import { useStudio } from '../studio';

export function ProjectMenu() {
  const { project, openProject, recent } = useStudio();
  const [dialog, setDialog] = useState<'new' | 'open' | 'rename' | 'delete' | null>(null);
  const entries: MenuEntry[] = [
    { label: 'New project…', onClick: () => setDialog('new') },
    { label: 'Open a project…', onClick: () => setDialog('open') },
    ...(project !== null
      ? [
          { label: 'Rename the project…', onClick: () => setDialog('rename') },
          { label: 'Delete the project…', onClick: () => setDialog('delete') },
          { label: 'Close the project', onClick: () => openProject(null) },
        ]
      : []),
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
      {dialog === 'rename' && project !== null ? (
        <RenameProjectDialog project={project} onClose={() => setDialog(null)} />
      ) : dialog === 'delete' && project !== null ? (
        <DeleteProjectDialog project={project} onClose={() => setDialog(null)} />
      ) : (
        <ProjectDialogs dialog={dialog === 'new' || dialog === 'open' ? dialog : null} onClose={() => setDialog(null)} />
      )}
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

// Move the open project to another folder of the workspace (its blocks
// name their files relative to it, so they follow)
function RenameProjectDialog({ project, onClose }: { project: string; onClose: () => void }) {
  const { movedProject } = useStudio();
  const rename = useRenameProject();
  const [path, setPath] = useState(project);
  const folder = path.trim();
  return (
    <Dialog title="Rename the project" onClose={onClose}>
      <form
        className="dialog-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (!folder || folder === project) return;
          rename.mutate(
            { path: project, to: folder },
            {
              onSuccess: (moved) => {
                onClose();
                movedProject(project, moved.path);
              },
            },
          );
        }}
      >
        <TextField autoFocus mono showLabel value={path} label="Folder of the project, in the workspace" onChange={setPath} />
        <p className="note">Its folder moves, with its data and runs. Not while a job of it runs.</p>
        {rename.error && <p className="error">{rename.error.message}</p>}
        <div className="dialog-actions">
          <Button onClick={onClose}>Cancel</Button>
          <Button kind="primary" submit disabled={!folder || folder === project || rename.isPending}>
            Rename
          </Button>
        </div>
      </form>
    </Dialog>
  );
}

// Delete the open project and all its files, once its path is typed
function DeleteProjectDialog({ project, onClose }: { project: string; onClose: () => void }) {
  const { movedProject } = useStudio();
  const info = useProject(project);
  const remove = useDeleteProject();
  const [typed, setTyped] = useState('');
  const blocks = Object.values(info.data?.pieces ?? {}).reduce((n, names) => n + names.length, 0);
  const runs = info.data?.runs.length ?? 0;
  return (
    <Dialog title="Delete the project" onClose={onClose}>
      <form
        className="dialog-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (typed !== project) return;
          remove.mutate(project, {
            onSuccess: () => {
              onClose();
              movedProject(project, null);
            },
          });
        }}
      >
        <p>
          This deletes the folder <b className="mono">{project}</b> and everything in it: its {blocks}{' '}
          {blocks === 1 ? 'block' : 'blocks'}, its {runs} {runs === 1 ? 'run' : 'runs'}, and its copies of the data. It cannot be undone.
        </p>
        <TextField autoFocus mono showLabel value={typed} label="Type the folder of the project to confirm" onChange={setTyped} />
        {remove.error && <p className="error">{remove.error.message}</p>}
        <div className="dialog-actions">
          <Button onClick={onClose}>Cancel</Button>
          <Button kind="danger" submit disabled={typed !== project || remove.isPending}>
            Delete the project
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
