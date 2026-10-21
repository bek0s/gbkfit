// The explorer: the projects of the workspace, with their pieces, setups
// and runs. A piece or a setup opens in an editor, a run in its panel.
// New projects, and new pieces and setups (from templates), are made here.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, type ReactNode } from 'react';
import { api, data } from '../api/client';
import { useStudio } from '../studio';

// The folders of a project, in the order shown
const FOLDERS = ['setups', 'galaxies', 'observations', 'params', 'fitters'];

export function Explorer() {
  const client = useQueryClient();
  const projects = useQuery({
    queryKey: ['projects'],
    queryFn: () => data(api.GET('/api/v1/projects')),
  });
  const [adding, setAdding] = useState(false);
  const create = useMutation({
    mutationFn: (path: string) => data(api.POST('/api/v1/projects', { body: { path } })),
    onSuccess: () => {
      setAdding(false);
      void client.invalidateQueries({ queryKey: ['projects'] });
    },
  });
  return (
    <aside className="side">
      <div className="colhead">
        Explorer <span className="sub">{projects.data?.length ?? 0} projects</span>
        <button className="ghost small" onClick={() => setAdding(!adding)} title="New project">
          + project
        </button>
      </div>
      <div className="tree">
        {adding && (
          <NameInput
            placeholder="folder of the new project"
            error={create.error?.message}
            onSubmit={(path) => create.mutate(path)}
            onCancel={() => setAdding(false)}
          />
        )}
        {projects.error && <p className="error">{projects.error.message}</p>}
        {projects.data?.map((path) => <ProjectTree key={path} path={path} />)}
        {projects.data?.length === 0 && (
          <p className="hint">No projects here: a project is a folder with a gbkfit-project.yaml.</p>
        )}
      </div>
    </aside>
  );
}

function ProjectTree({ path }: { path: string }) {
  const { openFile, openRun } = useStudio();
  const [open, setOpen] = useState(true);
  const project = useQuery({
    queryKey: ['project', path],
    queryFn: () => data(api.GET('/api/v1/projects/{path}', { params: { path: { path } } })),
  });
  const prefix = path === '.' ? '' : `${path}/`;
  const client = useQueryClient();
  const createPiece = useMutation({
    mutationFn: ({ folder, name }: { folder: string; name: string }) =>
      data(api.POST('/api/v1/pieces', { body: { project: path, folder, name } })),
    onSuccess: (file) => {
      void client.invalidateQueries({ queryKey: ['project', path] });
      openFile(file);
    },
  });
  return (
    <>
      <button className="row project" onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} {path === '.' ? 'workspace' : path}
      </button>
      {open &&
        FOLDERS.map((folder) => (
          <Folder
            key={folder}
            name={folder}
            count={project.data?.pieces[folder]?.length ?? 0}
            create={(name) => createPiece.mutateAsync({ folder, name })}
          >
            {project.data?.pieces[folder]?.map((name) => (
              <button
                key={name}
                className="row file"
                onClick={() => openFile(`${prefix}${folder}/${name}.yaml`)}
              >
                {name}
              </button>
            ))}
          </Folder>
        ))}
      {open && (
        <Folder name="runs" count={project.data?.runs.length ?? 0}>
          {project.data?.runs.map((run) => (
            <button key={run} className="row file" onClick={() => openRun(run)}>
              {run.slice(`${prefix}runs/`.length)}
            </button>
          ))}
        </Folder>
      )}
    </>
  );
}

function Folder({
  name,
  count,
  create,
  children,
}: {
  name: string;
  count: number;
  // Create a piece of the folder by name (none for runs)
  create?: (name: string) => Promise<unknown>;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(name === 'setups' || name === 'runs');
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | undefined>();
  return (
    <>
      <div className="row folder">
        <button className="row-main" onClick={() => setOpen(!open)}>
          {open ? '▾' : '▸'} {name}
        </button>
        <span className="meta">{count}</span>
        {create ? (
          <button
            className="add"
            title={`New piece of ${name}`}
            onClick={() => {
              setOpen(true);
              setAdding(true);
            }}
          >
            +
          </button>
        ) : (
          <span className="add" />
        )}
      </div>
      {adding && create && (
        <NameInput
          placeholder="name (e.g. disk, or ngc1313/hi)"
          error={error}
          onSubmit={(piece) =>
            create(piece)
              .then(() => {
                setAdding(false);
                setError(undefined);
              })
              .catch((e: Error) => setError(e.message))
          }
          onCancel={() => {
            setAdding(false);
            setError(undefined);
          }}
        />
      )}
      {open && children}
    </>
  );
}

// A field for a name: Enter submits it, Escape cancels
function NameInput({
  placeholder,
  error,
  onSubmit,
  onCancel,
}: {
  placeholder: string;
  error?: string;
  onSubmit: (name: string) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState('');
  return (
    <div className="name-input">
      <input
        autoFocus
        value={name}
        placeholder={placeholder}
        onChange={(event) => setName(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && name.trim()) onSubmit(name.trim());
          if (event.key === 'Escape') onCancel();
        }}
      />
      {error && <p className="error">{error}</p>}
    </div>
  );
}
