// The explorer: the projects of the workspace, with their pieces, setups
// and runs. A piece or a setup opens in an editor, a run in its panel.
import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { api, data } from '../api/client';
import { useStudio } from '../studio';

// The folders of a project, in the order shown
const FOLDERS = ['setups', 'galaxies', 'observations', 'params', 'fitters'];

export function Explorer() {
  const projects = useQuery({
    queryKey: ['projects'],
    queryFn: () => data(api.GET('/api/v1/projects')),
  });
  return (
    <aside className="side">
      <div className="colhead">
        Explorer <span className="sub">{projects.data?.length ?? 0} projects</span>
      </div>
      <div className="tree">
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
  return (
    <>
      <button className="row project" onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} {path === '.' ? 'workspace' : path}
      </button>
      {open &&
        FOLDERS.map((folder) => (
          <Folder key={folder} name={folder} count={project.data?.pieces[folder]?.length ?? 0}>
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

function Folder({ name, count, children }: { name: string; count: number; children: React.ReactNode }) {
  const [open, setOpen] = useState(name === 'setups' || name === 'runs');
  return (
    <>
      <button className="row folder" onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} {name}
        <span className="meta">{count}</span>
      </button>
      {open && children}
    </>
  );
}
