// The Studio: a bar, a rail, the explorer of the projects, a dock of panels
// (editors, the live preview, the jobs, runs), and the parameters of the
// live session.
import { useQuery } from '@tanstack/react-query';
import {
  DockviewReact,
  themeDark,
  type DockviewApi,
  type DockviewReadyEvent,
  type IDockviewPanelProps,
} from 'dockview-react';
import { useCallback, useMemo, useRef, useState, type FunctionComponent } from 'react';
import { api, data } from './api/client';
import { useServerEvents } from './api/events';
import { useLiveSession } from './api/session';
import { FilesIcon, JobsIcon, Logo, PreviewIcon } from './components/icons';
import { EditorPanel } from './panels/EditorPanel';
import { Explorer } from './panels/Explorer';
import { JobsPanel } from './panels/JobsPanel';
import { Parameters } from './panels/Parameters';
import { PreviewPanel } from './panels/PreviewPanel';
import { RunPanel } from './panels/RunPanel';
import { basename, StudioContext, type Studio } from './studio';

// The panels of the dock, by component name
const PANELS: Record<string, FunctionComponent<IDockviewPanelProps>> = {
  editor: EditorPanel,
  preview: PreviewPanel,
  jobs: JobsPanel,
  run: RunPanel,
};

export function App() {
  const connected = useServerEvents();
  const live = useLiveSession();
  const dock = useRef<DockviewApi | null>(null);
  const [explorer, setExplorer] = useState(true);
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: () => data(api.GET('/api/v1/jobs')) });
  const running = (jobs.data ?? []).filter((job) => job.state === 'running').length;

  const open = useCallback((id: string, component: string, title: string, params = {}) => {
    const panel = dock.current?.getPanel(id) ?? dock.current?.addPanel({ id, component, title, params });
    panel?.api.setActive();
  }, []);

  const studio = useMemo<Studio>(
    () => ({
      openFile: (path) => open(`file:${path}`, 'editor', basename(path), { path }),
      openRun: (path) => open(`run:${path}`, 'run', path.replace(/^.*\/runs\//, 'runs/'), { path }),
      openPanel: (kind) => open(kind, kind, kind === 'preview' ? 'Live preview' : 'Jobs'),
      live,
    }),
    [open, live],
  );

  const onReady = (event: DockviewReadyEvent) => {
    dock.current = event.api;
  };

  return (
    <StudioContext.Provider value={studio}>
      <div className="app">
        <header className="bar">
          <div className="logo">
            <Logo />
            GBKFIT <span>Studio</span>
          </div>
          <div className="status" data-kind={connected ? (running ? 'run' : 'ok') : 'err'}>
            <span className="dot" />
            {connected ? `connected · ${running} running` : 'not connected'}
          </div>
        </header>
        <div className="body">
          <nav className="rail">
            <button aria-pressed={explorer} title="Explorer" onClick={() => setExplorer(!explorer)}>
              <FilesIcon />
            </button>
            <button title="Live preview" onClick={() => studio.openPanel('preview')}>
              <PreviewIcon />
            </button>
            <button title="Jobs" onClick={() => studio.openPanel('jobs')}>
              <JobsIcon />
            </button>
          </nav>
          {explorer && <Explorer />}
          <main className="dock">
            <DockviewReact components={PANELS} onReady={onReady} theme={themeDark} />
          </main>
          {live.config !== null && <Parameters />}
        </div>
      </div>
    </StudioContext.Provider>
  );
}
