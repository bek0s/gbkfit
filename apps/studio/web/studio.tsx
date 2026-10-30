// The state of the Studio that its parts share: the open project and the
// recent ones, the screen on the stage, the blocks opened last, the tree
// folded or not, and the folder where a new block is being named.
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { useProjects } from './api/queries';
import { address, parseAddress, sameScreen, type Screen } from './shell/navigation';
import { latestFirst, useRemembered } from './shell/remember';

// The projects, and the blocks of each, remembered as recent
const RECENT = 6;

export interface Studio {
  // The path of the open project in the workspace (null: none)
  project: string | null;
  openProject: (path: string | null) => void;
  recent: string[];
  screen: Screen;
  go: (screen: Screen) => void;
  // The blocks of the open project opened last, the latest first
  visited: Screen[];
  folded: boolean;
  setFolded: (folded: boolean) => void;
  adding: string | null;
  startAdding: (folder: string | null) => void;
}

const StudioContext = createContext<Studio | null>(null);

export function useStudio(): Studio {
  const studio = useContext(StudioContext);
  if (studio === null) throw new Error('useStudio is used out of a StudioProvider');
  return studio;
}

export function StudioProvider({ children }: { children: ReactNode }) {
  const projects = useProjects();
  const [project, setProject] = useRemembered<string | null>('project', null);
  const [recent, setRecent] = useRemembered<string[]>('recent-projects', []);
  const [visited, setVisited] = useRemembered<Record<string, Screen[]>>('visited', {});
  const [folded, setFolded] = useRemembered('tree-folded', false);
  const [screen, setScreen] = useState<Screen>(() => parseAddress(window.location.hash));
  const [adding, setAdding] = useState<string | null>(null);

  useEffect(() => {
    const onPop = () => setScreen(parseAddress(window.location.hash));
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  // (projects gone from the workspace are closed and forgotten)
  useEffect(() => {
    const found = projects.data;
    if (found === undefined) return;
    if (project !== null && !found.includes(project)) setProject(null);
    if (recent.some((path) => !found.includes(path))) setRecent(recent.filter((path) => found.includes(path)));
  }, [projects.data, project, recent, setProject, setRecent]);

  const show = useCallback((next: Screen) => {
    setScreen(next);
    window.history.pushState(null, '', address(next));
  }, []);

  const go = useCallback(
    (next: Screen) => {
      show(next);
      if (project !== null && next.kind === 'block') {
        setVisited((all) => ({ ...all, [project]: latestFirst(all[project] ?? [], next, RECENT, sameScreen) }));
      }
    },
    [show, project, setVisited],
  );

  const openProject = useCallback(
    (path: string | null) => {
      setProject(path);
      if (path !== null) setRecent((paths) => latestFirst(paths, path, RECENT));
      show({ kind: 'home' });
    },
    [setProject, setRecent, show],
  );

  const startAdding = useCallback(
    (folder: string | null) => {
      setAdding(folder);
      // (the field of its name is in the tree)
      if (folder !== null) setFolded(false);
    },
    [setFolded],
  );

  const studio = useMemo<Studio>(
    () => ({
      project,
      openProject,
      recent,
      screen,
      go,
      visited: project === null ? [] : (visited[project] ?? []),
      folded,
      setFolded,
      adding,
      startAdding,
    }),
    [project, openProject, recent, screen, go, visited, folded, setFolded, adding, startAdding],
  );
  return <StudioContext.Provider value={studio}>{children}</StudioContext.Provider>;
}
