// The state of the Studio that its parts share: the open project and the
// recent ones, the screen on the stage, the blocks opened last, the tree
// folded or not, and the folder where a new block is being named.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
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
  // What the page remembers of a project or a block follows it as it is
  // renamed (to: its new path or name) or deleted (to: null)
  movedProject: (path: string, to: string | null) => void;
  movedBlock: (folder: string, name: string, to: string | null) => void;
  // Whether the screen has edits not saved, which leaving it asks about
  // (see useUnsaved)
  setUnsaved: (unsaved: boolean) => void;
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

  // (back and forward ask first too, as the edits of the screen are not
  // saved: kept, the address goes back to the screen)
  const shown = useRef(screen);
  shown.current = screen;
  useEffect(() => {
    const onPop = () => {
      if (!leave()) {
        window.history.pushState(null, '', address(shown.current));
        return;
      }
      unsaved.current = false;
      setScreen(parseAddress(window.location.hash));
    };
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

  // (the screen has edits not saved: leaving it is asked first)
  const unsaved = useRef(false);
  const setUnsaved = useCallback((value: boolean) => {
    unsaved.current = value;
  }, []);
  const leave = () => !unsaved.current || window.confirm('The edits of this screen are not saved. Leave it anyway?');

  const show = useCallback((next: Screen) => {
    if (!leave()) return false;
    unsaved.current = false;
    setScreen(next);
    window.history.pushState(null, '', address(next));
    return true;
  }, []);

  const go = useCallback(
    (next: Screen) => {
      if (!show(next)) return;
      if (project !== null && next.kind === 'block') {
        setVisited((all) => ({ ...all, [project]: latestFirst(all[project] ?? [], next, RECENT, sameScreen) }));
      }
    },
    [show, project, setVisited],
  );

  const openProject = useCallback(
    (path: string | null) => {
      if (!show({ kind: 'home' })) return;
      setProject(path);
      if (path !== null) setRecent((paths) => latestFirst(paths, path, RECENT));
    },
    [setProject, setRecent, show],
  );

  const movedProject = useCallback(
    (path: string, to: string | null) => {
      setRecent((paths) => (to === null ? paths.filter((p) => p !== path) : paths.map((p) => (p === path ? to : p))));
      setVisited(({ [path]: moved, ...others }) => (to === null || moved === undefined ? others : { ...others, [to]: moved }));
      if (project === path) openProject(to);
    },
    [project, openProject, setRecent, setVisited],
  );

  const movedBlock = useCallback(
    (folder: string, name: string, to: string | null) => {
      if (project === null) return;
      const was = (other: Screen) => other.kind === 'block' && other.folder === folder && other.name === name;
      const next: Screen | null = to === null ? null : { kind: 'block', folder, name: to };
      setVisited((all) => ({
        ...all,
        [project]: (all[project] ?? []).flatMap((other) => (was(other) ? (next ? [next] : []) : [other])),
      }));
      // (the screen of the block shows it as renamed, or the home)
      const shown: Screen | null = was(screen) ? (next ?? { kind: 'home' }) : null;
      if (shown !== null) {
        setScreen(shown);
        window.history.replaceState(null, '', address(shown));
      }
    },
    [project, screen, setVisited],
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
      movedProject,
      movedBlock,
      setUnsaved,
    }),
    [project, openProject, recent, screen, go, visited, folded, setFolded, adding, startAdding, movedProject, movedBlock, setUnsaved],
  );
  return <StudioContext.Provider value={studio}>{children}</StudioContext.Provider>;
}
