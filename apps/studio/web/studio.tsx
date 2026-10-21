// What the panels of the Studio share: opening files, runs and panels in
// the dock, and the live session.
import { createContext, useContext } from 'react';
import type { LiveSession } from './api/session';

export interface Studio {
  openFile: (path: string) => void;
  openRun: (path: string) => void;
  openPanel: (kind: 'preview' | 'jobs') => void;
  live: LiveSession;
}

export const StudioContext = createContext<Studio | null>(null);

export function useStudio(): Studio {
  const studio = useContext(StudioContext);
  if (studio === null) throw new Error('useStudio is used out of the Studio');
  return studio;
}

// Whether a path is a setup of a project (a file in its folder setups)
export function isSetup(path: string): boolean {
  return /(^|\/)setups\/.+\.yaml$/.test(path);
}
