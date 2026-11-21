// The folds of a page: parts that can hide what they hold (e.g. a section
// of the dock), each giving what it holds the way to open it. Folds nest,
// so that what is inside one (e.g. the field a problem leads to) can open
// all those around it to be seen. A fold hides what it holds rather than
// dropping it, so that what is inside stays known while folded.
import { createContext, useContext, useMemo, type ReactNode } from 'react';

export interface Fold {
  // Its id in the page (e.g. that of its section)
  id: string;
  open: () => void;
  // The fold around it (null: none)
  outer: Fold | null;
}

const Context = createContext<Fold | null>(null);

// What a fold holds: inside it, its own and those around it (open should
// be stable, e.g. a state setter's call kept with useCallback)
export function FoldScope({ id, open, children }: { id: string; open: () => void; children: ReactNode }) {
  const outer = useContext(Context);
  const fold = useMemo(() => ({ id, open, outer }), [id, open, outer]);
  return <Context.Provider value={fold}>{children}</Context.Provider>;
}

// The innermost fold around the caller (null: none)
export function useFold(): Fold | null {
  return useContext(Context);
}

// Opens a fold and all those around it
export function openFolds(fold: Fold | null): void {
  for (let at = fold; at !== null; at = at.outer) at.open();
}

// Whether a fold is, or is inside, the fold of an id
export function isWithin(fold: Fold | null, id: string): boolean {
  for (let at = fold; at !== null; at = at.outer) if (at.id === id) return true;
  return false;
}
