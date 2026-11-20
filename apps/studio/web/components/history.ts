// The edits of a block, with a history: the states before, the one shown
// (null: as saved, the block as the server has it) and those undone.
// Changes made within a moment of each other (the keys of a number as it
// is typed, a slider as it is dragged) are one step; a revert is a step of
// its own, undone like any other. The states kept are whole values, never
// "as saved", so that undo gives back what was shown even after a save.
import { useCallback, useRef, useState } from 'react';

// The time within which changes are one step (ms), and the steps kept
const STEP_MS = 600;
const STEPS = 100;

interface History<T> {
  past: T[];
  present: T | null;
  future: T[];
}

export interface Edits<T> {
  // The value as edited (null: as saved)
  value: T | null;
  // Change it: to a value, to a change of the value shown (as saved, if not
  // edited), or to null, as saved (a revert)
  set: (next: T | null | ((current: T | null) => T | null)) => void;
  // As a save has made the value edited the block as saved: the history
  // stays
  settle: () => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
}

export function useEdits<T>(saved: T | undefined): Edits<T> {
  const [history, setHistory] = useState<History<T>>({ past: [], present: null, future: [] });
  // (the value as saved, for the states kept; when the last change was)
  const savedRef = useRef(saved);
  savedRef.current = saved;
  const last = useRef(0);
  const set = useCallback((next: T | null | ((current: T | null) => T | null)) => {
    const now = Date.now();
    const merge = now - last.current < STEP_MS;
    last.current = now;
    setHistory(({ past, present, future }) => {
      const value = typeof next === 'function' ? (next as (current: T | null) => T | null)(present) : next;
      if (value === present) return { past, present, future };
      const shown = present ?? savedRef.current;
      if (merge && value !== null && present !== null && past.length > 0) return { past, present: value, future: [] };
      return { past: shown === undefined ? past : [...past, shown].slice(-STEPS), present: value, future: [] };
    });
  }, []);
  const settle = useCallback(() => {
    last.current = 0;
    setHistory(({ past, future }) => ({ past, present: null, future }));
  }, []);
  const undo = useCallback(() => {
    last.current = 0;
    setHistory(({ past, present, future }) => {
      const shown = present ?? savedRef.current;
      if (past.length === 0 || shown === undefined) return { past, present, future };
      return { past: past.slice(0, -1), present: past[past.length - 1], future: [shown, ...future] };
    });
  }, []);
  const redo = useCallback(() => {
    last.current = 0;
    setHistory(({ past, present, future }) => {
      const shown = present ?? savedRef.current;
      if (future.length === 0 || shown === undefined) return { past, present, future };
      return { past: [...past, shown], present: future[0], future: future.slice(1) };
    });
  }, []);
  return { value: history.present, set, settle, undo, redo, canUndo: history.past.length > 0, canRedo: history.future.length > 0 };
}
