// What the page remembers in this browser (e.g. the open project, the
// recent ones, the lists hidden). Storage can be missing or
// refused (e.g. a private window): the page then forgets.
import { useEffect, useState } from 'react';

const PREFIX = 'gbkfit-studio:';

export function recall<T>(key: string, fallback: T): T {
  try {
    const text = localStorage.getItem(PREFIX + key);
    return text === null ? fallback : (JSON.parse(text) as T);
  } catch {
    return fallback;
  }
}

export function remember(key: string, value: unknown): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // (no storage: forgotten)
  }
}

// A state that the page remembers by a key
export function useRemembered<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => recall(key, initial));
  useEffect(() => remember(key, value), [key, value]);
  return [value, setValue] as const;
}

// A list with an item first, without its other copies, cut to a length
// (e.g. the projects opened last)
export function latestFirst<T>(list: T[], item: T, length: number, same: (a: T, b: T) => boolean = Object.is): T[] {
  return [item, ...list.filter((other) => !same(other, item))].slice(0, length);
}
