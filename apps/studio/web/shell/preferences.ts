// The preferences of the user (e.g. the theme, the colour scheme, the
// colour map), which the server keeps in their folder (~/.gbkfit; see its
// /preferences): the same whatever the folder it serves, or its port,
// unlike what the page remembers in the browser (see remember.ts), which
// is that of an address. The page reads them before it starts (see
// main.tsx), so that it starts in them. One set is sent to the server,
// which tells every page open (its event 'preferences', see
// api/events.ts), so that they all follow it. Each is checked as it is
// read (e.g. a theme no longer known, or a file edited by hand): one not
// valid is taken as not set.
import { useSyncExternalStore } from 'react';
import { api, data } from '../api/client';

type Values = Record<string, unknown>;

let values: Values = {};
// (where they are kept, as the server said, or why they could not be read
// or kept)
let status: { file: string | null; problem: string | null } = { file: null, problem: null };
// (the changes sent and not yet answered: until they are, what the server
// tells is older than what the page has)
let sending = 0;
const followers = new Set<() => void>();

function take(next: Values, file: string | null = status.file): void {
  values = next;
  status = { file, problem: null };
  followers.forEach((onChange) => onChange());
}

function fail(problem: string): void {
  status = { ...status, problem };
  followers.forEach((onChange) => onChange());
}

const reason = (error: unknown) => (error instanceof Error ? error.message : String(error));

// Read the preferences from the server (if it cannot give them, those the
// page has stay: none as it starts)
export async function loadPreferences(): Promise<void> {
  try {
    const info = await data(api.GET('/api/v1/preferences'));
    if (sending === 0) take(info.values, info.file);
  } catch (error) {
    fail(reason(error));
  }
}

// The preferences as the server tells they changed (see api/events.ts)
export function preferencesChanged(next: Values): void {
  if (sending === 0) take(next);
}

// Set a preference (null: not set), and send it to the server
export function setPreference(name: string, value: unknown): void {
  const next = { ...values };
  if (value === null) delete next[name];
  else next[name] = value;
  take(next);
  sending += 1;
  data(api.PATCH('/api/v1/preferences', { body: { [name]: value } })).then(
    (info) => {
      sending -= 1;
      if (sending === 0) take(info.values, info.file);
    },
    (error: unknown) => {
      sending -= 1;
      fail(reason(error));
    },
  );
}

// A preference, or a fallback if it is not set or not valid
export function preference<T>(name: string, valid: (value: unknown) => value is T, fallback: T): T {
  const value = values[name];
  return valid(value) ? value : fallback;
}

// Follow the preferences as they change (returns what stops following)
export function followPreferences(onChange: () => void): () => void {
  followers.add(onChange);
  return () => followers.delete(onChange);
}

// A preference, followed as it changes
export function usePreference<T>(name: string, valid: (value: unknown) => value is T, fallback: T): T {
  const all = useSyncExternalStore(followPreferences, () => values);
  const value = all[name];
  return valid(value) ? value : fallback;
}

// The file the preferences are kept in (null if the server did not say),
// and why they could not be read or kept, if they could not
export function usePreferencesStatus() {
  return useSyncExternalStore(followPreferences, () => status);
}
