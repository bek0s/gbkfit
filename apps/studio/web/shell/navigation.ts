// The screens of the stage, and their addresses (#home, #fit,
// #block/<folder>/<name>, #new-dataset/<kind>, #section/<key>), so
// that the browser's back, forward and reload keep the screen.
import type { SectionKey } from './sections';

export type Screen =
  | { kind: 'home' }
  // A section of blocks with none open (e.g. a project without galaxies)
  | { kind: 'section'; key: SectionKey }
  | { kind: 'block'; folder: string; name: string }
  // A dataset being added: the key of its kind of data (see DATA_KINDS)
  | { kind: 'new-dataset'; data: string }
  | { kind: 'compare' }
  | { kind: 'fit' };

export function address(screen: Screen): string {
  if (screen.kind === 'block') return `#block/${screen.folder}/${encodeURIComponent(screen.name)}`;
  if (screen.kind === 'new-dataset') return `#new-dataset/${screen.data}`;
  if (screen.kind === 'section') return `#section/${screen.key}`;
  return `#${screen.kind}`;
}

// The screen of an address; the home for one that is not a screen
export function parseAddress(hash: string): Screen {
  const [kind, ...rest] = hash.replace(/^#/, '').split('/');
  if (kind === 'block' && rest.length === 2) return { kind, folder: rest[0], name: decodeURIComponent(rest[1]) };
  if (kind === 'new-dataset' && rest.length === 1) return { kind, data: rest[0] };
  if (kind === 'section' && rest.length === 1 && isSection(rest[0])) return { kind, key: rest[0] };
  if (kind === 'compare' || kind === 'fit') return { kind };
  return { kind: 'home' };
}

function isSection(key: string): key is SectionKey {
  return ['home', 'data', 'galaxies', 'observations', 'compare', 'fit'].includes(key);
}

export function sameScreen(a: Screen, b: Screen): boolean {
  return address(a) === address(b);
}
