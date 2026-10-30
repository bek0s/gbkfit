// The screens of the stage, and their addresses (#home, #fit,
// #block/<folder>/<name>), so that the browser's back, forward and reload
// keep the screen.

export type Screen =
  | { kind: 'home' }
  | { kind: 'block'; folder: string; name: string }
  | { kind: 'observe' }
  | { kind: 'compare' }
  | { kind: 'fit' };

export function address(screen: Screen): string {
  if (screen.kind === 'block') return `#block/${screen.folder}/${encodeURIComponent(screen.name)}`;
  return `#${screen.kind}`;
}

// The screen of an address; the home for one that is not a screen
export function parseAddress(hash: string): Screen {
  const [kind, ...rest] = hash.replace(/^#/, '').split('/');
  if (kind === 'block' && rest.length === 2) return { kind, folder: rest[0], name: decodeURIComponent(rest[1]) };
  if (kind === 'observe' || kind === 'compare' || kind === 'fit') return { kind };
  return { kind: 'home' };
}

export function sameScreen(a: Screen, b: Screen): boolean {
  return address(a) === address(b);
}
