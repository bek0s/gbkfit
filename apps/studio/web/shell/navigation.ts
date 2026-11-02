// The screens of the stage, and their addresses (#home, #fit,
// #block/<folder>/<name>, #new-dataset/<kind>, #draft/<id>/<name>), so
// that the browser's back, forward and reload keep the screen.

export type Screen =
  | { kind: 'home' }
  | { kind: 'block'; folder: string; name: string }
  // A dataset being added: the key of its kind of data (see DATA_KINDS)
  | { kind: 'new-dataset'; data: string }
  // A draft of the screen of a block (see shell/drafts), of a block of its
  // folder (null: the first)
  | { kind: 'draft'; id: string; name: string | null }
  | { kind: 'observe' }
  | { kind: 'compare' }
  | { kind: 'fit' };

export function address(screen: Screen): string {
  if (screen.kind === 'block') return `#block/${screen.folder}/${encodeURIComponent(screen.name)}`;
  if (screen.kind === 'new-dataset') return `#new-dataset/${screen.data}`;
  if (screen.kind === 'draft') return `#draft/${screen.id}${screen.name === null ? '' : `/${encodeURIComponent(screen.name)}`}`;
  return `#${screen.kind}`;
}

// The screen of an address; the home for one that is not a screen
export function parseAddress(hash: string): Screen {
  const [kind, ...rest] = hash.replace(/^#/, '').split('/');
  if (kind === 'block' && rest.length === 2) return { kind, folder: rest[0], name: decodeURIComponent(rest[1]) };
  if (kind === 'new-dataset' && rest.length === 1) return { kind, data: rest[0] };
  if (kind === 'draft' && (rest.length === 1 || rest.length === 2)) {
    return { kind, id: rest[0], name: rest.length === 2 ? decodeURIComponent(rest[1]) : null };
  }
  if (kind === 'observe' || kind === 'compare' || kind === 'fit') return { kind };
  return { kind: 'home' };
}

export function sameScreen(a: Screen, b: Screen): boolean {
  return address(a) === address(b);
}
