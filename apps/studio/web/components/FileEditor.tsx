// A file read by Studio (e.g. the data of a dataset, or the image of a
// PSF), as lines of a row, a fact each: the file (or what else is read,
// e.g. one value), its array (of a file of several), and how it is read
// (e.g. 'σ'). Read only, the lines are text (FileSummary): each part is
// cut in its middle, so any length fits, and marked by its problems; a
// reading a row needs and has not is asked in its place (e.g. 'unit?').
// Editing, each fact is set where it is said, as quiet text with ▾ that
// opens its menu (SourceButton: where it comes from, e.g. a file of the
// workspace or of this computer; ArrayButton: the array of the file), or
// a small control in the line (e.g. what the errors are).
import type { ReactNode } from 'react';
import type { ArrayInfo } from '../api/client';
import { arrayLabel, kept } from './arrays';
import type { FileSource } from './FileSources';
import type { Severity } from './Problems';
import { MiddleCut } from './ui/cut';
import { DropMenu, type MenuEntry } from './ui/overlays';

// What is read: the name of the file (all its path on hover), its array,
// if it is one of several, and how it is read, if anything is said (or
// what is to be said); each part marked by the worst of its problems
export function FileSummary({
  path,
  array,
  fact,
  marks = {},
}: {
  path: string;
  array?: string;
  fact?: string;
  marks?: { name?: Severity | null; array?: Severity | null; fact?: Severity | null };
}) {
  return (
    <span className="file-sum" data-name={marks.name ?? undefined} data-array={marks.array ?? undefined} data-fact={marks.fact ?? undefined}>
      <MiddleCut text={path.split('/').pop()!} title={path} className="file-sum-name" />
      {(array !== undefined || fact) && (
        <span className="file-sum-more">
          {array !== undefined && <MiddleCut text={array} className="file-sum-array" />}
          {array !== undefined && fact && <span className="file-sum-sep">·</span>}
          {fact && <span className="file-sum-fact">{fact}</span>}
        </span>
      )}
    </span>
  );
}

// The ways to give a file, each copying it into the project: of the
// workspace, or of this computer (entries of a menu)
export function sourceEntries(source: Pick<FileSource, 'fromWorkspace' | 'fromComputer'>): MenuEntry[] {
  return [
    { label: 'A file of the workspace…', description: 'The project keeps a copy', onClick: source.fromWorkspace },
    { label: 'A file of this computer…', description: 'Uploaded into the project', onClick: source.fromComputer },
  ];
}

// The arrays of a file as entries of a menu, each with its shape, the one
// in use marked, one taken (e.g. the data, of the file of the data) not
// to be chosen; any other can be (what reads it says if it cannot be read
// there)
export function arrayEntries(
  arrays: ArrayInfo[],
  current: ArrayInfo | undefined,
  onChoose: (array: ArrayInfo) => void,
  taken?: { array: ArrayInfo; by: string },
): MenuEntry[] {
  return arrays.map((array) => {
    const isTaken = array.index === taken?.array.index;
    return {
      label: arrayLabel(array),
      description: isTaken ? `${shapeOf(array)} · ${taken!.by}` : shapeOf(array),
      current: array.index === current?.index,
      disabled: isTaken,
      onClick: () => onChoose(array),
    };
  });
}

// The shape of an array, as a menu says it
function shapeOf(array: ArrayInfo): string {
  return kept(array).join(' × ') || 'one value';
}

// Where what a row reads comes from, as quiet text with ▾ (a name of a
// file, cut in its middle, all its path on hover; or words, e.g. 'One
// value', or '+ Add' while nothing is given), marked by its problems; its
// menu offers where else it can come from
export function SourceButton({
  file,
  text,
  label,
  entries,
  marked,
}: {
  // The file read (none: text says what is)
  file?: string;
  text?: string;
  // What the menu is (e.g. 'Where the errors come from')
  label: string;
  entries: MenuEntry[];
  marked?: Severity | null;
}) {
  return (
    <DropMenu
      label={label}
      width={300}
      entries={entries}
      target={
        <button type="button" className="file-q" data-words={file === undefined || undefined} data-severity={marked ?? undefined} title={file}>
          {file !== undefined ? <MiddleCut text={file.split('/').pop()!} className="file-q-text" /> : <span className="file-q-text">{text}</span>}
          <span className="file-q-tri" aria-hidden>
            ▾
          </span>
        </button>
      }
    />
  );
}

// The array of a file of several, as quiet text with ▾, marked by its
// problems; its menu has every array of the file (a field finds them, of
// many)
export function ArrayButton({
  arrays,
  current,
  label,
  taken,
  marked,
  onChoose,
}: {
  arrays: ArrayInfo[];
  current: ArrayInfo | undefined;
  // What the menu is (e.g. 'The array of the errors')
  label: string;
  taken?: { array: ArrayInfo; by: string };
  marked?: Severity | null;
  onChoose: (array: ArrayInfo) => void;
}) {
  return (
    <DropMenu
      label={label}
      width={300}
      entries={arrayEntries(arrays, current, onChoose, taken)}
      target={
        <button type="button" className="file-q" data-small data-severity={marked ?? undefined}>
          <MiddleCut text={current ? arrayLabel(current) : 'Choose'} className="file-q-text" />
          <span className="file-q-tri" aria-hidden>
            ▾
          </span>
        </button>
      }
    />
  );
}

// A line of a row: its facts, side by side, wrapping between them
export function FileLine({ children }: { children: ReactNode }) {
  return <span className="file-line">{children}</span>;
}

// A label and the control it names, never apart (e.g. '≠ 0' and what it
// marks; wide: it takes the rest of its line, e.g. a field of a unit),
// marked by the problems of what it sets
export function FilePair({ label, marked, wide, children }: { label?: string; marked?: Severity | null; wide?: boolean; children: ReactNode }) {
  return (
    <span className="file-pair" data-wide={wide || undefined} data-severity={marked ?? undefined}>
      {label && <span className="file-pair-label">{label}</span>}
      {children}
    </span>
  );
}
