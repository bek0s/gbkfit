// A file read by Studio (e.g. the data of a dataset, or the image of a
// PSF), as a row of text and a popover that sets it. The row says what is
// read (FileSummary): the name of the file, and under it its array (of a
// file of several) and how it is read (e.g. 'σ'); each part is cut in its
// middle, so any length fits, and marked by its problems; a reading a row
// needs and has not is asked in its place (e.g. 'unit?'). The popover
// (FilePopover), beside the row, has the room the row has not: the file
// and the ways to give one (FileSection), the arrays of the file with
// their shapes (ArraySection), and whatever else says how it is read
// (PopSection). It is the same whether something is given or not: what is
// given is marked in it, and nothing changes until something is chosen.
import { Popover } from '@mantine/core';
import { cloneElement, useEffect, useRef, useState, type KeyboardEvent, type ReactElement, type ReactNode } from 'react';
import type { ArrayInfo } from '../api/client';
import { arrayLabel, kept } from './arrays';
import type { FileSource } from './FileSources';
import type { Severity } from './Problems';
import { TextField } from './ui/basics';
import { MiddleCut } from './ui/cut';
import { SEARCHED } from './ui/overlays';

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

// The popover of a row (its target), at its right (over the screen beside
// the dock), titled by what the row is (and, e.g., the map it is of), if
// its label does not say it (e.g. a field of a form: its parts do); Escape
// (in it, or on its row), or a click beside it, closes it. What it holds
// is made only while it is open.
export function FilePopover({
  opened,
  onChange,
  target,
  title,
  context,
  children,
}: {
  opened: boolean;
  onChange: (opened: boolean) => void;
  target: ReactElement<{ onKeyDown?: (event: KeyboardEvent) => void }>;
  title?: string;
  context?: string;
  children: ReactNode;
}) {
  return (
    <Popover
      opened={opened}
      onChange={onChange}
      position="right-start"
      offset={10}
      width={340}
      middlewares={{ flip: true, shift: true }}
      trapFocus={false}
      returnFocus
      classNames={{ dropdown: 'file-pop' }}
    >
      <Popover.Target>
        {cloneElement(target, {
          onKeyDown: (event: KeyboardEvent) => {
            if (event.key === 'Escape' && opened) onChange(false);
          },
        })}
      </Popover.Target>
      <Popover.Dropdown aria-label={[title ?? 'The file', context].filter(Boolean).join(' · ')}>
        {title && (
          <div className="file-pop-title">
            {title}
            {context && <small>{context}</small>}
          </div>
        )}
        {children}
      </Popover.Dropdown>
    </Popover>
  );
}

// A part of a popover: its label, a note or an action at its right (e.g.
// a count), and what it holds
export function PopSection({ label, aside, children }: { label: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="file-pop-part">
      <div className="file-pop-label">
        <span className="eyebrow">{label}</span>
        {aside && <small>{aside}</small>}
      </div>
      {children}
    </section>
  );
}

// The file given (its name, all of it, and its folder), or none, and the
// ways to give one, each copying it into the project: of the workspace, or
// of this computer
export function FileSection({ path, source }: { path: string | undefined; source: Pick<FileSource, 'fromWorkspace' | 'fromComputer'> }) {
  const parts = path?.split('/') ?? [];
  const name = parts.pop();
  return (
    <PopSection label="File">
      {name ? <span className="file-pop-name">{name}</span> : <span className="file-pop-none">None chosen</span>}
      <div className="file-pop-foot">
        <span className="file-pop-path">{parts.length > 0 ? `${parts.join('/')}/` : ''}</span>
        <span className="file-pop-another">
          {name ? 'Another:' : 'Choose:'}
          <button type="button" className="file-pop-link" title="A file of the workspace; the project keeps a copy" onClick={source.fromWorkspace}>
            workspace…
          </button>
          <button type="button" className="file-pop-link" title="A file of this computer, uploaded into the project" onClick={source.fromComputer}>
            this computer…
          </button>
        </span>
      </div>
    </PopSection>
  );
}

// An option of a popover: its label (mono: a name of the data, e.g. of an
// array), and what it is, at its right; one that cannot be chosen is
// faint, and says why on hover
function PopOption({
  label,
  description,
  current,
  mono,
  unavailable,
  onClick,
}: {
  label: string;
  description?: string;
  current?: boolean;
  mono?: boolean;
  unavailable?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="file-pop-item"
      data-current={current || undefined}
      data-mono={mono || undefined}
      disabled={unavailable !== undefined}
      title={unavailable}
      onClick={onClick}
    >
      <span>{label}</span>
      {description && <small>{description}</small>}
    </button>
  );
}

// The arrays of a file, each with its shape, the one read marked (any can
// be chosen: what reads it says if it cannot be read there), but one taken
// (e.g. the data, of the file of the data); a file of many has a field
// that finds them (Enter chooses the first found)
export function ArraySection({
  file,
  arrays,
  current,
  taken,
  onChoose,
}: {
  // The file, named under the label (e.g. that of the data, for its
  // errors)
  file?: string;
  arrays: ArrayInfo[];
  current?: ArrayInfo;
  taken?: { array: ArrayInfo; by: string };
  onChoose: (array: ArrayInfo) => void;
}) {
  const [query, setQuery] = useState('');
  const list = useRef<HTMLDivElement>(null);
  // (the one read in view as the popover opens; the effect returns
  // nothing: scrollIntoView may return a promise)
  useEffect(() => {
    list.current?.querySelector('[data-current]')?.scrollIntoView({ block: 'nearest' });
  }, []);
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const shown = arrays.filter((array) => words.every((word) => arrayLabel(array).toLowerCase().includes(word)));
  const choosable = shown.filter((array) => array.index !== taken?.array.index);
  return (
    <PopSection label="Array" aside={`${arrays.length} in the file`}>
      {file && <span className="file-pop-path">{file.split('/').pop()}</span>}
      {arrays.length > SEARCHED && (
        <TextField
          search
          value={query}
          onChange={setQuery}
          placeholder="Find"
          label="Find an array"
          autoFocus
          onKeyDown={(event) => {
            if (event.key === 'Enter' && choosable.length > 0) onChoose(choosable[0]);
            if (event.key === 'ArrowDown') list.current?.querySelector<HTMLButtonElement>('button:enabled')?.focus();
          }}
        />
      )}
      <div
        ref={list}
        className="file-pop-list file-pop-scroll"
        onKeyDown={(event) => {
          // (the arrows move over the arrays)
          const step = event.key === 'ArrowDown' ? 'nextElementSibling' : event.key === 'ArrowUp' ? 'previousElementSibling' : null;
          const next = step && (document.activeElement?.[step] as HTMLElement | null);
          if (next) {
            event.preventDefault();
            next.focus();
          }
        }}
      >
        {shown.map((array) => (
          <PopOption
            key={array.index}
            label={arrayLabel(array)}
            description={kept(array).join(' × ') || 'one value'}
            current={array.index === current?.index}
            mono
            unavailable={array.index === taken?.array.index ? `It is ${taken.by}` : undefined}
            onClick={() => onChoose(array)}
          />
        ))}
        {shown.length === 0 && <span className="note">Nothing found</span>}
      </div>
    </PopSection>
  );
}
