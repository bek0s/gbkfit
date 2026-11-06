// The files of a dataset, in the dock: for each item (the cube, the flux
// map, or each moment map), its data, its errors (a file, or one value)
// and its mask, a row each. A file is given from the workspace or from
// this computer (each copied into the project), or is another array of
// the file of the data. Read only, the rows say what the dataset reads.
import { useState, type ReactNode } from 'react';
import type { ArrayInfo } from '../../api/client';
import { useHeader } from '../../api/queries';
import { useFileSource } from '../../components/FileSources';
import { significant } from '../../components/format';
import { Close } from '../../components/ui/basics';
import { Choice } from '../../components/ui/inputs';
import { DropMenu, type MenuEntry } from '../../components/ui/overlays';
import { HeaderFacts } from './Header';
import {
  arrayLabel,
  arrayRef,
  ERROR_KINDS,
  errorKindOf,
  isFlags,
  kept,
  KINDS,
  NO_ARRAYS,
  headerProblems,
  presentSlots,
  restOf,
  sameShape,
  siblings,
  slotTitle,
  type Arrays,
  type ErrorKind,
  type FileArray,
  type Item,
  type Kind,
  type Resolved,
  type Role,
  type SpectralType,
} from './model';

export function Files({
  project,
  resolved,
  onItem,
  onAdd,
  onRemove,
}: {
  project: string;
  resolved: Resolved;
  // How the items change (none: read only)
  onItem?: (key: string, item: Item) => void;
  onAdd?: (key: string) => void;
  onRemove?: (key: string) => void;
}) {
  const { kind, draft, arrays } = resolved;
  const slots = presentSlots(kind, draft);
  const problems = headerProblems(resolved);
  const absent = KINDS[kind].slots.filter((slot) => draft.items[slot.key] === undefined);
  return (
    <div className="ds-files">
      {slots.map((slot) => (
        <div key={slot.key} className="ds-item" data-group={kind === 'moments' || undefined}>
          {kind === 'moments' && (
            <div className="ds-item-head">
              <span className="ds-item-short">{slot.short}</span>
              <span className="ds-item-title">{slotTitle(slot, kind).replace(/^Moment \d · /, '')}</span>
              {onRemove && <Close label={`Remove the map of ${slot.title.toLowerCase()}`} onClick={() => onRemove(slot.key)} />}
            </div>
          )}
          <ItemRows
            project={project}
            header={{
              rest: kind === 'cube' ? restOf(draft.spectral) : null,
              type: kind === 'cube' ? draft.spectral.type : null,
              problems: problems[slot.key] ?? [],
            }}
            kind={kind}
            item={draft.items[slot.key]}
            arrays={arrays[slot.key] ?? NO_ARRAYS}
            onChange={onItem && ((item) => onItem(slot.key, item))}
          />
        </div>
      ))}
      {onAdd && absent.length > 0 && (
        <DropMenu
          label="Add a map"
          entries={absent.map((slot) => ({ label: slotTitle(slot, kind), onClick: () => onAdd(slot.key) }))}
          target={<button className="ds-add">+ Add a map</button>}
        />
      )}
    </div>
  );
}

// What the facts of the header of the data are read with: the rest and the
// type of the spectral axis as given, and the problems a dataset of them
// has (see headerProblems)
interface HeaderContext {
  rest: string | null;
  type: SpectralType | null;
  problems: string[];
}

function ItemRows({
  project,
  header,
  kind,
  item,
  arrays,
  onChange,
}: {
  project: string;
  header: HeaderContext;
  kind: Kind;
  item: Item;
  arrays: Arrays;
  onChange?: (item: Item) => void;
}) {
  // (a file removed is no longer among the files, rather than one of none)
  const setFile = (role: Role, file: FileArray | undefined) => {
    const { [role]: _gone, ...others } = item.files;
    onChange?.({ ...item, files: file ? { ...others, [role]: file } : others, ...(role === 'error' ? { errorKind: null } : {}) });
  };
  return (
    <div className="ds-rows">
      <Row title="Data">
        <FileSlot project={project} header={header} role="data" kind={kind} item={item} arrays={arrays} onFile={onChange && ((file) => setFile('data', file))} />
      </Row>
      <Row title="Errors">
        {item.errorValue !== null ? (
          <ErrorValue item={item} arrays={arrays} onChange={onChange} />
        ) : (
          <FileSlot
            project={project}
            header={header}
            role="error"
            kind={kind}
            item={item}
            arrays={arrays}
            onFile={onChange && ((file) => setFile('error', file))}
            onValue={onChange && ((value) => onChange({ ...item, errorValue: value }))}
            extra={
              item.files.error && (
                <ErrorKindChoice item={item} arrays={arrays} onChange={onChange && ((errorKind) => onChange({ ...item, errorKind }))} />
              )
            }
          />
        )}
      </Row>
      <Row title="Mask">
        <FileSlot project={project} header={header} role="mask" kind={kind} item={item} arrays={arrays} onFile={onChange && ((file) => setFile('mask', file))} />
        {arrays.found.mask && isFlags(arrays.found.mask) && (
          <p className="ds-warn">
            {arrays.found.mask.name} flags the bad pixels (0 where the data are good), but GBKFIT keeps the pixels where a mask is not 0: this mask keeps
            only the flagged ones.
          </p>
        )}
      </Row>
    </div>
  );
}

function Row({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="ds-row">
      <span className="ds-row-label">{title}</span>
      <div className="ds-row-body">{children}</div>
    </div>
  );
}

// The file of a role: its name and its array (chosen among those of its
// file), or the ways to give one; a chevron opens what GBKFIT reads from
// the header of its array (one header for each array, an HDU: the data
// and the errors of one file have each their own), and the header as it
// is in the file (open at first if GBKFIT finds a problem in it)
function FileSlot({
  project,
  header,
  role,
  kind,
  item,
  arrays,
  onFile,
  onValue,
  extra,
}: {
  project: string;
  header: HeaderContext;
  role: Role;
  kind: Kind;
  item: Item;
  arrays: Arrays;
  onFile?: (file: FileArray | undefined) => void;
  onValue?: (value: string) => void;
  extra?: ReactNode;
}) {
  const source = useFileSource(project, (file) => onFile?.({ file }));
  const chosen = item.files[role];
  const inFile = arrays.inFile[role] ?? [];
  const found = arrays.found[role];
  const error = arrays.errors[role];
  // (whether the header shows: as the user chose; its problems are in the
  // head of the screen too)
  const [open, setOpen] = useState(false);
  if (chosen !== undefined) {
    const name = chosen.file.split('/').pop();
    return (
      <>
        <div className="ds-slot">
          <button
            type="button"
            className="ds-chevron"
            aria-expanded={open}
            disabled={found === undefined}
            title={open ? 'Hide the header' : 'What GBKFIT reads from the header, and the header'}
            onClick={() => setOpen(!open)}
          >
            ▸
          </button>
          <span className="ds-file" title={chosen.file}>
            {name}
          </span>
          {onFile && inFile.length > 1 ? (
            <Choice
              items={inFile.map((array) => {
                const why = unusable(role, array, arrays.found.data, KINDS[kind].ndim);
                return {
                  value: String(array.index),
                  label: arrayLabel(array),
                  description: `${kept(array).join(' × ') || 'one value'}${why ? ` · ${why}` : ''}`,
                  // (the one chosen stays, to show what is wrong with it)
                  disabled: why !== null && array.index !== found?.index,
                };
              })}
              value={String(found?.index ?? '')}
              label={`The array of the ${role === 'data' ? 'data' : role === 'error' ? 'errors' : 'mask'}`}
              onChange={(index) => {
                const array = inFile.find((a) => a.index === Number(index));
                if (array) onFile({ file: chosen.file, array: arrayRef(inFile, array) });
              }}
            />
          ) : (
            found && inFile.length > 1 && <span className="ds-array">{arrayLabel(found)}</span>
          )}
          {onFile && <Close label={`Remove the ${role === 'data' ? 'data' : role === 'error' ? 'errors' : 'mask'}`} onClick={() => onFile(undefined)} />}
        </div>
        {open && found && <HeaderOf path={chosen.file} role={role} array={found} data={arrays.found.data} header={header} />}
        {extra}
        {error && <p className="error">{error}</p>}
      </>
    );
  }
  if (!onFile) return <span className="ds-none">{role === 'data' ? 'None' : '—'}</span>;
  const others = role === 'data' ? [] : siblings(arrays, role);
  const dataFile = item.files.data?.file;
  const entries: MenuEntry[] = [
    { label: 'A file of the workspace…', description: 'The project keeps a copy', onClick: source.fromWorkspace },
    { label: 'A file of this computer…', description: 'Uploaded into the project', onClick: source.fromComputer },
  ];
  if (others.length > 0 && dataFile !== undefined) {
    entries.push({
      label: `An array of ${dataFile.split('/').pop()}`,
      entries: others.map((array) => ({ label: arrayLabel(array), onClick: () => onFile({ file: dataFile, array: arrayRef(arrays.inFile.data ?? [], array) }) })),
    });
  }
  if (onValue) {
    const rms = arrays.found.data?.rms;
    entries.push({ section: 'Or' });
    entries.push({
      label: 'One value for all the data',
      description: rms != null ? `The header says ${significant(rms, 3)}` : 'e.g. the rms of a radio cube',
      onClick: () => onValue(rms != null ? String(rms) : ''),
    });
  }
  return (
    <>
      <DropMenu
        label={`Add the ${role === 'data' ? 'data' : role === 'error' ? 'errors' : 'mask'}`}
        entries={entries}
        target={<button className="ds-add-file">+ Add</button>}
      />
      {source.status}
      {source.hidden}
    </>
  );
}

// What GBKFIT reads from the header of an array, and the header as it is
// in the file, read as it is asked for (in a box of its own, which
// scrolls: a header can be thousands of lines). Of the data, all that
// GBKFIT reads (see HeaderFacts): the dataset is on their grid. Of their
// errors and mask, which are on that grid whatever their headers say
// (GBKFIT reads the coordinates of the data alone), whether their shape is
// that of the data, and what the errors are, if their header says.
function HeaderOf({ path, role, array, data, header }: { path: string; role: Role; array: ArrayInfo; data: ArrayInfo | undefined; header: HeaderContext }) {
  const [full, setFull] = useState(false);
  const cards = useHeader(full ? path : null, array.index);
  const holds = ERROR_KINDS.find((k) => k.kind === array.holds);
  return (
    <div className="ds-header-of">
      {role === 'data' ? (
        <HeaderFacts array={array} rest={header.rest} type={header.type} problems={header.problems} />
      ) : (
        <dl className="ds-facts">
          <dt>Shape</dt>
          <dd>
            {kept(array).join(' × ')}
            {data && (sameShape(array, data) ? ', that of the data' : `: not that of the data (${kept(data).join(' × ')})`)}
          </dd>
          <dt>Coordinates</dt>
          <dd>those of the data</dd>
          {role === 'error' && (
            <>
              <dt>They are</dt>
              <dd>{holds ? `${holds.text.toLowerCase()} (${holds.title}), as the header says` : 'not said by the header'}</dd>
            </>
          )}
        </dl>
      )}
      <button type="button" className="ds-full-header" aria-expanded={full} onClick={() => setFull(!full)}>
        <span className="ds-chevron-sign">▸</span> The header of {array.name || `HDU ${array.index}`}
        {cards.data && <span className="ds-header-count"> · {cards.data.cards.length} lines</span>}
      </button>
      {full &&
        (cards.data ? (
          <pre className="ds-header-text">{cards.data.cards.join('\n')}</pre>
        ) : cards.error ? (
          <p className="error">{cards.error.message}</p>
        ) : (
          <p className="note">Reading the header…</p>
        ))}
    </div>
  );
}

// Why an array cannot be of a role (null: it can): the data need the
// number of axes of the kind, their errors and mask their shape
function unusable(role: Role, array: ArrayInfo, data: ArrayInfo | undefined, ndim: number): string | null {
  const count = kept(array).length;
  if (role === 'data') return count === ndim ? null : `${count} ${count === 1 ? 'axis' : 'axes'}`;
  if (data === undefined || kept(array).join() === kept(data).join()) return null;
  return 'another shape';
}

function ErrorValue({ item, arrays, onChange }: { item: Item; arrays: Arrays; onChange?: (item: Item) => void }) {
  const unit = arrays.found.data?.unit;
  return (
    <div className="ds-slot">
      {onChange ? (
        <input
          className="ds-value"
          type="number"
          step="any"
          min="0"
          autoFocus
          value={item.errorValue ?? ''}
          aria-label="The error of every value"
          onChange={(event) => onChange({ ...item, errorValue: event.target.value })}
        />
      ) : (
        <span className="ds-file">{item.errorValue}</span>
      )}
      <span className="ds-array">{unit ? `${unit} · ` : ''}every value</span>
      {onChange && <Close label="Remove the error" onClick={() => onChange({ ...item, errorValue: null })} />}
    </div>
  );
}

// What the errors are: as chosen, or as their header says, or else to be
// chosen (nothing is assumed)
function ErrorKindChoice({ item, arrays, onChange }: { item: Item; arrays: Arrays; onChange?: (kind: ErrorKind) => void }) {
  const kind = errorKindOf(item, arrays);
  const known = ERROR_KINDS.find((k) => k.kind === kind);
  if (!onChange) return <span className="ds-array">{known?.text ?? 'Not said'}</span>;
  return (
    <div className="ds-error-kind">
      <span>They are</span>
      <Choice
        items={ERROR_KINDS.map(({ kind: value, title, text }) => ({ value, label: `${text} (${title})` }))}
        value={kind ?? ''}
        placeholder="Choose: the header does not say"
        label="What the errors are"
        onChange={(value) => onChange(value as ErrorKind)}
      />
    </div>
  );
}
