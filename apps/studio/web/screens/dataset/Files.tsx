// The files of a dataset, in the dock: for each item (the cube, the flux
// map, or each moment map), its data, its errors (a file, or one value)
// and its mask, a row each. A file is given from the workspace or from
// this computer (each copied into the project), or is another array of
// the file of the data. Read only, the rows say what the dataset reads.
import type { ReactNode } from 'react';
import type { ArrayInfo } from '../../api/client';
import { useFileSource } from '../../components/FileSources';
import { significant } from '../../components/format';
import { Button, Close } from '../../components/ui/basics';
import { Choice } from '../../components/ui/inputs';
import { DropMenu, type MenuEntry } from '../../components/ui/overlays';
import {
  arrayLabel,
  arrayRef,
  ERROR_KINDS,
  errorKindOf,
  isFlags,
  kept,
  KINDS,
  NO_ARRAYS,
  presentSlots,
  siblings,
  slotTitle,
  type Arrays,
  type ErrorKind,
  type FileArray,
  type Item,
  type Kind,
  type Resolved,
  type Role,
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

function ItemRows({
  project,
  kind,
  item,
  arrays,
  onChange,
}: {
  project: string;
  kind: Kind;
  item: Item;
  arrays: Arrays;
  onChange?: (item: Item) => void;
}) {
  const setFile = (role: Role, file: FileArray | undefined) => onChange?.({ ...item, files: { ...item.files, [role]: file }, ...(role === 'error' ? { errorKind: null } : {}) });
  return (
    <div className="ds-rows">
      <Row title="Data">
        <FileSlot project={project} role="data" kind={kind} item={item} arrays={arrays} onFile={onChange && ((file) => setFile('data', file))} />
      </Row>
      <Row title="Errors">
        {item.errorValue !== null ? (
          <ErrorValue item={item} arrays={arrays} onChange={onChange} />
        ) : (
          <FileSlot
            project={project}
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
        <FileSlot project={project} role="mask" kind={kind} item={item} arrays={arrays} onFile={onChange && ((file) => setFile('mask', file))} />
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
// file), or the ways to give one
function FileSlot({
  project,
  role,
  kind,
  item,
  arrays,
  onFile,
  onValue,
  extra,
}: {
  project: string;
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
  if (chosen !== undefined) {
    const name = chosen.file.split('/').pop();
    return (
      <>
        <div className="ds-slot">
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
    entries.push({ section: `In ${dataFile.split('/').pop()}` });
    for (const array of others) {
      entries.push({ label: arrayLabel(array), onClick: () => onFile({ file: dataFile, array: arrayRef(arrays.inFile.data ?? [], array) }) });
    }
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
      {role === 'data' ? (
        <div className="ds-slot ds-slot-empty">
          <Button kind="small" onClick={source.fromWorkspace} title="A file of the server's workspace; the project keeps a copy">
            From the workspace
          </Button>
          <Button kind="small" onClick={source.fromComputer} title="A file of this computer, uploaded into the project">
            From this computer
          </Button>
        </div>
      ) : (
        <DropMenu label={`Add the ${role === 'error' ? 'errors' : 'mask'}`} width={250} entries={entries} target={<button className="ds-add-file">+ Add</button>} />
      )}
      {source.status}
      {source.hidden}
    </>
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

function ErrorKindChoice({ item, arrays, onChange }: { item: Item; arrays: Arrays; onChange?: (kind: ErrorKind) => void }) {
  const kind = errorKindOf(item, arrays);
  const known = ERROR_KINDS.find((k) => k.kind === kind)!;
  if (!onChange) return <span className="ds-array">{known.text}</span>;
  return (
    <div className="ds-error-kind">
      <span>They are</span>
      <Choice
        items={ERROR_KINDS.map(({ kind: value, title, text }) => ({ value, label: `${text} (${title})` }))}
        value={kind}
        label="What the errors are"
        onChange={(value) => onChange(value as ErrorKind)}
      />
    </div>
  );
}
