// The files of a dataset, in the dock: for each item (the cube, the flux
// map, or each moment map), its data, its errors (a file, or one value)
// and its mask, a row each. A file is given from the workspace or from
// this computer (each copied into the project), or is another array of
// the file of the data. Read only, the rows say what the dataset reads.
import { useState, type ReactNode } from 'react';
import type { ArrayInfo } from '../../api/client';
import { useFileSource } from '../../components/FileSources';
import { ProblemField } from '../../components/Problems';
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
  kept,
  KINDS,
  MASK_KINDS,
  NO_ARRAYS,
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
  type MaskKind,
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
            itemKey={slot.key}
            project={project}
            header={{
              rest: kind === 'cube' ? restOf(draft.spectral) : null,
              type: kind === 'cube' ? draft.spectral.type : null,
            }}
            kind={kind}
            item={draft.items[slot.key]}
            arrays={arrays[slot.key] ?? NO_ARRAYS}
            onChange={onItem && ((item) => onItem(slot.key, item))}
          />
        </div>
      ))}
      {onAdd && absent.length > 0 && (
        <ProblemField path={['files']} label="Maps">
          <DropMenu
            label="Add a map"
            entries={absent.map((slot) => ({ label: slotTitle(slot, kind), onClick: () => onAdd(slot.key) }))}
            target={<button className="ds-add">+ Add a map</button>}
          />
        </ProblemField>
      )}
    </div>
  );
}

// What the facts of the header of the data are read with: the rest and the
// type of the spectral axis as given
interface HeaderContext {
  rest: string | null;
  type: SpectralType | null;
}

function ItemRows({
  itemKey,
  project,
  header,
  kind,
  item,
  arrays,
  onChange,
}: {
  itemKey: string;
  project: string;
  header: HeaderContext;
  kind: Kind;
  item: Item;
  arrays: Arrays;
  onChange?: (item: Item) => void;
}) {
  // (a file removed is no longer among the files, rather than one of none;
  // what was said of a file goes with it)
  const setFile = (role: Role, file: FileArray | undefined) => {
    const { [role]: _gone, ...others } = item.files;
    const reset = role === 'data' ? { unit: '' } : role === 'error' ? { errorKind: null } : { maskKind: 'valid' as const };
    onChange?.({ ...item, files: file ? { ...others, [role]: file } : others, ...reset });
  };
  // (the unit of the values: a field once the library finds a problem of
  // it, or one is given)
  const unitProblem = arrays.found.data?.problems.some((problem) => problem.path[0] === 'unit') ?? false;
  return (
    <div className="ds-rows">
      <Row title="Data" path={[itemKey, 'data']} catches={[itemKey]}>
        <FileSlot project={project} header={header} role="data" kind={kind} item={item} arrays={arrays} onFile={onChange && ((file) => setFile('data', file))} />
      </Row>
      {(unitProblem || item.unit !== '') && (
        <Row title="Unit" path={[itemKey, 'unit']}>
          {onChange ? (
            <input className="ds-value ds-unit-input" value={item.unit} placeholder="FITS unit" aria-label="The unit of the values" onChange={(event) => onChange({ ...item, unit: event.target.value })} />
          ) : (
            <span className="ds-file">{item.unit}</span>
          )}
        </Row>
      )}
      <Row title="Errors" path={[itemKey, 'error']}>
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
      <Row title="Mask" path={[itemKey, 'mask']}>
        <FileSlot
          project={project}
          header={header}
          role="mask"
          kind={kind}
          item={item}
          arrays={arrays}
          onFile={onChange && ((file) => setFile('mask', file))}
          extra={item.files.mask && <MaskKindChoice kind={item.maskKind} onChange={onChange && ((maskKind) => onChange({ ...item, maskKind }))} />}
        />
      </Row>
    </div>
  );
}

// A row of an item: the field of the problems of its path (see
// components/Problems)
function Row({ title, path, catches, children }: { title: string; path: (string | number)[]; catches?: (string | number)[]; children: ReactNode }) {
  return (
    <ProblemField path={path} catches={catches} label={title} className="ds-row">
      <span className="ds-row-label">{title}</span>
      <div className="ds-row-body">{children}</div>
    </ProblemField>
  );
}

// The file of a role: its name and its array (chosen among those of its
// file), or the ways to give one; a chevron opens what GBKFIT reads from
// the header of its array (one header for each array, an HDU: the data
// and the errors of one file have each their own; the header as it is in
// the file is a view of the viewer)
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
  // (whether what GBKFIT reads of the header shows: as the user chose)
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
            title={open ? 'Hide' : 'What GBKFIT reads from the header'}
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
        {open && found && <HeaderOf role={role} array={found} data={arrays.found.data} header={header} />}
        {extra}
      </>
    );
  }
  if (!onFile) return <span className="ds-none">{role === 'data' ? 'None' : '—'}</span>;
  const others = role === 'data' ? [] : siblings(arrays);
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

// What GBKFIT reads from the header of an array. Of the data, all that
// GBKFIT reads (see HeaderFacts): the dataset is on their grid. Of their
// errors and mask, which are on that grid whatever their headers say
// (GBKFIT reads the coordinates of the data alone), whether their shape is
// that of the data, and what the errors are, if their header says.
function HeaderOf({ role, array, data, header }: { role: Role; array: ArrayInfo; data: ArrayInfo | undefined; header: HeaderContext }) {
  const holds = ERROR_KINDS.find((k) => k.kind === array.holds);
  return (
    <div className="ds-header-of">
      {role === 'data' ? (
        <HeaderFacts array={array} rest={header.rest} type={header.type} />
      ) : (
        <dl className="ds-facts">
          <dt>Shape</dt>
          <dd>
            {kept(array).join(' × ')}
            {data && !sameShape(array, data) && ` · data ${kept(data).join(' × ')}`}
          </dd>
          <dt>Coordinates</dt>
          <dd>of the data</dd>
          {role === 'error' && (
            <>
              <dt>Kind</dt>
              <dd>{holds ? `${holds.title} (header)` : '—'}</dd>
            </>
          )}
        </dl>
      )}
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
        placeholder="Choose"
        label="What the errors are"
        onChange={(value) => onChange(value as ErrorKind)}
      />
    </div>
  );
}

// What the mask marks where it is not 0 (see the library's mask_kind)
function MaskKindChoice({ kind, onChange }: { kind: MaskKind; onChange?: (kind: MaskKind) => void }) {
  const known = MASK_KINDS.find((k) => k.kind === kind);
  if (!onChange) return <span className="ds-array">{known?.text}</span>;
  return (
    <div className="ds-error-kind">
      <span>≠ 0 marks</span>
      <Choice items={MASK_KINDS.map(({ kind: value, text }) => ({ value, label: text }))} value={kind} label="What the mask marks" onChange={(value) => onChange(value as MaskKind)} />
    </div>
  );
}
