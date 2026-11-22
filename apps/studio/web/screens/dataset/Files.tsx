// The files of a dataset, in the dock: for each item (the cube, the flux
// map, or each moment map), its data, its errors (a file, or one value)
// and its mask, a row each. A file is given from the workspace or from
// this computer (each copied into the project), or is another array of
// the file of the data. Read only, the rows say what the dataset reads.
import type { ReactNode } from 'react';
import type { ArrayInfo } from '../../api/client';
import { ArrayPick } from '../../components/ArrayPick';
import { arrayLabel, arrayRef, kept } from '../../components/arrays';
import { useFileSource } from '../../components/FileSources';
import { ProblemField } from '../../components/Problems';
import { significant } from '../../components/format';
import { Close } from '../../components/ui/basics';
import { MiddleCut } from '../../components/ui/cut';
import { Choice } from '../../components/ui/inputs';
import { DropMenu, type MenuEntry } from '../../components/ui/overlays';
import {
  ERROR_KINDS,
  errorKindOf,
  KINDS,
  MASK_KINDS,
  NO_ARRAYS,
  presentSlots,
  siblings,
  slotTitle,
  type Arrays,
  type ErrorKind,
  type FileArray,
  type Item,
  type MaskKind,
  type Resolved,
  type Role,
} from './model';

export function Files({
  project,
  resolved,
  onItem,
  onAdd,
  onRemove,
  onFootprint,
}: {
  project: string;
  resolved: Resolved;
  // How the items change (none: read only)
  onItem?: (key: string, item: Item) => void;
  onAdd?: (key: string) => void;
  onRemove?: (key: string) => void;
  onFootprint?: (file: FileArray | undefined) => void;
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
            item={draft.items[slot.key]}
            arrays={arrays[slot.key] ?? NO_ARRAYS}
            onChange={onItem && ((item) => onItem(slot.key, item))}
          />
        </div>
      ))}
      <div className="ds-rows">
        <Row title="Footprint" path={['footprint']}>
          <FootprintSlot project={project} resolved={resolved} onFile={onFootprint} />
        </Row>
      </div>
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

function ItemRows({
  itemKey,
  project,
  item,
  arrays,
  onChange,
}: {
  itemKey: string;
  project: string;
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
        <FileSlot project={project} role="data" item={item} arrays={arrays} onFile={onChange && ((file) => setFile('data', file))} />
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
            role="error"
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
          role="mask"
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
// file), or the ways to give one
function FileSlot({
  project,
  role,
  item,
  arrays,
  onFile,
  onValue,
  extra,
}: {
  project: string;
  role: Role;
  item: Item;
  arrays: Arrays;
  onFile?: (file: FileArray | undefined) => void;
  onValue?: (value: string) => void;
  extra?: ReactNode;
}) {
  const chosen = item.files[role];
  const name = ROLE_NAMES[role];
  if (chosen !== undefined) {
    return (
      <>
        <ChosenFile name={name} chosen={chosen} inFile={arrays.inFile[role] ?? []} found={arrays.found[role]} onFile={onFile} />
        {extra}
      </>
    );
  }
  if (!onFile) return <span className="ds-none">{role === 'data' ? 'None' : '—'}</span>;
  const rms = arrays.found.data?.rms;
  const more: MenuEntry[] = onValue
    ? [
        { section: 'Or' },
        {
          label: 'One value for all the data',
          description: rms != null ? `The header says ${significant(rms, 3)}` : 'e.g. the rms of a radio cube',
          onClick: () => onValue(rms != null ? String(rms) : ''),
        },
      ]
    : [];
  return (
    <AddFile
      project={project}
      name={name}
      dataFile={item.files.data?.file}
      dataArrays={arrays.inFile.data ?? []}
      others={role === 'data' ? [] : siblings(arrays)}
      more={more}
      onFile={onFile}
    />
  );
}

// How the page names the file of each role
const ROLE_NAMES: Record<Role, string> = { data: 'data', error: 'errors', mask: 'mask' };

// A file given: its name, the whole width of its box, and a way to remove
// it; of a file of several arrays, the one read under it (all of the
// file can be chosen: the library says if one cannot be read there)
function ChosenFile({
  name,
  chosen,
  inFile,
  found,
  onFile,
}: {
  name: string;
  chosen: FileArray;
  inFile: ArrayInfo[];
  found: ArrayInfo | undefined;
  onFile?: (file: FileArray | undefined) => void;
}) {
  return (
    <div className="ds-slot" data-arrays={inFile.length > 1 || undefined}>
      <MiddleCut text={chosen.file.split('/').pop()!} title={chosen.file} className="ds-file" />
      {onFile && <Close label={`Remove the ${name}`} onClick={() => onFile(undefined)} />}
      {inFile.length > 1 && (
        <div className="ds-slot-array">
          {onFile ? (
            <ArrayPick arrays={inFile} chosen={found} label={`The array of the ${name}`} onChoose={(array) => onFile({ file: chosen.file, array: arrayRef(inFile, array) })} />
          ) : (
            found && <span className="ds-array">{arrayLabel(found)}</span>
          )}
        </div>
      )}
    </div>
  );
}

// The ways to give a file: of the workspace, of this computer (each copied
// into the project), another array of the file of the data (others), and
// any more (e.g. one value instead)
function AddFile({
  project,
  name,
  dataFile,
  dataArrays,
  others,
  more,
  onFile,
}: {
  project: string;
  name: string;
  dataFile: string | undefined;
  dataArrays: ArrayInfo[];
  others: ArrayInfo[];
  more: MenuEntry[];
  onFile: (file: FileArray) => void;
}) {
  const source = useFileSource(project, (file) => onFile({ file }));
  const entries: MenuEntry[] = [
    { label: 'A file of the workspace…', description: 'The project keeps a copy', onClick: source.fromWorkspace },
    { label: 'A file of this computer…', description: 'Uploaded into the project', onClick: source.fromComputer },
  ];
  if (others.length > 0 && dataFile !== undefined) {
    entries.push({
      label: `An array of ${dataFile.split('/').pop()}`,
      entries: others.map((array) => ({
        label: arrayLabel(array),
        description: kept(array).join(' × ') || 'one value',
        onClick: () => onFile({ file: dataFile, array: arrayRef(dataArrays, array) }),
      })),
    });
  }
  return (
    <>
      <DropMenu label={`Add the ${name}`} entries={[...entries, ...more]} target={<button className="ds-add-file">+ Add</button>} />
      {source.status}
      {source.hidden}
    </>
  );
}

// The footprint of a dataset (one for all its items: see the library's
// DatasetGrid): its file, or the ways to give one (or another array of the
// file of the data of the first item)
function FootprintSlot({ project, resolved, onFile }: { project: string; resolved: Resolved; onFile?: (file: FileArray | undefined) => void }) {
  const { kind, draft, arrays, footprint } = resolved;
  const first = presentSlots(kind, draft)[0];
  const of = first ? (arrays[first.key] ?? NO_ARRAYS) : NO_ARRAYS;
  if (draft.footprint !== undefined) {
    return <ChosenFile name="footprint" chosen={draft.footprint} inFile={footprint.inFile} found={footprint.found} onFile={onFile} />;
  }
  if (!onFile) return <span className="ds-none">—</span>;
  return (
    <AddFile
      project={project}
      name="footprint"
      dataFile={first ? draft.items[first.key].files.data?.file : undefined}
      dataArrays={of.inFile.data ?? []}
      others={siblings(of)}
      more={[]}
      onFile={onFile}
    />
  );
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
      <span>Given as</span>
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
