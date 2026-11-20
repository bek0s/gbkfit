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
  fitsFootprint,
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
  const chosen = item.files[role];
  const found = arrays.found[role];
  const name = ROLE_NAMES[role];
  if (chosen !== undefined) {
    return (
      <>
        <ChosenFile
          name={name}
          chosen={chosen}
          inFile={arrays.inFile[role] ?? []}
          found={found}
          why={(array) => unusable(role, array, arrays.found.data, KINDS[kind].ndim)}
          facts={found ? <HeaderOf role={role} array={found} data={arrays.found.data} header={header} /> : null}
          onFile={onFile}
        />
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

// A file given: its name, its array (chosen among those of its file; why
// one cannot be, if it cannot), and a way to remove it; a chevron opens
// what GBKFIT reads from it (facts), if anything
function ChosenFile({
  name,
  chosen,
  inFile,
  found,
  why,
  facts,
  onFile,
}: {
  name: string;
  chosen: FileArray;
  inFile: ArrayInfo[];
  found: ArrayInfo | undefined;
  why: (array: ArrayInfo) => string | null;
  facts?: ReactNode;
  onFile?: (file: FileArray | undefined) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <div className="ds-slot">
        {facts !== undefined && (
          <button type="button" className="ds-chevron" aria-expanded={open} disabled={!facts} title={open ? 'Hide' : 'What GBKFIT reads from the header'} onClick={() => setOpen(!open)}>
            ▸
          </button>
        )}
        <span className="ds-file" title={chosen.file}>
          {chosen.file.split('/').pop()}
        </span>
        {onFile && inFile.length > 1 ? (
          <Choice
            items={inFile.map((array) => {
              const reason = why(array);
              return {
                value: String(array.index),
                label: arrayLabel(array),
                description: `${kept(array).join(' × ') || 'one value'}${reason ? ` · ${reason}` : ''}`,
                // (the one chosen stays, to show what is wrong with it)
                disabled: reason !== null && array.index !== found?.index,
              };
            })}
            value={String(found?.index ?? '')}
            label={`The array of the ${name}`}
            onChange={(index) => {
              const array = inFile.find((a) => a.index === Number(index));
              if (array) onFile({ file: chosen.file, array: arrayRef(inFile, array) });
            }}
          />
        ) : (
          found && inFile.length > 1 && <span className="ds-array">{arrayLabel(found)}</span>
        )}
        {onFile && <Close label={`Remove the ${name}`} onClick={() => onFile(undefined)} />}
      </div>
      {open && facts}
    </>
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
      entries: others.map((array) => ({ label: arrayLabel(array), onClick: () => onFile({ file: dataFile, array: arrayRef(dataArrays, array) }) })),
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
// DatasetGrid): its file, or the ways to give one (an array of the file of
// the data offered if of their shape, or of the sky of a cube)
function FootprintSlot({ project, resolved, onFile }: { project: string; resolved: Resolved; onFile?: (file: FileArray | undefined) => void }) {
  const { kind, draft, arrays, footprint } = resolved;
  const first = presentSlots(kind, draft)[0];
  const of = first ? (arrays[first.key] ?? NO_ARRAYS) : NO_ARRAYS;
  const data = of.found.data;
  if (draft.footprint !== undefined) {
    return (
      <ChosenFile
        name="footprint"
        chosen={draft.footprint}
        inFile={footprint.inFile}
        found={footprint.found}
        why={(array) => (data !== undefined && !fitsFootprint(array, data) ? 'another shape' : null)}
        onFile={onFile}
      />
    );
  }
  if (!onFile) return <span className="ds-none">—</span>;
  const others = data === undefined ? [] : (of.inFile.data ?? []).filter((array) => array.index !== data.index && fitsFootprint(array, data));
  return (
    <AddFile
      project={project}
      name="footprint"
      dataFile={first ? draft.items[first.key].files.data?.file : undefined}
      dataArrays={of.inFile.data ?? []}
      others={others}
      more={[]}
      onFile={onFile}
    />
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
