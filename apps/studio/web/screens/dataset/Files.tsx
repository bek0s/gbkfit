// The files of a dataset, in the dock: for each item (the cube, the flux
// map, or each moment map), its data, its errors (a file, or one value)
// and its mask, a row each, and the footprint of the dataset. A row says
// what is read (see components/FileEditor); editing, a click on it opens
// the popover that sets it: its file (from the workspace or from this
// computer, each copied into the project, or another array of the file of
// the data), the array read, and how it is read (the unit of the data,
// what the errors are, what the mask marks).
import { useState, type ReactNode } from 'react';
import type { ArrayInfo } from '../../api/client';
import { arrayLabel, arrayRef } from '../../components/arrays';
import { ArraySection, FilePopover, FileSection, FileSummary, PopOption, PopSection, SourceSection } from '../../components/FileEditor';
import { useFileSource, type FileSource } from '../../components/FileSources';
import { ProblemField } from '../../components/Problems';
import { significant } from '../../components/format';
import { Close } from '../../components/ui/basics';
import { Segments } from '../../components/ui/inputs';
import { DropMenu } from '../../components/ui/overlays';
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
  type FileArray,
  type Item,
  type Resolved,
  type Role,
} from './model';

type Path = (string | number)[];

// The ways to give a file, as a popover offers them (see Row)
type Sources = Pick<FileSource, 'fromWorkspace' | 'fromComputer'>;

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
      {slots.map((slot) => {
        const title = slotTitle(slot, kind).replace(/^Moment \d · /, '');
        return (
          <div key={slot.key} className="ds-item" data-group={kind === 'moments' || undefined}>
            {kind === 'moments' && (
              <div className="ds-item-head">
                <span className="ds-item-short">{slot.short}</span>
                <span className="ds-item-title">{title}</span>
                {onRemove && <Close label={`Remove the map of ${slot.title.toLowerCase()}`} onClick={() => onRemove(slot.key)} />}
              </div>
            )}
            <ItemRows
              itemKey={slot.key}
              project={project}
              item={draft.items[slot.key]}
              arrays={arrays[slot.key] ?? NO_ARRAYS}
              context={kind === 'moments' ? title : undefined}
              onChange={onItem && ((item) => onItem(slot.key, item))}
            />
          </div>
        );
      })}
      <div className="ds-rows">
        <FootprintRow project={project} resolved={resolved} onFile={onFootprint} />
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
  context,
  onChange,
}: {
  itemKey: string;
  project: string;
  item: Item;
  arrays: Arrays;
  // The map the item is (of moment maps), named in the popovers
  context?: string;
  onChange?: (item: Item) => void;
}) {
  // (a file removed is no longer among the files, rather than one of none;
  // what was said of a file goes with it, and with its array)
  const setFile = (role: Role, file: FileArray | undefined) => {
    const { [role]: _gone, ...others } = item.files;
    const reset = role === 'data' ? { unit: '' } : role === 'error' ? { errorKind: null } : { maskKind: 'valid' as const };
    onChange?.({ ...item, files: file ? { ...others, [role]: file } : others, ...reset });
  };
  // (another array of the file of the data, for the errors or the mask)
  const dataFile = item.files.data?.file;
  const others = siblings(arrays);
  const ofData = (array: ArrayInfo): FileArray => ({ file: dataFile!, array: arrayRef(arrays.inFile.data ?? [], array) });
  // (an array of the file of a role)
  const ofFile = (role: Role, array: ArrayInfo): FileArray => ({ file: item.files[role]!.file, array: arrayRef(arrays.inFile[role] ?? [], array) });
  const data = arrays.found.data;
  // (the unit of the values: a field once the library finds a problem of
  // it, or one is given)
  const unitProblem = data?.problems.some((problem) => problem.path[0] === 'unit') ?? false;
  const errorKind = errorKindOf(item, arrays);
  const knownKind = ERROR_KINDS.find((k) => k.kind === errorKind);
  const maskKind = MASK_KINDS.find((k) => k.kind === item.maskKind);
  return (
    <div className="ds-rows">
      <Row
        title="Data"
        path={[itemKey, 'data']}
        catches={[itemKey]}
        context={context}
        project={project}
        summary={summaryOf(item.files.data, arrays.inFile.data, data, item.unit || undefined)}
        onFile={onChange && ((file) => setFile('data', { file }))}
        onRemove={onChange && (() => setFile('data', undefined))}
        editor={
          onChange &&
          ((sources) =>
            item.files.data ? (
              <>
                <FileSection path={item.files.data.file} source={sources} />
                <ArraySection arrays={arrays.inFile.data ?? []} current={data} onChoose={(array) => setFile('data', ofFile('data', array))} />
                {(unitProblem || item.unit !== '') && (
                  <PopSection label="Unit">
                    <ProblemField path={[itemKey, 'unit']} label="Unit">
                      <input
                        className="file-pop-field"
                        value={item.unit}
                        placeholder="FITS unit"
                        aria-label="The unit of the values"
                        onChange={(event) => onChange({ ...item, unit: event.target.value })}
                      />
                    </ProblemField>
                  </PopSection>
                )}
              </>
            ) : (
              <SourceSection source={sources} />
            ))
        }
      />
      <Row
        title="Errors"
        path={[itemKey, 'error']}
        context={context}
        project={project}
        summary={item.errorValue !== null ? <ValueSummary value={item.errorValue} unit={data?.unit} /> : summaryOf(item.files.error, arrays.inFile.error, arrays.found.error, knownKind?.title)}
        onFile={onChange && ((file) => setFile('error', { file }))}
        onRemove={onChange && (() => (item.errorValue !== null ? onChange({ ...item, errorValue: null }) : setFile('error', undefined)))}
        editor={
          onChange &&
          ((sources) =>
            item.errorValue !== null ? (
              <PopSection label="One value">
                <div className="file-pop-value">
                  <input
                    className="file-pop-field"
                    type="number"
                    step="any"
                    min="0"
                    autoFocus
                    value={item.errorValue}
                    aria-label="The error of every value"
                    onChange={(event) => onChange({ ...item, errorValue: event.target.value })}
                  />
                  <span>{data?.unit ? `${data.unit} · ` : ''}every value</span>
                </div>
              </PopSection>
            ) : item.files.error ? (
              <>
                <FileSection path={item.files.error.file} source={sources} />
                <ArraySection arrays={arrays.inFile.error ?? []} current={arrays.found.error} onChoose={(array) => setFile('error', ofFile('error', array))} />
                <PopSection label="Given as">
                  <div className="file-pop-choice">
                    <Segments
                      items={ERROR_KINDS.map(({ kind, title, text }) => ({ value: kind, label: title, title: text }))}
                      value={errorKind}
                      label="What the errors are"
                      onChange={(kind) => onChange({ ...item, errorKind: kind })}
                    />
                    {knownKind && (
                      <small>
                        {knownKind.text}
                        {item.errorKind === null && ', as the header says'}
                      </small>
                    )}
                  </div>
                </PopSection>
              </>
            ) : (
              <AddSections
                dataFile={dataFile}
                others={others}
                onArray={(array) => setFile('error', ofData(array))}
                value={{
                  label: 'One value for all the data',
                  description: data?.rms != null ? `The header says ${significant(data.rms, 3)}` : 'e.g. the rms of a radio cube',
                  onClick: () => onChange({ ...item, errorValue: data?.rms != null ? String(data.rms) : '' }),
                }}
                sources={sources}
              />
            ))
        }
      />
      <Row
        title="Mask"
        path={[itemKey, 'mask']}
        context={context}
        project={project}
        summary={summaryOf(item.files.mask, arrays.inFile.mask, arrays.found.mask, maskKind?.fact)}
        onFile={onChange && ((file) => setFile('mask', { file }))}
        onRemove={onChange && (() => setFile('mask', undefined))}
        editor={
          onChange &&
          ((sources) =>
            item.files.mask ? (
              <>
                <FileSection path={item.files.mask.file} source={sources} />
                <ArraySection arrays={arrays.inFile.mask ?? []} current={arrays.found.mask} onChoose={(array) => setFile('mask', ofFile('mask', array))} />
                <PopSection label="≠ 0 marks">
                  <div className="file-pop-choice">
                    <Segments
                      items={MASK_KINDS.map(({ kind, text }) => ({ value: kind, label: text }))}
                      value={item.maskKind}
                      label="What the mask marks where it is not 0"
                      onChange={(kind) => onChange({ ...item, maskKind: kind })}
                    />
                  </div>
                </PopSection>
              </>
            ) : (
              <AddSections dataFile={dataFile} others={others} onArray={(array) => setFile('mask', ofData(array))} sources={sources} />
            ))
        }
      />
    </div>
  );
}

// The footprint of a dataset (one for all its items: see the library's
// DatasetGrid): its file, or the ways to give one (or another array of the
// file of the data of the first item)
function FootprintRow({ project, resolved, onFile }: { project: string; resolved: Resolved; onFile?: (file: FileArray | undefined) => void }) {
  const { kind, draft, arrays, footprint } = resolved;
  const first = presentSlots(kind, draft)[0];
  const of = first ? (arrays[first.key] ?? NO_ARRAYS) : NO_ARRAYS;
  const dataFile = first ? draft.items[first.key].files.data?.file : undefined;
  return (
    <Row
      title="Footprint"
      path={['footprint']}
      project={project}
      summary={summaryOf(draft.footprint, footprint.inFile, footprint.found)}
      onFile={onFile && ((file) => onFile({ file }))}
      onRemove={onFile && (() => onFile(undefined))}
      editor={
        onFile &&
        ((sources) =>
          draft.footprint ? (
            <>
              <FileSection path={draft.footprint.file} source={sources} />
              <ArraySection
                arrays={footprint.inFile}
                current={footprint.found}
                onChoose={(array) => onFile({ file: draft.footprint!.file, array: arrayRef(footprint.inFile, array) })}
              />
            </>
          ) : (
            <AddSections
              dataFile={dataFile}
              others={siblings(of)}
              onArray={(array) => onFile({ file: dataFile!, array: arrayRef(of.inFile.data ?? [], array) })}
              sources={sources}
            />
          ))
      }
    />
  );
}

// A row of the files: its label and what it reads (none: nothing given);
// the field of the problems of its path (see components/Problems).
// Editing, a click on it opens its popover, beside it, and what it reads
// can be removed; a file given from the workspace or from this computer
// is chosen with the popover closed (the dialog that picks it takes its
// place), and the progress of its copy is said under the row.
function Row({
  title,
  path,
  catches,
  context,
  project,
  summary,
  editor,
  onFile,
  onRemove,
}: {
  title: string;
  path: Path;
  catches?: Path;
  context?: string;
  project: string;
  summary: ReactNode | null;
  // What its popover holds, given the ways to give a file (none: read
  // only)
  editor?: (sources: Sources) => ReactNode;
  onFile?: (path: string) => void;
  onRemove?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const source = useFileSource(project, (file) => onFile?.(file));
  const sources: Sources = {
    fromWorkspace: () => {
      setOpen(false);
      source.fromWorkspace();
    },
    fromComputer: () => {
      setOpen(false);
      source.fromComputer();
    },
  };
  const label = <span className="ds-row-label">{title}</span>;
  return (
    <ProblemField path={path} catches={catches} label={title} className="ds-row">
      <FilePopover
        opened={open}
        onChange={setOpen}
        title={title}
        context={context}
        target={
          <div className="ds-row-line" data-editable={editor ? true : undefined} data-open={open || undefined}>
            {editor ? (
              <button type="button" className="ds-row-main" aria-haspopup="dialog" aria-expanded={open} onClick={() => setOpen(!open)}>
                {label}
                {summary ?? <span className="file-add">+ Add</span>}
              </button>
            ) : (
              <div className="ds-row-main">
                {label}
                {summary ?? <span className="file-none">—</span>}
              </div>
            )}
            {editor && summary !== null && onRemove && <Close label={`Remove the ${title.toLowerCase()}`} onClick={onRemove} />}
          </div>
        }
      >
        {editor?.(sources)}
      </FilePopover>
      {source.status}
      {source.hidden}
    </ProblemField>
  );
}

// What a row of a file reads (null: no file): the file, its array, if it
// is one of several, and how it is read
function summaryOf(chosen: FileArray | undefined, inFile: ArrayInfo[] | undefined, found: ArrayInfo | undefined, fact?: string): ReactNode {
  if (chosen === undefined) return null;
  const several = inFile !== undefined && inFile.length > 1;
  return <FileSummary path={chosen.file} array={several && found ? arrayLabel(found) : undefined} fact={fact} />;
}

// One error for every value: the value, and under it its unit
function ValueSummary({ value, unit }: { value: string; unit: string | null | undefined }) {
  return (
    <span className="file-sum">
      <span className="file-sum-name">{value || '—'}</span>
      <span className="file-sum-more">
        <span className="file-sum-fact">{unit ? `${unit} · ` : ''}every value</span>
      </span>
    </span>
  );
}

// The ways to give a row a file: another array of the file of the data,
// one value (of errors), or a file of the workspace or of this computer
function AddSections({
  dataFile,
  others,
  onArray,
  value,
  sources,
}: {
  dataFile: string | undefined;
  others: ArrayInfo[];
  onArray: (array: ArrayInfo) => void;
  value?: { label: string; description: string; onClick: () => void };
  sources: Sources;
}) {
  const fromData = dataFile !== undefined && others.length > 0;
  return (
    <>
      {fromData && <ArraySection label="In the file of the data" aside={`${others.length} more`} file={dataFile} arrays={others} onChoose={onArray} />}
      {value && (
        <PopSection label="One value">
          <div className="file-pop-list">
            <PopOption label={value.label} description={value.description} onClick={value.onClick} />
          </div>
        </PopSection>
      )}
      <SourceSection label={fromData || value ? 'Another file' : 'A file'} source={sources} />
    </>
  );
}
