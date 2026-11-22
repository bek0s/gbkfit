// The files of a dataset, in the dock: for each item (the cube, the flux
// map, or each moment map), its data, its errors (a file, or one value)
// and its mask, a row each, and the footprint of the dataset. A row says
// what is read (see components/FileEditor), its parts with problems
// marked, and what it needs asked in its place (e.g. 'unit?'); the field
// of the problems of the row says why. Editing, a click on a row opens
// its popover, the same whether something is given or not: for the data,
// their file, its array and their unit; for a companion of the data (its
// errors, its mask, the footprint), where it comes from (an array of the
// file of the data, another file, or, of errors, one value), what of it
// is read, and how (what the errors are, what the mask marks). A file is
// given from the workspace or from this computer, each copied into the
// project.
import { useState, type ReactNode } from 'react';
import type { ArrayInfo } from '../../api/client';
import { arrayLabel, arrayRef } from '../../components/arrays';
import { ArraySection, FilePopover, FileSection, FileSummary, PopSection } from '../../components/FileEditor';
import { useFileSource, type FileSource } from '../../components/FileSources';
import { ProblemField, ProblemPart, useProblemsAt } from '../../components/Problems';
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
  // what was said of a file goes with it, and with its array; a file of
  // errors replaces their one value)
  const setFile = (role: Role, file: FileArray | undefined) => {
    const { [role]: _gone, ...others } = item.files;
    const reset = role === 'data' ? { unit: '' } : role === 'error' ? { errorKind: null, errorValue: null } : { maskKind: 'valid' as const };
    onChange?.({ ...item, files: file ? { ...others, [role]: file } : others, ...reset });
  };
  // (one value of errors replaces their file)
  const setErrorValue = (value: string) => {
    const { error: _gone, ...others } = item.files;
    onChange?.({ ...item, files: others, errorKind: null, errorValue: value });
  };
  const data = arrays.found.data;
  const ofData = { file: item.files.data?.file, arrays: arrays.inFile.data ?? [], array: data };
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
        summary={
          item.files.data && (
            <RowSummary
              path={[itemKey, 'data']}
              chosen={item.files.data}
              inFile={arrays.inFile.data}
              found={data}
              fact={item.unit || undefined}
              factPath={[itemKey, 'unit']}
              headerFact={data?.unit ?? undefined}
              asked="unit?"
            />
          )
        }
        onFile={onChange && ((file) => setFile('data', { file }))}
        onRemove={onChange && (() => setFile('data', undefined))}
        editor={
          onChange &&
          ((sources) => (
            <>
              <ProblemPart path={[itemKey, 'data']} exact>
                <FileSection path={item.files.data?.file} source={sources} />
                {item.files.data && arrays.inFile.data && (
                  <ArraySection arrays={arrays.inFile.data} current={data} onChoose={(array) => setFile('data', { file: item.files.data!.file, array: arrayRef(arrays.inFile.data!, array) })} />
                )}
              </ProblemPart>
              {item.files.data && (unitProblem || item.unit !== '') && (
                <PopSection label="Unit">
                  <ProblemPart path={[itemKey, 'unit']}>
                    <input
                      className="file-pop-field"
                      value={item.unit}
                      placeholder="FITS unit"
                      aria-label="The unit of the values"
                      onChange={(event) => onChange({ ...item, unit: event.target.value })}
                    />
                  </ProblemPart>
                </PopSection>
              )}
            </>
          ))
        }
      />
      <Row
        title="Errors"
        path={[itemKey, 'error']}
        catches={[itemKey, 'error_kind']}
        context={context}
        project={project}
        summary={
          item.errorValue !== null ? (
            <ValueSummary path={[itemKey, 'error']} value={item.errorValue} unit={data?.unit} />
          ) : (
            item.files.error && (
              <RowSummary
                path={[itemKey, 'error']}
                chosen={item.files.error}
                inFile={arrays.inFile.error}
                found={arrays.found.error}
                fact={knownKind?.title}
                factPath={[itemKey, 'error_kind']}
                asked="σ, σ² or 1/σ²?"
              />
            )
          )
        }
        onFile={onChange && ((file) => setFile('error', { file }))}
        onRemove={onChange && (() => setFile('error', undefined))}
        editor={
          onChange &&
          ((sources) => (
            <CompanionEditor
              path={[itemKey, 'error']}
              readingPath={[itemKey, 'error_kind']}
              chosen={item.files.error}
              value={item.errorValue}
              data={ofData}
              rms={data?.rms}
              unit={data?.unit}
              inFile={arrays.inFile.error}
              found={arrays.found.error}
              sources={sources}
              onFile={(file) => setFile('error', file)}
              onValue={setErrorValue}
              reading={
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
              }
            />
          ))
        }
      />
      <Row
        title="Mask"
        path={[itemKey, 'mask']}
        context={context}
        project={project}
        summary={
          item.files.mask && (
            <RowSummary path={[itemKey, 'mask']} chosen={item.files.mask} inFile={arrays.inFile.mask} found={arrays.found.mask} fact={maskKind?.fact} factPath={[itemKey, 'mask_kind']} />
          )
        }
        onFile={onChange && ((file) => setFile('mask', { file }))}
        onRemove={onChange && (() => setFile('mask', undefined))}
        editor={
          onChange &&
          ((sources) => (
            <CompanionEditor
              path={[itemKey, 'mask']}
              readingPath={[itemKey, 'mask_kind']}
              chosen={item.files.mask}
              data={ofData}
              inFile={arrays.inFile.mask}
              found={arrays.found.mask}
              sources={sources}
              onFile={(file) => setFile('mask', file)}
              reading={
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
              }
            />
          ))
        }
      />
    </div>
  );
}

// The footprint of a dataset (one for all its items: see the library's
// DatasetGrid), a companion of the data of the first item
function FootprintRow({ project, resolved, onFile }: { project: string; resolved: Resolved; onFile?: (file: FileArray | undefined) => void }) {
  const { kind, draft, arrays, footprint } = resolved;
  const first = presentSlots(kind, draft)[0];
  const of = first ? (arrays[first.key] ?? NO_ARRAYS) : NO_ARRAYS;
  const data = { file: first ? draft.items[first.key].files.data?.file : undefined, arrays: of.inFile.data ?? [], array: of.found.data };
  return (
    <Row
      title="Footprint"
      path={['footprint']}
      project={project}
      summary={draft.footprint && <RowSummary path={['footprint']} chosen={draft.footprint} inFile={footprint.inFile} found={footprint.found} />}
      onFile={onFile && ((file) => onFile({ file }))}
      onRemove={onFile && (() => onFile(undefined))}
      editor={
        onFile &&
        ((sources) => (
          <CompanionEditor
            path={['footprint']}
            chosen={draft.footprint}
            data={data}
            inFile={draft.footprint && footprint.inFile}
            found={footprint.found}
            sources={sources}
            onFile={onFile}
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
  // What it reads (none: nothing given)
  summary: ReactNode;
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
                {summary || <NothingRead path={path} editing />}
              </button>
            ) : (
              <div className="ds-row-main">
                {label}
                {summary || <NothingRead path={path} />}
              </div>
            )}
            {editor && summary && onRemove && <Close label={`Remove the ${title.toLowerCase()}`} onClick={onRemove} />}
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

// Where a companion of the data comes from: an array of the file of the
// data, another file, or (of errors) one value
type Source = 'data' | 'file' | 'value';

// The popover of a companion of the data (its errors, its mask, the
// footprint): where it comes from, that given marked (none, if nothing
// is), and, of the one shown, what of it is read: an array of the file of
// the data (the data's own taken), another file and its array, or one
// value; then, of an array given, how it is read (e.g. what the errors
// are). Showing where it can come from changes nothing; choosing an
// array, a file, or typing a value does.
function CompanionEditor({
  path,
  readingPath,
  chosen,
  value,
  data,
  rms,
  unit,
  inFile,
  found,
  sources,
  onFile,
  onValue,
  reading,
}: {
  // The path of the row, whose problems mark what is given, and that of
  // how it is read, whose problems mark that (see ProblemPart)
  path: Path;
  readingPath?: Path;
  // The file given (none: none, or a value)
  chosen: FileArray | undefined;
  // The value given, of errors (null: none)
  value?: string | null;
  // The data: their file, its arrays, and their own
  data: { file: string | undefined; arrays: ArrayInfo[]; array: ArrayInfo | undefined };
  // What the header of the data says of its noise, and its unit (for one
  // value of errors)
  rms?: number | null;
  unit?: string | null;
  // The arrays of the file given, and the one read
  inFile: ArrayInfo[] | undefined;
  found: ArrayInfo | undefined;
  sources: Sources;
  onFile: (file: FileArray) => void;
  // How one value is given (none: it cannot be)
  onValue?: (value: string) => void;
  // How an array given is read
  reading?: ReactNode;
}) {
  const given: Source | null = value != null ? 'value' : chosen === undefined ? null : chosen.file === data.file ? 'data' : 'file';
  const [shown, setShown] = useState<Source | null>(given);
  const others = data.file !== undefined && data.arrays.length > 1;
  const items: { value: Source; label: string; disabled?: boolean; title?: string }[] = [
    {
      value: 'data',
      label: 'File of the data',
      disabled: !others,
      title: data.file === undefined ? 'No data yet' : !others ? 'The file of the data has no other array' : undefined,
    },
    { value: 'file', label: 'Another file' },
    ...(onValue ? [{ value: 'value' as const, label: 'One value' }] : []),
  ];
  return (
    <>
      <PopSection label="From">
        <Segments items={items} value={shown} onChange={setShown} label="Where it comes from" />
      </PopSection>
      {/* (what is given marked by the problems of the row: shown, not
          what else can be) */}
      <ProblemPart path={shown === given ? path : null} exact>
        {shown === 'data' && data.file !== undefined && (
          <ArraySection
            file={data.file}
            arrays={data.arrays}
            current={given === 'data' ? found : undefined}
            taken={data.array && { array: data.array, by: 'the data' }}
            onChoose={(array) => onFile({ file: data.file!, array: arrayRef(data.arrays, array) })}
          />
        )}
        {shown === 'file' && (
          <>
            <FileSection path={given === 'file' ? chosen!.file : undefined} source={sources} />
            {given === 'file' && inFile && <ArraySection arrays={inFile} current={found} onChoose={(array) => onFile({ file: chosen!.file, array: arrayRef(inFile, array) })} />}
          </>
        )}
        {shown === 'value' && onValue && <ValueSection value={given === 'value' ? (value ?? '') : ''} rms={rms} unit={unit} onValue={onValue} />}
      </ProblemPart>
      {shown !== null && shown === given && given !== 'value' && readingPath && <ProblemPart path={readingPath}>{reading}</ProblemPart>}
    </>
  );
}

// One error for every value of the data: typed, or that the header of the
// data says, if it says one
function ValueSection({ value, rms, unit, onValue }: { value: string; rms?: number | null; unit?: string | null; onValue: (value: string) => void }) {
  return (
    <PopSection
      label="Value"
      aside={
        rms != null && (
          <button type="button" className="file-pop-link" title="The noise the header of the data says" onClick={() => onValue(String(rms))}>
            header: {significant(rms, 3)}
          </button>
        )
      }
    >
      <div className="file-pop-value">
        <input
          className="file-pop-field"
          type="number"
          step="any"
          min="0"
          autoFocus
          value={value}
          aria-label="The error of every value"
          onChange={(event) => onValue(event.target.value)}
        />
        <span>{unit ? `${unit} · ` : ''}every value</span>
      </div>
    </PopSection>
  );
}

// What a row of a file reads: its file, its array, if it is one of
// several, and how it is read; the file (its array, if shown) marked by
// the problems of the row's own path (e.g. of the wrong shape), how it is
// read by those of its option (factPath, e.g. [key, 'unit']). While that
// option has a problem and nothing is said of it, what the header says is
// shown in its place (e.g. a unit that is not one), or else it is asked
// for (e.g. 'unit?')
function RowSummary({
  path,
  chosen,
  inFile,
  found,
  fact,
  factPath,
  headerFact,
  asked,
}: {
  path: Path;
  chosen: FileArray;
  inFile: ArrayInfo[] | undefined;
  found: ArrayInfo | undefined;
  fact?: string;
  factPath?: Path;
  headerFact?: string;
  asked?: string;
}) {
  const ofFile = useProblemsAt(path, true);
  const ofReading = useProblemsAt(factPath ?? null);
  const array = inFile !== undefined && inFile.length > 1 && found ? arrayLabel(found) : undefined;
  return (
    <FileSummary
      path={chosen.file}
      array={array}
      fact={fact ?? (ofReading ? (headerFact ?? asked) : undefined)}
      marks={{ name: array === undefined ? ofFile : null, array: ofFile, fact: ofReading }}
    />
  );
}

// Nothing read: a dash, or, editing, the way to give it; marked by the
// problems of the row (e.g. the data missing)
function NothingRead({ path, editing }: { path: Path; editing?: boolean }) {
  const severity = useProblemsAt(path, true);
  return editing ? (
    <span className="file-add" data-severity={severity ?? undefined}>
      + Add
    </span>
  ) : (
    <span className="file-none">—</span>
  );
}

// One error for every value: the value (marked by the problems of the
// row), and under it its unit
function ValueSummary({ path, value, unit }: { path: Path; value: string; unit: string | null | undefined }) {
  const severity = useProblemsAt(path, true);
  return (
    <span className="file-sum" data-name={severity ?? undefined}>
      <span className="file-sum-name">{value || '—'}</span>
      <span className="file-sum-more">
        <span className="file-sum-fact">{unit ? `${unit} · ` : ''}every value</span>
      </span>
    </span>
  );
}
