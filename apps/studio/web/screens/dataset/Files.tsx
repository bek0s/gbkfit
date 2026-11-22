// The files of a dataset, in the dock: for each item (the cube, the flux
// map, or each moment map), its data, its errors (a file, or one value)
// and its mask, a row each, and the footprint of the dataset. A row says
// what is read, a line a fact (see components/FileEditor): where it comes
// from, its array (of a file of several), and how it is read (the unit
// of the data, what the errors are, what the mask marks); its parts with
// problems are marked, and the field of the problems of the row says why.
// Editing, each fact is set where it is said: where it comes from by its
// menu (a file of the workspace or of this computer, each copied into the
// project; of a companion of the data, an array of the file of the data;
// of the errors, one value), its array by its menu, how it is read by a
// small control in the line.
import type { ReactNode } from 'react';
import type { ArrayInfo } from '../../api/client';
import { arrayLabel, arrayRef } from '../../components/arrays';
import { ArrayButton, arrayEntries, FileLine, FilePair, FileSummary, SourceButton, sourceEntries } from '../../components/FileEditor';
import { useFileSource, type FileSource } from '../../components/FileSources';
import { ProblemField, useProblemsAt } from '../../components/Problems';
import { significant } from '../../components/format';
import { Close } from '../../components/ui/basics';
import { Segments } from '../../components/ui/inputs';
import { DropMenu, type MenuEntry } from '../../components/ui/overlays';
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

// The ways to give a file (see Row)
type Sources = Pick<FileSource, 'fromWorkspace' | 'fromComputer'>;

// The data of an item, as its companions see them: their file, its
// arrays, and their own
interface OfData {
  file: string | undefined;
  arrays: ArrayInfo[];
  array: ArrayInfo | undefined;
}

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
  onChange,
}: {
  itemKey: string;
  project: string;
  item: Item;
  arrays: Arrays;
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
  const ofData: OfData = { file: item.files.data?.file, arrays: arrays.inFile.data ?? [], array: data };
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
        project={project}
        onFile={onChange && ((file) => setFile('data', { file }))}
        onRemove={onChange && item.files.data ? () => setFile('data', undefined) : undefined}
      >
        {(sources) => (
          <DataLines
            path={[itemKey, 'data']}
            unitPath={[itemKey, 'unit']}
            chosen={item.files.data}
            inFile={arrays.inFile.data}
            found={data}
            unit={item.unit}
            headerUnit={data?.unit ?? undefined}
            unitNeeded={unitProblem || item.unit !== ''}
            sources={sources}
            onArray={(array) => setFile('data', { file: item.files.data!.file, array: arrayRef(arrays.inFile.data!, array) })}
            onUnit={(unit) => onChange?.({ ...item, unit })}
          />
        )}
      </Row>
      <Row
        title="Errors"
        path={[itemKey, 'error']}
        catches={[itemKey, 'error_kind']}
        project={project}
        onFile={onChange && ((file) => setFile('error', { file }))}
        onRemove={onChange && (item.files.error || item.errorValue !== null) ? () => setFile('error', undefined) : undefined}
      >
        {(sources) => (
          <CompanionLines
            name="errors"
            path={[itemKey, 'error']}
            readingPath={[itemKey, 'error_kind']}
            chosen={item.files.error}
            value={item.errorValue}
            ofData={ofData}
            inFile={arrays.inFile.error}
            found={arrays.found.error}
            sources={sources}
            onFile={(file) => setFile('error', file)}
            onValue={setErrorValue}
            rms={data?.rms}
            unit={data?.unit}
            fact={knownKind?.title}
            asked="σ, σ² or 1/σ²?"
            reading={
              <Segments
                small
                items={ERROR_KINDS.map(({ kind, title, text }) => ({ value: kind, label: title, title: text }))}
                value={errorKind}
                label="What the errors are"
                onChange={(kind) => onChange?.({ ...item, errorKind: kind })}
              />
            }
          />
        )}
      </Row>
      <Row
        title="Mask"
        path={[itemKey, 'mask']}
        project={project}
        onFile={onChange && ((file) => setFile('mask', { file }))}
        onRemove={onChange && item.files.mask ? () => setFile('mask', undefined) : undefined}
      >
        {(sources) => (
          <CompanionLines
            name="mask"
            path={[itemKey, 'mask']}
            readingPath={[itemKey, 'mask_kind']}
            chosen={item.files.mask}
            ofData={ofData}
            inFile={arrays.inFile.mask}
            found={arrays.found.mask}
            sources={sources}
            onFile={(file) => setFile('mask', file)}
            fact={maskKind?.fact}
            readingLabel="≠ 0"
            reading={
              <Segments
                small
                items={MASK_KINDS.map(({ kind, text }) => ({ value: kind, label: text.split(' ')[0].toLowerCase(), title: text }))}
                value={item.maskKind}
                label="What the mask marks where it is not 0"
                onChange={(kind) => onChange?.({ ...item, maskKind: kind })}
              />
            }
          />
        )}
      </Row>
    </div>
  );
}

// The footprint of a dataset (one for all its items: see the library's
// DatasetGrid), a companion of the data of the first item
function FootprintRow({ project, resolved, onFile }: { project: string; resolved: Resolved; onFile?: (file: FileArray | undefined) => void }) {
  const { kind, draft, arrays, footprint } = resolved;
  const first = presentSlots(kind, draft)[0];
  const of = first ? (arrays[first.key] ?? NO_ARRAYS) : NO_ARRAYS;
  const ofData: OfData = { file: first ? draft.items[first.key].files.data?.file : undefined, arrays: of.inFile.data ?? [], array: of.found.data };
  return (
    <Row
      title="Footprint"
      path={['footprint']}
      project={project}
      onFile={onFile && ((file) => onFile({ file }))}
      onRemove={onFile && draft.footprint ? () => onFile(undefined) : undefined}
    >
      {(sources) => (
        <CompanionLines
          name="footprint"
          path={['footprint']}
          chosen={draft.footprint}
          ofData={ofData}
          inFile={draft.footprint && footprint.inFile}
          found={footprint.found}
          sources={sources}
          onFile={(file) => onFile?.(file)}
        />
      )}
    </Row>
  );
}

// A row of the files: its label, the lines of what it reads, and, if
// something is given, editing, the way to remove it; the field of the
// problems of its path (see components/Problems). A file chosen of the
// workspace or of this computer is given to it (onFile; none: read only),
// and the progress of its copy is said under it.
function Row({
  title,
  path,
  catches,
  project,
  onFile,
  onRemove,
  children,
}: {
  title: string;
  path: Path;
  catches?: Path;
  project: string;
  onFile?: (path: string) => void;
  onRemove?: () => void;
  // Its lines, given the ways to give a file (none: read only)
  children: (sources: Sources | null) => ReactNode;
}) {
  const source = useFileSource(project, (file) => onFile?.(file));
  return (
    <ProblemField path={path} catches={catches} label={title} className="ds-row">
      <div className="ds-row-line">
        <span className="ds-row-label">{title}</span>
        <div className="file-lines">{children(onFile ? source : null)}</div>
        {onRemove && <Close label={`Remove the ${title.toLowerCase()}`} onClick={onRemove} />}
      </div>
      {source.status}
      {source.hidden}
    </ProblemField>
  );
}

// The data of an item: their file (from the workspace or this computer),
// its array of several, and their unit while it is needed (the library
// finds a problem of that of their header, or one is given); read only,
// as text
function DataLines({
  path,
  unitPath,
  chosen,
  inFile,
  found,
  unit,
  headerUnit,
  unitNeeded,
  sources,
  onArray,
  onUnit,
}: {
  path: Path;
  unitPath: Path;
  chosen: FileArray | undefined;
  inFile: ArrayInfo[] | undefined;
  found: ArrayInfo | undefined;
  unit: string;
  headerUnit: string | undefined;
  unitNeeded: boolean;
  sources: Sources | null;
  onArray: (array: ArrayInfo) => void;
  onUnit: (unit: string) => void;
}) {
  const ofFile = useProblemsAt(path, true);
  const ofUnit = useProblemsAt(unitPath);
  if (sources === null) {
    return chosen ? (
      <RowSummary path={path} chosen={chosen} inFile={inFile} found={found} fact={unit || undefined} factPath={unitPath} headerFact={headerUnit} asked="unit?" />
    ) : (
      <span className="file-none">—</span>
    );
  }
  const several = inFile !== undefined && inFile.length > 1;
  return (
    <>
      <FileLine>
        <SourceButton file={chosen?.file} text="+ Add" label="Where the data come from" entries={sourceEntries(sources)} marked={several ? null : ofFile} />
      </FileLine>
      {chosen && several && (
        <FileLine>
          <ArrayButton arrays={inFile} current={found} label="The array of the data" marked={ofFile} onChoose={onArray} />
        </FileLine>
      )}
      {chosen && unitNeeded && (
        <FileLine>
          <FilePair label="unit" wide marked={ofUnit}>
            <input className="file-field" value={unit} placeholder="FITS unit" aria-label="The unit of the values" onChange={(event) => onUnit(event.target.value)} />
          </FilePair>
        </FileLine>
      )}
    </>
  );
}

// A companion of the data (their errors, their mask, or the footprint):
// where it comes from (a file of the workspace or of this computer, an
// array of the file of the data, or, of errors, one value), its array (of
// a file of several), and how it is read (e.g. what the errors are); read
// only, as text. What is given, and how it is read, are marked by the
// problems of the row at their paths; one value is typed in its line.
function CompanionLines({
  name,
  path,
  readingPath,
  chosen,
  value,
  ofData,
  inFile,
  found,
  sources,
  onFile,
  onValue,
  rms,
  unit,
  fact,
  asked,
  readingLabel,
  reading,
}: {
  // What it is to the data (e.g. 'errors'), in the labels of its menus
  name: string;
  path: Path;
  readingPath?: Path;
  chosen: FileArray | undefined;
  // One value given, of errors (null: none; undefined: none can be)
  value?: string | null;
  ofData: OfData;
  inFile: ArrayInfo[] | undefined;
  found: ArrayInfo | undefined;
  sources: Sources | null;
  onFile: (file: FileArray) => void;
  onValue?: (value: string) => void;
  // What the header of the data says of their noise, and their unit (of
  // one value of errors)
  rms?: number | null;
  unit?: string | null;
  // How it is read, as text, and as asked while not said and needed
  fact?: string;
  asked?: string;
  // The control of how it is read, and its label
  readingLabel?: string;
  reading?: ReactNode;
}) {
  const ofFile = useProblemsAt(path, true);
  const ofReading = useProblemsAt(readingPath ?? null);
  if (sources === null) {
    if (value != null) return <ValueSummary path={path} value={value} unit={unit} />;
    return chosen ? (
      <RowSummary path={path} chosen={chosen} inFile={inFile} found={found} fact={fact} factPath={readingPath} asked={asked} />
    ) : (
      <span className="file-none">—</span>
    );
  }
  const taken = ofData.array && { array: ofData.array, by: 'the data' };
  const fromData = ofData.file !== undefined && ofData.arrays.length > 1;
  // (the ways of few first, the arrays of the file of the data, of many,
  // last)
  const entries: MenuEntry[] = [
    ...sourceEntries(sources),
    ...(onValue
      ? [
          {
            label: 'One value…',
            description: rms != null ? `The header says ${significant(rms, 3)}` : 'e.g. the rms of a radio cube',
            current: value != null,
            onClick: () => onValue(value ?? (rms != null ? String(rms) : '')),
          },
        ]
      : []),
    ...(fromData
      ? [
          { section: 'Arrays of the file of the data' },
          ...arrayEntries(ofData.arrays, chosen?.file === ofData.file ? found : undefined, (array) => onFile({ file: ofData.file!, array: arrayRef(ofData.arrays, array) }), taken),
        ]
      : []),
  ];
  const label = `Where the ${name} come${name === 'errors' ? '' : 's'} from`;
  if (value != null && onValue) {
    return (
      <>
        <FileLine>
          <SourceButton text="One value" label={label} entries={entries} />
        </FileLine>
        <FileLine>
          <FilePair marked={ofFile}>
            <input
              className="file-field file-value"
              type="number"
              step="any"
              min="0"
              value={value}
              aria-label={`The ${name} of every value`}
              onChange={(event) => onValue(event.target.value)}
            />
            {unit && <span className="file-text">{unit}</span>}
          </FilePair>
        </FileLine>
      </>
    );
  }
  if (!chosen) {
    return (
      <FileLine>
        <SourceButton text="+ Add" label={label} entries={entries} />
      </FileLine>
    );
  }
  const several = inFile !== undefined && inFile.length > 1;
  return (
    <>
      <FileLine>
        <SourceButton file={chosen.file} label={label} entries={entries} marked={several ? null : ofFile} />
      </FileLine>
      {(several || reading) && (
        <FileLine>
          {several && (
            <ArrayButton
              arrays={inFile}
              current={found}
              label={`The array of the ${name}`}
              taken={chosen.file === ofData.file ? taken : undefined}
              marked={ofFile}
              onChoose={(array) => onFile({ file: chosen.file, array: arrayRef(inFile, array) })}
            />
          )}
          {reading && (
            <FilePair label={readingLabel} marked={ofReading}>
              {reading}
            </FilePair>
          )}
        </FileLine>
      )}
    </>
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

// One error for every value: the value (marked by the problems of the
// row), and under it its unit
function ValueSummary({ path, value, unit }: { path: Path; value: string; unit: string | null | undefined }) {
  const severity = useProblemsAt(path, true);
  return (
    <span className="file-sum" data-name={severity ?? undefined}>
      <span className="file-sum-name">{value || '—'}</span>
      {unit && (
        <span className="file-sum-more">
          <span className="file-sum-fact">{unit}</span>
        </span>
      )}
    </span>
  );
}
