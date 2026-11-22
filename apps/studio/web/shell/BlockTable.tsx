// The parts of a table of the blocks of a folder, the overview of their
// section (e.g. the datasets of a project): the heads of its columns,
// each sorting it by its values; its rows, each a block (see BlockRow);
// and the values each row reports, by column, that the table sorts by
// (see useRowValues): a row reads its block itself, so the table knows
// them only as each row tells it.
import { Tooltip } from '@mantine/core';
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { MoreIcon } from '../components/icons';
import { ProblemSign, type Severity } from '../components/Problems';
import { Cut } from '../components/ui/cut';
import { DropMenu, PointMenu } from '../components/ui/overlays';
import { rowKeys, SortHead, type RowFocus, type Sort, type SortValue } from '../components/ui/table';
import { useStudio } from '../studio';
import { NameInput, useBlockActions } from './BlockMenu';
import { kindOf } from './sections';

// A column: the key of the value of a row that sorts by it, its title,
// what it sorts by (e.g. 'the area of the beam'), whether its values are
// numbers (set to the right), and whether it sorts down first (e.g. the
// latest first)
export interface Column {
  key: string;
  title: string;
  by: string;
  numeric?: boolean;
  downFirst?: boolean;
}

// The values of a row, by the key of each column
export type RowValues = Record<string, SortValue>;

// The longest name shown whole (characters)
const NAME_CAP = 28;

// The heads of the columns of a table, each sorting it by its column, and
// that of the menus of its rows, empty
export function ColumnHeads({ columns, sort, onSort }: { columns: readonly Column[]; sort: Sort; onSort: (sort: Sort) => void }) {
  return (
    <thead>
      <tr>
        {columns.map(({ key, title, by, numeric, downFirst }) => (
          <SortHead key={key} column={key} sort={sort} onSort={onSort} by={by} downFirst={downFirst} className={numeric ? 'ov-num' : undefined}>
            {title}
          </SortHead>
        ))}
        <th aria-label="Actions" />
      </tr>
    </thead>
  );
}

// The values of the rows of a table, by the name of each, and how a row
// reports its own (null: a row gone)
export function useRowValues(): [Record<string, RowValues>, (name: string, row: RowValues | null) => void] {
  const [values, setValues] = useState<Record<string, RowValues>>({});
  const report = useCallback(
    (name: string, row: RowValues | null) =>
      setValues((all) => {
        if (row !== null) return sameRow(all[name], row) ? all : { ...all, [name]: row };
        const { [name]: _gone, ...others } = all;
        return others;
      }),
    [],
  );
  return [values, report];
}

// A row telling its table its values (before it is drawn: it is in its
// place from the first), and that it is gone as it goes
export function useReportRow(name: string, row: RowValues, report: (name: string, row: RowValues | null) => void) {
  useLayoutEffect(() => report(name, row), [name, row, report]);
  useEffect(() => () => report(name, null), [name, report]);
}

const sameRow = (a: RowValues | undefined, b: RowValues) => a !== undefined && Object.keys(b).every((key) => a[key] === b[key]);

// A row of a block: a click (or Enter) opens it; its menu, at its end, of
// a right click, or of the menu key, renames or deletes it. Its name comes
// first, after a sign of the worst of its problems (all of them on hover),
// cut if long (all of it on hover); then its cells.
export function BlockRow({
  project,
  folder,
  name,
  focus,
  problems,
  children,
}: {
  project: string;
  folder: string;
  name: string;
  // How the keyboard reaches it (see useRowFocus)
  focus: RowFocus;
  problems: readonly { severity: Severity; message: string }[];
  children: ReactNode;
}) {
  const { go } = useStudio();
  const actions = useBlockActions(project, folder, name);
  const label = `The ${kindOf(folder).toLowerCase()} ${name}`;
  // (the menu of a right click, where it was in the box of the menu)
  const [pointed, setPointed] = useState<[number, number] | null>(null);
  const menuBox = useRef<HTMLDivElement>(null);
  const rowRef = useRef<HTMLTableRowElement>(null);
  const worst = problems.find((problem) => problem.severity === 'error')?.severity ?? problems[0]?.severity;
  const open = () => go({ kind: 'block', folder, name });
  return (
    <tr
      ref={rowRef}
      className="ov-row"
      data-renaming={actions.renaming || undefined}
      onClick={actions.renaming ? undefined : open}
      {...focus}
      {...rowKeys(open)}
      onContextMenu={(event) => {
        const box = menuBox.current?.getBoundingClientRect();
        const button = menuBox.current?.querySelector('.ov-more-button')?.getBoundingClientRect();
        if (!box || !button) return;
        event.preventDefault();
        // (of the right button, where it was pressed; of the keyboard, e.g.
        // the menu key, under the button of the menu)
        setPointed(event.button === 2 ? [event.clientX - box.left, event.clientY - box.top] : [button.left - box.left, button.bottom - box.top]);
      }}
    >
      <td className="ov-name">
        <span className="ov-name-box">
          <span className="ov-sign">
            {worst && (
              <Tooltip label={problems.map((problem) => problem.message).join('\n')} multiline maw={360} position="top-start" openDelay={120} classNames={{ tooltip: 'problem-tip' }}>
                <span>
                  <ProblemSign severity={worst} />
                </span>
              </Tooltip>
            )}
          </span>
          <Cut max={NAME_CAP} title={name} className="ov-name-text" />
          {actions.renaming && <NameInput project={project} folder={folder} name={name} path={actions.path} onDone={() => actions.setRenaming(false)} />}
        </span>
      </td>
      {children}
      {/* (what is done here is not a click on the row) */}
      <td className="ov-more" onClick={(event) => event.stopPropagation()} onContextMenu={(event) => event.stopPropagation()}>
        {/* (the box the menu of a right click is placed in: the cell is
            not, so that the focus ring of the row is drawn over it) */}
        <div ref={menuBox} className="ov-more-box">
          <DropMenu
            label={label}
            at="end"
            returnFocus={false}
            entries={actions.entries}
            target={
              // (not a stop of Tab: the menu key of the row opens it)
              <button type="button" className="ov-more-button" tabIndex={-1} title={`Rename or delete the ${kindOf(folder).toLowerCase()}`} aria-label={label}>
                <MoreIcon />
              </button>
            }
          />
          {pointed && (
            <PointMenu
              at={pointed}
              entries={actions.entries}
              label={label}
              onClose={() => {
                setPointed(null);
                // (the focus back to the row, as it was before the menu)
                rowRef.current?.focus({ preventScroll: true });
              }}
            />
          )}
        </div>
        {actions.dialog}
      </td>
    </tr>
  );
}
