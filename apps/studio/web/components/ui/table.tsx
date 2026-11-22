// The parts of a table of blocks (e.g. the datasets of a project): its
// heading (what it lists, how many, and the way to add one: at its top,
// however long it is), the heads of its columns, which sort it, and its
// rows, which the keyboard reaches. Its cells are never wrapped: a long
// text is cut (see Cut), a table too wide for its place scrolls.
import { useState, type FocusEvent, type KeyboardEvent, type ReactNode } from 'react';

export function TableHeading({ title, count, onAdd, addLabel }: { title: string; count: number; onAdd?: () => void; addLabel: string }) {
  return (
    <div className="ui-table-heading">
      <h2>{title}</h2>
      <span className="ui-table-count">{count}</span>
      {onAdd && (
        <button type="button" className="ui-table-add" aria-label={addLabel} title={addLabel} onClick={onAdd}>
          + Add
        </button>
      )}
    </div>
  );
}

// The order of a table: by the values of one column, up (smallest first)
// or down
export interface Sort {
  column: string;
  down: boolean;
}

// The value of a row at a column that it sorts by (null: none, or not
// known yet)
export type SortValue = string | number | null;

// The head of a column, which sorts the table by it (up, or down first:
// e.g. the latest first; a click again: the other way), and says by what
// (e.g. 'the area of the beam')
export function SortHead({
  column,
  sort,
  onSort,
  by,
  downFirst = false,
  className,
  children,
}: {
  column: string;
  sort: Sort;
  onSort: (sort: Sort) => void;
  by: string;
  downFirst?: boolean;
  className?: string;
  children: ReactNode;
}) {
  const active = sort.column === column;
  return (
    <th className={className} aria-sort={active ? (sort.down ? 'descending' : 'ascending') : 'none'}>
      <button type="button" className="ui-sort" data-active={active || undefined} title={`Sort by ${by}`} onClick={() => onSort({ column, down: active ? !sort.down : downFirst })}>
        {children}
        <span className="ui-sort-sign" aria-hidden>
          {active ? (sort.down ? '▼' : '▲') : ''}
        </span>
      </button>
    </th>
  );
}

// The keys of rows in the order of their values at the column sorted by;
// those of no value last, either way, and ties by key
export function sortedKeys(keys: readonly string[], values: Readonly<Record<string, Readonly<Record<string, SortValue>>>>, { column, down }: Sort): string[] {
  const valueOf = (key: string) => values[key]?.[column] ?? null;
  return [...keys].sort((a, b) => {
    const [x, y] = [valueOf(a), valueOf(b)];
    if (x === null || y === null) return x === y ? byText(a, b) : x === null ? 1 : -1;
    const order = typeof x === 'number' && typeof y === 'number' ? x - y : byText(String(x), String(y));
    return (down ? -order : order) || byText(a, b);
  });
}

// The rows of a table as the keyboard reaches them: one stop of Tab, the
// row it was last at (else the first), and the arrows from row to row (see
// rowKeys). Of the key of each row, what makes it that stop.
export interface RowFocus {
  tabIndex: number;
  onFocus: (event: FocusEvent<HTMLTableRowElement>) => void;
}

export function useRowFocus(keys: readonly string[]): (key: string) => RowFocus {
  const [at, setAt] = useState<string | null>(null);
  const current = at !== null && keys.includes(at) ? at : keys[0];
  return (key) => ({
    tabIndex: key === current ? 0 : -1,
    onFocus: (event) => {
      if (event.target === event.currentTarget) setAt(key);
    },
  });
}

// What makes a row that a click opens open by the keyboard too: Enter
// opens it, and the arrows go to the row above or below, from it or from
// what it holds (but keys typed in a field of it, e.g. a name, and those
// of what it opens over the page, e.g. a menu, are theirs)
export function rowKeys(open: () => void) {
  return {
    onKeyDown: (event: KeyboardEvent<HTMLTableRowElement>) => {
      const row = event.currentTarget;
      const target = event.target as HTMLElement;
      if (!row.contains(target) || target.closest('input, textarea, select')) return;
      if (event.key === 'Enter' && target === row) open();
      const to = { ArrowDown: row.nextElementSibling, ArrowUp: row.previousElementSibling }[event.key];
      if (to instanceof HTMLElement) {
        event.preventDefault();
        to.focus();
      }
    },
  };
}

const byText = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
