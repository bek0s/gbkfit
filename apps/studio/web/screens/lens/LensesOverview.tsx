// The lenses of the project at a glance, the screen of the section before
// any lens is opened: one row each (see BlockRow), with its pieces (their
// kinds; their names on hover), its redshift and that of the source its
// quoted strengths are of, how many parameters it has, and when its file
// last changed; a sign before its name says the worst of its problems,
// all of them on hover. What the library says of it (its parameters and
// problems) is of the lens as saved. A click on the head of a column
// sorts by it, as the page remembers.
import { useMemo } from 'react';
import { useLensDescription, usePiece, useProject, useSchema } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import { when } from '../../components/format';
import type { Severity } from '../../components/Problems';
import { titleOf, type Defs } from '../../components/SchemaForm';
import { sortedKeys, TableHeading, useRowFocus, type RowFocus, type Sort } from '../../components/ui/table';
import { BlockRow, ColumnHeads, useReportRow, useRowValues, type Column, type RowValues } from '../../shell/BlockTable';
import { useRemembered } from '../../shell/remember';
import { ScreenHead } from '../../shell/ScreenHead';
import { sectionNamed } from '../../shell/sections';
import { useStudio } from '../../studio';
import { asList, isObject } from '../galaxy/options';

const COLUMNS: Column[] = [
  { key: 'name', title: 'Lens', by: 'name' },
  { key: 'pieces', title: 'Pieces', by: 'pieces' },
  { key: 'redshift', title: 'Redshift', by: 'redshift', numeric: true },
  { key: 'source', title: 'Source redshift', by: 'the redshift of the source', numeric: true },
  { key: 'parameters', title: 'Parameters', by: 'the number of parameters', numeric: true },
  { key: 'modified', title: 'Modified', by: 'when its file last changed', downFirst: true },
];

export function LensesOverview() {
  const { project, startAdding } = useStudio();
  const info = useProject(project).data;
  const names = info?.pieces.lenses ?? [];
  const defs = useSchema().data?.$defs as Defs | undefined;
  const [sort, setSort] = useRemembered<Sort>('lenses:sort', { column: 'name', down: false });
  // (the values of each row, as it reads them)
  const [values, report] = useRowValues();
  const sorted = sortedKeys(names, values, sort);
  const focusOf = useRowFocus(sorted);
  if (project === null) return null;
  return (
    <section className="screen ov-screen">
      <ScreenHead title={sectionNamed('lenses').overview?.title ?? 'Lenses'} />
      <div className="ov-scroll">
        <TableHeading title="Lenses" count={names.length} addLabel="Add a lens" onAdd={() => startAdding('lenses')} />
        {names.length === 0 ? (
          <p className="ov-empty">The project has no lens yet: the mass in front of a galaxy that bends its light, made of pieces such as an SIE and a shear.</p>
        ) : (
          <table className="ov-table">
            <ColumnHeads columns={COLUMNS} sort={sort} onSort={setSort} />
            <tbody>
              {sorted.map((name) => (
                <LensRow key={name} project={project} name={name} defs={defs} modified={info?.modified.lenses?.[name] ?? null} focus={focusOf(name)} onValues={report} />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

function LensRow({
  project,
  name,
  defs,
  modified,
  focus,
  onValues,
}: {
  project: string;
  name: string;
  defs: Defs | undefined;
  // When its file last changed (ISO 8601; null: not known)
  modified: string | null;
  // How the keyboard reaches it (see useRowFocus)
  focus: RowFocus;
  onValues: (name: string, row: RowValues | null) => void;
}) {
  const piece = usePiece(joinPath(project, `lenses/${name}.yaml`)).data;
  const options = isObject(piece) ? piece : undefined;
  const description = useLensDescription(project, options);
  const described = description.isSuccess && !description.isPlaceholderData;
  const params = described ? description.data.parameters : null;
  const problems = described ? (description.data.problems as unknown as { severity: Severity; message: string }[]) : [];
  const reading = options === undefined;
  const dash = <span className="ov-faint">{reading ? '…' : '—'}</span>;
  // (of what the library says: while it is to come; none if it failed)
  const later = <span className="ov-faint">{description.isError ? '—' : '…'}</span>;

  const pieces = options ? asList(options.pieces) : [];
  const kinds = pieces.map((p) => titleOf(defs, 'lens_piece', String(p.type)));
  const redshift = typeof options?.redshift === 'number' ? options.redshift : null;
  const source = typeof options?.source_redshift === 'number' ? options.source_redshift : null;
  // (what the row sorts by)
  const time = modified === null ? null : Date.parse(modified);
  const kindsText = kinds.join(' · ') || null;
  const row = useMemo<RowValues>(
    () => ({ name, pieces: kindsText, redshift, source, parameters: params?.length ?? null, modified: time }),
    [name, kindsText, redshift, source, params?.length, time],
  );
  useReportRow(name, row, onValues);
  return (
    <BlockRow project={project} folder="lenses" name={name} focus={focus} problems={problems}>
      <td title={pieces.map((p) => String(p.name ?? p.type)).join('\n') || undefined}>{kindsText ?? dash}</td>
      <td className="ov-num">{redshift ?? dash}</td>
      <td className="ov-num">{source ?? dash}</td>
      <td className="ov-num">{params?.length ?? later}</td>
      <td className="ov-time">{modified === null ? dash : when(modified)}</td>
    </BlockRow>
  );
}
