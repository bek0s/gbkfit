// The galaxies of the project at a glance, the screen of the section
// before any galaxy is opened: one row each (see BlockRow), with its
// components (their kinds, in their colours of its screen; their names on
// hover), the light they give (their emission, as its screen says it),
// its mass (its gravity and how many parts; the parts on hover), its
// distance and redshift, how many parameters it has, and when its file
// last changed; a sign before its name says the worst of its problems,
// all of them on hover. What the library says of it (its parameters and
// problems) is of the galaxy as saved. A click on the head of a column
// sorts by it, as the page remembers.
import { useMemo } from 'react';
import { useGalaxyParameters, useOptionalPiece, usePiece, useProject, useSchema } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import { when } from '../../components/format';
import type { Severity } from '../../components/Problems';
import type { Defs } from '../../components/SchemaForm';
import { sortedKeys, TableHeading, useRowFocus, type RowFocus, type Sort } from '../../components/ui/table';
import { BlockRow, ColumnHeads, useReportRow, useRowValues, type Column, type RowValues } from '../../shell/BlockTable';
import { useRemembered } from '../../shell/remember';
import { ScreenHead } from '../../shell/ScreenHead';
import { sectionNamed } from '../../shell/sections';
import { useStudio } from '../../studio';
import { COLOURS } from './GalaxyScreen';
import { valueText } from './Numbers';
import { asList, componentName, componentsOf, isObject, itemText, type Options } from './options';
import { unitText } from './ParamField';
import { settingOf, type Param, type Properties } from './params';
import { titled } from './schema';

const COLUMNS: Column[] = [
  { key: 'name', title: 'Galaxy', by: 'name' },
  { key: 'components', title: 'Components', by: 'components' },
  { key: 'light', title: 'Light', by: 'light' },
  { key: 'mass', title: 'Mass', by: 'the number of parts of the mass' },
  { key: 'distance', title: 'Distance', by: 'distance', numeric: true },
  { key: 'redshift', title: 'Redshift', by: 'redshift', numeric: true },
  { key: 'parameters', title: 'Parameters', by: 'the number of parameters', numeric: true },
  { key: 'modified', title: 'Modified', by: 'when its file last changed', downFirst: true },
];

export function GalaxiesOverview() {
  const { project, startAdding } = useStudio();
  const info = useProject(project).data;
  const names = info?.pieces.galaxies ?? [];
  const defs = useSchema().data?.$defs as Defs | undefined;
  const [sort, setSort] = useRemembered<Sort>('galaxies:sort', { column: 'name', down: false });
  // (the values of each row, as it reads them)
  const [values, report] = useRowValues();
  const sorted = sortedKeys(names, values, sort);
  const focusOf = useRowFocus(sorted);
  if (project === null) return null;
  return (
    <section className="screen ov-screen">
      <ScreenHead title={sectionNamed('galaxies').overview?.title ?? 'Galaxies'} />
      <div className="ov-scroll">
        <TableHeading title="Galaxies" count={names.length} addLabel="Add a galaxy" onAdd={() => startAdding('galaxies')} />
        {names.length === 0 ? (
          <p className="ov-empty">The project has no galaxy yet: the components that shine, how they move, and the mass that moves them.</p>
        ) : (
          <table className="ov-table">
            <ColumnHeads columns={COLUMNS} sort={sort} onSort={setSort} />
            <tbody>
              {sorted.map((name) => (
                <GalaxyRow
                  key={name}
                  project={project}
                  name={name}
                  hasParams={info?.pieces.params?.includes(name) ?? false}
                  defs={defs}
                  modified={info?.modified.galaxies?.[name] ?? null}
                  focus={focusOf(name)}
                  onValues={report}
                />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

function GalaxyRow({
  project,
  name,
  hasParams,
  defs,
  modified,
  focus,
  onValues,
}: {
  project: string;
  name: string;
  // Whether it has a params block (of its name), with the values of its
  // parameters
  hasParams: boolean;
  defs: Defs | undefined;
  // When its file last changed (ISO 8601; null: not known)
  modified: string | null;
  // How the keyboard reaches it (see useRowFocus)
  focus: RowFocus;
  onValues: (name: string, row: RowValues | null) => void;
}) {
  const piece = usePiece(joinPath(project, `galaxies/${name}.yaml`)).data;
  const options = isObject(piece) ? piece : undefined;
  const paramsBlock = useOptionalPiece(joinPath(project, `params/${name}.yaml`), hasParams).data;
  const properties = isObject(paramsBlock) && isObject(paramsBlock.properties) ? (paramsBlock.properties as Properties) : {};
  const description = useGalaxyParameters(project, options);
  const described = description.isSuccess && !description.isPlaceholderData;
  const params = described ? (description.data.parameters as unknown as Param[]) : null;
  const problems = described ? (description.data.problems as unknown as { severity: Severity; message: string }[]) : [];
  const reading = options === undefined;
  const dash = <span className="ov-faint">{reading ? '…' : '—'}</span>;
  // (of what the library says: while it is to come; none if it failed)
  const later = <span className="ov-faint">{description.isError ? '—' : '…'}</span>;

  const components = options ? componentsOf(options) : [];
  const kinds = components.map((component) => titled(defs, 'component', String(component.type)).label);
  const light = options ? lightOf(components) : null;
  const mass = options ? asList(options.mass) : [];
  const gravity = titled(defs, 'gravity', String((isObject(options?.gravity) && options.gravity.type) || 'newton')).label;
  const distance = options && params ? distanceOf(options, params, properties, defs) : null;
  const redshift = options ? (typeof options.redshift === 'number' ? options.redshift : 0) : null;
  // (what the row sorts by)
  const time = modified === null ? null : Date.parse(modified);
  const kindsText = kinds.join(', ') || null;
  const row = useMemo<RowValues>(
    () => ({ name, components: kindsText, light, mass: mass.length || null, distance: distance?.mpc ?? null, redshift, parameters: params?.length ?? null, modified: time }),
    [name, kindsText, light, mass.length, distance?.mpc, redshift, params?.length, time],
  );
  useReportRow(name, row, onValues);
  return (
    <BlockRow project={project} folder="galaxies" name={name} focus={focus} problems={problems}>
      <td>
        {components.length > 0 ? (
          <span className="ov-parts">
            {components.map((component, i) => (
              <span key={i} className="ov-part" title={componentName(component, i)}>
                <span className="ov-dot" style={{ background: `var(--${COLOURS[i % COLOURS.length]})` }} />
                {kinds[i]}
              </span>
            ))}
          </span>
        ) : (
          dash
        )}
      </td>
      <td>{light ?? dash}</td>
      <td>
        {mass.length > 0 ? (
          <span title={mass.map((part, i) => `${part.name ?? `part ${i + 1}`} · ${titled(defs, 'mass_component', String(part.type)).label}`).join('\n')}>
            {gravity}
            <span className="ov-faint"> · {mass.length}</span>
          </span>
        ) : (
          dash
        )}
      </td>
      <td className="ov-num" title={options && !options.distance ? 'none: angles only' : undefined}>
        {distance?.text ?? (options?.distance ? later : dash)}
      </td>
      <td className="ov-num">{redshift ?? dash}</td>
      <td className="ov-num">{params?.length ?? later}</td>
      <td className="ov-time">{modified === null ? dash : when(modified)}</td>
    </BlockRow>
  );
}

// The light of the components, as the screen of a galaxy says it (e.g.
// 'lines hi21', 'band'): each emission once (none: they give none)
function lightOf(components: Options[]): string | null {
  const texts = components.flatMap((component) => asList(component.emission).map(itemText));
  return [...new Set(texts)].join(' · ') || null;
}

// The distance of a galaxy, as the screen of a galaxy says it: the value
// of its parameter of distance (and it in Mpc, what the column sorts by),
// or the kind of its distance, if it has no such parameter (none: it has
// no distance, its lengths are angles)
function distanceOf(options: Options, params: Param[], properties: Properties, defs: Defs | undefined): { text: string; mpc: number | null } | null {
  if (!isObject(options.distance)) return null;
  const param = params.find((p) => p.owner.length === 1 && p.owner[0] === 'distance');
  if (!param) return { text: titled(defs, 'distance', String(options.distance.type)).label.toLowerCase(), mpc: null };
  const setting = settingOf(properties, param);
  const value = setting.kind === 'fixed' || setting.kind === 'free' ? setting.values[0] : undefined;
  const unit = unitText(param.unit);
  return { text: unit ? `${valueText(setting)} ${unit}` : valueText(setting), mpc: param.unit === 'Mpc' && typeof value === 'number' ? value : null };
}
