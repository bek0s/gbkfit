// The fits of the project at a glance: each fit (a setup), what it fits
// (its galaxies and observations) and with what (its fitter), its latest
// run (its state and when) and the χ² reduced of its latest result; what
// runs one, or stops the run of one that runs. A fit opens from its row.
// The fits of a folder (e.g. the models of one galaxy) are listed under
// it.
import { useMemo } from 'react';
import { useFileText, usePiece, useProject, useRun, useStopJob, useSubmitJob } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import { when } from '../../components/format';
import { Button } from '../../components/ui/basics';
import { Cut, rowKeys, TableHeading, useRowFocus, type RowFocus } from '../../components/ui/table';
import { ScreenHead } from '../../shell/ScreenHead';
import { sectionNamed } from '../../shell/sections';
import { useStudio } from '../../studio';
import { isObject } from '../galaxy/options';
import { itemsOf } from './recipe';
import { runState, useFitRuns } from './Runs';

export function FitsOverview() {
  const { project, startAdding } = useStudio();
  const fits = useProject(project).data?.pieces.setups ?? [];
  const focusOf = useRowFocus(fits);
  if (project === null) return null;
  return (
    <section className="screen fx-overview">
      <ScreenHead title={sectionNamed('fit').overview?.title ?? 'Fits'} />
      <div className="ov-scroll">
        <TableHeading title="Fits" count={fits.length} addLabel="Add a fit" onAdd={() => startAdding('setups')} />
        {fits.length === 0 ? (
          <p className="ob-text">The project has no fit yet: a fit puts galaxies and observations together, with a fitter.</p>
        ) : (
          <table className="ov-table">
            <thead>
              <tr>
                <th>Fit</th>
                <th>Galaxies</th>
                <th>Observations</th>
                <th>Fitter</th>
                <th>Latest run</th>
                <th className="ov-num">χ² reduced</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {fits.map((name) => (
                <FitRow key={name} project={project} name={name} focus={focusOf(name)} />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

// The longest name of a fit, and list of its galaxies or observations,
// shown whole (characters)
const NAME_CAP = 28;
const LIST_CAP = 24;

function FitRow({ project, name, focus }: { project: string; name: string; focus: RowFocus }) {
  const { go } = useStudio();
  const path = joinPath(project, `setups/${name}.yaml`);
  const setup = usePiece(path).data;
  const entries = useFitRuns(project, name, path);
  const latest = entries.at(-1);
  const manifest = useRun(latest?.run ?? null).data;
  const state = latest ? runState(latest, manifest?.status) : null;
  const result = manifest?.products.find((p) => p.kind === 'fit_result');
  const text = useFileText(result && latest?.run ? `${latest.run}/${String(result.path)}` : null, false, manifest?.updated ?? '');
  const chi2 = useMemo(() => {
    try {
      return text.data ? ((JSON.parse(text.data) as { solutions: { reduced_chi2: number | null }[] }).solutions[0]?.reduced_chi2 ?? null) : null;
    } catch {
      return null;
    }
  }, [text.data]);
  const submit = useSubmitJob();
  const stop = useStopJob();
  const options = isObject(setup) ? setup : {};
  const names = (section: 'galaxies' | 'observations') =>
    itemsOf(options, section)
      .map((item) => item.name ?? item.piece ?? 'in place')
      .join(', ') || '—';
  const fitter = typeof options.fitter === 'string' ? options.fitter : isObject(options.fitter) ? String(options.fitter.type ?? '') : '—';
  const folder = name.includes('/') ? name.slice(0, name.lastIndexOf('/') + 1) : '';
  const running = state?.state === 'running' || state?.state === 'queued';
  const open = () => go({ kind: 'block', folder: 'setups', name });
  return (
    <tr className="ov-row" onClick={open} {...focus} {...rowKeys(open)}>
      <td className="ov-name">
        <Cut max={NAME_CAP} title={name}>
          {folder && <span className="fx-faint">{folder}</span>}
          {name.slice(folder.length)}
        </Cut>
      </td>
      <td className="fx-small">
        <Cut max={LIST_CAP} title={names('galaxies')} />
      </td>
      <td className="fx-small">
        <Cut max={LIST_CAP} title={names('observations')} />
      </td>
      <td className="fx-small">{fitter}</td>
      <td>
        {latest && state ? (
          <span className="fx-state" data-state={state.state}>
            <span className="fx-state-sign">{state.sign}</span>
            {state.word} · {latest.title}
            {manifest ? <span className="fx-faint"> · {when(manifest.created)}</span> : null}
          </span>
        ) : (
          <span className="fx-faint fx-small">not run yet</span>
        )}
      </td>
      <td className="ov-num">{chi2 === null ? '—' : chi2.toFixed(4)}</td>
      <td onClick={(event) => event.stopPropagation()}>
        {running && latest?.job ? (
          <Button kind="small" onClick={() => stop.mutate(latest.job!.id)}>
            Stop
          </Button>
        ) : (
          <Button kind="small" disabled={options.fitter === undefined || submit.isPending} title={options.fitter === undefined ? 'The fit needs a fitter' : 'Runs it on the server (it waits if the server is busy)'} onClick={() => submit.mutate({ task: 'fit', options: { config: path }, preview: false })}>
            Run
          </Button>
        )}
      </td>
    </tr>
  );
}
