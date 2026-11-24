// The screen of an observation of the project: how an instrument sees the
// galaxies, on a sampling of its own (the data a fit pairs it with have
// theirs). Its sheet sets it (see Sheet: its kind first, which decides the
// rest; its sampling; its instrument; what it sees; the noise of its mock
// data; more), and its picture shows a galaxy as it sees it beside the
// galaxy itself (see Picture). The problems the library finds in it,
// alone, are at their rows, and counted in the head. Its options are
// edited as a whole, and saved or reverted (see editing).
import { useEffect, useState } from 'react';
import { useDatasetGives, useLsfView, useObservationProblems, useObservationSampling, usePiece, usePsfView } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import { ProblemCounter, ProblemsProvider, type Problem } from '../../components/Problems';
import { propertiesOf, resolve, titleOf, type Defs, type Options } from '../../components/SchemaForm';
import { SheetLayout } from '../../components/ui/sheet';
import { useBlockHead } from '../../shell/BlockMenu';
import { SaveControls, ScreenHead } from '../../shell/ScreenHead';
import { useStudio } from '../../studio';
import { useEditedBlock } from '../editing';
import { problemText } from '../galaxy/Model';
import { InstrumentBlock } from './Instrument';
import { hasOption, isObject, noiseOf, objectAt, observationMember, psfEllipse, samplingKeys, textAt, withKey } from './model';
import type { RowsContext } from './OptionRows';
import { ObservationPicture } from './Picture';
import { KindBlock, MoreBlock, NoiseBlock, SamplingBlock, SeesBlock } from './Sheet';

export function ObservationScreen({ name }: { name: string }) {
  const { project, go } = useStudio();
  const root = project ?? '.';
  const edited = useEditedBlock(root, 'observations', name);
  const head = useBlockHead(root, 'observations', name);
  const options = edited.options;
  const defs = edited.defs;
  const blocks = edited.blocks;
  const named = (folder: string, value: unknown) => (typeof value === 'string' && (blocks[folder] ?? []).includes(value) ? joinPath(root, `${folder}/${value}.yaml`) : null);

  // (its PSF, LSF and noise, of the project or given in place, and their
  // widths as the server draws them)
  const instrument = options ? objectAt(options, 'instrument') : {};
  const psfPiece = usePiece(named('psfs', instrument.psf)).data as Options | undefined;
  const lsfPiece = usePiece(named('lsfs', instrument.lsf)).data as Options | undefined;
  const noisePiece = usePiece(named('noise', options?.noise)).data as Options | undefined;
  const psfOptions = isObject(instrument.psf) ? instrument.psf : psfPiece;
  const lsfOptions = isObject(instrument.lsf) ? instrument.lsf : lsfPiece;
  const psfView = usePsfView(root, psfOptions ?? {}, null, psfOptions !== undefined && typeof psfOptions.type === 'string');
  const spectral = options ? objectAt(options, 'spectral') : {};
  const lsfView = useLsfView(root, lsfOptions ?? {}, { step: null, rest: textAt(spectral, 'rest'), unit: null, channel: null }, lsfOptions !== undefined && typeof lsfOptions.type === 'string');
  // (its problems as the library finds them, alone, and its sampling;
  // asked again as the pieces it names change)
  const pieces = [psfPiece, lsfPiece, noisePiece];
  const described = useObservationProblems(root, options, pieces);
  const sampling = useObservationSampling(root, options, pieces);

  // (a dataset copied: its sampling, or the beam of its data as its PSF;
  // what went wrong, at the row of what was copied, until the next edit)
  const gives = useDatasetGives();
  const [copyProblem, setCopyProblem] = useState<Problem | null>(null);
  // (the parts of its instrument hidden in the picture, not saved)
  const [hidden, setHidden] = useState<ReadonlySet<string>>(new Set());
  useEffect(() => setCopyProblem(null), [options]);
  if (project === null) return null;
  const set = edited.set;
  const copySampling = async (dataset: string) => {
    if (!options) return;
    try {
      const given = await gives.mutateAsync({ project, type: String(options.type), dataset });
      const keys = samplingKeys(observationMember(defs, options.type));
      const kept = Object.fromEntries(Object.entries(options).filter(([key]) => !keys.includes(key)));
      set({ ...kept, ...given.sampling });
    } catch (e) {
      setCopyProblem({ severity: 'error', path: [], message: `${dataset}: ${(e as Error).message}` });
    }
  };
  const copyBeam = async (dataset: string) => {
    if (!options) return;
    try {
      const given = await gives.mutateAsync({ project, type: String(options.type), dataset });
      if (!given.psf) throw new Error(given.psf_note ?? 'the data have no beam');
      set(withKey(options, 'instrument', { ...instrument, psf: given.psf }));
    } catch (e) {
      setCopyProblem({ severity: 'error', path: ['instrument', 'psf'], message: `${dataset}: ${(e as Error).message}` });
    }
  };

  const title = (family: string, type: unknown) => (type === undefined ? '—' : titleOf(defs, family, String(type)));
  const member = options ? observationMember(defs, options.type) : undefined;
  const problems: Problem[] = [...(described.data ?? []).map((p) => ({ ...p, message: problemText(p.message) })), ...(copyProblem ? [copyProblem] : [])];
  // (whether it can be drawn: none of its problems is an error; null
  // while the problems of the options as they are come)
  const canDraw = described.data && !described.isPlaceholderData ? !described.data.some((p) => p.severity === 'error') : null;
  // (a quantity of an observation with channels starts in their unit:
  // e.g. the width of its LSF in km/s)
  const channelUnit = member && hasOption(member, 'spectral') ? unitOfChannels(defs) : null;
  const ctx: RowsContext | null = defs ? { defs, project, blocks, openBlock: (folder, block) => go({ kind: 'block', folder, name: block }), units: channelUnit ? [channelUnit] : [] } : null;
  const noise = options?.noise == null ? null : isObject(options.noise) ? noiseOf(options.noise, title) : noisePiece ? `${options.noise}: ${noiseOf(noisePiece, title)}` : String(options.noise);
  const datasets = blocks.datasets ?? [];
  const psf = instrument.psf == null ? null : psfEllipse(psfView.data?.axes as [number, number, number | null] | undefined);
  const lsfFwhm = instrument.lsf == null ? null : (lsfView.data?.fwhm ?? null);
  // (the observation in the picture: without the parts hidden; without a
  // PSF, without its unit too, of light per beam, which needs one)
  const shown = options && hidden.size ? pictured(options, instrument, hidden) : options;
  const hide = (key: string) => setHidden((now) => (now.has(key) ? new Set([...now].filter((k) => k !== key)) : new Set([...now, key])));

  return (
    <ProblemsProvider problems={problems}>
      <section className="screen ob-screen">
        <ScreenHead up="observations" title={head.title}>
          <ProblemCounter />
          <SaveControls saving={edited} />
          {head.menu}
        </ScreenHead>
        {edited.error && <p className="error screen-error">{edited.error.message}</p>}
        {!options || !ctx ? (
          <p className="note screen-error">Reading the observation…</p>
        ) : (
          <SheetLayout
            className="ob-body"
            width={400}
            sheet={
              <aside className="lp-sheet ob-sheet" aria-label="The observation">
                <KindBlock ctx={ctx} options={options} member={member} onChange={set} />
                {member && (
                  <>
                    <SamplingBlock ctx={ctx} options={options} member={member} sampling={sampling.data} onChange={set} datasets={datasets} onCopy={copySampling} />
                    {hasOption(member, 'instrument') && (
                      <InstrumentBlock
                        ctx={ctx}
                        options={options}
                        onChange={set}
                        shared={{ psf: psfPiece, lsf: lsfPiece }}
                        psf={psf}
                        lsfFwhm={lsfFwhm}
                        channel={sampling.data?.channels?.step ?? null}
                        datasets={datasets}
                        onCopyBeam={copyBeam}
                        hidden={hidden}
                        onHide={hide}
                      />
                    )}
                    <SeesBlock ctx={ctx} options={options} member={member} galaxies={blocks.galaxies ?? []} onChange={set} />
                    <NoiseBlock ctx={ctx} options={options} member={member} shared={noisePiece} onChange={set} titleOf={title} />
                    <MoreBlock ctx={ctx} options={options} member={member} onChange={set} />
                  </>
                )}
              </aside>
            }
            picture={
              <ObservationPicture
                project={project}
                options={shown ?? options}
                canDraw={canDraw}
                kindTitle={title('observation', options.type)}
                sampling={sampling.data}
                psf={hidden.has('psf') ? null : psf}
                lsf={instrument.lsf != null && !hidden.has('lsf') && lsfView.data ? { values: lsfView.data.values, step: lsfView.data.step } : null}
                lsfFwhm={hidden.has('lsf') ? null : lsfFwhm}
                noise={noise}
                galaxies={blocks.galaxies ?? []}
              />
            }
          />
        )}
      </section>
    </ProblemsProvider>
  );
}

// An observation as its picture draws it: without the parts of its
// instrument that are hidden; without a PSF, without its unit too (light
// per beam needs a beam; the picture then shows its model in its own
// unit)
function pictured(options: Options, instrument: Options, hidden: ReadonlySet<string>): Options {
  const kept = Object.fromEntries(Object.entries(instrument).filter(([key]) => !hidden.has(key)));
  const next = withKey(options, 'instrument', Object.keys(kept).length ? kept : undefined);
  return hidden.has('psf') ? withKey(next, 'unit', undefined) : next;
}

// The unit of the channels of an observation, as its schema gives it
// (that of the width of the channels of a spectral axis), if any
function unitOfChannels(defs: Defs | undefined): string | null {
  const step = defs ? propertiesOf(resolve({ $ref: '#/$defs/SpectralAxis' }, defs)).step : undefined;
  return typeof step?.unit === 'string' ? step.unit : null;
}
