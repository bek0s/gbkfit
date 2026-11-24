// The screen of an observation of the project: how an instrument sees the
// galaxies, on a sampling of its own (the data it is fitted to are paired
// with it by the setups). Its view shows it at a glance (its stages, the
// sky of its grid with its PSF to scale, its spectral resolution, a patch
// of the noise of its mock data, and its checks); its dock sets it, a
// section for each stage, in the order of the path of the light (what it
// sees, its instrument, what it records, the noise of its mock data), and
// then how accurately and fast its model is made. The problems the library
// finds in it, alone, lead from the head to their sections. Its options
// are edited as a whole, and saved or reverted (see editing).
import { useLsfView, useObservationProblems, usePiece, usePsfView } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import { membersOf, SchemaForm, titleOf, typeOf, type Options } from '../../components/SchemaForm';
import { Dock, DockSection, NeedsNote } from '../../components/ui/dock';
import { useBlockHead } from '../../shell/BlockMenu';
import { SaveControls, ScreenHead } from '../../shell/ScreenHead';
import { useStudio } from '../../studio';
import { useEditedBlock } from '../editing';
import { problemText } from '../galaxy/Model';
import { ACCURACY_KEYS, checksOf, HANDLED_KEYS, isObject, isSpectral, LIGHT_KEYS, noiseText, objectAt, pairOf, psfEllipse, sectionOf, skyKeys, textAt } from './model';
import { AccuracyFields, InstrumentFields, NoiseFields, OptionFields, RecordsFields, SeesFields } from './Sections';
import { Checks, NoisePatch, Resolution, Sky, Stages } from './View';
import { SheetLayout } from '../../components/ui/sheet';

export function ObservationScreen({ name }: { name: string }) {
  const { project, go } = useStudio();
  const root = project ?? '.';
  const edited = useEditedBlock(root, 'observations', name);
  const head = useBlockHead(root, 'observations', name);
  const options = edited.options;
  const defs = edited.defs;
  const named = (folder: string, value: unknown) => (typeof value === 'string' && (edited.blocks[folder] ?? []).includes(value) ? joinPath(root, `${folder}/${value}.yaml`) : null);

  // (its PSF and its LSF, of the project or given in place, as the server
  // draws them)
  const instrument = options ? objectAt(options, 'instrument') : {};
  const spectral = options ? objectAt(options, 'spectral') : {};
  const psfPiece = usePiece(named('psfs', instrument.psf)).data as Options | undefined;
  const lsfPiece = usePiece(named('lsfs', instrument.lsf)).data as Options | undefined;
  const psfOptions = isObject(instrument.psf) ? instrument.psf : psfPiece;
  const lsfOptions = isObject(instrument.lsf) ? instrument.lsf : lsfPiece;
  const psfView = usePsfView(root, psfOptions ?? {}, null, psfOptions !== undefined && typeof psfOptions.type === 'string');
  // (its LSF at its rest: of its widths of wavelength or frequency)
  const lsfView = useLsfView(root, lsfOptions ?? {}, { step: null, rest: textAt(spectral, 'rest'), unit: null, channel: null }, lsfOptions !== undefined && typeof lsfOptions.type === 'string');
  // (the noise of its mock data, of the project or in place)
  const noisePiece = usePiece(named('noise', options?.noise)).data as Options | undefined;
  const noiseOptions = isObject(options?.noise) ? options.noise : noisePiece;
  // (its problems as the library finds them, alone; asked again as the
  // pieces it names change)
  const described = useObservationProblems(root, options, [psfPiece, lsfPiece, noisePiece]);

  if (project === null) return null;
  const title = (family: string, type: unknown) => (type === undefined ? '—' : titleOf(defs, family, String(type)));
  // (of a kind of data on the sky; a profile has the form of the schema)
  const sky = defs && options && options.type !== 'profile' ? membersOf(defs.observation ?? {}, defs).find((m) => typeOf(m) === options.type) : undefined;
  const psf = instrument.psf == null ? null : psfEllipse(psfView.data?.axes as [number, number, number | null] | undefined);
  const lsfWidth = instrument.lsf == null ? null : (lsfView.data?.fwhm ?? null);
  const channel = typeof spectral.step === 'number' ? spectral.step : null;
  const grid = options && (Array.isArray(options.size) || typeof options.size === 'number') ? { size: pairOf(options.size), step: pairOf(options.step ?? 1) } : null;
  const noise = noiseText(options?.noise, noiseOptions, title);
  const checks = options ? checksOf({ options, psf, lsfWidth, noise }) : [];
  // (the options of every observation on the sky that no section sets)
  const others = sky && defs ? skyKeys(defs).filter((key) => !HANDLED_KEYS.includes(key)) : [];
  const needs = (described.data ?? []).map((problem) => ({ text: problemText(problem.message), section: sky ? sectionOf(problem.path, others) : 'ob-all' }));
  const needed = (section: string) => needs.some((need) => need.section === section);
  const common = defs && options ? { defs, project, options, onChange: edited.set, blocks: edited.blocks, openBlock: (folder: string, block: string) => go({ kind: 'block', folder, name: block }) } : null;

  return (
    <section className="screen ob-screen">
      <ScreenHead up="observations" tags={options ? [title('observation', options.type)] : []} title={head.title}>
        <NeedsNote needs={needs.map((need) => ({ text: need.text, at: need.section }))} />
        <SaveControls saving={edited} />
        {head.menu}
      </ScreenHead>
      {edited.error && <p className="error screen-error">{edited.error.message}</p>}
      {!options || !common ? (
        <p className="note screen-error">Reading the observation…</p>
      ) : !sky ? (
        <SheetLayout
          className="ob-body"
          width={400}
          picture={
            <div className="ob-view">
              <p className="ob-empty">An observation of {title('observation', options.type).toLowerCase()}: its options are at the right.</p>
            </div>
          }
          sheet={
            <Dock label="The observation">
              <DockSection id="ob-all" title="Options" needed={needed('ob-all')}>
                <div className="ob-schema">
                  <SchemaForm defs={common.defs} def="observation" project={project} value={options} onChange={edited.set} blocks={edited.blocks} openBlock={common.openBlock} />
                </div>
              </DockSection>
            </Dock>
          }
        />
      ) : (
        <SheetLayout
          className="ob-body"
          width={400}
          picture={
            <div className="ob-view">
              <Stages options={options} titleOf={title} noise={noise} />
              <div className="ob-figures">
                <Sky grid={grid} psf={psf} />
                <div className="ob-side">
                  {isSpectral(options.type) && <Resolution fwhm={lsfWidth} channel={channel} named={instrument.lsf != null} />}
                  <NoisePatch noise={noiseOptions} text={noise} psf={psf} pixel={grid ? grid.step[0] : null} />
                  <Checks checks={checks} />
                </div>
              </div>
            </div>
          }
          sheet={
            <Dock label="The observation">
              <DockSection id="ob-sees" title="Sees" note={setCount(options, LIGHT_KEYS)} needed={needed('ob-sees')}>
                <SeesFields {...common} />
              </DockSection>
              <DockSection id="ob-instrument" title="Instrument" needed={needed('ob-instrument')}>
                <InstrumentFields {...common} psf={psf} lsfWidth={lsfWidth} channel={channel} titleOf={title} />
              </DockSection>
              <DockSection id="ob-records" title="Records" needed={needed('ob-records')}>
                <RecordsFields {...common} titleOf={(type) => title('observation', type)} />
              </DockSection>
              <DockSection id="ob-noise" title="Noise" note={options.noise == null ? undefined : noise} needed={needed('ob-noise')}>
                <NoiseFields {...common} titleOf={title} />
              </DockSection>
              <DockSection id="ob-accuracy" title="Accuracy and speed" note={setCount(options, ACCURACY_KEYS) ?? 'as by default'} needed={needed('ob-accuracy')} folded>
                <AccuracyFields {...common} member={sky} />
              </DockSection>
              {others.length > 0 && (
                <DockSection id="ob-more" title="Other options" note={setCount(options, others)} needed={needed('ob-more')}>
                  <OptionFields {...common} member={sky} keys={others} />
                </DockSection>
              )}
            </Dock>
          }
        />
      )}
    </section>
  );
}

function setCount(options: Options, keys: string[]): string | undefined {
  const count = keys.filter((key) => options[key] != null).length;
  return count ? `${count} set` : undefined;
}
