// The screen of an observation of the project: how an instrument sees the
// galaxies, and the data it compares them with. Its view shows it at a
// glance (its stages, the sky with its PSF and the beam of the data to
// scale, its spectral resolution, a patch of the noise of its mock data,
// and its checks); its dock sets it, a section for each stage, in the order
// of the path of the light (what it sees, its instrument, what it records,
// the noise of its mock data, its data), and then how accurately and fast
// its model is made. Its options are edited as a whole, and saved or
// reverted (see editing).
import { useDescriptions, useLsfView, usePiece, usePsfView, type Reading, type SpectralTypeOption } from '../../api/queries';
import { kept } from '../../components/arrays';
import { joinPath } from '../../components/FilePicker';
import { membersOf, SchemaForm, titleOf, typeOf, type Options } from '../../components/SchemaForm';
import { Dock, DockSection, NeedsNote } from '../../components/ui/dock';
import { useBlockHead } from '../../shell/BlockMenu';
import { SaveControls, ScreenHead } from '../../shell/ScreenHead';
import { useStudio } from '../../studio';
import { arraysOf, draftOfDataset, EMPTY_ITEM, KINDS, kindOfType } from '../dataset/model';
import { useEditedBlock } from '../editing';
import { ACCURACY_KEYS, checksOf, datasetHasErrors, HANDLED_KEYS, headerBeam, isObject, isSpectral, LIGHT_KEYS, needsOf, noiseText, objectAt, psfEllipse, skyKeys, textAt } from './model';
import { AccuracyFields, DataFields, InstrumentFields, NoiseFields, OptionFields, RecordsFields, SeesFields } from './Sections';
import { Checks, NoisePatch, Resolution, Sky, Stages, type SkyData } from './View';
import { SheetLayout } from '../../components/ui/sheet';

export function ObservationScreen({ name }: { name: string }) {
  const { project, go } = useStudio();
  const root = project ?? '.';
  const edited = useEditedBlock(root, 'observations', name);
  const head = useBlockHead(root, 'observations', name);
  const options = edited.options;
  const defs = edited.defs;
  const named = (folder: string, value: unknown) => (typeof value === 'string' && (edited.blocks[folder] ?? []).includes(value) ? joinPath(root, `${folder}/${value}.yaml`) : null);

  // (the dataset it names, and the array of the data of its first item,
  // read as the dataset reads it: of its rest or that of the observation,
  // the convention of velocities its header leaves ambiguous, the type of
  // a spectral axis FITS does not define and the unit of one whose header
  // gives none)
  const dataset = usePiece(named('datasets', options?.dataset)).data as Options | undefined;
  const kind = kindOfType(dataset?.type);
  const first = kind && dataset ? Object.values(draftOfDataset(kind, dataset, root).items)[0]?.files.data : undefined;
  const axis = dataset ? objectAt(dataset, 'spectral') : {};
  const given = textAt(objectAt(options ?? {}, 'spectral'), 'rest') ?? textAt(axis, 'rest');
  const velocity = axis.convention === 'radio' || axis.convention === 'optical' ? axis.convention : null;
  const reading: Reading = { rest: given, velocity, type: textAt(axis, 'type') as SpectralTypeOption | null, spectralUnit: textAt(axis, 'unit'), dataUnit: null };
  const describe = useDescriptions(first ? [first.file] : [], reading);
  const array = kind && first ? arraysOf({ ...EMPTY_ITEM, files: { data: first } }, describe, KINDS[kind].ndim).found.data : undefined;

  // (its PSF and its LSF, of the project or given in place, as the server
  // draws them)
  const instrument = options ? objectAt(options, 'instrument') : {};
  const psfPiece = usePiece(named('psfs', instrument.psf)).data as Options | undefined;
  const lsfPiece = usePiece(named('lsfs', instrument.lsf)).data as Options | undefined;
  const psfOptions = isObject(instrument.psf) ? instrument.psf : psfPiece;
  const lsfOptions = isObject(instrument.lsf) ? instrument.lsf : lsfPiece;
  const psfView = usePsfView(root, psfOptions ?? {}, null, psfOptions !== undefined && typeof psfOptions.type === 'string');
  // (its LSF at its rest: of its widths of wavelength or frequency)
  const lsfView = useLsfView(root, lsfOptions ?? {}, { step: null, rest: array?.rest ?? given, unit: null, channel: null }, lsfOptions !== undefined && typeof lsfOptions.type === 'string');
  // (the noise of its mock data, of the project or in place)
  const noisePiece = usePiece(named('noise', options?.noise)).data as Options | undefined;
  const noiseOptions = isObject(options?.noise) ? options.noise : noisePiece;

  if (project === null) return null;
  const title = (family: string, type: unknown) => (type === undefined ? '—' : titleOf(defs, family, String(type)));
  // (of a kind of data on the sky; a profile has the form of the schema)
  const sky = defs && options && options.type !== 'profile' ? membersOf(defs.observation ?? {}, defs).find((m) => typeOf(m) === options.type) : undefined;
  const psf = instrument.psf == null ? null : psfEllipse(psfView.data?.axes as [number, number, number | null] | undefined);
  const beam = headerBeam(array);
  const withData = options?.dataset != null;
  const lsfWidth = instrument.lsf == null ? null : (lsfView.data?.fwhm ?? null);
  const spectral = options ? objectAt(options, 'spectral') : {};
  const channel = array?.step?.length === 3 ? Math.abs(array.step[2]) : typeof spectral.step === 'number' ? spectral.step : null;
  const skyData: SkyData | null = first && array ? { path: first.file, array, cube: kept(array).length === 3, reading } : null;
  const grid = !withData && options && (Array.isArray(options.size) || typeof options.size === 'number') ? { size: pairOf(options.size), step: pairOf(options.step ?? 1) } : null;
  const errors = datasetHasErrors(dataset);
  const noise = noiseText(options?.noise, noiseOptions, title);
  const checks = options
    ? checksOf({ options, datasets: edited.blocks.datasets ?? [], datasetType: dataset?.type as string | undefined, data: array, psf, lsfWidth, errors, noise, titleOf: title })
    : [];
  const needs = defs && options ? needsOf({ options, defs, datasets: edited.blocks.datasets ?? [], datasetType: dataset?.type as string | undefined, titleOf: title }) : [];
  const needed = (section: string) => needs.some((need) => need.section === section);
  const common = defs && options ? { defs, project, options, onChange: edited.set, blocks: edited.blocks, openBlock: (folder: string, block: string) => go({ kind: 'block', folder, name: block }) } : null;
  // (the options of every observation on the sky that no section sets)
  const others = sky && defs ? skyKeys(defs).filter((key) => !HANDLED_KEYS.includes(key)) : [];

  return (
    <section className="screen ob-screen">
      <ScreenHead
        eyebrow={`Observation${options ? ` · ${title('observation', options.type)}` : ''}`}
        title={head.title}
      >
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
              <DockSection id="ob-all" title="Options">
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
                <Sky data={skyData} grid={grid} psf={psf} beam={beam} />
                <div className="ob-side">
                  {isSpectral(options.type) && <Resolution fwhm={lsfWidth} channel={channel} named={instrument.lsf != null} />}
                  <NoisePatch noise={noiseOptions} text={noise} psf={psf} pixel={array?.step ? Math.abs(array.step[0]) : grid ? grid.step[0] : null} />
                  <Checks checks={checks} />
                </div>
              </div>
            </div>
          }
          sheet={
            <Dock label="The observation">
              <DockSection id="ob-sees" title="Sees" note={setCount(options, LIGHT_KEYS)}>
                <SeesFields {...common} />
              </DockSection>
              <DockSection id="ob-instrument" title="Instrument">
                <InstrumentFields {...common} beam={beam} psf={psf} lsfWidth={lsfWidth} channel={channel} titleOf={title} />
              </DockSection>
              <DockSection id="ob-records" title="Records" needed={needed('ob-records')}>
                <RecordsFields {...common} datasetType={dataset?.type as string | undefined} titleOf={(type) => title('observation', type)} array={array} />
              </DockSection>
              <DockSection id="ob-noise" title="Noise" note={options.noise == null ? undefined : noise}>
                <NoiseFields {...common} titleOf={title} errors={errors} rms={array?.rms ?? null} />
              </DockSection>
              <DockSection id="ob-data" title="Data" needed={needed('ob-data')}>
                <DataFields {...common} datasetType={dataset?.type as string | undefined} titleOf={(type) => title('observation', type)} member={sky} />
              </DockSection>
              <DockSection id="ob-accuracy" title="Accuracy and speed" note={setCount(options, ACCURACY_KEYS) ?? 'as by default'} folded>
                <AccuracyFields {...common} member={sky} />
              </DockSection>
              {others.length > 0 && (
                <DockSection id="ob-more" title="Other options" note={setCount(options, others)}>
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

function pairOf(value: unknown): [number, number] {
  const [a, b] = Array.isArray(value) ? value : [value, value];
  return [Number(a) || 1, Number(b ?? a) || 1];
}

function setCount(options: Options, keys: string[]): string | undefined {
  const count = keys.filter((key) => options[key] != null).length;
  return count ? `${count} set` : undefined;
}
