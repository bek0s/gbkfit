// The screen of an observation of the project: how an instrument sees the
// galaxies, and the data it compares them with. Its view shows it at a
// glance (its stages, the sky with its PSF and the beam of the data to
// scale, its spectral resolution, and its checks); its dock sets it, a
// section for each stage. Its options are edited as a whole, and saved or
// reverted (see editing).
import { useDescriptions, useLsfView, usePiece, usePsfView, type SpectralTypeOption } from '../../api/queries';
import { joinPath } from '../../components/FilePicker';
import { membersOf, SchemaForm, titleOf, typeOf, type Options } from '../../components/SchemaForm';
import { Dock, DockSection, NeedsNote } from '../../components/ui/dock';
import { useBlockHead } from '../../shell/BlockMenu';
import { useStudio } from '../../studio';
import { BlockLinks } from '../../shell/BlockLinks';
import { arraysOf, draftOfDataset, EMPTY_ITEM, kept, KINDS, kindOfType } from '../dataset/model';
import { SaveActions, useEditedBlock } from '../editing';
import { checksOf, HANDLED_KEYS, headerBeam, isObject, isSpectral, LIGHT_KEYS, needsOf, objectAt, psfEllipse, textAt } from './model';
import { DataFields, InstrumentFields, LightFields, LikelihoodFields, LineOfSightFields, ObservableFields, OptionFields } from './Sections';
import { Checks, Resolution, Sky, Stages, type SkyData } from './View';
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
  // the convention of velocities its header leaves ambiguous and the type
  // of a spectral axis FITS does not define)
  const dataset = usePiece(named('datasets', options?.data)).data as Options | undefined;
  const kind = kindOfType(dataset?.type);
  const first = kind && dataset ? Object.values(draftOfDataset(kind, dataset, root).items)[0]?.files.data : undefined;
  const axis = dataset ? objectAt(dataset, 'spectral') : {};
  const given = textAt(objectAt(objectAt(options ?? {}, 'observable'), 'spectral'), 'rest') ?? textAt(axis, 'rest');
  const velocity = axis.convention === 'radio' || axis.convention === 'optical' ? axis.convention : null;
  const describe = useDescriptions(first ? [first.file] : [], velocity, given, textAt(axis, 'type') as SpectralTypeOption | null);
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

  if (project === null) return null;
  const title = (family: string, type: unknown) => (type === undefined ? '—' : titleOf(defs, family, String(type)));
  const sky = defs && options ? membersOf(defs.observation ?? {}, defs).find((m) => typeOf(m) === options.type) : undefined;
  const psf = instrument.psf == null ? null : psfEllipse(psfView.data?.axes as [number, number, number | null] | undefined);
  const beam = headerBeam(array);
  const observable = options ? objectAt(options, 'observable') : {};
  const withData = options?.data != null;
  const lsfWidth = instrument.lsf == null ? null : (lsfView.data?.fwhm ?? null);
  const channel = array?.step?.length === 3 ? Math.abs(array.step[2]) : typeof objectAt(observable, 'spectral').step === 'number' ? (objectAt(observable, 'spectral').step as number) : null;
  const skyData: SkyData | null = first && array ? { path: first.file, array, cube: kept(array).length === 3, reading: { rest: given, velocity, type: textAt(axis, 'type') as SpectralTypeOption | null } } : null;
  const grid = !withData && Array.isArray(observable.size) ? { size: pairOf(observable.size), step: pairOf(observable.step ?? [1, 1]) } : !withData && typeof observable.size === 'number' ? { size: pairOf(observable.size), step: pairOf(observable.step ?? 1) } : null;
  const checks = options
    ? checksOf({ options, datasets: edited.blocks.datasets ?? [], datasetType: dataset?.type as string | undefined, data: array, psf, lsfWidth, titleOf: title })
    : [];
  const needs = defs && options ? needsOf({ options, defs, datasets: edited.blocks.datasets ?? [], datasetType: dataset?.type as string | undefined, titleOf: title }) : [];
  const needed = (section: string) => needs.some((need) => need.section === section);
  const common = defs && options ? { defs, project, options, onChange: edited.set, blocks: edited.blocks, openBlock: (folder: string, block: string) => go({ kind: 'block', folder, name: block }) } : null;
  const others = sky ? Object.keys((sky.properties as Record<string, unknown>) ?? {}).filter((key) => !HANDLED_KEYS.includes(key)) : [];

  return (
    <section className="screen ob-screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">Observation{options ? ` · ${title('observation', options.type)}` : ''}</div>
          {head.title}
        </div>
        <BlockLinks folder="observations" name={name} />
        <div className="head-actions">
          <NeedsNote needs={needs.map((need) => ({ text: need.text, at: need.section }))} />
          <SaveActions edited={edited} />
          {head.menu}
        </div>
      </div>
      {edited.error && <p className="error screen-error">{edited.error.message}</p>}
      {!options || !common ? (
        <p className="note screen-error">Reading the observation…</p>
      ) : options.type !== 'sky' || !sky ? (
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
              <Stages options={options} titleOf={title} withData={withData} />
              <div className="ob-figures">
                <Sky data={skyData} grid={grid} psf={psf} beam={beam} />
                <div className="ob-side">
                  {isSpectral(observable.type) && <Resolution fwhm={lsfWidth} channel={channel} named={instrument.lsf != null} />}
                  <Checks checks={checks} />
                </div>
              </div>
            </div>
          }
          sheet={
            <Dock label="The observation">
              <DockSection id="ob-data" title="Data" needed={needed('ob-data')}>
                <DataFields {...common} datasetType={dataset?.type as string | undefined} titleOf={(type) => title('observable', type)} />
              </DockSection>
              <DockSection id="ob-observable" title="Observable" needed={needed('ob-observable')}>
                <ObservableFields {...common} datasetType={dataset?.type as string | undefined} titleOf={(type) => title('observable', type)} />
              </DockSection>
              <DockSection id="ob-instrument" title="Instrument">
                <InstrumentFields {...common} beam={beam} psf={psf} titleOf={title} />
              </DockSection>
              <DockSection id="ob-sight" title="Line of sight" needed={needed('ob-sight')}>
                <LineOfSightFields {...common} titleOf={title} />
              </DockSection>
              {withData && (
                <DockSection id="ob-likelihood" title="Likelihood">
                  <LikelihoodFields {...common} titleOf={title} />
                </DockSection>
              )}
              <DockSection id="ob-light" title="Galaxies and light" note={setCount(options, LIGHT_KEYS)}>
                <LightFields {...common} />
              </DockSection>
              <DockSection id="ob-more" title="More options" note={setCount(options, others)}>
                <OptionFields {...common} member={sky} keys={others} />
              </DockSection>
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
