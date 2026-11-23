// What the page reads from the server and changes on it, with the keys of
// its cache (which the events of the server refresh, see events.ts).
import { keepPreviousData, useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { createContext, useContext, useRef } from 'react';
import { api, data, errorMessage } from './client';
import { decodeFrame } from './frames';
import type { components, operations } from './schema';

export const keys = {
  projects: ['projects'],
  project: (path: string) => ['project', path],
  piece: (path: string) => ['piece', path],
  folder: (path: string) => ['folder', path],
  description: (path: string, reading: string) => ['description', path, reading],
  preparation: (path: string) => ['preparation', path],
  view: (...parts: (string | number)[]) => ['view', ...parts],
  jobs: ['jobs'],
  renders: ['renders'],
};

// The paths of the projects of the workspace
export function useProjects() {
  return useQuery({ queryKey: keys.projects, queryFn: () => data(api.GET('/api/v1/projects')) });
}

// A piece that a project may not have (e.g. the params block of the same
// name as a galaxy): asked for only if the project has it (exists;
// undefined: not known yet), so that a missing one is no error
export function useOptionalPiece(path: string, exists: boolean | undefined) {
  const piece = useQuery({
    queryKey: keys.piece(path),
    enabled: exists === true,
    retry: false,
    queryFn: () => data(api.GET('/api/v1/pieces/{path}', { params: { path: { path } } })),
  });
  return { data: exists ? piece.data : undefined, isPending: exists === undefined || (exists && piece.isPending), error: exists ? piece.error : null };
}

// A project: its blocks by folder (their names), and its runs
export function useProject(path: string | null) {
  return useQuery({
    queryKey: keys.project(path ?? ''),
    enabled: path !== null,
    queryFn: () => data(api.GET('/api/v1/projects/{path}', { params: { path: { path: path ?? '' } } })),
  });
}

// A block of a project (a piece): its options
export function usePiece(path: string | null) {
  return useQuery({
    queryKey: keys.piece(path ?? ''),
    enabled: path !== null,
    queryFn: () => data(api.GET('/api/v1/pieces/{path}', { params: { path: { path: path ?? '' } } })),
  });
}

// A folder of the workspace: its folders and files
export function useFolder(path: string) {
  return useQuery({
    queryKey: keys.folder(path),
    queryFn: () => data(api.GET('/api/v1/folders/{path}', { params: { path: { path } } })),
  });
}

// The convention of velocities that a header leaves ambiguous (see the
// server's ArrayInfo)
export type Velocity = 'radio' | 'optical';
// (what a spectral axis whose header gives it a type FITS does not define
// is said to be; see the server's describe)
export type SpectralTypeOption = NonNullable<NonNullable<operations['describe_api_v1_describe__path__get']['parameters']['query']>['spectral_type']>;

// How the spectral axis of a cube is read, as a dataset of it reads it:
// in a unit its header gives none of (or one not valid), as the type of
// an axis FITS does not define, of a convention of velocities its header
// leaves ambiguous, and as velocities of a rest (of wavelengths or
// frequencies); null: as its header says. Every view of a cube reads it
// alike, so that their channels are the same (a rest may reverse them).
// How a file of data is read (see the server's _reading): the rest and the
// convention of velocities of its spectral axis, its type and unit where
// its header does not say them, and the unit of its values (instead of
// BUNIT)
export type Reading = { rest: string | null; velocity: Velocity | null; type: SpectralTypeOption | null; spectralUnit: string | null; dataUnit: string | null };

// The query of a reading (see the server's _reading), and its part of the
// keys of the views
function readingQuery(reading: Reading | null): { rest?: string; velocity?: Velocity; spectral_type?: SpectralTypeOption; spectral_unit?: string; unit?: string } {
  return {
    ...(reading?.rest ? { rest: reading.rest } : {}),
    ...(reading?.velocity ? { velocity: reading.velocity } : {}),
    ...(reading?.type ? { spectral_type: reading.type } : {}),
    ...(reading?.spectralUnit ? { spectral_unit: reading.spectralUnit } : {}),
    ...(reading?.dataUnit ? { unit: reading.dataUnit } : {}),
  };
}

function readingKey(reading: Reading | null): string {
  return JSON.stringify(readingQuery(reading));
}

// The arrays of a file of data, from its headers, its spectral axis read
// so (see Reading)
export function useDescription(path: string | null, reading: Reading | null = null) {
  return useQuery({
    queryKey: keys.description(path ?? '', readingKey(reading)),
    enabled: path !== null,
    // (the description of the same file stays while another convention or
    // rest is read; never that of another file)
    placeholderData: (previous, query) => (query?.queryKey[1] === path ? previous : undefined),
    queryFn: () => describe(path ?? '', reading),
  });
}

// The header of an array (an HDU) of a file of data, as it is in the file
// (null: none asked for)
export function useHeader(path: string | null, hdu: number) {
  return useQuery({
    queryKey: ['header', path ?? '', hdu],
    enabled: path !== null,
    queryFn: () => data(api.GET('/api/v1/header/{path}', { params: { path: { path: path ?? '' }, query: { hdu } } })),
  });
}

// The descriptions of several files (as useDescription), as a lookup by
// path: its arrays once described, or why it cannot be, and whether they
// are of the inputs given (current). While a file is read again (e.g.
// with another rest), its last description stays, not current: what shows
// it stays, and nothing judges it (e.g. its problems are of the inputs it
// was read with).
export function useDescriptions(paths: string[], reading: Reading | null | ((path: string) => Reading | null) = null) {
  const unique = [...new Set(paths)];
  const readingOf = (path: string) => (typeof reading === 'function' ? reading(path) : reading);
  const results = useQueries({
    queries: unique.map((path) => ({
      queryKey: keys.description(path, readingKey(readingOf(path))),
      queryFn: () => describe(path, readingOf(path)),
    })),
  });
  const last = useRef(new Map<string, NonNullable<(typeof results)[number]['data']>>());
  const byPath = new Map(unique.map((path, i) => [path, results[i]]));
  for (const [path, result] of byPath) if (result.data) last.current.set(path, result.data);
  return (path: string) => {
    const result = byPath.get(path);
    return { arrays: result?.data ?? last.current.get(path), error: result?.error?.message, current: result?.data !== undefined };
  };
}

function describe(path: string, reading: Reading | null) {
  return data(api.GET('/api/v1/describe/{path}', { params: { path: { path }, query: readingQuery(reading) } }));
}

// How a file of data was prepared by the server (null: it was not)
export function usePreparation(path: string | null) {
  return useQuery({
    queryKey: keys.preparation(path ?? ''),
    enabled: path !== null,
    queryFn: () => data(api.GET('/api/v1/preparations/{path}', { params: { path: { path: path ?? '' } } })),
  });
}

// A channel of a cube (an array of a file of data, its spectral axis read
// so): its values (ny, nx), the number of channels of the cube, the
// coordinate of the channel along its spectral axis and its unit (null if
// unknown), and the range of the values of the cube for colours.
// The channel shown before stays while the next one comes; without a
// path, none.
export function useChannel(path: string | null, hdu: number, channel: number, reading: Reading | null = null) {
  return useQuery({
    queryKey: keys.view('channel', path ?? '', hdu, channel, readingKey(reading)),
    enabled: path !== null,
    placeholderData: keepPreviousData,
    queryFn: () => readChannel(path ?? '', hdu, channel, reading),
  });
}

// The values of maps (arrays of files of data, as the first channel of
// useChannel), in their order (undefined until read)
export function useMaps(maps: { path: string; hdu: number }[]) {
  return useQueries({
    queries: maps.map(({ path, hdu }) => ({
      queryKey: keys.view('channel', path, hdu, 0, readingKey(null)),
      queryFn: () => readChannel(path, hdu, 0, null),
    })),
  }).map((result) => result.data?.plane.values);
}

async function readChannel(path: string, hdu: number, channel: number, reading: Reading | null) {
  const frame = decodeFrame(
    await data(
      api.GET('/api/v1/views/channel/{path}', {
        params: { path: { path }, query: { hdu, channel, ...readingQuery(reading) } },
        parseAs: 'arrayBuffer',
      }),
    ),
  );
  const { nchannels, spectral, unit, range } = frame.header as {
    nchannels: number;
    spectral: number | null;
    unit: string | null;
    range: [number, number];
  };
  return { plane: frame.arrays[0], nchannels, spectral, unit, range };
}

// The spectrum of a pixel of a cube, or (none) its total spectrum: the sum
// of its pixels in each channel, of all of them or of a region (x0, x1,
// y0, y1, inclusive); and the coordinates of its channels along its
// spectral axis, read so, with their unit
export function useSpectrum(
  path: string,
  hdu: number,
  pixel: [number, number] | null,
  region: [number, number, number, number] | null,
  reading: Reading | null = null,
) {
  const where = pixel !== null ? { x: pixel[0], y: pixel[1] } : region !== null ? { x0: region[0], x1: region[1], y0: region[2], y1: region[3] } : {};
  const query = { hdu, ...where, ...readingQuery(reading) };
  return useQuery({
    queryKey: keys.view('spectrum', path, hdu, pixel?.join() ?? '', region?.join() ?? '', readingKey(reading)),
    placeholderData: keepPreviousData,
    queryFn: () => data(api.GET('/api/v1/views/spectrum/{path}', { params: { path: { path }, query } })),
  });
}

// The quick moment maps of a cube, by name, of the voxels where the cube,
// smoothed, is above a signal-to-noise ratio (0: all of them): measured as
// moments, its flux, velocity, dispersion, skewness and kurtosis; or of a
// Gauss-Hermite series fitted to each spectrum, its h3 and h4; its
// spectral axis read so (its channels must be velocities)
export function useMoments(path: string, hdu: number, snr: number, enabled: boolean, method: 'moments' | 'gauss_hermite' = 'moments', reading: Reading | null = null) {
  const query = { hdu, snr, method, ...readingQuery(reading) };
  return useQuery({
    queryKey: keys.view('moments', path, hdu, snr, method, readingKey(reading)),
    enabled,
    placeholderData: keepPreviousData,
    queryFn: async () =>
      decodeFrame(
        await data(
          api.GET('/api/v1/views/moments/{path}', {
            params: { path: { path }, query },
            parseAs: 'arrayBuffer',
          }),
        ),
      ).arrays,
  });
}

// A run of a project (see the server's /runs): its manifest (its task,
// status, times and products), asked again as the server says it changed
export function useRun(path: string | null) {
  return useQuery({
    queryKey: ['run', path ?? ''],
    enabled: path !== null,
    // (a run just made has no manifest yet: its event asks again)
    retry: false,
    queryFn: () => data(api.GET('/api/v1/runs/{path}', { params: { path: { path: path ?? '' } } })),
  });
}

// The text of a file of the workspace (e.g. the log of a run), asked again
// every few seconds while it may grow (none: once), and as its version
// changes (e.g. the time its run was last written)
export function useFileText(path: string | null, every: number | false = false, version = '') {
  return useQuery({
    queryKey: ['file', path ?? '', version],
    enabled: path !== null,
    refetchInterval: every,
    queryFn: async () => {
      const { data: text, error, response } = await api.GET('/api/v1/files/{path}', { params: { path: { path: path ?? '' } }, parseAs: 'text' });
      if (!response.ok) throw new Error(errorMessage(error, response.status, response.statusText));
      return text ?? '';
    },
  });
}

export function useJobs() {
  return useQuery({ queryKey: keys.jobs, queryFn: () => data(api.GET('/api/v1/jobs')) });
}

// Stop a job (that runs, or waits)
export function useStopJob() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => data(api.POST('/api/v1/jobs/{id_}/stop', { params: { path: { id_: id } } })),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.jobs }),
  });
}

// The renders that run long (refreshed by the events of the server)
export function useRenders() {
  return useQuery({ queryKey: keys.renders, queryFn: () => data(api.GET('/api/v1/renders')) });
}

// Cancel a render: dropped if it waits, stopped if it runs
export function useCancelRender() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => data(api.POST('/api/v1/renders/{id_}/cancel', { params: { path: { id_: id } } })),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.renders }),
  });
}

// A job (refreshed as its state changes, by the events of the server)
export function useJob(id: string | null) {
  return useQuery({
    queryKey: [...keys.jobs, id ?? ''],
    enabled: id !== null,
    queryFn: () => data(api.GET('/api/v1/jobs/{id_}', { params: { path: { id_: id ?? '' } } })),
  });
}

// Run a task of the library as a job of the server; resolves to the job
export function useSubmitJob() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (job: components['schemas']['JobRequest']) => data(api.POST('/api/v1/jobs', { body: job })),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.jobs }),
  });
}

// Keep the files of a preview of prep (see the server's JobRequest);
// resolves to the job, of the files kept
export function useKeepJob() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => data(api.POST('/api/v1/jobs/{id_}/keep', { params: { path: { id_: id } } })),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.jobs }),
  });
}

// Delete a file of data of a project (refused while the project names it)
export function useDeleteData() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (path: string) => {
      const { error, response } = await api.DELETE('/api/v1/data/{path}', { params: { path: { path } } });
      if (!response.ok) throw new Error(errorMessage(error, response.status, response.statusText));
    },
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ['folder'] });
      void client.invalidateQueries({ queryKey: ['description'] });
      void client.invalidateQueries({ queryKey: ['preparation'] });
    },
  });
}

// Make a project in a folder of the workspace
export function useCreateProject() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (path: string) => data(api.POST('/api/v1/projects', { body: { path } })),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.projects }),
  });
}

// Copy a file of data of the workspace into a project; resolves to the
// path of the copy
export function useImportData(project: string) {
  return useMutation({
    mutationFn: (path: string) => data(api.POST('/api/v1/imports', { body: { project, path } })),
  });
}

// Make a block of a project: with its content (its options), or from the
// template of its folder
export function useCreateBlock(project: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ folder, name, content }: { folder: string; name: string; content?: Record<string, unknown> }) =>
      data(api.POST('/api/v1/pieces', { body: { project, folder, name, content } })),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.project(project) }),
  });
}

// Move a project to another folder of the workspace; resolves to it
export function useRenameProject() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ path, to }: { path: string; to: string }) =>
      data(api.PATCH('/api/v1/projects/{path}', { params: { path: { path } }, body: { path: to } })),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.projects }),
  });
}

// Delete a project and all its files
export function useDeleteProject() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (path: string) => {
      const { error, response } = await api.DELETE('/api/v1/projects/{path}', { params: { path: { path } } });
      if (!response.ok) throw new Error(errorMessage(error, response.status, response.statusText));
    },
    onSuccess: () => client.invalidateQueries({ queryKey: keys.projects }),
  });
}

// The blocks of a project that name one of its blocks (by its path): their
// folders and names
export function useReferences(path: string | null) {
  return useQuery({
    queryKey: ['references', path ?? ''],
    enabled: path !== null,
    queryFn: () => data(api.GET('/api/v1/references/{path}', { params: { path: { path: path ?? '' } } })),
  });
}

// Rename a block of a project (by its path, its folder and its name), and
// the blocks that name it with it; resolves to its new path
export function useRenameBlock(project: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ path, to }: { path: string; folder: string; from: string; to: string }) =>
      data(api.PATCH('/api/v1/pieces/{path}', { params: { path: { path } }, body: { name: to } })),
    onSuccess: (renamed, { path, folder, from, to }) => {
      // (the block is known by its new name at once, as it was by its old,
      // while the project is read again: what shows it neither shows its
      // old name again nor reads it again)
      const piece = client.getQueryData(keys.piece(path));
      if (piece !== undefined) client.setQueryData(keys.piece(renamed), piece);
      client.setQueryData<ProjectInfo>(keys.project(project), (info) => info && withRenamed(info, folder, from, to));
      client.invalidateQueries({ queryKey: keys.project(project) });
      // (but its old path, which names nothing now)
      client.invalidateQueries({ queryKey: ['piece'], predicate: (query) => query.queryKey[1] !== path });
      client.invalidateQueries({ queryKey: ['references'] });
    },
  });
}

type ProjectInfo = components['schemas']['ProjectInfo'];

// A project as it is once a block of a folder of it is renamed, as far as
// its lists say (its blocks, their types and times; the rest is as it is
// read again)
function withRenamed(info: ProjectInfo, folder: string, from: string, to: string): ProjectInfo {
  // (what a list of the folder, by name, says of the block, by its new name)
  const renamed = (byFolder: Record<string, Record<string, string>>) => {
    const { [from]: value, ...others } = byFolder[folder] ?? {};
    return value === undefined ? byFolder : { ...byFolder, [folder]: { ...others, [to]: value } };
  };
  return {
    ...info,
    pieces: { ...info.pieces, [folder]: (info.pieces[folder] ?? []).map((name) => (name === from ? to : name)) },
    types: renamed(info.types),
    modified: renamed(info.modified),
  };
}

// Delete a block of a project (by its path)
export function useDeleteBlock(project: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (path: string) => {
      const { error, response } = await api.DELETE('/api/v1/pieces/{path}', { params: { path: { path } } });
      if (!response.ok) throw new Error(errorMessage(error, response.status, response.statusText));
    },
    onSuccess: (_, path) => {
      client.invalidateQueries({ queryKey: keys.project(project) });
      client.removeQueries({ queryKey: keys.piece(path) });
      client.invalidateQueries({ queryKey: ['references'] });
    },
  });
}

// The JSON Schema of the configurations of GBKFIT (see the server's
// /schema), which the forms of the blocks are made of; it changes with
// the library alone
export function useSchema() {
  return useQuery({ queryKey: ['schema'], staleTime: Infinity, queryFn: () => data(api.GET('/api/v1/schema')) });
}

// The emission lines of the catalogue of GBKFIT ({name, rest}; see the
// server's /lines), which emissions name
export function useLines() {
  return useQuery({ queryKey: ['lines'], staleTime: Infinity, queryFn: () => data(api.GET('/api/v1/lines')) });
}

// The types of components (see the server's /components), in the order of
// the library: by its type, its parts in the order people read them, and
// the traits it takes (the types of its traits of each kind, e.g. 'vpt',
// and the directions of its velocity and dispersion traits)
export function useComponentTypes() {
  return useQuery({ queryKey: ['components'], staleTime: Infinity, queryFn: () => data(api.GET('/api/v1/components')) });
}

// The notes of a project: the text of its README.md ('' if it has none)
export function useNotes(project: string | null) {
  return useQuery({
    queryKey: ['notes', project ?? ''],
    enabled: project !== null,
    queryFn: async () => {
      const { data: text, error, response } = await api.GET('/api/v1/files/{path}', {
        params: { path: { path: `${project}/README.md` } },
        parseAs: 'text',
      });
      if (response.status === 404) return '';
      if (!response.ok) throw new Error(errorMessage(error, response.status, response.statusText));
      // (an empty file has no body)
      return text ?? '';
    },
  });
}

// Write the notes of a project (its README.md)
export function useSaveNotes(project: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (text: string) => {
      const { error, response } = await api.PUT('/api/v1/files/{path}', {
        params: { path: { path: `${project}/README.md` } },
        body: text,
        bodySerializer: (body?: string) => body ?? '',
        headers: { 'Content-Type': 'text/plain' },
      });
      if (!response.ok) throw new Error(errorMessage(error, response.status, response.statusText));
    },
    // (the project says whether it has notes)
    onSuccess: () => Promise.all([client.invalidateQueries({ queryKey: ['notes', project] }), client.invalidateQueries({ queryKey: keys.project(project) })]),
  });
}

// Write the options of a block of a project (by its path)
// The text of a piece of a project of a content, as the project writes it
// (YAML, block style)
export function usePieceText() {
  return useMutation({
    mutationFn: async (content: Record<string, unknown>) => {
      const { data: text, error, response } = await api.POST('/api/v1/piece-text', { body: { content }, parseAs: 'text' });
      if (!response.ok || text === undefined) throw new Error(errorMessage(error, response.status, response.statusText));
      return text;
    },
  });
}

export function useUpdateBlock() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async ({ path, content }: { path: string; content: Record<string, unknown> }) => {
      const { error, response } = await api.PUT('/api/v1/pieces/{path}', { params: { path: { path } }, body: content });
      if (!response.ok) throw new Error(errorMessage(error, response.status, response.statusText));
    },
    onSuccess: (_, { path }) => {
      client.invalidateQueries({ queryKey: keys.piece(path) });
      // (its type and time, as its project says them)
      client.invalidateQueries({ queryKey: ['project'] });
    },
  });
}

// A PSF of options of a block of a project drawn on pixels of a size
// (arcsec; null: of its own scale), with its width; the last drawn stays
// while the next comes
export function usePsfView(project: string, options: Record<string, unknown>, step: [number, number] | null, enabled = true) {
  return useQuery({
    queryKey: keys.view('psf', project, JSON.stringify(options), step?.join() ?? ''),
    enabled,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: ({ signal }) => data(api.POST('/api/v1/views/psf', { signal, body: { project, options, step } })),
  });
}

// A lens of the options of a piece, as the library describes it: its
// parameters, by the names of the lens, and its problems (see the
// server's LensInfo)
// (with the options it describes, as the description of earlier options
// stays while that of the next comes)
export function useLensDescription(project: string, lens: Record<string, unknown> | undefined) {
  return useQuery({
    queryKey: keys.view('lens-describe', project, JSON.stringify(lens ?? null)),
    enabled: lens !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: async ({ signal }) => ({ ...(await data(api.POST('/api/v1/lenses/describe', { signal, body: { project, lens: lens ?? {} } }))), of: lens ?? {} }),
  });
}

// The problems of an observation of options of a piece of a project, as
// the library finds them from the headers of its data, without reading
// them (see the server's /observations/describe); of the pieces it names
// too (depends: what they are, read by the caller, which its key takes)
export function useObservationProblems(project: string, observation: Record<string, unknown> | undefined, depends: unknown = null) {
  return useQuery({
    queryKey: keys.view('observation-describe', project, JSON.stringify(observation ?? null), JSON.stringify(depends)),
    enabled: observation !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: ({ signal }) => data(api.POST('/api/v1/observations/describe', { signal, body: { project, observation: observation ?? {} } })),
  });
}

// The problems of a part that observations share (a PSF, an LSF or a
// noise) of options of a piece of a project (see the server's
// /observations/parts/describe)
export function usePartProblems(project: string, family: 'psf' | 'lsf' | 'noise', part: Record<string, unknown> | undefined) {
  return useQuery({
    queryKey: keys.view('part-describe', project, family, JSON.stringify(part ?? null)),
    enabled: part !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: ({ signal }) => data(api.POST('/api/v1/observations/parts/describe', { signal, body: { project, family, part: part ?? {} } })),
  });
}

// The problems of how the files of a dataset of options (its files
// relative to the project) fit together, from their headers alone (e.g.
// errors of another shape than their data), each at the path of its
// option (see the server's check_dataset)
export function useDatasetCheck(project: string, dataset: Record<string, unknown> | undefined) {
  return useQuery({
    queryKey: keys.view('dataset-check', project, JSON.stringify(dataset ?? null)),
    enabled: dataset !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: ({ signal }) => data(api.POST('/api/v1/datasets/check', { signal, body: { project, dataset: dataset ?? {} } })),
  });
}

// The field of a view of a lens: its half-width and its centre (x, y;
// arcsec)
export type LensFrame = { half: number; centre: [number, number] };

// What a lens makes of a source behind it, at the properties of its
// parameters, without its pieces hidden (by index): its deflection over a
// field (null: one the server fits to the lens), its critical curves and
// its caustics (see the server's LensViewInfo)
export function useLensView(project: string, lens: Record<string, unknown> | undefined, properties: Record<string, unknown>, frame: LensFrame | null, hidden: number[], size = 96) {
  const client = useQueryClient();
  const key = (of: LensFrame | null) => keys.view('lens', project, JSON.stringify(lens ?? null), JSON.stringify(properties), size, JSON.stringify(of), hidden.join(','));
  // (no placeholder: the screen draws the last view while the next comes)
  return useQuery({
    queryKey: key(frame),
    enabled: lens !== undefined,
    retry: false,
    queryFn: async ({ signal }) => {
      const view = await data(api.POST('/api/v1/views/lens', { signal, body: { project, lens: lens ?? {}, properties, size, half: frame?.half ?? null, centre: frame?.centre ?? null, hidden } }));
      // (a view of a field fitted to the lens is also the view of that
      // field, which is asked for next)
      if (frame === null) client.setQueryData(key({ half: view.half, centre: [view.centre[0], view.centre[1]] }), view);
      return view;
    },
  });
}

// A view of a lens (see useLensView)
export type LensView = NonNullable<ReturnType<typeof useLensView>['data']>;

// How an LSF is drawn (see the server's LSFRequest): on channels of a
// width (null: of its own scale), of velocities (km/s) at the rest of a
// line (e.g. '6562.8 Angstrom'; null: none, which its widths in a unit of
// wavelength or frequency need), or of another unit (e.g. 'Angstrom', or
// 'pix': channels; null: km/s), its widths in pixels and its Hanning
// smoothing of channels of the data of a width (null: those it is drawn
// on)
export type LsfAt = { step: number | null; rest: string | null; unit: string | null; channel: number | null };

// An LSF of options of a block of a project drawn as given, with its width
export function useLsfView(project: string, options: Record<string, unknown>, at: LsfAt, enabled = true) {
  return useQuery({
    queryKey: keys.view('lsf', project, JSON.stringify(options), at.step ?? '', at.rest ?? '', at.unit ?? '', at.channel ?? ''),
    enabled,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: ({ signal }) => data(api.POST('/api/v1/views/lsf', { signal, body: { project, options, ...at } })),
  });
}

// A galaxy of options of a block of a project as the library describes it:
// the parameters it has, and its problems; the last stays while the next
// comes
export function useGalaxyParameters(project: string, galaxy: Record<string, unknown> | undefined) {
  return useQuery({
    queryKey: keys.view('parameters', project, JSON.stringify(galaxy ?? null)),
    enabled: galaxy !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: ({ signal }) => data(api.POST('/api/v1/parameters', { signal, body: { project, galaxy: galaxy ?? {} } })),
  });
}

// A render of a galaxy of options of a block of a project at values of its
// parameters (see the server's /views/galaxy): its maps by name (light,
// velocity, vx, vy, dispersion, of all and '{component}/{map}'), the size
// of its pixels and their number across, its curves by name (with
// 'radius') and the problems of those not drawn; the last stays while the
// next comes
export interface GalaxyRender {
  maps: Record<string, Float32Array>;
  size: number;
  step: number;
  curves: Record<string, (number | null)[]>;
  problems: string[];
  // The extra outputs of its evaluation (see the library's Extras)
  extras: GalaxyExtraInfo[];
}

// An extra output of the evaluation of a galaxy, as a render lists it
export interface GalaxyExtraInfo {
  name: string;
  ids: Record<string, string>;
  title: string;
  description: string;
  unit: string;
  ndim: number;
}

// The light of a galaxy in 3d, face-on (see the server's
// /views/galaxy-volume): its values (z, y, x; z along its axis), the size
// of its voxels across and along z (arcsec), and the voxel of its plane
export interface GalaxyVolume {
  values: Float32Array;
  shape: [number, number, number];
  step: number;
  depthStep: number;
  zero: number;
}

// What the renders of a screen are of, for the Activity drawer (e.g. the
// name of a galaxy): given by the screen, and no part of their keys
export const RenderLabel = createContext<string | undefined>(undefined);

export function useGalaxyVolume(project: string, galaxy: Record<string, unknown> | undefined, properties: Record<string, unknown>, size = 96, enabled = true) {
  const label = useContext(RenderLabel);
  return useQuery({
    queryKey: keys.view('galaxy-volume', project, JSON.stringify(galaxy ?? null), JSON.stringify(properties), size),
    enabled: enabled && galaxy !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: async ({ signal }): Promise<GalaxyVolume> => {
      const frame = decodeFrame(await data(api.POST('/api/v1/views/galaxy-volume', { signal, body: { project, galaxy: galaxy ?? {}, properties, size, label }, parseAs: 'arrayBuffer' })));
      const header = frame.header as { step: number; depth_step: number; zero: number };
      const [array] = frame.arrays;
      return { values: array.values, shape: array.shape as [number, number, number], step: header.step, depthStep: header.depth_step, zero: header.zero };
    },
  });
}

// An extra output of a galaxy rendered (see the server's /views/galaxy-extra)
export interface GalaxyExtra {
  values: Float32Array;
  shape: number[];
  title: string;
  description: string;
  unit: string;
  step: number;
  depthStep: number | null;
}

export function useGalaxyExtra(
  project: string,
  galaxy: Record<string, unknown> | undefined,
  properties: Record<string, unknown>,
  extra: { name: string; ids: Record<string, string> } | null,
  view: 'sky' | 'face-on' | 'edge-on' = 'sky',
  size = 128,
) {
  const label = useContext(RenderLabel);
  return useQuery({
    queryKey: keys.view('galaxy-extra', project, JSON.stringify(galaxy ?? null), JSON.stringify(properties), JSON.stringify(extra), view, size),
    enabled: galaxy !== undefined && extra !== null,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: async ({ signal }): Promise<GalaxyExtra> => {
      const frame = decodeFrame(
        await data(api.POST('/api/v1/views/galaxy-extra', { signal, body: { project, galaxy: galaxy ?? {}, properties, name: extra?.name ?? '', ids: extra?.ids ?? {}, view, size, label }, parseAs: 'arrayBuffer' })),
      );
      const header = frame.header as { title: string; description: string; unit: string; step: number; depth_step: number | null };
      const [array] = frame.arrays;
      return { values: array.values, shape: array.shape as number[], title: header.title, description: header.description, unit: header.unit, step: header.step, depthStep: header.depth_step };
    },
  });
}

// The position-velocity diagram of a cube along a slit (see the server's
// /views/pv): its values (channels, positions), its positions (pixels from
// the centre, positive towards the position angle), the coordinates of
// its channels and their unit, and the size of the pixels of the cube
// (arcsec; null if not known)
export interface PVDiagram {
  values: Float32Array;
  positions: number[];
  spectral: number[];
  unit: string | null;
  step: number | null;
}

export function usePV(path: string | undefined, hdu: number, pa: number, centre: [number, number] | null, width = 1) {
  return useQuery({
    queryKey: keys.view('pv', path ?? '', hdu, pa, JSON.stringify(centre), width),
    enabled: path !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: async ({ signal }): Promise<PVDiagram> => {
      const query = { hdu, pa, width, ...(centre ? { x: centre[0], y: centre[1] } : {}) };
      const frame = decodeFrame(await data(api.GET('/api/v1/views/pv/{path}', { signal, params: { path: { path: path ?? '' }, query }, parseAs: 'arrayBuffer' })));
      const header = frame.header as { positions: number[]; spectral: number[]; unit: string | null; step: number | null };
      return { values: frame.arrays[0].values, positions: header.positions, spectral: header.spectral, unit: header.unit, step: header.step };
    },
  });
}

// The cube of a galaxy rendered (see the server's /views/galaxy-cube): the
// path the views of cubes take it at, its shape (x, y, channels) and the
// unit of its channels; made again as the galaxy or its values change
export function useGalaxyCube(
  project: string,
  galaxy: Record<string, unknown> | undefined,
  properties: Record<string, unknown>,
  view: 'sky' | 'face-on' | 'edge-on',
  size = 160,
) {
  const label = useContext(RenderLabel);
  return useQuery({
    queryKey: keys.view('galaxy-cube', project, JSON.stringify(galaxy ?? null), JSON.stringify(properties), view, size),
    enabled: galaxy !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
    // (a cube is kept by the server for a while only)
    gcTime: 0,
    queryFn: ({ signal }) => data(api.POST('/api/v1/views/galaxy-cube', { signal, body: { project, galaxy: galaxy ?? {}, properties, view, size, label } })),
  });
}

export function useGalaxyRender(
  project: string,
  galaxy: Record<string, unknown> | undefined,
  properties: Record<string, unknown>,
  view: 'sky' | 'face-on' | 'edge-on',
  size = 160,
) {
  const label = useContext(RenderLabel);
  return useQuery({
    queryKey: keys.view('galaxy', project, JSON.stringify(galaxy ?? null), JSON.stringify(properties), view, size),
    enabled: galaxy !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: async ({ signal }): Promise<GalaxyRender> => {
      const frame = decodeFrame(
        await data(
          api.POST('/api/v1/views/galaxy', {
            signal,
            body: { project, galaxy: galaxy ?? {}, properties, view, size, label },
            parseAs: 'arrayBuffer',
          }),
        ),
      );
      const header = frame.header as { size: number; step: number; curves: GalaxyRender['curves']; problems: string[]; extras?: GalaxyExtraInfo[] };
      const maps = Object.fromEntries(frame.arrays.map((a) => [String(a.info.name), a.values]));
      return { maps, size: header.size, step: header.step, curves: header.curves, problems: header.problems, extras: header.extras ?? [] };
    },
  });
}

// A galaxy as an observation of its project sees it (see the server's
// /views/observed): its model of each item (and the data of the
// observation, if it has them), each kept by the server for the views of
// cubes; the type of the observation; and what the library noted. Asked
// again as the galaxy, its values or the observation change.
// A setup of a project of the given options (e.g. as it is edited), as
// the library describes it (see the server's /setups/describe): its
// parameters, each with the properties its layers give it and the file of
// the layer that does, and its problems; the last stays while the next
// comes
export function useSetupDescription(path: string, setup: Record<string, unknown> | undefined) {
  return useQuery({
    queryKey: keys.view('setup', path, JSON.stringify(setup ?? null)),
    enabled: setup !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: ({ signal }) => data(api.POST('/api/v1/setups/describe', { signal, body: { path, setup: setup ?? {} } })),
  });
}

// A setup of a project of the given options at the starts of its
// parameters, through one of its observations (see the server's
// /views/setup-observed): as useObserved
export function useSetupObserved(path: string, setup: Record<string, unknown> | undefined, observation: number) {
  return useQuery({
    queryKey: keys.view('setup-observed', path, JSON.stringify(setup ?? null), observation),
    enabled: setup !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
    gcTime: 0,
    queryFn: ({ signal }) => data(api.POST('/api/v1/views/setup-observed', { signal, body: { path, setup: setup ?? {}, observation } })),
  });
}

export function useObserved(
  project: string,
  galaxy: Record<string, unknown> | undefined,
  properties: Record<string, unknown>,
  observation: string | null,
  // (the options of the observation, so that it is asked again as they
  // change)
  seen: unknown,
  // (what it is of, for the Activity drawer: else as the screen says)
  of?: string,
) {
  const screen = useContext(RenderLabel);
  const label = of ?? screen;
  return useQuery({
    queryKey: keys.view('observed', project, JSON.stringify(galaxy ?? null), JSON.stringify(properties), observation ?? '', JSON.stringify(seen ?? null)),
    enabled: galaxy !== undefined && observation !== null,
    placeholderData: keepPreviousData,
    retry: false,
    // (its arrays are kept by the server for a while only)
    gcTime: 0,
    queryFn: ({ signal }) => data(api.POST('/api/v1/views/observed', { signal, body: { project, galaxy: galaxy ?? {}, properties, observation: observation ?? '', label } })),
  });
}
