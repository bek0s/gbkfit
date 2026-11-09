// What the page reads from the server and changes on it, with the keys of
// its cache (which the events of the server refresh, see events.ts).
import { keepPreviousData, useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { useRef } from 'react';
import { api, data } from './client';
import { decodeFrame } from './frames';
import type { components, operations } from './schema';

export const keys = {
  projects: ['projects'],
  project: (path: string) => ['project', path],
  piece: (path: string) => ['piece', path],
  folder: (path: string) => ['folder', path],
  description: (path: string, velocity: string | null, rest: string | null = null, type: string | null = null) => ['description', path, velocity, rest, type],
  preparation: (path: string) => ['preparation', path],
  view: (...parts: (string | number)[]) => ['view', ...parts],
  jobs: ['jobs'],
};

// The paths of the projects of the workspace
export function useProjects() {
  return useQuery({ queryKey: keys.projects, queryFn: () => data(api.GET('/api/v1/projects')) });
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

// The arrays of a file of data, from its headers, its ambiguous velocities
// read by a convention, its wavelengths or frequencies as velocities of a
// rest, and its spectral axis of a type FITS does not define as what it
// is, if given
export function useDescription(path: string | null, velocity: Velocity | null = null, rest: string | null = null, type: SpectralTypeOption | null = null) {
  return useQuery({
    queryKey: keys.description(path ?? '', velocity, rest, type),
    enabled: path !== null,
    // (the description of the same file stays while another convention or
    // rest is read; never that of another file)
    placeholderData: (previous, query) => (query?.queryKey[1] === path ? previous : undefined),
    queryFn: () => describe(path ?? '', velocity, rest, type),
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
export function useDescriptions(paths: string[], velocity: Velocity | null = null, rest: string | null = null, type: SpectralTypeOption | null = null) {
  const unique = [...new Set(paths)];
  const results = useQueries({
    queries: unique.map((path) => ({
      queryKey: keys.description(path, velocity, rest, type),
      queryFn: () => describe(path, velocity, rest, type),
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

function describe(path: string, velocity: Velocity | null, rest: string | null, type: SpectralTypeOption | null = null) {
  const query: { velocity?: Velocity; rest?: string; spectral_type?: SpectralTypeOption } = {};
  if (velocity) query.velocity = velocity;
  if (rest) query.rest = rest;
  if (type) query.spectral_type = type;
  return data(api.GET('/api/v1/describe/{path}', { params: { path: { path }, query } }));
}

// How a file of data was prepared by the server (null: it was not)
export function usePreparation(path: string | null) {
  return useQuery({
    queryKey: keys.preparation(path ?? ''),
    enabled: path !== null,
    queryFn: () => data(api.GET('/api/v1/preparations/{path}', { params: { path: { path: path ?? '' } } })),
  });
}

// A channel of a cube (an array of a file of data): its values (ny, nx),
// the number of channels of the cube, the coordinate of the channel along
// its spectral axis and its unit (null if unknown), and the range of the
// values of the cube for colours.
// The channel shown before stays while the next one comes; without a
// path, none.
export function useChannel(path: string | null, hdu: number, channel: number) {
  return useQuery({
    queryKey: keys.view('channel', path ?? '', hdu, channel),
    enabled: path !== null,
    placeholderData: keepPreviousData,
    queryFn: () => readChannel(path ?? '', hdu, channel),
  });
}

// The values of maps (arrays of files of data, as the first channel of
// useChannel), in their order (undefined until read)
export function useMaps(maps: { path: string; hdu: number }[]) {
  return useQueries({
    queries: maps.map(({ path, hdu }) => ({
      queryKey: keys.view('channel', path, hdu, 0),
      queryFn: () => readChannel(path, hdu, 0),
    })),
  }).map((result) => result.data?.plane.values);
}

async function readChannel(path: string, hdu: number, channel: number) {
  const frame = decodeFrame(
    await data(
      api.GET('/api/v1/views/channel/{path}', {
        params: { path: { path }, query: { hdu, channel } },
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
// spectral axis, with their unit
export function useSpectrum(
  path: string,
  hdu: number,
  pixel: [number, number] | null,
  region: [number, number, number, number] | null,
) {
  const query =
    pixel !== null
      ? { hdu, x: pixel[0], y: pixel[1] }
      : region !== null
        ? { hdu, x0: region[0], x1: region[1], y0: region[2], y1: region[3] }
        : { hdu };
  return useQuery({
    queryKey: keys.view('spectrum', path, hdu, pixel?.join() ?? '', region?.join() ?? ''),
    placeholderData: keepPreviousData,
    queryFn: () => data(api.GET('/api/v1/views/spectrum/{path}', { params: { path: { path }, query } })),
  });
}

// The quick moment maps of a cube, by name, of the voxels where the cube,
// smoothed, is above a signal-to-noise ratio (0: all of them): measured as
// moments, its flux, velocity, dispersion, skewness and kurtosis; or of a
// Gauss-Hermite series fitted to each spectrum, its h3 and h4
export function useMoments(path: string, hdu: number, snr: number, enabled: boolean, method: 'moments' | 'gauss_hermite' = 'moments') {
  return useQuery({
    queryKey: keys.view('moments', path, hdu, snr, method),
    enabled,
    placeholderData: keepPreviousData,
    queryFn: async () =>
      decodeFrame(
        await data(
          api.GET('/api/v1/views/moments/{path}', {
            params: { path: { path }, query: { hdu, snr, method } },
            parseAs: 'arrayBuffer',
          }),
        ),
      ).arrays,
  });
}

export function useJobs() {
  return useQuery({ queryKey: keys.jobs, queryFn: () => data(api.GET('/api/v1/jobs')) });
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
      if (!response.ok) throw new Error(String((error as { detail?: unknown } | undefined)?.detail ?? response.statusText));
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
      if (!response.ok) throw new Error(String((error as { detail?: unknown } | undefined)?.detail ?? response.statusText));
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

// Rename a block of a project (by its path), and the blocks that name it
// with it; resolves to its new path
export function useRenameBlock(project: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ path, name }: { path: string; name: string }) =>
      data(api.PATCH('/api/v1/pieces/{path}', { params: { path: { path } }, body: { name } })),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: keys.project(project) });
      client.invalidateQueries({ queryKey: ['piece'] });
      client.invalidateQueries({ queryKey: ['references'] });
    },
  });
}

// Delete a block of a project (by its path)
export function useDeleteBlock(project: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (path: string) => {
      const { error, response } = await api.DELETE('/api/v1/pieces/{path}', { params: { path: { path } } });
      if (!response.ok) throw new Error(String((error as { detail?: unknown } | undefined)?.detail ?? response.statusText));
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

// The traits each type of component takes (see the server's /components):
// by its type, the types of its traits of each kind (e.g. 'vpt'), and the
// directions of its velocity and dispersion traits
export function useComponentTraits() {
  return useQuery({ queryKey: ['components'], staleTime: Infinity, queryFn: () => data(api.GET('/api/v1/components')) });
}

// Write the options of a block of a project (by its path)
export function useUpdateBlock() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async ({ path, content }: { path: string; content: Record<string, unknown> }) => {
      const { error, response } = await api.PUT('/api/v1/pieces/{path}', { params: { path: { path } }, body: content });
      if (!response.ok) throw new Error(String((error as { detail?: unknown } | undefined)?.detail ?? response.statusText));
    },
    onSuccess: (_, { path }) => client.invalidateQueries({ queryKey: keys.piece(path) }),
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
    queryFn: () => data(api.POST('/api/v1/views/psf', { body: { project, options, step } })),
  });
}

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
    queryFn: () => data(api.POST('/api/v1/views/lsf', { body: { project, options, ...at } })),
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
    queryFn: () => data(api.POST('/api/v1/parameters', { body: { project, galaxy: galaxy ?? {} } })),
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
  return useQuery({
    queryKey: keys.view('galaxy-cube', project, JSON.stringify(galaxy ?? null), JSON.stringify(properties), view, size),
    enabled: galaxy !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
    // (a cube is kept by the server for a while only)
    gcTime: 0,
    queryFn: () => data(api.POST('/api/v1/views/galaxy-cube', { body: { project, galaxy: galaxy ?? {}, properties, view, size } })),
  });
}

export function useGalaxyRender(
  project: string,
  galaxy: Record<string, unknown> | undefined,
  properties: Record<string, unknown>,
  view: 'sky' | 'face-on' | 'edge-on',
  size = 160,
) {
  return useQuery({
    queryKey: keys.view('galaxy', project, JSON.stringify(galaxy ?? null), JSON.stringify(properties), view, size),
    enabled: galaxy !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: async (): Promise<GalaxyRender> => {
      const frame = decodeFrame(
        await data(
          api.POST('/api/v1/views/galaxy', {
            body: { project, galaxy: galaxy ?? {}, properties, view, size },
            parseAs: 'arrayBuffer',
          }),
        ),
      );
      const header = frame.header as { size: number; step: number; curves: GalaxyRender['curves']; problems: string[] };
      const maps = Object.fromEntries(frame.arrays.map((a) => [String(a.info.name), a.values]));
      return { maps, size: header.size, step: header.step, curves: header.curves, problems: header.problems };
    },
  });
}

// A galaxy as an observation of its project sees it (see the server's
// /views/observed): its model of each item (and the data of the
// observation, if it has them), each kept by the server for the views of
// cubes; the type of the observable; and what the library noted. Asked
// again as the galaxy, its values or the observation change.
export function useObserved(
  project: string,
  galaxy: Record<string, unknown> | undefined,
  properties: Record<string, unknown>,
  observation: string | null,
  // (the options of the observation, so that it is asked again as they
  // change)
  seen: unknown,
) {
  return useQuery({
    queryKey: keys.view('observed', project, JSON.stringify(galaxy ?? null), JSON.stringify(properties), observation ?? '', JSON.stringify(seen ?? null)),
    enabled: galaxy !== undefined && observation !== null,
    placeholderData: keepPreviousData,
    retry: false,
    // (its arrays are kept by the server for a while only)
    gcTime: 0,
    queryFn: () => data(api.POST('/api/v1/views/observed', { body: { project, galaxy: galaxy ?? {}, properties, observation: observation ?? '' } })),
  });
}
