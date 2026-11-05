// What the page reads from the server and changes on it, with the keys of
// its cache (which the events of the server refresh, see events.ts).
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, data } from './client';
import { decodeFrame } from './frames';
import type { components } from './schema';

export const keys = {
  projects: ['projects'],
  project: (path: string) => ['project', path],
  piece: (path: string) => ['piece', path],
  folder: (path: string) => ['folder', path],
  description: (path: string, velocity: string | null) => ['description', path, velocity],
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

// The arrays of a file of data, from its headers, its ambiguous velocities
// read by a convention, if given
export function useDescription(path: string | null, velocity: Velocity | null = null) {
  return useQuery({
    queryKey: keys.description(path ?? '', velocity),
    enabled: path !== null,
    // (the description of the same file stays while another convention is
    // read; never that of another file)
    placeholderData: (previous, query) => (query?.queryKey[1] === path ? previous : undefined),
    queryFn: () =>
      data(
        api.GET('/api/v1/describe/{path}', {
          params: { path: { path: path ?? '' }, query: velocity ? { velocity } : {} },
        }),
      ),
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
    queryFn: async () => {
      const frame = decodeFrame(
        await data(
          api.GET('/api/v1/views/channel/{path}', {
            params: { path: { path: path ?? '' }, query: { hdu, channel } },
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
    },
  });
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

// The quick moment maps of a cube (its flux, velocity, dispersion,
// skewness and kurtosis, by name), of the voxels where the cube, smoothed,
// is above a signal-to-noise ratio (0: all of them)
export function useMoments(path: string, hdu: number, snr: number, enabled: boolean) {
  return useQuery({
    queryKey: keys.view('moments', path, hdu, snr),
    enabled,
    placeholderData: keepPreviousData,
    queryFn: async () =>
      decodeFrame(
        await data(
          api.GET('/api/v1/views/moments/{path}', {
            params: { path: { path }, query: { hdu, snr } },
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

// An LSF of options of a block of a project drawn on channels of a width
// (km/s; null: of its own scale), with its width
export function useLsfView(project: string, options: Record<string, unknown>, step: number | null, enabled = true) {
  return useQuery({
    queryKey: keys.view('lsf', project, JSON.stringify(options), step ?? ''),
    enabled,
    placeholderData: keepPreviousData,
    retry: false,
    queryFn: () => data(api.POST('/api/v1/views/lsf', { body: { project, options, step } })),
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
