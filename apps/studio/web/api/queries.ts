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
