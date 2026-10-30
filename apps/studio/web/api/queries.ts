// What the page reads from the server and changes on it, with the keys of
// its cache (which the events of the server refresh, see events.ts).
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, data } from './client';
import { decodeFrame } from './frames';

export const keys = {
  projects: ['projects'],
  project: (path: string) => ['project', path],
  piece: (path: string) => ['piece', path],
  folder: (path: string) => ['folder', path],
  description: (path: string) => ['description', path],
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

// The arrays of a file of data, from its headers
export function useDescription(path: string | null) {
  return useQuery({
    queryKey: keys.description(path ?? ''),
    enabled: path !== null,
    queryFn: () => data(api.GET('/api/v1/describe/{path}', { params: { path: { path: path ?? '' } } })),
  });
}

// A channel of a cube (an array of a file of data): its values (ny, nx),
// the number of channels of the cube, the coordinate of the channel along
// its spectral axis and its unit (null if unknown), and the range of the
// values of the cube for colours.
// The channel shown before stays while the next one comes.
export function useChannel(path: string, hdu: number, channel: number) {
  return useQuery({
    queryKey: keys.view('channel', path, hdu, channel),
    placeholderData: keepPreviousData,
    queryFn: async () => {
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
    },
  });
}

// The spectrum of a pixel of a cube, and the coordinates of its channels
// along its spectral axis, with their unit
export function useSpectrum(path: string, hdu: number, x: number, y: number) {
  return useQuery({
    queryKey: keys.view('spectrum', path, hdu, x, y),
    placeholderData: keepPreviousData,
    queryFn: () =>
      data(api.GET('/api/v1/views/spectrum/{path}', { params: { path: { path }, query: { hdu, x, y } } })),
  });
}

// The quick moment maps of a cube (flux, velocity and dispersion), by name
export function useMoments(path: string, hdu: number, enabled: boolean) {
  return useQuery({
    queryKey: keys.view('moments', path, hdu),
    enabled,
    queryFn: async () =>
      decodeFrame(
        await data(
          api.GET('/api/v1/views/moments/{path}', {
            params: { path: { path }, query: { hdu } },
            parseAs: 'arrayBuffer',
          }),
        ),
      ).arrays,
  });
}

export function useJobs() {
  return useQuery({ queryKey: keys.jobs, queryFn: () => data(api.GET('/api/v1/jobs')) });
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
