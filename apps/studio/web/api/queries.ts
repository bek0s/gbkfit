// What the page reads from the server and changes on it, with the keys of
// its cache (which the events of the server refresh, see events.ts).
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, data } from './client';

export const keys = {
  projects: ['projects'],
  project: (path: string) => ['project', path],
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

// Make a block of a project from the template of its folder
export function useCreateBlock(project: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ folder, name }: { folder: string; name: string }) =>
      data(api.POST('/api/v1/pieces', { body: { project, folder, name } })),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.project(project) }),
  });
}
