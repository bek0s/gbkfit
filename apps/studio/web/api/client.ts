// The client of the API of the server, and its token. The server prints a
// link with the token (?token=...): the page takes it out of the address,
// keeps it for the tab, and sends it with every request. The types of the
// API come from its contract (../server/openapi.json; npm run api).
import createClient from 'openapi-fetch';
import type { components, paths } from './schema';

export type ProjectInfo = components['schemas']['ProjectInfo'];
export type Job = components['schemas']['JobInfo'];
export type ArrayInfo = components['schemas']['ArrayInfo'];
export type CatalogueLine = components['schemas']['CatalogueLine'];

const TOKEN_KEY = 'gbkfit-token';
let token = '';

export function takeToken(): void {
  const url = new URL(window.location.href);
  const given = url.searchParams.get('token');
  try {
    if (given !== null) sessionStorage.setItem(TOKEN_KEY, given);
    token = sessionStorage.getItem(TOKEN_KEY) ?? '';
  } catch {
    // (no storage: the token lasts as long as the page)
    token = given ?? token;
  }
  if (given !== null) {
    url.searchParams.delete('token');
    window.history.replaceState(null, '', url);
  }
}

// The URL of a path of the API with the token, for what cannot send
// headers (the events)
export function tokenUrl(path: string): string {
  const url = new URL(path, window.location.href);
  url.searchParams.set('token', token);
  return url.toString();
}

export const api = createClient<paths>({ baseUrl: '' });
api.use({
  onRequest({ request }) {
    request.headers.set('Authorization', `Bearer ${token}`);
    return request;
  },
});

// The data of a response, or its error, with the detail the server gave
export async function data<T>(request: Promise<{ data?: T; error?: unknown; response: Response }>): Promise<T> {
  const { data: value, error, response } = await request;
  if (error !== undefined || value === undefined) {
    const detail = (error as { detail?: unknown } | undefined)?.detail;
    throw new Error(typeof detail === 'string' ? detail : `${response.status} ${response.statusText}`);
  }
  return value;
}

// Send a file of this computer into a project (see the server's
// /uploads), reporting the fraction sent; resolves to the path of the
// file. (A request of XMLHttpRequest, as fetch does not report what it
// has sent.)
export function upload(project: string, file: File, onProgress: (fraction: number) => void): Promise<string> {
  const url = new URL('/api/v1/uploads', window.location.href);
  url.searchParams.set('project', project);
  url.searchParams.set('name', file.name);
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', url);
    request.setRequestHeader('Authorization', `Bearer ${token}`);
    request.setRequestHeader('Content-Type', 'application/octet-stream');
    request.upload.onprogress = (event) => event.lengthComputable && onProgress(event.loaded / event.total);
    request.onload = () => {
      const body = JSON.parse(request.responseText || 'null') as unknown;
      if (request.status === 201 && typeof body === 'string') resolve(body);
      else reject(new Error(String((body as { detail?: unknown } | null)?.detail ?? `${request.status} ${request.statusText}`)));
    };
    request.onerror = () => reject(new Error('the file could not be sent'));
    request.send(file);
  });
}
