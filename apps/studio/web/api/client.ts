// The client of the API of the server, behind its token. Its types come
// from the contract of the API (../server/openapi.json; npm run api).
import createClient from 'openapi-fetch';
import type { components, paths } from './schema';

export type Project = components['schemas']['ProjectInfo'];
export type Run = components['schemas']['RunInfo'];
export type Job = components['schemas']['JobInfo'];
export type JobRequest = components['schemas']['JobRequest'];
export type Session = components['schemas']['SessionInfo'];

const TOKEN_KEY = 'gbkfit-token';

let memoryToken = '';

// Take the token of the link that the server printed (?token=...), keep it
// for this tab, and take it out of the address.
export function takeToken(): void {
  const url = new URL(window.location.href);
  const given = url.searchParams.get('token');
  if (given === null) return;
  memoryToken = given;
  try {
    sessionStorage.setItem(TOKEN_KEY, given);
  } catch {
    // (no storage: the token lasts as long as the page)
  }
  url.searchParams.delete('token');
  window.history.replaceState(null, '', url);
}

export function token(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? memoryToken;
  } catch {
    return memoryToken;
  }
}

// The URL of a path of the API with the token in its query, for what
// cannot send headers (events, WebSockets).
export function tokenUrl(path: string, protocol: 'http' | 'ws' = 'http'): string {
  const url = new URL(path, window.location.href);
  url.searchParams.set('token', token());
  if (protocol === 'ws') url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}

export const api = createClient<paths>({ baseUrl: '' });
api.use({
  onRequest({ request }) {
    request.headers.set('Authorization', `Bearer ${token()}`);
    return request;
  },
});

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

type Result<T> = Promise<{ data?: T; error?: unknown; response: Response }>;

// Return the data of a response of the client, or throw its error (with
// the detail the server gave).
export async function data<T>(result: Result<T>): Promise<T> {
  const { data, response } = await done(result);
  if (data === undefined) throw new ApiError(response.status, 'the response has no data');
  return data;
}

// Wait for a response of the client (e.g. without content); throw its
// error, if any.
export async function done<T>(result: Result<T>): Promise<Awaited<Result<T>>> {
  const response = await result;
  if (response.error !== undefined) {
    const detail = (response.error as { detail?: unknown }).detail;
    throw new ApiError(
      response.response.status,
      typeof detail === 'string' ? detail : `${response.response.status} ${response.response.statusText}`,
    );
  }
  return response;
}
