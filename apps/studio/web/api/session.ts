// A live session (see the server's /sessions): a configuration set up once
// on the server, evaluated at the values of its free parameters as they
// change. One evaluation is in flight at a time: values set meanwhile
// replace each other, and the latest goes when the result comes back.
import { useCallback, useEffect, useRef, useState } from 'react';
import { api, data, tokenUrl, type Session } from './client';
import { decode, type Frame } from './frames';

export interface LiveSession {
  config: string | null;
  info: Session | null;
  values: Record<string, number>;
  frame: Frame | null;
  error: string | null;
  starting: boolean;
  start: (config: string) => void;
  // Set the configuration up again (e.g. after it was saved)
  restart: () => void;
  stop: () => void;
  set: (name: string, value: number) => void;
}

export function useLiveSession(): LiveSession {
  const [config, setConfig] = useState<string | null>(null);
  const [generation, setGeneration] = useState(0);
  const [info, setInfo] = useState<Session | null>(null);
  const [values, setValues] = useState<Record<string, number>>({});
  const [frame, setFrame] = useState<Frame | null>(null);
  const [error, setError] = useState<string | null>(null);
  const socket = useRef<WebSocket | null>(null);
  const busy = useRef(false);
  const pending = useRef<Record<string, number> | null>(null);

  const send = useCallback((update: Record<string, number>) => {
    const ws = socket.current;
    if (ws === null || ws.readyState !== WebSocket.OPEN) return;
    if (busy.current) {
      pending.current = { ...(pending.current ?? {}), ...update };
      return;
    }
    busy.current = true;
    ws.send(JSON.stringify({ values: update }));
  }, []);

  useEffect(() => {
    if (config === null) return;
    let id: string | null = null;
    let closed = false;
    setInfo(null);
    setFrame(null);
    setError(null);
    data(api.POST('/api/v1/sessions', { body: { config } }))
      .then((session) => {
        id = session.id;
        if (closed) return void closeSession(id);
        setInfo(session);
        setValues(
          Object.fromEntries(
            Object.entries(session.free).map(([name, p]) => [name, Number(p.value)]),
          ),
        );
        const ws = new WebSocket(tokenUrl(`/api/v1/sessions/${session.id}/evaluate`, 'ws'));
        ws.binaryType = 'arraybuffer';
        ws.onmessage = (event: MessageEvent<ArrayBuffer>) => {
          const result = decode(event.data);
          if (result.header.type === 'error') setError(String(result.header.message));
          else {
            setFrame(result);
            setError(null);
          }
          busy.current = false;
          if (pending.current !== null) {
            const next = pending.current;
            pending.current = null;
            send(next);
          }
        };
        ws.onopen = () => send({});
        socket.current = ws;
      })
      .catch((e: Error) => setError(e.message));
    return () => {
      closed = true;
      socket.current?.close();
      socket.current = null;
      busy.current = false;
      pending.current = null;
      if (id !== null) void closeSession(id);
    };
  }, [config, generation]);

  const set = useCallback(
    (name: string, value: number) => {
      setValues((current) => ({ ...current, [name]: value }));
      send({ [name]: value });
    },
    [send],
  );

  return {
    config,
    info,
    values,
    frame,
    error,
    starting: config !== null && info === null && error === null,
    start: setConfig,
    restart: () => setGeneration((g) => g + 1),
    stop: () => setConfig(null),
    set,
  };
}

async function closeSession(id: string): Promise<void> {
  await api.DELETE('/api/v1/sessions/{id_}', { params: { path: { id_: id } } });
}
