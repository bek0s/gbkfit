// The check of the text of a file as it is edited (see the server's
// /check): a second after the last change, one check at a time, the
// latest text next.
import { useEffect, useRef, useState } from 'react';
import { api, data } from './client';
import type { components } from './schema';

export type Diagnostic = components['schemas']['Diagnostic'];

// The wait after the last change (ms)
const DELAY = 1000;

export interface Check {
  diagnostics: Diagnostic[];
  checking: boolean;
  error: string | null;
}

export function useCheck(path: string, text: string | null): Check {
  const [diagnostics, setDiagnostics] = useState<Diagnostic[]>([]);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const busy = useRef(false);
  const next = useRef<string | null>(null);

  useEffect(() => {
    if (text === null) return;
    const run = (checked: string) => {
      busy.current = true;
      setChecking(true);
      data(api.POST('/api/v1/check', { body: { path, text: checked } }))
        .then((result) => {
          setDiagnostics(result.diagnostics);
          setError(null);
        })
        .catch((e: Error) => setError(e.message))
        .finally(() => {
          busy.current = false;
          setChecking(false);
          if (next.current !== null) {
            const latest = next.current;
            next.current = null;
            run(latest);
          }
        });
    };
    const timer = setTimeout(() => {
      if (busy.current) next.current = text;
      else run(text);
    }, DELAY);
    return () => clearTimeout(timer);
  }, [path, text]);

  return { diagnostics, checking, error };
}
