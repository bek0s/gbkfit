// The events of the server (see its route /events): jobs whose state
// changed, runs that changed, and the progress of fits. Queries of what
// changed are refreshed; the reports of progress are kept, by job.
import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useState, useSyncExternalStore } from 'react';
import { tokenUrl } from './client';

export interface ProgressReport {
  stage: string;
  nevals: number;
  elapsed: number;
  chi2: number | null;
  best_chi2: number | null;
  best: Record<string, number> | null;
  final: boolean;
}

const reports = new Map<string, ProgressReport[]>();
const listeners = new Set<() => void>();

function addReport(job: string, report: ProgressReport): void {
  reports.set(job, [...(reports.get(job) ?? []), report]);
  listeners.forEach((listener) => listener());
}

// The reports of the progress of a job so far
export function useProgress(job: string): ProgressReport[] {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => reports.get(job) ?? EMPTY,
  );
}

const EMPTY: ProgressReport[] = [];

// Follow the events while mounted; return whether they are connected.
export function useServerEvents(): boolean {
  const client = useQueryClient();
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    const source = new EventSource(tokenUrl('/api/v1/events'));
    source.onopen = () => {
      setConnected(true);
      // (what changed while disconnected)
      void client.invalidateQueries();
    };
    source.onerror = () => setConnected(false);
    source.addEventListener('job', () => {
      void client.invalidateQueries({ queryKey: ['jobs'] });
      void client.invalidateQueries({ queryKey: ['project'] });
    });
    source.addEventListener('run', (event) => {
      const { run } = JSON.parse((event as MessageEvent<string>).data) as { run: string };
      void client.invalidateQueries({ queryKey: ['run', run] });
      // (a new run appears in its project)
      void client.invalidateQueries({ queryKey: ['project'] });
    });
    source.addEventListener('progress', (event) => {
      const { job, report } = JSON.parse((event as MessageEvent<string>).data) as {
        job: string;
        report: ProgressReport;
      };
      addReport(job, report);
    });
    return () => source.close();
  }, [client]);
  return connected;
}
