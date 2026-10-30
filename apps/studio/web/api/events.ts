// The events of the server (its route /events): what changed, so that the
// page asks again. Returns whether they are connected.
import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { tokenUrl } from './client';
import { keys } from './queries';

export function useServerEvents(): boolean {
  const client = useQueryClient();
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    const source = new EventSource(tokenUrl('/api/v1/events'));
    source.onopen = () => {
      setConnected(true);
      // (what changed while not connected)
      void client.invalidateQueries();
    };
    source.onerror = () => setConnected(false);
    // (a job changes state; a run appears in its project)
    const changed = () => {
      void client.invalidateQueries({ queryKey: keys.jobs });
      void client.invalidateQueries({ queryKey: ['project'] });
    };
    source.addEventListener('job', changed);
    source.addEventListener('run', changed);
    return () => source.close();
  }, [client]);
  return connected;
}
