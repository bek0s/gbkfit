// The events of the server (its route /events): what changed, so that the
// page asks again, and the preferences of the user, as they changed (see
// shell/preferences.ts). Returns whether they are connected.
import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { loadPreferences, preferencesChanged } from '../shell/preferences';
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
      void loadPreferences();
    };
    source.onerror = () => setConnected(false);
    // (a job changes state; a run appears in its project)
    const changed = () => {
      void client.invalidateQueries({ queryKey: keys.jobs });
      void client.invalidateQueries({ queryKey: ['project'] });
    };
    source.addEventListener('job', changed);
    source.addEventListener('run', changed);
    source.addEventListener('preferences', (event) => preferencesChanged((JSON.parse(event.data) as { values: Record<string, unknown> }).values));
    return () => source.close();
  }, [client]);
  return connected;
}
