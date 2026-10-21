// Follow the fit of the job that the live session watches (see
// LiveSession.watch): evaluate the session at the best values of each new
// report of the progress of the fit.
import { useEffect } from 'react';
import { useProgress } from './events';
import type { LiveSession } from './session';

export function useFitWatch(live: LiveSession): void {
  const reports = useProgress(live.watching ?? '');
  const best = reports.at(-1)?.best ?? null;
  const ready = live.info !== null;
  const { set, watching } = live;
  useEffect(() => {
    if (watching !== null && ready && best !== null) set(best);
  }, [best, ready, set, watching]);
}
