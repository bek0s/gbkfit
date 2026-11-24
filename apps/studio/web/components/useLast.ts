// The latest of a value that was given (e.g. the last picture drawn,
// while the next comes, or while what it is of cannot be drawn), so that
// a screen keeps what it shows rather than blanking it.
import { useRef } from 'react';

export function useLast<T>(value: T | undefined): T | undefined {
  const last = useRef(value);
  if (value !== undefined) last.current = value;
  return last.current;
}
