// Motion of the interface: content that opens and closes by its height.
import { useEffect, useState, type ReactNode } from 'react';

// The time of the closing of an Expand (ms), as its style draws it (see
// .ui-expand in styles.css)
const CLOSING_MS = 160;

// Content that opens from no height to its own, fading in, and closes back
// (kept while it closes, then gone); with reduced motion, at once
export function Expand({ open, children }: { open: boolean; children: ReactNode }) {
  const [closing, setClosing] = useState(false);
  const [shown, setShown] = useState(open);
  useEffect(() => {
    if (open) {
      setShown(true);
      setClosing(false);
      return;
    }
    if (!shown) return;
    setClosing(true);
    const timer = setTimeout(() => {
      setShown(false);
      setClosing(false);
    }, CLOSING_MS);
    return () => clearTimeout(timer);
  }, [open]);
  if (!open && !shown) return null;
  return (
    <div className="ui-expand" data-closing={(!open && closing) || undefined}>
      <div className="ui-expand-inner">{children}</div>
    </div>
  );
}
