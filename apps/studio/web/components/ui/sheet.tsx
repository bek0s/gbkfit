// The body of a screen that edits a block: its sheet (the dock of its
// options, see dock.tsx) beside its picture (e.g. the viewer of a
// dataset), the sheet on the side the user prefers (Settings › Layout;
// the left by default). The two swap in the page, not only on the
// screen, so that the keyboard (Tab) and readers of screens follow them.
import type { CSSProperties, ReactNode } from 'react';
import { setPreference, usePreference } from '../../shell/preferences';

export type SheetSide = 'left' | 'right';

const isSide = (value: unknown): value is SheetSide => value === 'left' || value === 'right';

// The side of the sheets, as the user prefers it
export function useSheetSide(): SheetSide {
  return usePreference('sheet-side', isSide, 'left');
}

export function setSheetSide(side: SheetSide): void {
  setPreference('sheet-side', side);
}

export function SheetLayout({
  sheet,
  picture,
  width = 400,
  className,
}: {
  sheet: ReactNode;
  picture: ReactNode;
  // The width of the sheet (px)
  width?: number;
  className?: string;
}) {
  const side = useSheetSide();
  return (
    <div className={`sheet-layout ${className ?? ''}`} data-side={side} style={{ '--sheet': `${width}px` } as CSSProperties}>
      {side === 'left' ? (
        <>
          {sheet}
          {picture}
        </>
      ) : (
        <>
          {picture}
          {sheet}
        </>
      )}
    </div>
  );
}
