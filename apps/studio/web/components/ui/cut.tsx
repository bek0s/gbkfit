// Text cut to a width with '…', all of it on hover once it is cut: at its
// end (e.g. a name in a cell of a table), or in its middle (e.g. the name
// of a file, whose end tells it from others alike: 'NGC2347.V…cube.fits').
// Text that is cut is never wrapped.
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';

// Cut at its end to a width (characters of its font). What it shows is its
// title, or else what it holds (e.g. parts of it in other colours).
export function Cut({ max, title, className, children }: { max: number; title: string; className?: string; children?: ReactNode }) {
  const [cut, setCut] = useState(false);
  return (
    <span
      className={['ui-cut', className].filter(Boolean).join(' ')}
      style={{ maxWidth: `${max}ch` }}
      title={cut ? title : undefined}
      onMouseEnter={(event) => setCut(event.currentTarget.scrollWidth > event.currentTarget.clientWidth)}
    >
      {children ?? title}
    </span>
  );
}

// Cut in its middle to the width its place gives it: as wide as all of it,
// at most (e.g. a tag), or as wide as its place (e.g. the rest of a row);
// cut again as that width changes. Its title is all of it once it is cut,
// unless one is given (e.g. the path of a file, always).
export function MiddleCut({ text, title, className }: { text: string; title?: string; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [shown, setShown] = useState(text);
  useLayoutEffect(() => {
    const element = ref.current!;
    const fit = () => setShown(middleCut(text, element.clientWidth, getComputedStyle(element).font));
    fit();
    // (again once the fonts are in: those of the page may come after it)
    let live = true;
    document.fonts.ready.then(() => live && fit());
    const observer = new ResizeObserver(fit);
    observer.observe(element);
    return () => {
      live = false;
      observer.disconnect();
    };
  }, [text]);
  return (
    <span ref={ref} className={['ui-middle-cut', className].filter(Boolean).join(' ')} title={title ?? (shown === text ? undefined : text)}>
      {/* (all of it, of no height: it makes the place as wide as all of
          it, at most, and is what is read aloud) */}
      <span className="ui-middle-cut-all">{text}</span>
      <span aria-hidden>{shown}</span>
    </span>
  );
}

// The most of a text, its start and its end with '…' between them, that
// is at most a width (px) in a font (as CSS gives it)
function middleCut(text: string, width: number, font: string): string {
  if (widthOf(text, font) <= width) return text;
  // (the most characters kept that fit; '…' alone, if none)
  let [fit, over] = [0, text.length];
  while (over - fit > 1) {
    const count = Math.floor((fit + over) / 2);
    if (widthOf(keptOf(text, count), font) <= width) fit = count;
    else over = count;
  }
  return keptOf(text, fit);
}

// (of a text, a count of characters: half of its start, half of its end,
// one more of the end if odd)
function keptOf(text: string, count: number): string {
  const start = Math.floor(count / 2);
  return `${text.slice(0, start)}…${text.slice(text.length - (count - start))}`;
}

let context: CanvasRenderingContext2D | null = null;

function widthOf(text: string, font: string): number {
  context ??= document.createElement('canvas').getContext('2d')!;
  context.font = font;
  return context.measureText(text).width;
}
