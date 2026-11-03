// Two panes side by side (a Splitter of Mantine; see theme.ts): the main
// one, which takes the space left, and one at its right, of a width that
// a drag of the gap between them changes (or a double click resets). The
// gap is that between the cards of the page, which shows a line as it is
// hovered, dragged or focused. With a least width of the main pane, too
// narrow for both (the main at its least, the side at its width), the side
// pane goes under the main one, which fills the height, and they scroll as
// one.
import { Splitter } from '@mantine/core';
import { useEffect, useRef, useState, type ReactNode } from 'react';

export function SidePanes({
  main,
  side,
  width = 400,
  least = 300,
  most = 640,
  mainLeast,
}: {
  main: ReactNode;
  side: ReactNode;
  // The width of the side pane at first, and its least and most (px)
  width?: number;
  least?: number;
  most?: number;
  // The least width of the main pane (px; none: 40% of both)
  mainLeast?: number;
}) {
  // (the width of both, of whether the side pane goes under the main one)
  const box = useRef<HTMLDivElement>(null);
  const [space, setSpace] = useState<number | null>(null);
  useEffect(() => {
    const element = box.current;
    if (!element || mainLeast === undefined) return;
    const observer = new ResizeObserver(([entry]) => setSpace(entry.contentRect.width));
    observer.observe(element);
    return () => observer.disconnect();
  }, [mainLeast]);
  const stacked = mainLeast !== undefined && space !== null && space < mainLeast + width + 14;
  if (stacked) {
    return (
      <div ref={box} className="ui-panes-stacked">
        <div className="ui-panes-stacked-main">{main}</div>
        {side}
      </div>
    );
  }
  return (
    <div ref={box} className="ui-panes-box">
      <Splitter className="ui-panes" lineSize={14} withHandle={false}>
        <Splitter.Pane defaultSize="100%" min={mainLeast === undefined ? '40%' : `${mainLeast}px`}>
          {main}
        </Splitter.Pane>
        <Splitter.Pane defaultSize={`${width}px`} min={`${least}px`} max={`${most}px`}>
          {side}
        </Splitter.Pane>
      </Splitter>
    </div>
  );
}
