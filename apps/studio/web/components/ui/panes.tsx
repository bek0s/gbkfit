// Two panes side by side (a Splitter of Mantine; see theme.ts): the main
// one, which takes the space left, and one at its right, of a width that
// a drag of the gap between them changes (or a double click resets), and
// that folds away (its button, or dragged below its least width). The gap
// is that between the cards of the page, which shows a line as it is
// hovered, dragged or focused; the button that folds the side pane, and
// unfolds it, is in it, at its top: by the pane it acts on, not in the
// main one.
import { Splitter } from '@mantine/core';
import type { UseSplitterReturnValue } from '@mantine/hooks';
import { useRef, useState, type ReactNode } from 'react';

export function SidePanes({
  main,
  side,
  width = 400,
  least = 300,
  most = 640,
}: {
  main: ReactNode;
  side: ReactNode;
  // The width of the side pane at first, and its least and most (px)
  width?: number;
  least?: number;
  most?: number;
}) {
  const splitter = useRef<UseSplitterReturnValue | null>(null);
  const [folded, setFolded] = useState(false);
  return (
    <Splitter
      splitterRef={splitter}
      onCollapseChange={(pane, collapsed) => pane === 1 && setFolded(collapsed)}
      className="ui-panes"
      lineSize={14}
      withHandle={false}
    >
      <Splitter.Pane defaultSize="100%" min="40%">
        {main}
      </Splitter.Pane>
      <Splitter.Pane defaultSize={`${width}px`} min={`${least}px`} max={`${most}px`} collapsible className="ui-panes-side" data-folded={folded || undefined}>
        <button
          type="button"
          className="ui-panes-fold"
          aria-label={folded ? 'Show the panels at the side' : 'Hide the panels at the side'}
          title={folded ? 'Show the panels at the side' : 'Hide the panels at the side'}
          onClick={() => splitter.current?.toggleCollapse(1)}
        >
          {folded ? '‹' : '›'}
        </button>
        {side}
      </Splitter.Pane>
    </Splitter>
  );
}
