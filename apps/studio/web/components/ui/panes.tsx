// Two panes side by side (a Splitter of Mantine; see theme.ts): the main
// one, which takes the space left, and one at its right, of a width that
// a drag of the gap between them changes (or a double click resets), and
// that folds away (toggle, or dragged below its least width). The gap is
// that between the cards of the page, which shows a line as it is
// hovered, dragged or focused.
import { Splitter } from '@mantine/core';
import type { UseSplitterReturnValue } from '@mantine/hooks';
import { useRef, useState, type ReactNode } from 'react';

export interface PanesControl {
  // Whether the side pane is folded, and how it folds or unfolds
  folded: boolean;
  toggle: () => void;
}

export function SidePanes({
  main,
  side,
  width = 400,
  least = 300,
  most = 640,
}: {
  // The main pane, given how the side one folds
  main: (control: PanesControl) => ReactNode;
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
        {main({ folded, toggle: () => splitter.current?.toggleCollapse(1) })}
      </Splitter.Pane>
      <Splitter.Pane defaultSize={`${width}px`} min={`${least}px`} max={`${most}px`} collapsible>
        {side}
      </Splitter.Pane>
    </Splitter>
  );
}
