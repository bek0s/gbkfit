// The first step of adding data: what kind of data it is. The dataset
// screen then asks for its files (see screens/dataset).
import { NavItem } from '../components/ui/basics';
import { Dialog } from '../components/ui/overlays';
import { useStudio } from '../studio';
import { DATA_KINDS } from './blocks';
import type { AdderProps } from './Making';

export function DataKindDialog({ onDone }: AdderProps) {
  const { go } = useStudio();
  return (
    <Dialog title="Add data: what kind of data?" onClose={onDone}>
      <div className="kinds">
        {DATA_KINDS.map((group) => (
          <div key={group.title}>
            <div className="eyebrow">{group.title}</div>
            {group.kinds.map((kind) => (
              <NavItem
                key={kind.key}
                disabled={!kind.ready}
                title={kind.ready ? undefined : 'Comes in a later step'}
                label={
                  <>
                    <b>{kind.title}</b>
                    <span>{kind.text}</span>
                  </>
                }
                onClick={() => {
                  onDone();
                  go({ kind: 'new-dataset', data: kind.key });
                }}
              />
            ))}
          </div>
        ))}
      </div>
    </Dialog>
  );
}
