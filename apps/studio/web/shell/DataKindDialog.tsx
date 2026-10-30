// The first step of adding data: what kind of data it is. The dataset
// screen then asks for its files (see screens/Dataset).
import { Dialog } from '../components/Dialog';
import { useStudio } from '../studio';
import { DATA_KINDS } from './blocks';
import type { AdderProps } from './Tree';

export function DataKindDialog({ onDone }: AdderProps) {
  const { go } = useStudio();
  return (
    <Dialog title="Add data: what kind of data?" onClose={onDone}>
      <div className="kinds">
        {DATA_KINDS.map((group) => (
          <div key={group.title}>
            <div className="eyebrow">{group.title}</div>
            {group.kinds.map((kind) => (
              <button
                key={kind.key}
                disabled={!kind.ready}
                title={kind.ready ? undefined : 'Comes in a later step'}
                onClick={() => {
                  onDone();
                  go({ kind: 'new-dataset', data: kind.key });
                }}
              >
                <b>{kind.title}</b>
                <span>{kind.text}</span>
              </button>
            ))}
          </div>
        ))}
      </div>
    </Dialog>
  );
}
