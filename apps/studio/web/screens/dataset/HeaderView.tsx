// The header of the data of an item, as it is in its file (the cards of
// its array, an HDU, in mono), in the room of the viewer: of the item
// chosen (by its title; one or several alike), its file and HDU beside it.
import { useState, type ReactNode } from 'react';
import { useHeader } from '../../api/queries';
import { MiddleCut } from '../../components/ui/cut';
import { Choice } from '../../components/ui/inputs';

export interface HeaderSource {
  key: string;
  title: string;
  path: string;
  hdu: number;
}

export function HeaderView({ sources, actions }: { sources: HeaderSource[]; actions: ReactNode }) {
  const [chosen, setChosen] = useState(sources[0]?.key ?? '');
  const source = sources.find((one) => one.key === chosen) ?? sources[0];
  const header = useHeader(source?.path ?? null, source?.hdu ?? 0);
  return (
    <div className="viewer">
      <div className="viewer-head">
        <Choice
          items={sources.map(({ key, title, path, hdu }) => ({ value: key, label: title, description: `${path.split('/').pop()} · HDU ${hdu}` }))}
          value={source?.key ?? ''}
          label="The header of"
          onChange={setChosen}
        />
        {source && (
          <MiddleCut text={`${source.path.split('/').pop()} · HDU ${source.hdu}`} title={source.path} className="ds-header-file" />
        )}
        <span className="viewer-head-gap" />
        {actions}
      </div>
      <pre className="ds-header-cards">{header.data ? header.data.cards.join('\n') : header.error ? header.error.message : 'Reading…'}</pre>
    </div>
  );
}
