// How a dataset of the project was made, in the dock: of its files as
// they are, or of copies that prep made (as the server recorded it): from
// which original files, with which edits, and when. The originals stay in
// the project, so that the dataset can be prepared again from them (Edit);
// they can be deleted once the copies are right.
import { useState } from 'react';
import { useDeleteData, useDescriptions } from '../../api/queries';
import { Button } from '../../components/ui/basics';
import { dataFiles, draftOfPreparation, editsInWords, inProject, type Kind, type Preparation } from './model';

export function MadeFrom({
  project,
  kind,
  dataset,
  record,
}: {
  project: string;
  kind: Kind;
  dataset: Record<string, unknown>;
  // How its files were prepared (null: they were not)
  record: Preparation | null;
}) {
  const originals = record ? originalsOf(record) : [];
  const describe = useDescriptions(originals);
  const remove = useDeleteData();
  const [confirming, setConfirming] = useState(false);
  if (record === null) {
    // (prep names its files prep_<name>: such files were prepared before
    // the server kept records, or by the command line)
    const prepared = Object.values(dataFiles(kind, dataset, project)).some((path) => path.split('/').pop()?.startsWith('prep_'));
    return (
      <p className="ds-text">
        {prepared
          ? 'The files look prepared (prep_), but the project has no record of how: they were prepared elsewhere, or before records were kept.'
          : 'The dataset reads its files as they are: no copies were prepared.'}
      </p>
    );
  }
  const lines = editsInWords(kind, draftOfPreparation(kind, dataset, record, project));
  const present = originals.filter((path) => !describe(path).error);
  const deleteAll = async () => {
    for (const path of present) await remove.mutateAsync(path).catch(() => undefined);
    setConfirming(false);
  };
  return (
    <div className="ds-made">
      <p className="ds-text">
        Prepared on {when(record.time)} from {originals.length === 1 ? 'the original' : 'the originals'}:
      </p>
      <ul className="ds-originals">
        {originals.map((path) => (
          <li key={path} data-gone={describe(path).error ? true : undefined} title={path}>
            {inProject(path, project)}
            {describe(path).error && <span> · deleted</span>}
          </li>
        ))}
      </ul>
      {lines.length > 0 ? (
        <ul className="ds-steps">
          {lines.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      ) : (
        <p className="note">With no edits.</p>
      )}
      {present.length > 0 &&
        (confirming ? (
          <div className="ds-confirm">
            <p>
              Delete {present.length === 1 ? 'the original' : `the ${present.length} originals`} from the project? The dataset keeps its copies, but can no
              longer be prepared again from {present.length === 1 ? 'it' : 'them'}.
            </p>
            <div className="ds-confirm-actions">
              <Button onClick={() => setConfirming(false)}>Keep</Button>
              <Button kind="danger" disabled={remove.isPending} onClick={() => void deleteAll()}>
                Delete
              </Button>
            </div>
          </div>
        ) : (
          <button className="ds-link ds-danger" onClick={() => setConfirming(true)}>
            Delete the {present.length === 1 ? 'original' : 'originals'}…
          </button>
        ))}
      {remove.error && <p className="error">{remove.error.message}</p>}
    </div>
  );
}

// The original files of a preparation (paths of the workspace), each once
export function originalsOf(record: Preparation): string[] {
  const paths = ['file_d', 'file_e', 'file_m'].flatMap((key) => {
    const value = record.options[key];
    const list = Array.isArray(value) ? value : [value];
    return list.flatMap((item) => (typeof item === 'string' ? [item] : item && typeof item === 'object' && 'file' in item ? [String(item.file)] : []));
  });
  return [...new Set(paths)];
}

// The files a preparation wrote (paths of the workspace)
export function preparedFilesOf(record: Preparation): string[] {
  return Object.values(record.files).flatMap((value) => (Array.isArray(value) ? value.filter((path): path is string => path !== null) : [value]));
}

function when(time: string): string {
  const date = new Date(time);
  return Number.isNaN(date.getTime()) ? time : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}
