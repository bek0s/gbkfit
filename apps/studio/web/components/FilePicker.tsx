// A dialog to choose a file of data of a project: its folders are browsed
// from a start (e.g. its folder data), never above the project, and only
// the files of data that GBKFIT reads are shown.
import { useEffect, useState } from 'react';
import { useFolder } from '../api/queries';
import { Dialog } from './Dialog';

// The endings of the names of the files of data that GBKFIT reads (FITS)
const DATA_FILES = ['.fits', '.fit', '.fts', '.fits.gz'];

export function isDataFile(name: string): boolean {
  return DATA_FILES.some((ending) => name.toLowerCase().endsWith(ending));
}

// A path of the workspace joined to a name in it ('.': the workspace)
export function joinPath(folder: string, name: string): string {
  return folder === '.' ? name : `${folder}/${name}`;
}

function parentOf(path: string): string {
  return path.split('/').slice(0, -1).join('/') || '.';
}

export function FilePicker({
  root,
  start,
  onChoose,
  onClose,
}: {
  // The folder of the project, and the folder shown first
  root: string;
  start: string;
  onChoose: (path: string) => void;
  onClose: () => void;
}) {
  const [folder, setFolder] = useState(start);
  const listing = useFolder(folder);
  // (a start that does not exist falls back to the project)
  useEffect(() => {
    if (listing.error && folder !== root) setFolder(root);
  }, [listing.error, folder, root]);
  const files = (listing.data?.files ?? []).filter((file) => isDataFile(file.name));
  const shown = folder === root ? '' : folder.slice(root === '.' ? 0 : root.length + 1);
  return (
    <Dialog title="Choose a file" onClose={onClose}>
      <div className="picker-path">{`project/${shown}`}</div>
      <div className="dialog-list">
        {folder !== root && <button onClick={() => setFolder(parentOf(folder))}>..</button>}
        {listing.data?.folders.map((name) => (
          <button key={name} onClick={() => setFolder(joinPath(folder, name))}>
            {name}/
          </button>
        ))}
        {files.map((file) => (
          <button key={file.name} className="picker-file" onClick={() => onChoose(joinPath(folder, file.name))}>
            {file.name}
            <small>{size(file.size)}</small>
          </button>
        ))}
        {listing.data && listing.data.folders.length + files.length === 0 && (
          <p className="note">No files of data here (FITS).</p>
        )}
      </div>
    </Dialog>
  );
}

// A size in bytes, as people read it
function size(bytes: number): string {
  const units = ['B', 'kB', 'MB', 'GB', 'TB'];
  const power = Math.min(units.length - 1, Math.floor(Math.log10(Math.max(bytes, 1)) / 3));
  return `${(bytes / 1000 ** power).toFixed(power ? 1 : 0)} ${units[power]}`;
}
