// A dialog to choose a file of data in a folder of the workspace, whose
// folders are browsed but never above it; only the files of data that
// GBKFIT reads are shown.
import { useState } from 'react';
import { useFolder } from '../api/queries';
import { NavItem } from './ui/basics';
import { Dialog } from './ui/overlays';

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
  title,
  root,
  label,
  onChoose,
  onClose,
}: {
  title: string;
  // The folder, and its name as the dialog shows it
  root: string;
  label: string;
  onChoose: (path: string) => void;
  onClose: () => void;
}) {
  const [folder, setFolder] = useState(root);
  const listing = useFolder(folder);
  const files = (listing.data?.files ?? []).filter((file) => isDataFile(file.name));
  const inside = folder === root ? '' : folder.slice(root === '.' ? 0 : root.length + 1);
  return (
    <Dialog title={title} onClose={onClose}>
      <div className="picker-path">{`${label}/${inside}`}</div>
      <div className="dialog-list">
        {folder !== root && <NavItem mono label=".." onClick={() => setFolder(parentOf(folder))} />}
        {listing.data?.folders.map((name) => (
          <NavItem key={name} mono label={`${name}/`} onClick={() => setFolder(joinPath(folder, name))} />
        ))}
        {files.map((file) => (
          <NavItem
            key={file.name}
            mono
            label={file.name}
            right={<small className="picker-size">{size(file.size)}</small>}
            onClick={() => onChoose(joinPath(folder, file.name))}
          />
        ))}
        {listing.error && <p className="note">Nothing here yet.</p>}
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
