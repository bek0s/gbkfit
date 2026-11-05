// The two ways to give a file of data, each copying it into the project:
// a file of the workspace of the server (import), or one of this computer
// (upload). useFileSource gives the actions and what they show (the
// progress, a failure, and the dialog that picks a file of the
// workspace), for a caller that places them (e.g. in a menu);
// FileSources is a pair of buttons.
import { useRef, useState, type ReactNode } from 'react';
import { upload } from '../api/client';
import { useImportData } from '../api/queries';
import { FilePicker } from './FilePicker';
import { Button } from './ui/basics';

export interface FileSource {
  fromWorkspace: () => void;
  fromComputer: () => void;
  // Whether a file is being copied
  busy: boolean;
  // The progress or the failure of a copy (none: nothing to say)
  status: ReactNode;
  // What the actions need in the page: the input of files, the dialog
  hidden: ReactNode;
}

export function useFileSource(project: string, onFile: (path: string) => void): FileSource {
  const importing = useImportData(project);
  const [picking, setPicking] = useState(false);
  // The fraction of the file sent while uploading, and why it failed
  const [sent, setSent] = useState<number | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const send = (file: File) => {
    setFailure(null);
    setSent(0);
    upload(project, file, setSent)
      .then(onFile)
      .catch((error: Error) => setFailure(error.message))
      .finally(() => setSent(null));
  };
  const error = failure ?? importing.error?.message;
  const status =
    sent !== null ? (
      <span className="note">Uploading {Math.round(100 * sent)}%</span>
    ) : importing.isPending ? (
      <span className="note">Copying into the project…</span>
    ) : error ? (
      <span className="error">{error}</span>
    ) : null;
  const hidden = (
    <>
      <input
        ref={input}
        type="file"
        accept=".fits,.fit,.fts,.gz"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = '';
          if (file) send(file);
        }}
      />
      {picking && (
        <FilePicker
          title="Import a file of the workspace"
          root="."
          label="Workspace"
          onClose={() => setPicking(false)}
          onChoose={(path) => {
            setPicking(false);
            setFailure(null);
            importing.mutate(path, { onSuccess: onFile });
          }}
        />
      )}
    </>
  );
  return {
    fromWorkspace: () => setPicking(true),
    fromComputer: () => input.current?.click(),
    busy: sent !== null || importing.isPending,
    status,
    hidden,
  };
}

export function FileSources({ project, onFile }: { project: string; onFile: (path: string) => void }) {
  const source = useFileSource(project, onFile);
  return (
    <>
      <Button kind="small" title="A file of the server's workspace; the project keeps a copy" onClick={source.fromWorkspace}>
        Import
      </Button>
      <Button kind="small" title="A file of this computer; the project keeps a copy" onClick={source.fromComputer}>
        Upload
      </Button>
      {source.status}
      {source.hidden}
    </>
  );
}
