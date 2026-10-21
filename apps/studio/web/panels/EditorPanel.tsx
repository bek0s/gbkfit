// The editor of a file of a project (YAML): Ctrl-S saves it. A setup can
// be previewed live, and evaluated or fitted as a job.
import { yaml } from '@codemirror/lang-yaml';
import { keymap } from '@codemirror/view';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import CodeMirror from '@uiw/react-codemirror';
import type { IDockviewPanelProps } from 'dockview-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { api, data, done, type JobRequest } from '../api/client';
import { isSetup, useStudio } from '../studio';

export function EditorPanel({ params, api: panel }: IDockviewPanelProps<{ path: string }>) {
  const { path } = params;
  const { live, openPanel } = useStudio();
  const client = useQueryClient();
  const [text, setText] = useState<string | null>(null);
  const saved = useRef<string | null>(null);

  const file = useQuery({
    queryKey: ['file', path],
    queryFn: () =>
      data(api.GET('/api/v1/files/{path}', { params: { path: { path } }, parseAs: 'text' })),
  });
  useEffect(() => {
    if (file.data !== undefined && text === null) {
      saved.current = file.data;
      setText(file.data);
    }
  }, [file.data, text]);

  const save = useMutation({
    mutationFn: (body: string) =>
      done(
        api.PUT('/api/v1/files/{path}', {
          params: { path: { path } },
          body,
          bodySerializer: (value: string) => value,
          headers: { 'Content-Type': 'text/plain' },
        }),
      ),
    onSuccess: (_, body) => {
      saved.current = body;
      setText(body);
      void client.invalidateQueries({ queryKey: ['project'] });
      if (live.config === path) live.restart();
    },
  });

  const dirty = text !== null && text !== saved.current;
  useEffect(() => {
    panel.setTitle(`${dirty ? '● ' : ''}${path.slice(path.lastIndexOf('/') + 1)}`);
  }, [dirty, panel, path]);

  // (the keys of the editor are set once: they save its latest text)
  const saveLatest = useRef(() => {});
  saveLatest.current = () => {
    if (text !== null) save.mutate(text);
  };
  const extensions = useMemo(
    () => [
      yaml(),
      keymap.of([{ key: 'Mod-s', run: () => (saveLatest.current(), true) }]),
    ],
    [],
  );

  const submit = useMutation({
    mutationFn: (job: JobRequest) => data(api.POST('/api/v1/jobs', { body: job })),
    onSuccess: () => openPanel('jobs'),
  });

  return (
    <section className="panel">
      <div className="panel-head">
        <span className="kind">{path}</span>
        <div className="actions">
          {dirty && (
            <button className="ghost" onClick={() => text !== null && save.mutate(text)}>
              Save
            </button>
          )}
          {isSetup(path) && (
            <>
              <button
                className="ghost"
                onClick={() => {
                  live.start(path);
                  openPanel('preview');
                }}
              >
                Preview
              </button>
              <button
                className="ghost"
                onClick={() =>
                  submit.mutate({ task: 'eval', options: { mode: 'model', config: path, profile_iters: 0 } })
                }
              >
                Eval
              </button>
              <button className="primary" onClick={() => submit.mutate({ task: 'fit', options: { config: path } })}>
                Fit
              </button>
            </>
          )}
        </div>
      </div>
      {(file.error ?? save.error ?? submit.error) && (
        <p className="error">{(file.error ?? save.error ?? submit.error)?.message}</p>
      )}
      {text !== null && (
        <CodeMirror
          className="editor"
          value={text}
          theme="none"
          extensions={extensions}
          onChange={setText}
          height="100%"
        />
      )}
    </section>
  );
}
