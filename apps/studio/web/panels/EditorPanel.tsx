// The editor of a file of a project (YAML): Ctrl-S saves it, and its text
// is checked as it is edited, its problems shown on their lines. A setup
// can be previewed live, and evaluated or fitted as a job.
import { yaml } from '@codemirror/lang-yaml';
import { lintGutter, setDiagnostics } from '@codemirror/lint';
import { keymap, type EditorView } from '@codemirror/view';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import CodeMirror from '@uiw/react-codemirror';
import type { IDockviewPanelProps } from 'dockview-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useCheck } from '../api/check';
import { api, data, done, type JobRequest } from '../api/client';
import { yamlStyle } from '../components/yamlStyle';
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
    panel.setTitle(`${dirty ? '● ' : ''}${path.split('/').slice(-2).join('/')}`);
  }, [dirty, panel, path]);

  // (the keys of the editor are set once: they save its latest text)
  const saveLatest = useRef(() => {});
  saveLatest.current = () => {
    if (text !== null) save.mutate(text);
  };
  const extensions = useMemo(
    () => [
      yaml(),
      yamlStyle,
      lintGutter(),
      keymap.of([{ key: 'Mod-s', run: () => (saveLatest.current(), true) }]),
    ],
    [],
  );

  // The problems of the text, on their lines
  const check = useCheck(path, text);
  const view = useRef<EditorView | null>(null);
  useEffect(() => {
    const editor = view.current;
    if (editor === null) return;
    const doc = editor.state.doc;
    const diagnostics = check.diagnostics
      .filter((d) => d.line <= doc.lines)
      .map((d) => {
        const line = doc.line(d.line);
        return { from: line.from, to: line.to, severity: d.severity, message: d.message };
      });
    editor.dispatch(setDiagnostics(editor.state, diagnostics));
  }, [check.diagnostics]);
  const errors = check.diagnostics.filter((d) => d.severity === 'error').length;
  const warnings = check.diagnostics.length - errors;

  const submit = useMutation({
    mutationFn: (job: JobRequest) => data(api.POST('/api/v1/jobs', { body: job })),
    onSuccess: () => openPanel('jobs'),
  });

  return (
    <section className="panel">
      <div className="panel-head">
        <span className="kind">{path}</span>
        <span className={errors ? 'badge bad' : warnings ? 'badge' : 'badge ok'}>
          {check.checking
            ? 'checking…'
            : errors + warnings === 0
              ? 'no problems'
              : [errors && `${errors} error${errors > 1 ? 's' : ''}`, warnings && `${warnings} warning${warnings > 1 ? 's' : ''}`]
                  .filter(Boolean)
                  .join(', ')}
        </span>
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
      {check.error && <p className="error">{check.error}</p>}
      {text !== null && (
        <CodeMirror
          className="editor"
          value={text}
          theme="none"
          extensions={extensions}
          onChange={setText}
          onCreateEditor={(editor) => (view.current = editor)}
          height="100%"
        />
      )}
    </section>
  );
}
