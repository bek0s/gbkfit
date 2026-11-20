// Copies the options of a dataset to the clipboard, as the project writes
// its piece (the YAML a configuration of the command line takes)
import { useState } from 'react';
import { usePieceText } from '../../api/queries';
import { Button } from '../../components/ui/basics';

export function CopyConfig({ options }: { options: Record<string, unknown> }) {
  const text = usePieceText();
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(await text.mutateAsync(options));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  };
  return (
    <Button className="copy-config" disabled={text.isPending} title={text.error?.message} onClick={() => void copy().catch(() => undefined)}>
      {copied ? 'Copied' : 'Copy config'}
    </Button>
  );
}
