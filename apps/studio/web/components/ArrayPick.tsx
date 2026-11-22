// The array of a file that is read (e.g. one of the HDUs of a FITS file),
// as a small button of its name, under the name of the file. Its menu has
// every array of the file, each with its shape (any can be chosen: what
// reads it says if it cannot be read there); the menu of many has a field
// at its top that finds them (see DropMenu).
import type { ArrayInfo } from '../api/client';
import { arrayLabel, kept } from './arrays';
import { DropMenu } from './ui/overlays';

export function ArrayPick({
  arrays,
  chosen,
  label,
  onChoose,
}: {
  // The arrays of the file
  arrays: ArrayInfo[];
  chosen: ArrayInfo | undefined;
  // What it is (e.g. 'The array of the errors')
  label: string;
  onChoose: (array: ArrayInfo) => void;
}) {
  return (
    <DropMenu
      label={label}
      width={280}
      entries={arrays.map((array) => ({
        label: arrayLabel(array),
        description: kept(array).join(' × ') || 'one value',
        current: array.index === chosen?.index,
        onClick: () => onChoose(array),
      }))}
      target={
        <button type="button" className="ui-array-pick" title={chosen ? `${label}: ${arrayLabel(chosen)}` : label}>
          <span>{chosen ? arrayLabel(chosen) : 'Choose'}</span>▾
        </button>
      }
    />
  );
}
