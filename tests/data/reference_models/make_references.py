"""
Regenerate the reference model outputs stored in this directory.

The references were produced with commit 6f79336 (the last version of
gbkfit that evaluated models end to end before the 2025 refactor, plus
two small NumPy 2 fixes), from these configurations in the config format
of that version. The configurations have since been converted to the
current format. The references are used by
tests/model/test_reference_models.py to check that the current code still
produces the same models.

Usage:

    python make_references.py PYTHON

where PYTHON is the interpreter of an environment with the reference
version of gbkfit installed. Every <case>.yaml in this directory is
evaluated with `gbkfit-cli eval model`, and its FITS outputs are stored
in <case>.npz, keyed by file name.

Note: commit 6f79336 only writes the first map of models with multiple
maps, so the mmaps reference covers the moment 0 map only. The other
moment maps are tested in tests/model/test_mmaps.py.
"""

import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from astropy.io import fits


HERE = pathlib.Path(__file__).parent


def evaluate_model(python, config, workdir):
    """Evaluate a model configuration and return its FITS outputs."""
    subprocess.run(
        [python, '-m', 'gbkfit.apps.cli', 'eval', 'model', str(config)],
        cwd=workdir, check=True, capture_output=True)
    # Older versions write their outputs to the working directory, newer
    # ones to an output directory, so search both
    return {
        path.stem: fits.getdata(path)
        for path in sorted(workdir.rglob('model_[0-9]*.fits'))}


def main():
    python = sys.argv[1]
    for config in sorted(HERE.glob('*.yaml')):
        with tempfile.TemporaryDirectory() as workdir:
            outputs = evaluate_model(python, config, pathlib.Path(workdir))
        np.savez_compressed(config.with_suffix('.npz'), **outputs)
        print(f"{config.name}: {', '.join(outputs)}")


if __name__ == '__main__':
    main()
