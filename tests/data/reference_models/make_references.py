"""
Regenerate the reference model outputs stored in this directory.

The references were produced with commit 6f79336 (the last version of
gbkfit that evaluated models end to end before the 2025 refactor, plus
two small NumPy 2 fixes), from these configurations in the config format
of that version. The configurations have since been converted to the
current format. The references are used by
tests/model/test_reference_models.py to check that the current code still
produces the same models.

Since then, the spectral lines are integrated over each channel instead
of sampled at its centre, so the references were made again with the
current code. Before that, the outputs were checked to differ from those
of 6f79336 by that change only: the same flux in each spaxel (to 2e-5),
and lines whose variance is larger by that of a channel (step^2 / 12).

They were made again when the Gaussian cdf that integrates the lines
was computed in float instead of double (2026-10-10). The cubes changed
by at most 2e-7 of their peak, but the moments 1 and 2 of faint spaxels,
ratios of small sums, by more than the tolerance of the tests: up to
2e-4 km/s where moment 0 is above 10% of its peak, 1.4e-3 km/s above 1%,
and more at the noise of the convolutions, where one spaxel at the
cutoff of moment 0 became masked.

Usage:

    python make_references.py PYTHON

where PYTHON is the interpreter of an environment with the version of
gbkfit that makes the references (now the current code). Every <case>.yaml in this directory is
evaluated with `gbkfit-cli eval model`, and its FITS outputs are stored
in <case>.npz, keyed by file name.
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
