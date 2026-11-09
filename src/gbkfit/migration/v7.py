"""
The rules of version 7 of the format (2026-11-09 on), which rewrite
configurations of version 6.
"""

import abc
from collections.abc import MutableMapping
from typing import Any

from .rules import Change, Rule


__all__ = [
    'RULES',
    'GaussBeams',
    'LSFWidths'
]


class _KernelRule(Rule):
    """
    A rule of the PSFs or of the LSFs of configurations: those of their
    options of kernels (e.g. lsf, of instruments, and
    spectral_correlation, of the noise), the terms of their sums and
    convolutions, and the pieces of projects (a list of a section of
    their kind: see migrate_project).
    """

    # The options that are kernels, the section of their pieces, and the
    # option of the terms of their sums and convolutions
    _OPTIONS: tuple[str, ...]
    _PIECES: str
    _TERMS: str

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        pieces = config.get(self._PIECES)
        if isinstance(pieces, list):
            for i, kernel in enumerate(pieces):
                if isinstance(kernel, MutableMapping):
                    changes += self._kernel(kernel, [self._PIECES, i])
        for key, value in config.items():
            if key != self._PIECES or not isinstance(value, list):
                changes += self._walk(value, [key])
        return changes

    def _walk(self, value: Any, path: list[str | int]) -> list[Change]:
        """Rewrite the kernels of a value and of its items, in place."""
        changes = []
        if isinstance(value, MutableMapping):
            for key, item in value.items():
                if key in self._OPTIONS and isinstance(item, MutableMapping):
                    changes += self._kernel(item, path + [key])
                else:
                    changes += self._walk(item, path + [key])
        elif isinstance(value, list):
            for i, item in enumerate(value):
                changes += self._walk(item, path + [i])
        return changes

    def _kernel(
            self, kernel: MutableMapping[str, Any], path: list[str | int]
    ) -> list[Change]:
        """Rewrite a kernel, and its terms, in place."""
        changes = []
        if kernel.get('type') in ('sum', 'convolution') and isinstance(
                terms := kernel.get(self._TERMS), list):
            for i, term in enumerate(terms):
                if isinstance(term, MutableMapping):
                    changes += self._kernel(term, path + [self._TERMS, i])
        return changes + self._rewrite_kernel(kernel, path)

    @abc.abstractmethod
    def _rewrite_kernel(
            self, kernel: MutableMapping[str, Any], path: list[str | int]
    ) -> list[Change]:
        """Rewrite a kernel itself (not its terms), in place."""


class LSFWidths(_KernelRule):
    """
    The widths of LSFs have units (2026-11-09; see gbkfit.instrument.lsfs):
    a width that was a number, of km/s of the light, is its text in km/s
    (e.g. '20 km/s'), and an inline table of widths without a unit gets
    unit: km/s;
    the Hanning smoothing is of the channels of the data (channels), not
    of a width of the axis (width). The LSFs are those of instruments
    (lsf) and of the noise (spectral_correlation).

    The spectral correlation of the noise was in km/s of the axis, and is
    now in km/s of the light, drawn at the middle channel of the data:
    the same, to 1 + v / c of that channel.
    """

    _OPTIONS = ('lsf', 'spectral_correlation')
    _PIECES = 'lsf'
    _TERMS = 'lsfs'

    # The widths of the LSFs, by type
    _WIDTHS = dict(
        gauss=('sigma',), ggauss=('alpha',), lorentz=('gamma',),
        moffat=('alpha',), image=('step',), images=('step',))

    @staticmethod
    def version() -> int:
        return 7

    def _rewrite_kernel(
            self, kernel: MutableMapping[str, Any], path: list[str | int]
    ) -> list[Change]:
        changes = []
        kind = kernel.get('type')
        if kind == 'hanning' and 'width' in kernel:
            width = kernel.pop('width')
            changes.append(Change(
                path + ['width'],
                f"width {width} (km/s of the axis) -> the channels of the "
                f"data; give channels if it was not their width"))
        for name in self._WIDTHS.get(kind, ()):
            value = kernel.get(name)
            if isinstance(value, int | float) and not isinstance(value, bool):
                kernel[name] = f'{value} km/s'
                changes.append(Change(path + [name], f"{value} -> {value} km/s"))
            elif isinstance(value, MutableMapping) and 'points' in value \
                    and 'unit' not in value:
                if 'file' in value:
                    changes.append(Change(
                        path + [name],
                        "the column of the widths must have a unit in its "
                        "file; if it has none, give it in units"))
                else:
                    value['unit'] = 'km/s'
                    changes.append(Change(
                        path + [name], "the values of the table: unit km/s"))
        return changes


class GaussBeams(_KernelRule):
    """
    A Gaussian PSF of a radio beam is a Gaussian (2026-11-09): type
    gauss_beam is gauss, of the same options (bmaj, bmin and bpa, or a
    file; see PSFGauss.from_beam and from_file). The PSFs are those of
    instruments (psf) and of the noise (spatial_correlation).
    """

    _OPTIONS = ('psf', 'spatial_correlation')
    _PIECES = 'psf'
    _TERMS = 'psfs'

    @staticmethod
    def version() -> int:
        return 7

    def _rewrite_kernel(
            self, kernel: MutableMapping[str, Any], path: list[str | int]
    ) -> list[Change]:
        if kernel.get('type') != 'gauss_beam':
            return []
        kernel['type'] = 'gauss'
        return [Change(path + ['type'], "gauss_beam -> gauss")]


# The rules, in the order of the changes
RULES: tuple[Rule, ...] = (
    LSFWidths(),
    GaussBeams(),
)
