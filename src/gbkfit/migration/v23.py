"""
The rules of version 23 of the format (2026-11-25 on), which rewrite
configurations of version 22: a point is not a type of PSF or LSF (one
alone was no PSF or LSF; a sharp core of a sum is a narrow Gaussian of
its real width).
"""

from collections.abc import MutableMapping
from typing import Any

from .rules import Change, PiecesView, Rule, items


__all__ = [
    'RULES',
    'NoPointResponses'
]


# The options that are a PSF or an LSF, and the options of the terms of a
# sum or a convolution of them
_RESPONSES = ('psf', 'lsf', 'spatial_correlation', 'spectral_correlation')
_TERMS = ('psfs', 'lsfs')


class NoPointResponses(Rule):
    """
    A point is not a type of PSF or LSF (2026-11-25): a PSF or an LSF of
    the type point (of an instrument, or the correlation of the noise of
    data) is left out, as it was none; a point term of a convolution is
    left out, as it did nothing; one of a sum, which kept a share of the
    light sharp, is replaced by hand by a narrow Gaussian of the real
    width of the core; and a PSF or LSF piece of a project of the type
    point is taken out by hand, from where it is named.
    """

    @staticmethod
    def version() -> int:
        return 23

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        if isinstance(config, PiecesView):
            for section in ('psf', 'lsf'):
                for path, piece in items(config, section):
                    if _is_point(piece):
                        changes.append(Change(path, (
                            f"by hand: a point is no {section.upper()}: "
                            f"take the piece out of what names it, and "
                            f"delete it")))
                    else:
                        changes += _rewrite(piece, path)
        for key, value in config.items():
            if not (isinstance(config, PiecesView) and key in ('psf', 'lsf')):
                changes += _rewrite(value, [key])
        return changes


def _is_point(value: Any) -> bool:
    """Check whether a value is a PSF or an LSF of the type point."""
    return isinstance(value, MutableMapping) and value.get('type') == 'point'


def _rewrite(value: Any, path: list[str | int]) -> list[Change]:
    """
    Rewrite the points of the PSFs and LSFs of a value of a configuration
    at a path, in place (see NoPointResponses), and return the changes.
    """
    changes = []
    if isinstance(value, MutableMapping):
        for key in [k for k in _RESPONSES if _is_point(value.get(k))]:
            del value[key]
            changes.append(Change(path + [key], (
                "removed (was a point, which did nothing: as none)")))
        for key in _TERMS:
            terms = value.get(key)
            if not isinstance(terms, list):
                continue
            points = [i for i, term in enumerate(terms) if _is_point(term)]
            if not points:
                continue
            if value.get('type') == 'convolution':
                for i in reversed(points):
                    del terms[i]
                changes.append(Change(path + [key], (
                    "its point terms removed (they did nothing)")))
            else:
                changes += [Change(path + [key, i], (
                    "by hand: replace the point with a narrow Gaussian of "
                    "the real width of the sharp core (e.g. the "
                    "diffraction limit)")) for i in points]
        for key, item in value.items():
            changes += _rewrite(item, path + [key])
    elif isinstance(value, list):
        for i, item in enumerate(value):
            changes += _rewrite(item, path + [i])
    return changes


RULES: tuple[Rule, ...] = (
    NoPointResponses(),
)
