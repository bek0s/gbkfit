"""
The ASDF format: reading an array of the tree of an ASDF file (e.g.
'roman/data'), with a FITS header of its unit and of the world
coordinates of the file (its gwcs, linearised, see linear_header).
"""

import asdf
import asdf.treeutil
import astropy.io.fits
import astropy.table
import gwcs
import numpy as np

import gbkfit.units
from ._linear_wcs import linear_header


__all__ = []


def read_image(
        filename: str, path: str | None
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Return the array of an ASDF file at a path of its tree (e.g.
    'roman/data'), and a FITS header of its unit (BUNIT, of a Quantity)
    and of the world coordinates of the only WCS of the file (a gwcs), if
    it has one (see linear_header).

    Raises
    ------
    ValueError
        If the path is not given, not in the tree or not of an array, or
        the file has more than one WCS.
    """
    if path is None:
        raise ValueError(
            f"{filename}: give the path of the array in the tree of the "
            f"file (path, e.g. roman/data)")
    with asdf.open(filename) as file:
        node = file.tree
        for key in str(path).split('/'):
            try:
                node = node[key]
            except (KeyError, IndexError, TypeError):
                raise ValueError(
                    f"{filename}: its tree has no {path!r}") from None
        if not hasattr(node, 'shape'):
            raise ValueError(
                f"{filename}: {path!r} of its tree is not an array; give "
                f"the path of one (e.g. roman/data)")
        unit = getattr(node, 'unit', None)
        values = np.array(getattr(node, 'value', node))
        wcss = [
            node for node in asdf.treeutil.iter_tree(file.tree)
            if isinstance(node, gwcs.WCS)]
    header = astropy.io.fits.Header()
    if unit is not None:
        header['BUNIT'] = gbkfit.units.format_unit(unit)
    if len(wcss) > 1:
        raise ValueError(
            f"{filename}: its tree has {len(wcss)} WCSs; files of one are "
            f"read")
    if wcss:
        header.update(linear_header(filename, wcss[0], values.shape))
    return values, header


def read_header(filename: str, path: str | None) -> astropy.io.fits.Header:
    """
    Return the FITS header of an array of the tree of an ASDF file (see
    read_image).
    """
    return read_image(filename, path)[1]


def read_channel_beams(
        filename: str, header: astropy.io.fits.Header
) -> astropy.table.QTable | None:
    """Return None: ASDF files have no table of the beams of channels."""
    return None
