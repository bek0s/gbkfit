"""
The types of the options of grids on the sky (see make_grid and Coords),
which the configurations of everything on a grid share (e.g. the
observations, the datasets, the images of PSFs and of primary beams, the
deflection maps of lenses): their numbers of pixels and their reference
pixels (in pixels), their lengths (arcsec), their reference positions
and their rotations (degrees). An option of the two axes on the sky (x,
y) is one value for both, or one for each; one of a grid of one axis on
the sky (along a slit) is one value, or a list of one.
"""

from collections.abc import Sequence
from typing import Annotated, TypeAlias

from annotated_types import Ge, Gt, Len

from gbkfit.config import Unit


__all__ = [
    'Count',
    'Length',
    'Lengths',
    'Rota',
    'Rpix',
    'Rval',
    'Size',
    'SkyPosition',
    'SlitRpix',
    'SlitRval',
    'SlitSize',
    'SlitStep',
    'Step'
]


# A number of pixels (or of cells), and a length on the sky (arcsec)
Count: TypeAlias = Annotated[int, Ge(1)]
Length: TypeAlias = Annotated[float, Gt(0)]

# Of the two axes on the sky
Size: TypeAlias = Count | Annotated[Sequence[Count], Len(2, 2)]
Lengths: TypeAlias = Annotated[
    Length | Annotated[Sequence[Length], Len(2, 2)], Unit('arcsec')]
Step: TypeAlias = Lengths
Rpix: TypeAlias = float | Annotated[Sequence[float], Len(2, 2)]
Rval: TypeAlias = Annotated[
    float | Annotated[Sequence[float], Len(2, 2)], Unit('deg')]
Rota: TypeAlias = Annotated[float, Unit('deg')]

# Of one axis on the sky (along a slit)
SlitSize: TypeAlias = Count | Annotated[Sequence[Count], Len(1, 1)]
SlitStep: TypeAlias = Annotated[
    Length | Annotated[Sequence[Length], Len(1, 1)], Unit('arcsec')]
SlitRpix: TypeAlias = float | Annotated[Sequence[float], Len(1, 1)]
SlitRval: TypeAlias = Annotated[
    float | Annotated[Sequence[float], Len(1, 1)], Unit('deg')]

# A position on the sky: its RA and Dec (degrees)
SkyPosition: TypeAlias = Annotated[
    Annotated[Sequence[float], Len(2, 2)], Unit('deg')]
