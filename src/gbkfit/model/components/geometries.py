"""
The geometric parameters of the components: their centre, orientation
and systemic velocity, each its own (components that share them tie
them in the parameter space).
"""

from typing import Any

from gbkfit.params import ParamDesc, ParamScalarDesc, ParamVectorDesc


__all__ = [
    'GEOMETRY_PARAMS',
    'GEOMETRY_UNITS',
    'geometry_pdesc'
]


# The geometric parameters of the components, and the warp that makes each
# of them node-wise (one value for each ring): loose (the centre and the
# systemic velocity) or tilted (the position angle and the inclination)
GEOMETRY_PARAMS = dict(
    vsys='loose', xpos='loose', ypos='loose', posa='tilted', incl='tilted')

# The units of the geometric parameters of the components
GEOMETRY_UNITS = dict(
    vsys='km/s', xpos='arcsec', ypos='arcsec', posa='deg', incl='deg')
# and their typical values (see ParamDesc): at the centre, inclined
_DEFAULTS = dict(vsys=0, xpos=0, ypos=0, posa=0, incl=45)


def geometry_pdesc(name: str, nnodes: int | None = None) -> ParamDesc:
    """
    Return the description of a geometric parameter (see GEOMETRY_UNITS):
    a scalar, or node-wise (one value for each of nnodes rings) if nnodes
    is given. The position angle repeats every 360 degrees.
    """
    options: dict[str, Any] = dict(
        unit=GEOMETRY_UNITS[name], default=_DEFAULTS[name],
        period=360 if name == 'posa' else None)
    return ParamScalarDesc(**options) if nnodes is None \
        else ParamVectorDesc(nnodes, **options)
