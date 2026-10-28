"""
The selection polar traits (spt): the parts of a component that it keeps
(see SPTrait).
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np

from gbkfit.params.pdescs import ParamDesc, ParamScalarDesc, ParamVectorDesc
from gbkfit.utils.parseutils import check_options
from .base import SPTrait
from .options import SAMPLINGS, SAMPLING_DEFAULT, Sampling


__all__ = [
    'SPTraitAzimuthalRange',
    'SPTraitNWAzimuthalRange',
    'SPTraitRadialRange'
]


# Their uids in the native kernels
SPT_UID_AZRANGE = 1
SPT_UID_RRANGE = 2
SPT_UID_NW_AZRANGE = 101


class SPTraitAzimuthalRange(SPTrait):
    """
    A wedge of the component: the azimuths within s / 2 of p. Its parameters:
    p, the azimuth of its centre, and s, its full width (fitted without
    slopes, see SPTrait).

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    @staticmethod
    def type() -> str:
        return 'azrange'

    @staticmethod
    def uid() -> int:
        return SPT_UID_AZRANGE

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            p=ParamScalarDesc(unit='deg'),
            s=ParamScalarDesc(unit='deg'))

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        p, s = params['p'], params['s']
        return np.tile([p - s / 2, p + s / 2], (len(r), 1))


class SPTraitRadialRange(SPTrait):
    """
    A range of radii of the component (a ring of a disk): the radii from
    rmin (inclusive) to rmax (exclusive). Its parameters: rmin and rmax
    (fitted without slopes, see SPTrait).

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters (see Trait).
    """

    @staticmethod
    def type() -> str:
        return 'rrange'

    @staticmethod
    def uid() -> int:
        return SPT_UID_RRANGE

    def axisymmetric(self) -> bool:
        return True

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            rmin=ParamScalarDesc(unit='arcsec'),
            rmax=ParamScalarDesc(unit='arcsec'))


class SPTraitNWAzimuthalRange(SPTrait):
    """
    A wedge of the component at each node: the azimuths within s(r) / 2
    of p(r). Its parameters, node-wise: p and s (fitted without slopes,
    see SPTrait).

    Parameters
    ----------
    sampling : str, optional
        Where the values of its node-wise parameters are given: 'rnodes',
        at the radii of the rings, or 'subrings', at the rings it is
        evaluated on (see the package docstring).
    name : str, optional
        Its name, which prefixes its parameters (see Trait).

    Raises
    ------
    ConfigError
        If an option is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'nw_azrange'

    @staticmethod
    def uid() -> int:
        return SPT_UID_NW_AZRANGE

    def __init__(
            self,
            sampling: Literal[SAMPLINGS] = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        check_options(SPTraitNWAzimuthalRange.__init__, locals())
        super().__init__(Sampling(sampling), name=name)

    def axisymmetric(self) -> bool:
        return False

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(
            p=ParamVectorDesc(nnodes, unit='deg'),
            s=ParamVectorDesc(nnodes, unit='deg'))

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        p = np.interp(r, subrnodes, params['p'])
        s = np.interp(r, subrnodes, params['s'])
        return np.column_stack((p - s / 2, p + s / 2))
