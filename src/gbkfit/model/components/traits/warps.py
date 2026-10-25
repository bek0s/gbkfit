"""
The vertical distortion polar traits (zpt): displacements of the plane of
a disk along z (see ZPTrait).
"""



from gbkfit.params.pdescs import ParamDesc
from .base import ZPTrait
from .options import Harmonic, SAMPLING_DEFAULT, Sampling, Zmax


__all__ = [
    'ZPTraitNWHarmonic',
    'ZPTraitNWUniform'
]


# Their uids in the native kernels
ZPT_UID_NW_UNIFORM = 101
ZPT_UID_NW_HARMONIC = 102


class ZPTraitNWUniform(ZPTrait):
    """
    A displacement of the plane given at each ring: a(r), along z. Its
    parameter: a, node-wise.

    Parameters
    ----------
    zmax : float
        The largest displacement (arcsec; see ZPTrait).
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
        return 'nw_uniform'

    @staticmethod
    def uid() -> int:
        return ZPT_UID_NW_UNIFORM

    def __init__(
            self,
            zmax: float,
            sampling: str = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        super().__init__(Zmax(zmax), Sampling(sampling), name=name)

    def consts(self) -> tuple[float, ...]:
        return (self.zmax(),)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(a=self._amplitude_desc(nnodes))


class ZPTraitNWHarmonic(ZPTrait):
    """
    A harmonic displacement of the plane: a(r) cos(m (theta - p(r))), of
    the order m; order 1 is a warp, whose line of nodes is 90 degrees from
    p. Its parameters, node-wise: a, the amplitude, and p (but for order
    0), the azimuth of a maximum.

    Parameters
    ----------
    order : int
        The order m of the harmonic: the pattern repeats every 360 / m
        degrees, and order 0 is an axisymmetric a(r).
    zmax : float
        The largest displacement (arcsec; see ZPTrait).
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
        return 'nw_harmonic'

    @staticmethod
    def uid() -> int:
        return ZPT_UID_NW_HARMONIC

    def __init__(
            self,
            order: int,
            zmax: float,
            sampling: str = SAMPLING_DEFAULT,
            name: str | None = None
    ):
        super().__init__(
            Harmonic(order), Zmax(zmax), Sampling(sampling), name=name)

    def consts(self) -> tuple[float, ...]:
        return (self.zmax(), self._option(Harmonic).order)

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return self._option(Harmonic).params(nnodes) | dict(
            a=self._amplitude_desc(nnodes))
