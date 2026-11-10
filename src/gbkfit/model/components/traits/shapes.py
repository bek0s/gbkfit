"""
The shapes that traits of several families share, each declared once: a
mixin of its type, title and uid in the native kernels (the same in every
family), whether it is axisymmetric, and its parameters. Its amplitude a
is what its family says (see Amplitude): a surface density (rpt), a
velocity dispersion (dpt), ...; the concrete classes of the families
(e.g. RPTraitGauss and DPTraitGauss) keep their own names, docstrings and
options, and the maths that only their family has (e.g. the cloud flux of
a density).
"""

from collections.abc import Sequence

import numpy as np

from gbkfit.params import ParamDesc
from gbkfit.params.pdescs import ParamScalarDesc
from .base import Amplitude, params_nw_distortion, splits_nw_distortion
from .options import Blobs, Harmonic


__all__ = [
    'PT_UID_EXP',
    'PT_UID_GAUSS',
    'PT_UID_GGAUSS',
    'PT_UID_LORENTZ',
    'PT_UID_MIXTURE_EXP',
    'PT_UID_MIXTURE_GAUSS',
    'PT_UID_MIXTURE_GGAUSS',
    'PT_UID_MIXTURE_MOFFAT',
    'PT_UID_MOFFAT',
    'PT_UID_NW_DISTORTION',
    'PT_UID_NW_HARMONIC',
    'PT_UID_NW_UNIFORM',
    'PT_UID_SECH2',
    'PT_UID_UNIFORM',
    'ExponentialShape',
    'GaussShape',
    'GGaussShape',
    'LorentzShape',
    'MixtureExponentialShape',
    'MixtureGaussShape',
    'MixtureGGaussShape',
    'MixtureMoffatShape',
    'MoffatShape',
    'NWDistortionShape',
    'NWHarmonicShape',
    'NWUniformShape',
    'Sech2Shape',
    'UniformShape'
]


# The uids of the shapes in the native kernels (traits.hpp PT_UID_*): the
# same in every family; the uids of the traits of one family alone are in
# blocks of their own (see each family)
PT_UID_UNIFORM = 1
PT_UID_EXP = 2
PT_UID_GAUSS = 3
PT_UID_GGAUSS = 4
PT_UID_LORENTZ = 5
PT_UID_MOFFAT = 6
PT_UID_SECH2 = 7
PT_UID_MIXTURE_EXP = 51
PT_UID_MIXTURE_GAUSS = 52
PT_UID_MIXTURE_GGAUSS = 53
PT_UID_MIXTURE_MOFFAT = 54
PT_UID_NW_UNIFORM = 101
PT_UID_NW_HARMONIC = 102
PT_UID_NW_DISTORTION = 103


class _Shape:
    """
    A shape that traits of several families share: its amplitude is that
    of the family of the trait (its class attribute AMPLITUDE).
    """
    AMPLITUDE: Amplitude

    def axisymmetric(self) -> bool:
        return True


class UniformShape(_Shape):
    """The uniform shape: a."""

    @staticmethod
    def type() -> str:
        return 'uniform'

    @staticmethod
    def type_title() -> str:
        return 'Uniform'

    @staticmethod
    def uid() -> int:
        return PT_UID_UNIFORM

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(a=self.AMPLITUDE.level())


class ExponentialShape(_Shape):
    """The exponential shape: a exp(-r / s)."""

    @staticmethod
    def type() -> str:
        return 'exponential'

    @staticmethod
    def type_title() -> str:
        return 'Exponential'

    @staticmethod
    def uid() -> int:
        return PT_UID_EXP

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(a=self.AMPLITUDE.central(), s=ParamScalarDesc(
            title='Scale length',
            desc=f"the radius where the {self.AMPLITUDE.quantity} falls by "
                 f"a factor e", min=0, unit='arcsec', default=1))

class GaussShape(_Shape):
    """The Gaussian shape: a exp(-r^2 / (2 s^2))."""

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def type_title() -> str:
        return 'Gaussian'

    @staticmethod
    def uid() -> int:
        return PT_UID_GAUSS

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(a=self.AMPLITUDE.central(), s=ParamScalarDesc(
            title='Width (sigma)',
            desc="the standard deviation of the Gaussian", min=0,
            unit='arcsec', default=1))


class GGaussShape(_Shape):
    """The generalised Gaussian shape: a exp(-(r / s)^b)."""

    @staticmethod
    def type() -> str:
        return 'ggauss'

    @staticmethod
    def type_title() -> str:
        return 'Generalised Gaussian'

    @staticmethod
    def uid() -> int:
        return PT_UID_GGAUSS

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=self.AMPLITUDE.central(),
            s=ParamScalarDesc(
                title='Scale length',
                desc=f"the radius where the {self.AMPLITUDE.quantity} falls "
                     f"by a factor e", min=0, unit='arcsec', default=1),
            b=ParamScalarDesc(
                title='Shape',
                desc="the shape: 1 is the exponential, 2 a Gaussian",
                min=0, unit='', default=1))


class LorentzShape(_Shape):
    """The Lorentzian shape: a / (1 + (r / s)^2)."""

    @staticmethod
    def type() -> str:
        return 'lorentz'

    @staticmethod
    def type_title() -> str:
        return 'Lorentzian'

    @staticmethod
    def uid() -> int:
        return PT_UID_LORENTZ

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(a=self.AMPLITUDE.central(), s=ParamScalarDesc(
            title='Half width at half maximum',
            desc=f"the radius where the {self.AMPLITUDE.quantity} falls to "
                 f"half", min=0, unit='arcsec', default=1))


class MoffatShape(_Shape):
    """The Moffat shape: a (1 + (r / s)^2)^-b."""

    @staticmethod
    def type() -> str:
        return 'moffat'

    @staticmethod
    def type_title() -> str:
        return 'Moffat'

    @staticmethod
    def uid() -> int:
        return PT_UID_MOFFAT

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(
            a=self.AMPLITUDE.central(),
            s=ParamScalarDesc(
                title='Core radius', desc="the radius of the flat core",
                min=0, unit='arcsec', default=1),
            b=ParamScalarDesc(
                title='Power index',
                desc="the power of the fall, as r^(-2b) far out",
                min=0, unit='', default=2))


class Sech2Shape(_Shape):
    """The sech^2 shape: a sech^2(r / s)."""

    @staticmethod
    def type() -> str:
        return 'sech2'

    @staticmethod
    def type_title() -> str:
        return 'Sech-squared'

    @staticmethod
    def uid() -> int:
        return PT_UID_SECH2

    def params_sm(self) -> dict[str, ParamDesc]:
        return dict(a=self.AMPLITUDE.central(), s=ParamScalarDesc(
            title='Scale length', desc="the scale length of the sech^2",
            min=0, unit='arcsec', default=1))


class _Blobs(_Shape):
    """
    A shape of elliptical blobs (see options.Blobs): the columns of their
    table, of the amplitudes of its family, and with the shape b of the
    7-parameter mixtures, of the valid range _SHAPE (None without b).
    """
    _SHAPE: dict | None = None

    def axisymmetric(self) -> bool:
        return False

    def params_sm(self) -> dict[str, ParamDesc]:
        return self._option(Blobs).params(
            shape=self._SHAPE, unit=self.AMPLITUDE.unit)


class MixtureExponentialShape(_Blobs):
    """Blobs, each an exponential of the elliptical distance."""

    @staticmethod
    def type() -> str:
        return 'mixture_exponential'

    @staticmethod
    def type_title() -> str:
        return 'Exponential blobs'

    @staticmethod
    def uid() -> int:
        return PT_UID_MIXTURE_EXP


class MixtureGaussShape(_Blobs):
    """Blobs, each a Gaussian of the elliptical distance."""

    @staticmethod
    def type() -> str:
        return 'mixture_gauss'

    @staticmethod
    def type_title() -> str:
        return 'Gaussian blobs'

    @staticmethod
    def uid() -> int:
        return PT_UID_MIXTURE_GAUSS


class MixtureGGaussShape(_Blobs):
    """Blobs, each a generalised Gaussian of the elliptical distance."""
    _SHAPE = dict(min=0)

    @staticmethod
    def type() -> str:
        return 'mixture_ggauss'

    @staticmethod
    def type_title() -> str:
        return 'Generalised Gaussian blobs'

    @staticmethod
    def uid() -> int:
        return PT_UID_MIXTURE_GGAUSS


class MixtureMoffatShape(_Blobs):
    """Blobs, each a Moffat profile of the elliptical distance."""
    _SHAPE = dict(min=0)

    @staticmethod
    def type() -> str:
        return 'mixture_moffat'

    @staticmethod
    def type_title() -> str:
        return 'Moffat blobs'

    @staticmethod
    def uid() -> int:
        return PT_UID_MIXTURE_MOFFAT


class NWUniformShape(_Shape):
    """The value at each ring: a(r)."""

    @staticmethod
    def type() -> str:
        return 'nw_uniform'

    @staticmethod
    def type_title() -> str:
        return 'Uniform (per ring)'

    @staticmethod
    def uid() -> int:
        return PT_UID_NW_UNIFORM

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return dict(a=self.AMPLITUDE.rings(nnodes))


class NWHarmonicShape(_Shape):
    """A harmonic at each ring: a(r) cos(m (theta - p(r)))."""

    @staticmethod
    def type() -> str:
        return 'nw_harmonic'

    @staticmethod
    def type_title() -> str:
        return 'Harmonic (per ring)'

    @staticmethod
    def uid() -> int:
        return PT_UID_NW_HARMONIC

    def axisymmetric(self) -> bool:
        # (a harmonic of order 0 is uniform)
        return self._option(Harmonic).order == 0

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return self._option(Harmonic).params(nnodes, unit=self.AMPLITUDE.unit)


class NWDistortionShape(_Shape):
    """
    A value around one azimuth at each ring: a(r) exp(-(dtheta r)^2 / (2
    s(r)^2)).
    """

    @staticmethod
    def type() -> str:
        return 'nw_distortion'

    @staticmethod
    def type_title() -> str:
        return 'Distortion (per ring)'

    @staticmethod
    def uid() -> int:
        return PT_UID_NW_DISTORTION

    def axisymmetric(self) -> bool:
        return False

    def params_rnw(self, nnodes: int) -> dict[str, ParamDesc]:
        return params_nw_distortion(nnodes, unit=self.AMPLITUDE.unit)

    def azimuthal_edges(
            self,
            r: np.ndarray,
            params: dict[str, np.ndarray],
            subrnodes: Sequence[float]
    ) -> np.ndarray | None:
        return splits_nw_distortion(r, params, subrnodes)
