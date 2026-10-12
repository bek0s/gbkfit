from collections.abc import Mapping, Sequence

import astropy.constants
import astropy.units
import numpy as np

from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Band',
    'Bands',
    'Line',
    'LineCatalogue',
    'Lines',
    'band_parser',
    'line_parser'
]


# The speed of light (km/s)
_C = astropy.constants.c.to_value('km/s')

# The lines of the catalogue (see LineCatalogue): the rest wavelength in
# vacuum of the optical lines (those of the SDSS line list) and of the
# near-infrared hydrogen lines, and the rest frequency of the radio and
# (sub)millimetre lines. Optical lines are named after their wavelength in
# air, as usual.
_CATALOGUE_LINES = {
    'oii3726': '3727.092 Angstrom',
    'oii3729': '3729.875 Angstrom',
    'hbeta': '4862.683 Angstrom',
    'oiii4959': '4960.295 Angstrom',
    'oiii5007': '5008.240 Angstrom',
    'nii6548': '6549.86 Angstrom',
    'halpha': '6564.61 Angstrom',
    'nii6583': '6585.27 Angstrom',
    'sii6716': '6718.29 Angstrom',
    'sii6731': '6732.67 Angstrom',
    'paalpha': '18756.1 Angstrom',
    'brgamma': '21661.2 Angstrom',
    'hi21': '1420.405751768 MHz',
    'co10': '115.2712018 GHz',
    'co21': '230.538 GHz',
    'co32': '345.7959899 GHz',
    'cii158': '1900.5369 GHz'}

# The bands of the catalogue (see LineCatalogue): the effective wavelength
# of the filters of SDSS, 2MASS and of the IRAC channels 1 and 2 of
# Spitzer
_CATALOGUE_BANDS = {
    'sdss_u': '3551 Angstrom',
    'sdss_g': '4686 Angstrom',
    'sdss_r': '6166 Angstrom',
    'sdss_i': '7480 Angstrom',
    'sdss_z': '8932 Angstrom',
    'twomass_j': '1.235 um',
    'twomass_h': '1.662 um',
    'twomass_ks': '2.159 um',
    'irac1': '3.550 um',
    'irac2': '4.493 um'}


def _air_refractive_index(wavelength: float) -> float:
    """
    Return the refractive index of standard air at a wavelength in vacuum
    (m), as the IAU adopted it (Morton 2000, ApJS 130, 403, eq. 8; from
    Edlen 1966 and Birch & Downs 1994): air wavelengths are those in
    vacuum over it. Raise ConfigError below 2000 Angstrom, where air
    wavelengths are not used.
    """
    if not wavelength >= 2e-7:
        raise ConfigError(
            f"wavelengths in air are used above 2000 Angstrom only; a line "
            f"is at {wavelength * 1e10:.1f} Angstrom (in vacuum)")
    # The wavenumber (1 / um), squared
    s2 = (1e-6 / wavelength) ** 2
    return 1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2)


class Line(parseutils.Serializable):
    """
    An emission line of a spectral component.

    Parameters
    ----------
    name : str
        Its name, which names its flux ratio (see Lines).
    rest : str or Quantity
        Its rest wavelength (in vacuum) or frequency (see
        gridutils.make_rest).

    Raises
    ------
    ConfigError
        If the name or the rest is not valid.
    """

    def dump(self) -> dict[str, str]:
        return dict(name=self._name, rest=str(self._rest))

    def __init__(self, name: str, rest: str | astropy.units.Quantity):
        parseutils.check_name(name)
        self._name = name
        self._rest = gridutils.make_rest(rest)

    def name(self) -> str:
        """Return its name."""
        return self._name

    def rest(self) -> astropy.units.Quantity:
        """Return its rest wavelength or frequency."""
        return self._rest


line_parser = parseutils.BasicParser(Line)


class Band(parseutils.Serializable):
    """
    A band of images: a filter, of an effective wavelength.

    Parameters
    ----------
    name : str
        Its name, which names the flux ratio of the light of a component
        in it (see Bands).
    wavelength : str or Quantity
        Its effective wavelength (e.g. '6166 Angstrom').

    Raises
    ------
    ConfigError
        If the name or the wavelength is not valid.
    """

    def dump(self) -> dict[str, str]:
        return dict(name=self._name, wavelength=str(self._wavelength))

    def __init__(self, name: str, wavelength: str | astropy.units.Quantity):
        parseutils.check_name(name)
        error = ConfigError(
            f"the wavelength of a band must be a positive length with units "
            f"(e.g. '6166 Angstrom'); it is {wavelength!r}")
        try:
            wavelength = astropy.units.Quantity(wavelength)
        except (TypeError, ValueError) as e:
            raise error from e
        if not (wavelength.unit.is_equivalent(astropy.units.m)
                and wavelength.isscalar and wavelength.value > 0):
            raise error
        self._name = name
        self._wavelength = wavelength.to(astropy.units.m)

    def name(self) -> str:
        """Return its name."""
        return self._name

    def wavelength(self) -> astropy.units.Quantity:
        """Return its effective wavelength (m)."""
        return self._wavelength


band_parser = parseutils.BasicParser(Band)


class LineCatalogue:
    """
    The emission lines and the bands that configurations name: those of
    GBKFIT, and those of the user, which add to or replace them. The lines
    of GBKFIT are the brightest optical lines (oii3726, oii3729, hbeta,
    oiii4959, oiii5007, nii6548, halpha, nii6583, sii6716, sii6731), the
    near-infrared hydrogen lines paalpha and brgamma, and the radio and
    (sub)millimetre lines hi21, co10, co21, co32 and cii158; their rest
    wavelengths are in vacuum. Its bands are those of SDSS (sdss_u, sdss_g,
    sdss_r, sdss_i, sdss_z), of 2MASS (twomass_j, twomass_h, twomass_ks)
    and the IRAC channels 1 and 2 of Spitzer (irac1, irac2).

    Parameters
    ----------
    lines : Mapping of str to str or Quantity, optional
        The user's lines: the rest wavelength (in vacuum) or frequency of
        each, by name (see Line).
    bands : Mapping of str to str or Quantity, optional
        The user's bands: the effective wavelength of each, by name (see
        Band).

    Raises
    ------
    ConfigError
        If a name or a rest is not valid.

    Examples
    --------
    >>> catalogue = LineCatalogue({'mgii2796': '2796.35 Angstrom'})
    >>> catalogue.line('halpha').rest()
    <Quantity 6.56461e-07 m>
    """

    def __init__(
            self,
            lines: Mapping[str, str | astropy.units.Quantity] | None = None,
            bands: Mapping[str, str | astropy.units.Quantity] | None = None
    ):
        self._lines = {
            name: Line(name, rest)
            for name, rest in (_CATALOGUE_LINES | dict(lines or {})).items()}
        self._bands = {
            name: Band(name, wavelength)
            for name, wavelength in (
                _CATALOGUE_BANDS | dict(bands or {})).items()}

    def names(self) -> tuple[str, ...]:
        """Return the names of its lines."""
        return tuple(self._lines)

    def band_names(self) -> tuple[str, ...]:
        """Return the names of its bands."""
        return tuple(self._bands)

    def band(self, name: str) -> Band:
        """
        Return its band of the given name.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        Band
            The band.

        Raises
        ------
        ConfigError
            If it has no band of the name.
        """
        if name not in self._bands:
            raise ConfigError(
                f"there is no band {name!r} in the catalogue; its bands: "
                f"{list(self._bands)}")
        return self._bands[name]

    def line(self, name: str) -> Line:
        """
        Return its line of the given name.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        Line
            The line.

        Raises
        ------
        ConfigError
            If it has no line of the name.
        """
        if name not in self._lines:
            raise ConfigError(
                f"there is no line {name!r} in the catalogue; its lines: "
                f"{list(self._lines)}")
        return self._lines[name]


class Lines:
    """
    The emission lines of a spectral component. Without lines, it has one
    line at the velocity of the spectral axis, whatever the line is. With
    lines, each is at its place on the spectral axis, whose velocities
    refer to a rest (see gridutils.Coords), and the flux of each line
    after the first, relative to the first, is a parameter
    ('{name}_ratio').

    Parameters
    ----------
    lines : Sequence of Line, optional
        The lines; one line at the velocity of the spectral axis if None.

    Raises
    ------
    ConfigError
        If there are no lines, or their names repeat.
    """

    def __init__(self, lines: Sequence[Line] | None):
        if lines is not None:
            lines = tuple(lines)
            if not lines:
                raise ConfigError("lines needs at least one line")
            names = [line.name() for line in lines]
            if repeated := sorted({n for n in names if names.count(n) > 1}):
                raise ConfigError(
                    f"the lines must have different names; repeated: "
                    f"{repeated}")
        self._lines = lines
        self._ratios = {} if lines is None else {
            f'{line.name()}_ratio': ParamScalarDesc(
                desc=f"the flux of line {line.name()} relative to line "
                     f"{lines[0].name()}")
            for line in lines[1:]}

    def lines(self) -> tuple[Line, ...] | None:
        """Return the lines, if given."""
        return self._lines

    def names(self) -> tuple[str, ...]:
        """Return the names of the lines (none without lines)."""
        return () if self._lines is None else tuple(
            line.name() for line in self._lines)

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """Return the parameters of the lines: their flux ratios."""
        return self._ratios

    def select(self, names: Sequence[str] | None) -> tuple[int, ...]:
        """
        Return the indices of the lines of the given names (all if None). The
        one line of a component without lines is always selected. Raise
        ConfigError if none of the lines is selected.
        """
        if self._lines is None:
            return (0,)
        if names is None:
            return tuple(range(len(self._lines)))
        selected = tuple(
            i for i, name in enumerate(self.names()) if name in names)
        if not selected:
            raise ConfigError(
                f"a component of the lines {list(self.names())} has none of "
                f"the selected lines {list(names)}; leave it out with the "
                f"components of the observation")
        return selected

    def ratio_name(self, index: int) -> str | None:
        """
        Return the name of the flux ratio of the line of the given index:
        none for the first line, whose flux is that of the component.
        """
        return None if index == 0 else f'{self.names()[index]}_ratio'

    def values(self, spectral: gridutils.Grid) -> np.ndarray:
        """
        Return the offset, scale and flux of each line on a spectral axis
        (a grid of one axis; an array of shape (nlines, 3); see
        DiskPlan.evaluate), such that a line of velocity v is at offset +
        scale * v on the axis, with a flux of 1 (the ratios are
        parameters).

        With optical velocities (a rest wavelength), a line of rest
        wavelength w at velocity v is at w (1 + v / c), which is at the
        velocity c (k - 1) + k v of the axis, where k is w over the rest of
        the axis. With radio velocities (a rest frequency), a line of rest
        frequency f at velocity v is at f (1 - v / c), which is at the
        velocity c (1 - k) + k v, where k is f over the rest.

        The rest wavelengths of the lines are in vacuum. If those of the
        axis are in air (see gridutils.Coords), w is that of each line in
        air: divided by the refractive index of air at the wavelength (in
        vacuum) where the line is at the velocity of the centre of the
        axis. (The index changes by less than 1e-7 across 2000 km/s.)
        """
        if self._lines is None:
            return np.array([[0.0, 1.0, 1.0]])
        rest = spectral.coords.rest
        if rest is None:
            raise ConfigError(
                "the lines of a component need the rest wavelength or "
                "frequency that the velocities of the spectral axis refer "
                "to: RESTWAV or RESTFRQ in the header of the data, or the "
                "rest option of the observable (spec_rest for moments and "
                "spectra in regions)")
        optical = rest.unit == astropy.units.m
        air = spectral.coords.medium == 'air'
        centre = spectral.zero()[0] \
            + (spectral.size[0] - 1) / 2 * spectral.coords.step[0]
        values = []
        for line in self._lines:
            ratio = float(line.rest().to_value(
                rest.unit, astropy.units.spectral()) / rest.value)
            if air:
                wavelength = line.rest().to_value(
                    astropy.units.m, astropy.units.spectral())
                ratio /= _air_refractive_index(wavelength * (1 + centre / _C))
            offset = _C * (ratio - 1) if optical else _C * (1 - ratio)
            values.append((offset, ratio, 1.0))
        return np.array(values)


class Bands:
    """
    The bands of a component that emits in bands. Without bands, its light
    is the same in every band. With bands, it emits in those only: its
    light in the first band is its density, and that in each other band,
    relative to the first, is a parameter ('{name}_ratio').

    Parameters
    ----------
    bands : Sequence of Band, optional
        The bands; every band if None.

    Raises
    ------
    ConfigError
        If there are no bands, or their names repeat.
    """

    def __init__(self, bands: Sequence[Band] | None):
        if bands is not None:
            bands = tuple(bands)
            if not bands:
                raise ConfigError("bands needs at least one band")
            names = [band.name() for band in bands]
            if repeated := sorted({n for n in names if names.count(n) > 1}):
                raise ConfigError(
                    f"the bands must have different names; repeated: "
                    f"{repeated}")
        self._bands = bands
        self._ratios = {} if bands is None else {
            f'{band.name()}_ratio': ParamScalarDesc(
                desc=f"the flux in band {band.name()} relative to that in "
                     f"band {bands[0].name()}")
            for band in bands[1:]}

    def bands(self) -> tuple[Band, ...] | None:
        """Return the bands, if given."""
        return self._bands

    def names(self) -> tuple[str, ...]:
        """Return the names of the bands (none without bands)."""
        return () if self._bands is None else tuple(
            band.name() for band in self._bands)

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """Return the parameters of the bands: their flux ratios."""
        return self._ratios

    def emits_in(self, band: Band | None) -> bool:
        """
        Check whether the component emits in a band (that of an image;
        None if not known).

        Raises
        ------
        ConfigError
            If the band is not known, and the component has bands.
        """
        if self._bands is None:
            return True
        if band is None:
            raise ConfigError(
                f"a component emits in the bands {list(self.names())}; the "
                f"image that sees it needs a band")
        return band.name() in self.names()

    def ratio_name(self, band: Band | None) -> str | None:
        """
        Return the name of the flux ratio of the light in a band in which
        the component emits (see emits_in): none in its first band, or
        without bands.
        """
        names = self.names()
        if not names or band.name() == names[0]:
            return None
        return f'{band.name()}_ratio'
