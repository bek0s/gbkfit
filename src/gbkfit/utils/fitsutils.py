import typing

import astropy.io.fits
import astropy.units
import astropy.wcs
import numpy as np

from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Coords',
    'Grid',
    'GridData',
    'SpectraData',
    'VELOCITY_TYPES',
    'centre_missing_crpix',
    'make_grid',
    'make_rest',
    'read_data',
    'sky_positions',
    'write_data',
    'write_spectra'
]


# The spectral axis types that are velocities (FITS WCS Paper III)
VELOCITY_TYPES = ('VRAD', 'VOPT', 'VELO')

_KM_S = astropy.units.km / astropy.units.s


class Coords(typing.NamedTuple):
    """
    The world coordinates of the pixels of data, in the units of the
    model, for each axis in FITS order (x, y and, for spectral cubes, the
    velocity):

    - step: arcsec per pixel (spatial axes), km/s per channel (velocity)
    - rpix: the reference pixel (0-based)
    - rval: the world position at rpix: RA and Dec (degrees), and the
      velocity (km/s). The model measures the spatial axes from rpix.
    - rota: the rotation of the pixel grid on the sky: the position angle
      (degrees, north through east) of the +y axis.
    - rest: the rest wavelength or frequency that the velocities of the
      spectral axis refer to (an astropy Quantity in m or Hz; see
      make_rest), or None. The velocities of a rest wavelength are optical
      (VOPT), and those of a rest frequency radio (VRAD).

    Axes without a known type have the values of their header.
    """
    step: tuple[float, ...]
    rpix: tuple[float, ...]
    rval: tuple[float, ...]
    rota: float
    rest: astropy.units.Quantity | None = None

    def axes(self, *indices: int) -> 'Coords':
        """The coordinates of the given axes."""
        def pick(values):
            return tuple(values[i] for i in indices)
        return Coords(
            pick(self.step), pick(self.rpix), pick(self.rval), self.rota,
            self.rest)


class Grid(typing.NamedTuple):
    """
    A grid of pixels: its size and world coordinates (see Coords) for each
    axis in FITS order, and the index of its spectral axis (or None).
    """
    size: tuple[int, ...]
    coords: Coords
    spectral_axis: int | None

    def zero(self) -> tuple[float, ...]:
        """
        The world position of the first pixel on each axis, in model units:
        the spatial axes are measured from the reference pixel, and the
        spectral axis from its world value there.
        """
        step, rpix, rval = self.coords.step, self.coords.rpix, self.coords.rval
        return tuple(
            (rval[axis] if axis == self.spectral_axis else 0)
            - rpix[axis] * step[axis]
            for axis in range(len(self.size)))

    def spatial(self) -> 'Grid':
        """The grid of the x and y axes."""
        return Grid(
            self.size[:2], self.coords.axes(0, 1)._replace(rest=None), None)

    def spectral(self) -> 'Grid':
        """The grid of the spectral axis (one axis)."""
        axis = self.spectral_axis
        return Grid((self.size[axis],), self.coords.axes(axis), 0)


def make_grid(
        size: typing.Sequence[int],
        step: float | typing.Sequence[float] | None = None,
        rpix: float | typing.Sequence[float] | None = None,
        rval: float | typing.Sequence[float] | None = None,
        rota: float | None = None,
        spectral_axis: int | None = None,
        rest: typing.Any = None
) -> Grid:
    """
    A grid of the given size (FITS order) with the given world
    coordinates (see Coords), each a value or one per axis, or their
    defaults: step 1, the reference pixel at the centre, reference value 0
    and no rotation. rest (see make_rest) needs a spectral axis.
    """
    ndim = len(size)
    if step is None:
        step = 1
    if rpix is None:
        rpix = tuple((np.asarray(size) / 2 - 0.5).tolist())
    if rval is None:
        rval = 0
    if rota is None:
        rota = 0
    step, rpix, rval = (
        (value,) * ndim if np.ndim(value) == 0 else tuple(value)
        for value in (step, rpix, rval))
    for name, value in dict(step=step, rpix=rpix, rval=rval).items():
        if len(value) != ndim:
            raise RuntimeError(
                f"the grid has {ndim} axes, but {name} has {len(value)} "
                f"values")
    if not all(value > 0 for value in step):
        raise RuntimeError(f"step must be positive; it is {step}")
    if rest is not None and spectral_axis is None:
        raise RuntimeError("a grid without a spectral axis has no rest")
    return Grid(
        tuple(size), Coords(step, rpix, rval, rota, make_rest(rest)),
        spectral_axis)


def make_rest(value: typing.Any) -> astropy.units.Quantity | None:
    """
    The rest wavelength or frequency of a spectral axis (see Coords), in m
    or Hz, from a Quantity or a string with units (e.g. '6562.8 Angstrom',
    '1420.405752 MHz'), or None.
    """
    if value is None:
        return None
    error = ConfigError(
        f"the rest of a spectral axis must be a positive wavelength or "
        f"frequency with units (e.g. '6562.8 Angstrom'); it is {value!r}")
    try:
        rest = astropy.units.Quantity(value)
    except (TypeError, ValueError) as e:
        raise error from e
    for unit in (astropy.units.m, astropy.units.Hz):
        if rest.unit.is_equivalent(unit) and rest.isscalar and rest.value > 0:
            return rest.to(unit)
    raise error


def sky_positions(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """
    The positions on the sky of the centres of the pixels of the x and y
    axes of a grid, in the frame of the model (arcsec from the reference
    pixel, x and y like xpos and ypos): two arrays of shape (ny, nx). The
    pixel grid is rotated on the sky by rota (see Coords).
    """
    size_x, size_y = grid.size[:2]
    zero_x, zero_y = grid.zero()[:2]
    step_x, step_y = grid.coords.step[:2]
    j, i = np.mgrid[0:size_y, 0:size_x]
    x = zero_x + i * step_x
    y = zero_y + j * step_y
    rota = np.radians(grid.coords.rota)
    return (x * np.cos(rota) - y * np.sin(rota),
            x * np.sin(rota) + y * np.cos(rota))


class GridData(typing.NamedTuple):
    """
    Data on a grid with world coordinates (see Coords and write_data).
    spectral_axis is the index of its spectral axis (FITS order), or None.
    """
    data: np.ndarray
    coords: Coords
    spectral_axis: int | None


class SpectraData(typing.NamedTuple):
    """
    Spectra of regions of the sky (data of shape (nchannels, nregions);
    FITS: the regions along x, the velocity along y), with the world
    coordinates of their spectral axis (a Coords of one axis; see
    write_spectra).
    """
    data: np.ndarray
    coords: Coords


def read_data(
        filename: str,
        hdu: int | str = 0,
        rpix: typing.Sequence[float] | None = None,
        rval: typing.Sequence[float] | None = None,
        rest: typing.Any = None
) -> tuple[np.ndarray, Coords]:
    """
    The data of a FITS file (from the given HDU) and its world
    coordinates in the units of the model (see Coords).

    rpix is CRPIX - 1 (the centre of the axes without CRPIX), and rval is
    CRVAL, the world position at rpix. Either can be given instead (in
    model units), and the other is computed from the header. The rest of
    the spectral axis (see _spectral_rest) is that of the header (RESTWAV
    or RESTFRQ), or rest if given.

    Raise ConfigError for coordinates the model cannot represent: a
    header it cannot read, a mirrored (east to the right of north) or
    skewed pixel grid, celestial axes coupled to other axes, and
    spectral axes that are not velocities.
    """
    with astropy.io.fits.open(filename) as hdulist:
        data = hdulist[hdu].data
        header = hdulist[hdu].header
    if data is None:
        raise ConfigError(
            f"{filename}: HDU {hdu} has no data; choose the HDU with the "
            f"data (e.g. hdu: SCI)")
    header = centre_missing_crpix(header, data.shape)
    try:
        wcs = astropy.wcs.WCS(header)
    except Exception as e:
        raise ConfigError(
            f"{filename}: invalid world coordinates: {e}") from e
    if wcs.naxis != data.ndim:
        raise ConfigError(
            f"{filename}: the header has world coordinates for {wcs.naxis} "
            f"axes, but the data has {data.ndim}")
    # The linear transformation from pixels to intermediate world
    # coordinates: one row for each world axis, one column for each
    # pixel axis
    linear = wcs.wcs.get_cdelt()[:, None] * wcs.wcs.get_pc()
    # The factor from the world units of each axis to model units
    scale = [_axis_scale(filename, wcs, axis) for axis in range(wcs.naxis)]
    rota = 0.0
    if wcs.wcs.lng >= 0:
        rota = _celestial_rotation(filename, linear, wcs.wcs.lng, wcs.wcs.lat)
    _check_uncoupled(filename, wcs, linear)
    step = [
        np.hypot(*linear[:, axis][[wcs.wcs.lng, wcs.wcs.lat]]) * 3600
        if axis in (wcs.wcs.lng, wcs.wcs.lat)
        else linear[axis, axis] * scale[axis]
        for axis in range(wcs.naxis)]
    if wcs.wcs.spec >= 0 and step[wcs.wcs.spec] < 0:
        raise ConfigError(
            f"{filename}: the velocity decreases along the spectral axis; "
            f"reverse the axis with gbkfit-cli prep")
    if rpix is not None:
        rpix = np.broadcast_to(np.asarray(rpix, float), wcs.naxis)
    if rval is not None:
        rval = np.broadcast_to(np.asarray(rval, float), wcs.naxis)
    if rpix is None and rval is None:
        rpix = (wcs.wcs.crpix - 1).tolist()
        rval = np.multiply(wcs.wcs.crval, scale).tolist()
    elif rpix is None:
        world = np.divide(rval, scale)
        rpix = np.ravel(wcs.world_to_pixel_values(*world)).tolist()
    elif rval is None:
        world = np.ravel(wcs.pixel_to_world_values(*rpix))
        rval = np.multiply(world, scale).tolist()
    coords = Coords(
        tuple(float(x) for x in step),
        tuple(float(x) for x in rpix),
        tuple(float(x) for x in rval),
        float(rota),
        _spectral_rest(filename, wcs, make_rest(rest)))
    return data, coords


def centre_missing_crpix(
        header: astropy.io.fits.Header,
        shape: tuple[int, ...]
) -> astropy.io.fits.Header:
    """
    A copy of the header of data of the given shape (numpy order) in which
    the axes without a reference pixel (CRPIXn) have it at their centre.
    This is the convention of the model; FITS would put it at 0.
    """
    header = header.copy()
    for n, size in enumerate(shape[::-1], start=1):
        if f'CRPIX{n}' not in header:
            header[f'CRPIX{n}'] = size / 2 + 0.5
    return header


def write_data(
        filename: str,
        data: np.ndarray,
        coords: Coords,
        spectral_axis: int | None = None,
        overwrite: bool = False
) -> None:
    """
    Write data with the world coordinates of the model (see Coords) to a
    FITS file. The axes of the data are the x and y axes of the sky (RA
    and Dec, TAN projection, rotated with a PC matrix), followed by the
    spectral axis (spectral_axis = 2, a radio velocity) or by a position
    along the line of sight (spectral_axis = None, an offset in arcsec
    from its reference pixel), or a position along a slit followed by the
    spectral axis (spectral_axis = 1), or the spectral axis alone
    (spectral_axis = 0).
    """
    header = astropy.io.fits.Header()
    if spectral_axis == 0 and data.ndim == 1:
        header.update(_velocity_header(coords, 1))
    elif spectral_axis is None and data.ndim == 2:
        header.update(_sky_header(coords))
    elif spectral_axis == 2 and data.ndim == 3:
        header.update(_sky_header(coords))
        header.update(_velocity_header(coords, 3))
    elif spectral_axis is None and data.ndim == 3:
        header.update(_sky_header(coords))
        header.update(_offset_header(
            3, coords.step[2], coords.rpix[2], coords.rval[2]))
    elif spectral_axis == 1 and data.ndim == 2:
        # The position along the slit is an offset from its reference
        header.update(_offset_header(
            1, coords.step[0], coords.rpix[0], 0.0))
        header.update(_velocity_header(coords, 2))
    else:
        raise ValueError(
            f"unsupported data: {data.ndim} axes, of which the spectral "
            f"axis is {spectral_axis}")
    astropy.io.fits.writeto(
        filename, data, header,
        output_verify='exception', overwrite=overwrite, checksum=True)


def write_spectra(
        filename: str,
        data: np.ndarray,
        coords: Coords,
        overwrite: bool = False
) -> None:
    """
    Write spectra of regions (see SpectraData) to a FITS file: the regions
    along x (an index, without world coordinates), and the velocity along
    y, with the world coordinates of a spectral axis (a Coords of one
    axis).
    """
    header = astropy.io.fits.Header(_velocity_header(coords, 2, 0))
    astropy.io.fits.writeto(
        filename, data, header,
        output_verify='exception', overwrite=overwrite, checksum=True)


def _sky_header(coords):
    """The header keywords of the x and y axes on the sky."""
    step_x, step_y = coords.step[:2]
    rota = np.radians(coords.rota)
    # The +y axis points to the position angle rota, and the +x axis to
    # rota - 90 (east is to the left of north). Intermediate world x
    # points east and y north. The rows of the PC matrix are scaled by
    # CDELT, so PC is a rotation only for square pixels.
    cd = np.array([
        [-step_x * np.cos(rota), step_y * np.sin(rota)],
        [step_x * np.sin(rota), step_y * np.cos(rota)]]) / 3600
    cdelt = np.array([-step_x, step_y]) / 3600
    pc = cd / cdelt[:, None]
    return dict(
        CTYPE1='RA---TAN', CUNIT1='deg', CDELT1=cdelt[0],
        CRPIX1=coords.rpix[0] + 1, CRVAL1=coords.rval[0],
        CTYPE2='DEC--TAN', CUNIT2='deg', CDELT2=cdelt[1],
        CRPIX2=coords.rpix[1] + 1, CRVAL2=coords.rval[1],
        PC1_1=pc[0, 0], PC1_2=pc[0, 1],
        PC2_1=pc[1, 0], PC2_2=pc[1, 1])


def _offset_header(n, step, rpix, rval):
    """The header keywords of an offset in arcsec, FITS axis n."""
    return {
        f'CTYPE{n}': 'OFFSET', f'CUNIT{n}': 'arcsec', f'CDELT{n}': step,
        f'CRPIX{n}': rpix + 1, f'CRVAL{n}': rval}


def _velocity_header(coords, n, index=None):
    """
    The header keywords of the spectral axis, FITS axis n, whose world
    coordinates are those of axis index of coords (by default n - 1):
    optical velocities with a rest wavelength, radio velocities with a
    rest frequency or without a rest.
    """
    index = n - 1 if index is None else index
    rest = coords.rest
    header = {
        f'CTYPE{n}': 'VRAD', f'CUNIT{n}': 'km/s',
        f'CDELT{n}': coords.step[index],
        f'CRPIX{n}': coords.rpix[index] + 1,
        f'CRVAL{n}': coords.rval[index]}
    if rest is not None and rest.unit == astropy.units.m:
        header.update({f'CTYPE{n}': 'VOPT', 'RESTWAV': rest.value})
    elif rest is not None:
        header.update(RESTFRQ=rest.value)
    return header


def _spectral_rest(filename, wcs, rest):
    """
    The rest of the spectral axis (see Coords): that given (rest, if not
    None), or that of the header (RESTWAV or RESTFRQ), as a wavelength for
    optical velocities (VOPT) and as a frequency for radio velocities
    (VRAD). Without a spectral axis type, a given rest keeps its kind.
    Relativistic velocities (VELO) have none: their lines are not at
    linear offsets, so a given rest is an error.
    """
    if wcs.wcs.spec < 0:
        return rest
    kind = wcs.wcs.ctype[wcs.wcs.spec][:4]
    if kind == 'VELO':
        if rest is not None:
            raise ConfigError(
                f"{filename}: the spectral axis has relativistic velocities "
                f"(VELO), which have no rest; convert it to optical (VOPT) "
                f"or radio (VRAD) velocities")
        return None
    if rest is None and wcs.wcs.restwav > 0:
        rest = wcs.wcs.restwav * astropy.units.m
    if rest is None and wcs.wcs.restfrq > 0:
        rest = wcs.wcs.restfrq * astropy.units.Hz
    if rest is None:
        return None
    unit = astropy.units.m if kind == 'VOPT' else astropy.units.Hz
    return rest.to(unit, astropy.units.spectral())


def _axis_scale(filename, wcs, axis):
    """
    The factor from the world units of an axis to model units: celestial
    axes stay in degrees (only their steps become arcsec), velocity axes
    become km/s, and the other axes keep their units.
    """
    if axis != wcs.wcs.spec:
        return 1.0
    ctype = wcs.wcs.ctype[axis]
    if ctype[:4] not in VELOCITY_TYPES:
        raise ConfigError(
            f"{filename}: the spectral axis is of type '{ctype}'; only "
            f"velocity axes ({', '.join(VELOCITY_TYPES)}) are supported")
    return astropy.units.Unit(wcs.wcs.cunit[axis]).to(_KM_S)


def _celestial_rotation(filename, linear, lng, lat):
    """
    The position angle (degrees, north through east) of the +y pixel axis,
    from the linear transformation of the celestial axes. Intermediate
    world x points east and y north.
    """
    matrix = linear[np.ix_([lng, lat], [lng, lat])]
    column_x, column_y = matrix.T
    if np.linalg.det(matrix) > 0:
        raise ConfigError(
            f"{filename}: the image is mirrored (east is to the right of "
            f"north), which the model does not support")
    cosine = column_x @ column_y / np.hypot(*column_x) / np.hypot(*column_y)
    if abs(cosine) > 1e-6:
        raise ConfigError(
            f"{filename}: the pixel grid is skewed (its axes are not "
            f"perpendicular on the sky), which the model does not support")
    return float(np.degrees(np.arctan2(column_y[0], column_y[1])))


def _check_uncoupled(filename, wcs, linear):
    """
    Raise ConfigError if a pixel axis other than the celestial pair is
    rotated into another world axis (e.g. velocity varying along x).
    """
    celestial = {wcs.wcs.lng, wcs.wcs.lat} - {-1}
    coupled = linear.copy()
    for i in celestial:
        for j in celestial:
            coupled[i, j] = 0
    np.fill_diagonal(coupled, 0)
    if np.any(coupled):
        raise ConfigError(
            f"{filename}: the world coordinates couple axes that the model "
            f"keeps independent (e.g. the velocity varies along a spatial "
            f"axis)")
