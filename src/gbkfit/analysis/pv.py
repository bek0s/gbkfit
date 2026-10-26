"""
Position-velocity diagrams of spectral cubes along slits, and the
rotation of a galaxy traced on them (as GIPSY and NEMO users do): by the
envelope of the emission (Sofue 1996), its peak, or its mean.
"""

import dataclasses
import warnings

import numpy as np
import numpy.typing as npt
import scipy.ndimage

from ._fields import galaxy_axes


__all__ = [
    'PVDiagram',
    'TracedRotation',
    'pv_diagram',
    'trace_rotation'
]


# How the rotation is traced on a position-velocity diagram
TRACE_METHODS = ('envelope', 'peak', 'mean')


@dataclasses.dataclass(frozen=True)
class PVDiagram:
    """
    A position-velocity diagram (see pv_diagram).

    Attributes
    ----------
    positions : np.ndarray
        The positions along the slit, from the centre (positive towards
        its position angle).
    velocities : np.ndarray
        The velocities of the channels.
    data : np.ndarray
        The mean intensity in the slit at each velocity and position
        (velocities, positions); NaN where the slit has no pixel.
    """
    positions: np.ndarray
    velocities: np.ndarray
    data: np.ndarray


@dataclasses.dataclass(frozen=True)
class TracedRotation:
    """
    The rotation of a galaxy traced on a position-velocity diagram along
    its major axis (see trace_rotation).

    Attributes
    ----------
    positions : np.ndarray
        The positions along the slit (see PVDiagram).
    velocity : np.ndarray
        The traced velocity at each position (NaN where none).
    radius : np.ndarray
        The radius of each position, |position|.
    rotation : np.ndarray
        The rotation at each position: (velocity - systemic) / sin(i) on
        the receding side (positive positions), its opposite on the
        approaching side, less the dispersion of the envelope method.
    """
    positions: np.ndarray
    velocity: np.ndarray
    radius: np.ndarray
    rotation: np.ndarray


def pv_diagram(
        cube: npt.ArrayLike,
        velocities: npt.ArrayLike,
        centre: tuple[float, float],
        pa: float,
        width: int = 1,
        step: float = 1.0
) -> PVDiagram:
    """
    Return the position-velocity diagram of a spectral cube along a slit
    through the centre at a position angle, on the pixels of the cube (x
    its column, y its row; the positions, the width and the step in
    pixels): at each position along the slit, every step, the mean of the
    spectra at width points across it, a pixel apart, each interpolated
    linearly between the pixels (NaN beyond the cube).

    Parameters
    ----------
    cube : ArrayLike
        The cube (channels, ny, nx; NaN where it has no value).
    velocities : ArrayLike
        The velocities of its channels.
    centre : tuple of float
        The centre (x, y) of the slit (pixels).
    pa : float
        The position angle (degrees) of the slit, as that of the models
        (posa): from +y towards -x.
    width : int, optional
        The width of the slit (pixels).
    step : float, optional
        The step of its positions (pixels).

    Returns
    -------
    PVDiagram
        The diagram.

    Raises
    ------
    ValueError
        If the cube is not of the velocities, the width is less than 1,
        or the step is not positive.
    """
    cube = np.asarray(cube, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if cube.ndim != 3 or velocities.shape != (len(cube),):
        raise ValueError(
            "the cube must be (channels, ny, nx), of one velocity per "
            "channel")
    if width < 1 or not step > 0:
        raise ValueError(
            f"the width must be at least 1 and the step positive; they are "
            f"{width} and {step}")
    _, ny, nx = cube.shape
    major, minor = galaxy_axes(pa)
    corners = np.array([[0, 0], [nx - 1, 0], [0, ny - 1], [nx - 1, ny - 1]])
    reach = np.max(np.abs((corners - np.asarray(centre)) @ major))
    count = np.floor(reach / step)
    positions = step * np.arange(-count, count + 1)
    offsets = np.arange(width) - (width - 1) / 2
    px = centre[0] + positions[:, None] * major[0] + offsets * minor[0]
    py = centre[1] + positions[:, None] * major[1] + offsets * minor[1]
    data = np.empty((len(velocities), len(positions)))
    with warnings.catch_warnings():
        # (the mean of a position beyond the cube is NaN)
        warnings.simplefilter('ignore', RuntimeWarning)
        for k, channel in enumerate(cube):
            values = scipy.ndimage.map_coordinates(
                channel, [py.ravel(), px.ravel()], order=1, cval=np.nan)
            data[k] = np.nanmean(values.reshape(px.shape), axis=1)
    return PVDiagram(positions=positions, velocities=velocities, data=data)


def trace_rotation(
        pv: PVDiagram,
        systemic: float,
        incl: float,
        method: str = 'envelope',
        eta: float = 0.2,
        noise: float = 0.0,
        nsigma: float = 3.0,
        dispersion: float = 0.0
) -> TracedRotation:
    """
    Return the rotation of a galaxy traced on a position-velocity diagram
    along its major axis (see pv_diagram, at the position angle of its
    receding side), at each position:

    - 'envelope' (Sofue 1996; Sofue & Rubin 2001): the terminal velocity,
      the farthest from the systemic on the side of the position where
      the intensity reaches max(eta I_max, nsigma noise), I_max the peak
      of the spectrum (interpolated linearly between the channels); the
      rotation (V_t - V_sys) / sin(i) - dispersion, with dispersion =
      sqrt(sigma_ISM^2 + sigma_obs^2) of the gas and of the instrument;
    - 'peak': the velocity of the peak of the spectrum;
    - 'mean': the intensity-weighted mean velocity of the spectrum (of the
      intensities above nsigma noise).

    Parameters
    ----------
    pv : PVDiagram
        The diagram.
    systemic : float
        The systemic velocity.
    incl : float
        The inclination (degrees).
    method : str, optional
        'envelope', 'peak' or 'mean'.
    eta : float, optional
        The fraction of the peak of the envelope, in (0, 1).
    noise, nsigma : float, optional
        The noise of the intensities, and the multiple of it that they
        must reach.
    dispersion : float, optional
        The dispersion that the envelope method takes from the rotation.

    Returns
    -------
    TracedRotation
        The rotation.

    Raises
    ------
    ValueError
        If the method is not known, or eta is not in (0, 1).
    """
    if method not in TRACE_METHODS:
        raise ValueError(
            f"method must be one of {list(TRACE_METHODS)}; it is {method!r}")
    if not 0 < eta < 1:
        raise ValueError(f"eta must be in (0, 1); it is {eta}")
    v = pv.velocities
    velocity = np.full(len(pv.positions), np.nan)
    for i, position in enumerate(pv.positions):
        spectrum = pv.data[:, i]
        if position == 0 or not np.any(spectrum > nsigma * noise):
            continue
        if method == 'peak':
            velocity[i] = v[np.nanargmax(spectrum)]
        elif method == 'mean':
            weights = np.where(spectrum > nsigma * noise, spectrum, 0)
            velocity[i] = np.sum(weights * v) / np.sum(weights)
        else:
            velocity[i] = _terminal_velocity(
                v, spectrum, max(eta * np.nanmax(spectrum), nsigma * noise),
                receding=position > 0)
    sign = np.sign(pv.positions)
    rotation = sign * (velocity - systemic) / np.sin(np.radians(incl))
    if method == 'envelope':
        rotation = rotation - dispersion
    return TracedRotation(
        positions=pv.positions, velocity=velocity,
        radius=np.abs(pv.positions), rotation=rotation)


def _terminal_velocity(
        velocities: np.ndarray,
        spectrum: np.ndarray,
        threshold: float,
        receding: bool
) -> float:
    """
    Return the velocity farthest on the receding (or approaching) side
    where a spectrum reaches the threshold, interpolated linearly between
    the last channel above it and the next one beyond.
    """
    order = np.argsort(velocities)
    if receding:
        order = order[::-1]
    v, s = velocities[order], np.nan_to_num(spectrum[order], nan=-np.inf)
    above = np.flatnonzero(s >= threshold)
    first = int(above[0])
    # (without a channel outside to interpolate with: at the edge, or one
    # not a number, which gave inf / inf)
    if first == 0 or not np.isfinite(s[first - 1]):
        return float(v[first])
    # (between the channel outside and the first inside)
    fraction = (threshold - s[first - 1]) / (s[first] - s[first - 1])
    return float(v[first - 1] + fraction * (v[first] - v[first - 1]))
