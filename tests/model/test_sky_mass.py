"""
Tests of the mass of galaxies laid out on the sky (see
gbkfit.model.mass.sky): the layers of each mass component against direct
integrals of its density (projected masses, lines of sight), the slices
of thick disks, the bands of traced disks, the placement and its errors,
and the unit shapes that a change of a scale keeps.
"""

import math

import numpy as np
import pytest
import scipy.integrate

from gbkfit.model.mass import (
    BandLayer, ContractionBlumenthal86, EllipticalLayer, Halo, MassContext,
    MassDisk, MassModel, MassPoint, MassSpheroid, MassTable, MassUser,
    PointLayer, ProfileNFW, ProfileNodewise, ProfilePseudoIsothermal,
    ProfileSersic, RingGeometry, RoundLayer, ShapeFlattened, SkyPlacement,
    SpheroidAxis, SpheroidDensity, SurfaceExponential, SurfaceNodewise,
    TracedDisk, TracedSpheroid, VerticalComponent, VerticalSech2,
    VerticalThin, sky)
from gbkfit.model.mass.profiles import ProfileGauss
from gbkfit.model.mass.shapes import ShapeSphere
from gbkfit.params import ModelEvaluationError


def _context(**densities) -> MassContext:
    return MassContext(
        critical_density=136.0, omega_m=0.3, ricci_scalar=0.0,
        ricci_scalar_today=0.0, densities=densities, kpc_per_arcsec=5.0)


class _Rings:
    """The rings of a disk, of one geometry (see sky.RingPlacement)."""

    def __init__(self, x=0.0, y=0.0, posa=0.0, incl=0.0, dincl=0.0):
        self._values = (x, y, posa, incl, dincl)

    def rings(self, radii):
        x, y, posa, incl, dincl = self._values
        z = np.zeros(len(radii))
        return RingGeometry(
            z + x, z + y, z + posa, incl + dincl * radii, z, z, z)


def _plane(posa=30.0, incl=50.0, centre=(0.4, -0.3)) -> SkyPlacement:
    return SkyPlacement(
        centre, disks={'gas': _Rings(*centre, posa=posa, incl=incl)})


def _layers(component, params, context=None, placement=None):
    model = MassModel(component)
    full = {f'mass_{component.name()}_{name}': value
            for name, value in params.items()}
    mass = model.sky_mass(
        full, context or _context(), placement or _plane())
    return mass.parts[component.name()]


def _sky_density(layer, x, y):
    """
    Return the surface density of the sky (Msun/kpc^2) of an elliptical
    layer or of bands at points (kpc), times its scale.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    total = np.zeros(x.shape)

    def principal(dx, dy, posa):
        p = np.radians(posa)
        return (-dx * np.sin(p) + dy * np.cos(p),
                -dx * np.cos(p) - dy * np.sin(p))
    if isinstance(layer, EllipticalLayer):
        sigma1, sigma2 = layer.stretch
        table = layer.sigma
        for row, height in enumerate(layer.heights):
            u, v = principal(
                x - layer.centre[0] - height * layer.normal[0],
                y - layer.centre[1] - height * layer.normal[1], layer.posa)
            R = np.hypot(u, v / (sigma2 / sigma1)) / sigma1
            values = sky.LogTable(
                table.start, table.step, table.values[row])(R)
            end = np.exp(table.start + table.step * (
                table.values.shape[-1] - 1))
            total += np.where(R <= end, values, 0) / (sigma1 * sigma2)
    else:
        for row, height in enumerate(layer.heights):
            for k in range(len(layer.posa)):
                sigma1, sigma2 = layer.stretch[k]
                centre = layer.centres[k] + height * layer.normals[k]
                u, v = principal(
                    x - centre[0], y - centre[1], layer.posa[k])
                R = np.hypot(u, v / (sigma2 / sigma1)) / sigma1
                c0, c1, c2 = layer.density[row, k]
                inside = (R >= layer.edges[k, 0]) & (R < layer.edges[k, 1])
                total += np.where(inside, c0 + c1 * R + c2 * R * R, 0) \
                    / (sigma1 * sigma2)
    return layer.scale * total


def _line_of_sight(density, x, y, centre, posa, incl):
    """
    Return the integral of a density of a frame (a function of its R and
    z) along the lines of sight through points of the sky (kpc), of a
    frame at a centre of the sky, oriented as a disk (posa, incl; see the
    native transform_cpos_posa_incl).
    """
    p, i = np.radians(posa), np.radians(incl)
    out = []
    for xx, yy in zip(x, y):
        dx, dy = xx - centre[0], yy - centre[1]
        xr = -dx * np.sin(p) + dy * np.cos(p)
        y1 = -dx * np.cos(p) - dy * np.sin(p)

        def rho(s):
            yd = y1 * np.cos(i) + s * np.sin(i)
            zd = -y1 * np.sin(i) + s * np.cos(i)
            return float(density(np.hypot(xr, yd), zd))
        out.append(sum(scipy.integrate.quad(
            rho, a, b, limit=400, epsabs=0, epsrel=1e-11)[0]
            for a, b in ((-np.inf, -50), (-50, 0), (0, 50), (50, np.inf))))
    return np.array(out)


def test_point():
    (layer,) = _layers(MassPoint('bh'), dict(log_m=9.0))
    assert isinstance(layer, PointLayer)
    assert layer.centre == (0.4, -0.3) and layer.scale == 1e9


def test_round_nfw_halo_against_closed_form():
    halo = Halo('halo', ProfileNFW('density_radius'))
    (layer,) = _layers(halo, dict(log_rho=7.0, rs=12.0))
    assert isinstance(layer, RoundLayer)
    R = np.geomspace(0.01, 1e4, 60)
    x = R / 12.0
    f = np.where(x < 1, np.arccosh(1 / np.minimum(x, 1)) / np.sqrt(
        np.abs(1 - x * x)), np.arccos(1 / np.maximum(x, 1)) / np.sqrt(
        np.abs(x * x - 1)))
    expected = 4 * np.pi * 1e7 * 12.0 ** 3 * (np.log(x / 2) + f)
    np.testing.assert_allclose(layer.scale * layer.mass(R), expected,
                               rtol=1e-8)


def test_round_sersic_spheroid_against_its_projection():
    # The projection of the deprojected tables of a Sersic sphere: its
    # projected mass within R, against the 2D integral of its closed
    # surface density
    profile = ProfileSersic()
    params = dict(log_m=10.5, re=2.0, n=3.0)
    (layer,) = _layers(MassSpheroid('bulge', profile), params)
    for R in (0.05, 0.5, 2.0, 8.0, 40.0):
        expected = scipy.integrate.quad(
            lambda r: 2 * np.pi * r * float(
                profile.surface_density(r, params, _context())),
            0, R, limit=400, epsrel=1e-11, points=[R / 10])[0]
        assert layer.scale * layer.mass(R) == pytest.approx(
            expected, rel=2e-6)


def test_round_contracted_halo_against_quadrature():
    # A halo that responds to its baryons: its projected mass of its
    # enclosed mass after its contraction, against adaptive quadrature
    disk = MassDisk('disk', SurfaceExponential(), VerticalThin())
    halo = Halo('halo', ProfileNFW(), baryons=['disk'],
                contraction=ContractionBlumenthal86())
    model = MassModel([disk, halo])
    params = {'mass_disk_log_m': 10.7, 'mass_disk_rd': 3.0,
              'mass_halo_log_m': 12.0, 'mass_halo_c': 10.0}
    mass = model.sky_mass(params, _context(), _plane())
    (layer,) = mass.parts['halo']
    own = model.component_params('halo', params)
    context = model._context(halo, params, _context())
    for R in (0.3, 3.0, 30.0):
        def integrand(t):
            return float(halo.enclosed_mass(
                np.array([R / np.cos(t)]), own, context)[0]) * np.cos(t)
        expected = scipy.integrate.quad(
            integrand, 0, np.arccos(1e-7), limit=500, epsrel=1e-10)[0]
        assert layer.mass(R) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize('q', [0.5, 1.6])
def test_flattened_halo_against_lines_of_sight(q):
    # A flattened (q < 1) or elongated (q > 1) NFW halo in the plane of the
    # mass: the q' identity, against the density integrated along lines of
    # sight
    halo = Halo('halo', ProfileNFW('density_radius'), shape=ShapeFlattened())
    params = dict(log_rho=7.0, rs=8.0, q=q)
    (layer,) = _layers(halo, params)
    assert isinstance(layer, EllipticalLayer)
    rng = np.random.default_rng(1)
    x, y = 0.4 + rng.uniform(-20, 20, 8), -0.3 + rng.uniform(-20, 20, 8)
    expected = _line_of_sight(
        lambda R, z: halo.density(R, z, params, _context()), x, y,
        (0.4, -0.3), 30.0, 50.0)
    np.testing.assert_allclose(_sky_density(layer, x, y), expected,
                               rtol=1e-6)


def test_flattened_sersic_spheroid_against_lines_of_sight():
    bulge = MassSpheroid('bulge', ProfileSersic(), shape=ShapeFlattened())
    params = dict(log_m=10.0, re=2.0, n=2.0, q=0.6)
    (layer,) = _layers(bulge, params)
    rng = np.random.default_rng(2)
    x, y = 0.4 + rng.uniform(-6, 6, 8), -0.3 + rng.uniform(-6, 6, 8)
    expected = _line_of_sight(
        lambda R, z: bulge.density(R, z, params, _context()), x, y,
        (0.4, -0.3), 30.0, 50.0)
    # (the density of the deprojected tables, to about 1e-6)
    np.testing.assert_allclose(_sky_density(layer, x, y), expected,
                               rtol=1e-5)


def test_nodewise_spheroid_as_bands():
    profile = ProfileNodewise([0, 1, 3, 6])
    bulge = MassSpheroid('bulge', profile, shape=ShapeFlattened())
    params = dict(log_sigma=np.array([9.0, 8.4, 7.5]), q=0.7)
    (layer,) = _layers(bulge, params)
    assert isinstance(layer, BandLayer)
    rng = np.random.default_rng(3)
    x, y = 0.4 + rng.uniform(-7, 7, 50), -0.3 + rng.uniform(-7, 7, 50)
    qp = float(sky.projected_ratio(0.7, 50.0))
    p = np.radians(30.0)
    dx, dy = x - 0.4, y + 0.3
    m = np.hypot(-dx * np.sin(p) + dy * np.cos(p),
                 (-dx * np.cos(p) - dy * np.sin(p)) / qp)
    expected = profile.surface_density(m, params, _context()) / qp
    np.testing.assert_allclose(_sky_density(layer, x, y), expected,
                               rtol=1e-12, atol=1e-6)


def test_thin_disk_inclined():
    disk = MassDisk('disk', SurfaceExponential(), VerticalThin())
    (layer,) = _layers(disk, dict(log_m=10.0, rd=3.0))
    assert isinstance(layer, EllipticalLayer)
    assert layer.stretch == pytest.approx((1.0, math.cos(math.radians(50))))
    assert layer.posa == 30.0 and layer.scale == 1e10
    # (its mass: that of its table, to the tail it drops)
    R = sky.table_radii()[:layer.sigma.values.shape[1]]
    mass = scipy.integrate.trapezoid(
        2 * np.pi * R ** 2 * layer.sigma(R)[0], np.log(R))
    assert layer.scale * mass == pytest.approx(1e10, rel=1e-6)


def test_thick_disk_slices():
    # A sech2 disk in slices: of its surface density at every radius, and
    # of the moments of its pdf of the height
    disk = MassDisk('disk', SurfaceExponential(), VerticalSech2())
    (layer,) = _layers(disk, dict(log_m=10.0, rd=3.0, hz=0.4))
    assert len(layer.heights) == 84
    R = np.array([0.1, 1.0, 5.0, 20.0])
    shares = layer.sigma(R) / np.exp(-R / 3.0) * (2 * np.pi * 9.0)
    np.testing.assert_allclose(shares.sum(axis=0), 1, rtol=1e-6)
    z2 = np.sum(shares[:, 1] * layer.heights ** 2)
    assert z2 == pytest.approx(np.pi ** 2 / 12 * 0.4 ** 2, rel=1e-4)
    p = np.radians(30.0)
    i = np.radians(50.0)
    assert layer.normal == pytest.approx(
        (np.sin(i) * np.cos(p), np.sin(i) * np.sin(p)))


def test_nodewise_disk_as_bands():
    surface = SurfaceNodewise([1, 3, 6, 10])
    disk = MassDisk('disk', surface, VerticalThin())
    params = dict(log_sigma=np.array([8.0, 7.5, 7.0]))
    (layer,) = _layers(disk, params)
    assert isinstance(layer, BandLayer)
    np.testing.assert_array_equal(layer.edges[:, 0], [1, 3, 6])
    rng = np.random.default_rng(4)
    x, y = 0.4 + rng.uniform(-12, 12, 60), -0.3 + rng.uniform(-12, 12, 60)
    i, p = np.radians(50.0), np.radians(30.0)
    dx, dy = x - 0.4, y + 0.3
    R = np.hypot(-dx * np.sin(p) + dy * np.cos(p),
                 (-dx * np.cos(p) - dy * np.sin(p)) / np.cos(i))
    expected = surface.surface_density(R, params, _context()) / np.cos(i)
    np.testing.assert_allclose(_sky_density(layer, x, y), expected,
                               rtol=1e-12, atol=1e-6)


class _DiskDensity:
    """The density of a disk at knots (see mass.DiskDensity)."""

    def __init__(self, knots, sigmas, heights=None, cut=None):
        self._knots = np.asarray(knots, dtype=float)
        self._sigmas = np.atleast_2d(sigmas)
        # (the scale height of a sech2 profile of each term, at each knot)
        self._heights = heights
        self._cut = cut

    def knots(self, rstep):
        return self._knots

    def thick(self):
        return self._heights is not None

    def surface_densities(self, R):
        return np.stack([np.interp(R, self._knots, s, right=0)
                         for s in self._sigmas])

    def height_pdfs(self, R, z):
        h = np.stack([np.interp(R, self._knots, row)
                      for row in self._heights])[:, :, None]
        z = np.asarray(z)[None, None, :]
        pdf = 1 / (2 * h * np.cosh(z / h) ** 2) / np.tanh(self._cut / h)
        return np.where(np.abs(z) <= self._cut, pdf, 0)

    def height_cuts(self):
        return (self._cut,)

    def mass_per_unit(self):
        return 2.0


def test_traced_thin_disk_as_bands_on_its_rings():
    knots = np.linspace(0, 10, 11)
    density = _DiskDensity(knots, 100 * (1 - knots / 10))
    rings = _Rings(0.1, 0.2, posa=40.0, incl=30.0, dincl=4.0)
    placement = SkyPlacement((0.1, 0.2), disks={'gas': rings})
    (layer,) = _layers(TracedDisk('stars', 'gas'), dict(upsilon=3.0),
                       _context(gas=density), placement)
    assert isinstance(layer, BandLayer) and layer.scale == 6.0
    np.testing.assert_array_equal(layer.edges[:, 0], knots[:-1])
    # (each band of the inclination of the ring at its middle)
    np.testing.assert_allclose(
        layer.stretch[:, 1], np.cos(np.radians(30 + 4 * (knots[:-1] + 0.5))))
    np.testing.assert_allclose(
        layer.density[0, :, 0] + layer.density[0, :, 1] * knots[:-1],
        100 * (1 - knots[:-1] / 10), atol=1e-12)


def test_traced_thick_disk_slices_of_each_term():
    # In its own layer: each term its surface density times the pdf of its
    # height at each knot, linear between them; the slices add up to its
    # surface density
    knots = np.linspace(0, 10, 21)
    sigmas = [100 * np.exp(-knots / 3), 30 * np.exp(-knots / 5)]
    heights = [0.2 + 0.05 * knots, 0.8 + 0 * knots]
    density = _DiskDensity(knots, sigmas, heights, cut=4.0)
    placement = SkyPlacement((0.0, 0.0), disks={'gas': _Rings(incl=60.0)})
    (layer,) = _layers(TracedDisk('stars', 'gas'), dict(upsilon=1.0),
                       _context(gas=density), placement)
    assert len(layer.heights) == 84
    R = knots[:-1]
    at_knots = layer.density[:, :, 0] + layer.density[:, :, 1] * R
    np.testing.assert_allclose(
        at_knots.sum(axis=0), sigmas[0][:-1] + sigmas[1][:-1], rtol=1e-12)
    # (the shares: the pdfs at the slices, times their widths, of the
    # scale height of the mean pdf of the layer)
    z = np.linspace(0, 4.0, 513)
    scale = sky.layer_scale(z, np.einsum(
        'tk,tkz->z', np.array(sigmas), density.height_pdfs(knots, z)))
    _, widths = sky.height_slices(scale)
    pdfs = np.einsum('tk,tkz->zk', np.array(sigmas)[:, :-1],
                     density.height_pdfs(R, layer.heights))
    np.testing.assert_allclose(
        at_knots, pdfs * widths[:, None] / (pdfs * widths[:, None]).sum(0)
        * at_knots.sum(0), rtol=1e-10)


def test_traced_disk_in_another_layer_is_quadratic():
    # Its surface density, linear between its knots, times the shape of
    # the layer of another component, linear between those knots: bands
    # at both, of quadratics, exactly
    knots = np.linspace(0, 10, 6)
    other = np.array([0.0, 2.5, 7.0, 12.0])
    density = _DiskDensity(knots, 50 * (1 - knots / 10))
    layer_density = _DiskDensity(
        other, np.ones(4), [0.3 + 0.1 * other], cut=3.0)
    disk = TracedDisk('mol', 'gas', vertical=VerticalComponent('hi'))
    placement = SkyPlacement((0.0, 0.0), disks={'gas': _Rings(incl=40.0)})
    (layer,) = _layers(disk, dict(upsilon=1.0),
                       _context(gas=density, hi=layer_density), placement)
    np.testing.assert_allclose(
        layer.edges[:, 0], np.union1d(knots, other[(other > 0)
                                                   & (other < 10)])[:-1])
    vertical = VerticalComponent('hi')
    z = np.linspace(0, 3.0, 513)
    heights, widths = sky.height_slices(sky.layer_scale(z, vertical.shapes(
        other, z, _context(hi=layer_density)).mean(axis=0)))
    shares = sky.slice_weights(vertical.shapes(
        other, heights, _context(hi=layer_density)).T, widths)
    for R in (0.7, 2.6, 6.1, 9.3):
        k = np.searchsorted(layer.edges[:, 1], R)
        c = layer.density[:, k]
        expected = np.interp(R, knots, 50 * (1 - knots / 10)) * np.array(
            [np.interp(R, other, row) for row in shares])
        np.testing.assert_allclose(
            c[:, 0] + c[:, 1] * R + c[:, 2] * R * R, expected, rtol=1e-10)


def test_traced_spheroid_terms():
    terms = [
        (ShapeSphere(), ProfileSersic(), dict(log_m=0.0, re=1.0, n=4.0)),
        (ShapeFlattened(), ProfileGauss(), dict(
            log_m=0.0, sigma=0.5, q=0.5))]
    density = SpheroidDensity(terms, None)
    axis = SpheroidAxis(1.0, 2.0, 20.0, 70.0)
    placement = SkyPlacement((1.0, 2.0), spheroids={'bulge': axis})
    layers = _layers(TracedSpheroid('stars', 'bulge'), dict(upsilon=4.0),
                     _context(bulge=density), placement)
    round_, ellipse = layers
    assert isinstance(round_, RoundLayer) and round_.centre == (1.0, 2.0)
    assert round_.scale == 4.0
    assert round_.mass(1e4) == pytest.approx(1.0, rel=1e-6)
    assert isinstance(ellipse, EllipticalLayer)
    assert ellipse.posa == 20.0
    assert ellipse.stretch[1] == pytest.approx(
        float(sky.projected_ratio(0.5, 70.0)))


def test_plane_must_agree_when_needed():
    disk = MassDisk('disk', SurfaceExponential(), VerticalThin())
    placement = SkyPlacement((0.0, 0.0), disks={
        'gas': _Rings(incl=50.0), 'stars': _Rings(incl=50.0, posa=1e-6)})
    with pytest.raises(ModelEvaluationError, match='disagree'):
        _layers(disk, dict(log_m=10.0, rd=3.0), placement=placement)
    # (a part on the rings of its component, or spherical, needs none)
    knots = np.linspace(0, 5, 6)
    traced = TracedDisk('traced', 'gas')
    _layers(traced, dict(upsilon=1.0),
            _context(gas=_DiskDensity(knots, 1 - knots / 5)), placement)
    _layers(Halo('halo', ProfileNFW()), dict(log_m=12.0, c=10.0),
            placement=placement)
    # (a warp beyond the innermost ring does not move the plane)
    warped = SkyPlacement((0.0, 0.0), disks={
        'gas': _Rings(incl=50.0, dincl=3.0),
        'stars': _Rings(incl=50.0)})
    _layers(disk, dict(log_m=10.0, rd=3.0), placement=warped)


def test_plane_needs_a_component():
    disk = MassDisk('disk', SurfaceExponential(), VerticalThin())
    placement = SkyPlacement((0.0, 0.0), spheroids={
        'bulge': SpheroidAxis(0.0, 0.0, None, None)})
    with pytest.raises(ModelEvaluationError, match='none of those'):
        _layers(disk, dict(log_m=10.0, rd=3.0), placement=placement)


def test_projection_gaps():
    # (tables and functions of the user do not say where their mass is)
    model = MassModel([
        MassPoint('bh'), MassUser('user', lambda R, params: R * 0)])
    assert "['user']" in model.projection_gaps()
    with pytest.raises(ValueError, match='do not say where'):
        model.sky_mass({'mass_bh_log_m': 8.0}, _context(), _plane())
    assert not issubclass(MassTable, sky.SkyProjected)
    assert MassModel(MassPoint('bh')).projection_gaps() is None


def test_takes_mass_plane():
    assert MassModel([
        MassPoint('bh'), Halo('halo', ProfileNFW()),
        TracedDisk('traced', 'gas')]).takes_mass_plane() is False
    assert MassModel(Halo(
        'halo', ProfileNFW(), shape=ShapeFlattened())).takes_mass_plane()
    assert MassModel(MassDisk(
        'disk', SurfaceExponential(), VerticalThin())).takes_mass_plane()


def test_a_scale_keeps_the_unit_shape():
    disk = MassDisk('disk', SurfaceExponential(), VerticalSech2())
    (first,) = _layers(disk, dict(log_m=10.0, rd=3.0, hz=0.3))
    (heavier,) = _layers(disk, dict(log_m=10.3, rd=3.0, hz=0.3))
    (longer,) = _layers(disk, dict(log_m=10.0, rd=3.1, hz=0.3))
    assert heavier.same_shape(first) and not longer.same_shape(first)
    assert heavier.scale / first.scale == pytest.approx(10 ** 0.3)
    knots = np.linspace(0, 5, 6)
    context = _context(gas=_DiskDensity(knots, 1 - knots / 5))
    traced = TracedDisk('traced', 'gas')
    (light,) = _layers(traced, dict(upsilon=1.0), context)
    (heavy,) = _layers(traced, dict(upsilon=2.0), context)
    assert heavy.same_shape(light) and heavy.scale == 2 * light.scale
    halo = Halo('halo', ProfilePseudoIsothermal())
    params = dict(log_m=12.0, c=10.0)
    (one,) = _layers(halo, params)
    (two,) = _layers(halo, params)
    assert two.same_shape(one) and not two.same_shape(first)


def test_layer_scale():
    # (the scale height of an exponential pdf, and of sech2 its median
    # height over ln 2)
    z = np.linspace(0, 20, 20001)
    assert sky.layer_scale(z, np.exp(-z / 0.4) / 0.8) == pytest.approx(
        0.4, rel=1e-6)
    assert sky.layer_scale(z, 1 / (2 * 0.4 * np.cosh(z / 0.4) ** 2)) \
        == pytest.approx(0.4 * np.arctanh(0.5) / np.log(2), rel=1e-5)


@pytest.mark.parametrize('incl', [70.0, 85.0, 89.0])
def test_thick_disks_too_edge_on_are_refused(incl):
    # (hz 0.4 and rd 3: the median height 0.22 kpc, limit atan(1.5 x 3 /
    # 0.22) = 87.2 degrees; thin, any inclination)
    thick = MassDisk('disk', SurfaceExponential(), VerticalSech2())
    params = dict(log_m=10.0, rd=3.0, hz=0.4)
    placement = _plane(incl=incl)
    if incl < 87:
        _layers(thick, params, placement=placement)
    else:
        with pytest.raises(ModelEvaluationError, match=r"laid out to 87\.\d"):
            _layers(thick, params, placement=placement)
    thin = MassDisk('disk', SurfaceExponential(), VerticalThin())
    _layers(thin, dict(log_m=10.0, rd=3.0), placement=placement)


def test_thick_traced_disks_too_edge_on_are_refused():
    # (bands: of the inclination of the band most edge-on that holds mass)
    knots = np.linspace(0, 10, 21)
    density = _DiskDensity(
        knots, 100 * np.exp(-knots / 3), [0.3 + 0 * knots], cut=4.0)
    for incl, refused in ((60.0, False), (89.0, True)):
        placement = SkyPlacement(
            (0.0, 0.0), disks={'gas': _Rings(incl=incl)})
        if refused:
            with pytest.raises(ModelEvaluationError, match="edge-on"):
                _layers(TracedDisk('stars', 'gas'), dict(upsilon=1.0),
                        _context(gas=density), placement)
        else:
            _layers(TracedDisk('stars', 'gas'), dict(upsilon=1.0),
                    _context(gas=density), placement)


def test_planes():
    # (the angle between planes, exact near 0; a plane of its normal
    # flipped, posa + 180 and -incl, is the same)
    assert sky.plane_tilt((30.0, 50.0), (30.0, 50.0)) == 0
    assert sky.plane_tilt((0.0, 30.0), (180.0, -30.0)) == pytest.approx(
        0, abs=1e-12)
    assert sky.plane_tilt((0.0, 10.0), (180.0, 10.0)) == pytest.approx(20)
    assert sky.plane_tilt((30.0, 50.0), (30.0, 50.0 + 1e-6)) \
        == pytest.approx(1e-6, rel=1e-6)
    np.testing.assert_allclose(
        sky.plane_normal(90.0, 90.0), [0, 1, 0], atol=1e-15)
    assert sky.check_planes([('a', 30.0, 50.0), ('b', 30.0, 50.0)]) \
        == sky.MassPlane(30.0, 50.0)
    with pytest.raises(ModelEvaluationError, match="'b' at posa 31"):
        sky.check_planes([('a', 30.0, 50.0), ("'b'", 31.0, 50.0)])
    with pytest.raises(ModelEvaluationError, match="none of those"):
        sky.check_planes([])
