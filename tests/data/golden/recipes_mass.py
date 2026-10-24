"""
The recipes of the golden cases of the galaxy stage (mass, gravity,
equilibria, flows), of profile observations and several galaxies, and of
the parameters (modes, ties, extra parameters, shared geometries,
derived values and priors on several parameters) (see make_golden.Case).
"""

import pathlib
from typing import Any


# --- Shared pieces -------------------------------------------------------

_DISTANCE = dict(type='direct', cosmology='Planck18')

_STARS = dict(
    type='disk', name='stars', surface=dict(type='exponential'),
    vertical=dict(type='thin'))

_NFW = dict(type='spheroid', name='halo', profile=dict(type='nfw'))

# The rings of a disk of its own geometry
_RINGS = dict(loose=False, tilted=False, rnodes=list(range(0, 13)))


def _disk(
        name: str, vptraits: Any, dptraits: Any = None,
        rings: dict[str, Any] = _RINGS, **options: Any
) -> dict[str, Any]:
    """A thin exponential disk that emits lines, of the given traits."""
    return dict(type='smdisk', name=name) | rings | dict(
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(
            vptraits=vptraits, dptraits=dptraits or dict(type='uniform')),
        emission=dict(type='lines')) | options


def _thick(disk: dict[str, Any], **kinematics: Any) -> dict[str, Any]:
    """The disk, thick (sech2 heights), with more options of kinematics."""
    density = disk['density'] | dict(htraits=dict(type='sech2', zmax=6))
    return disk | dict(
        density=density, kinematics=disk['kinematics'] | kinematics)


_INSTRUMENT = dict(
    psf=dict(type='gauss', sigma=0.6), lsf=dict(type='gauss', sigma=20))


def _moments(
        size: int = 24, line_of_sight: str = 'flat', **options: Any
) -> dict[str, Any]:
    """A sky observation of moment maps, float64."""
    return dict(
        type='sky', line_of_sight=dict(type=line_of_sight), dtype='float64',
        observable=dict(
            type='pixel_moments', size=[size, size], spec_size=60,
            spec_step=10),
        instrument=_INSTRUMENT) | options


# The likelihood of moment maps (whose model can lack values that the
# data have)
_MOMENTS_LIKELIHOOD = dict(type='gaussian', missing_residual=3)


def _cube(
        size: list[int], line_of_sight: str = 'grid', **options: Any
) -> dict[str, Any]:
    """A sky observation of a spectral cube, float64."""
    return dict(
        type='sky', line_of_sight=dict(type=line_of_sight), dtype='float64',
        observable=dict(type='pixel_spectra', size=size, step=[1, 1, 15]),
        instrument=_INSTRUMENT) | options


def _prefixed(prefix: str, properties: dict[str, Any]) -> dict[str, Any]:
    """The properties with their names prefixed."""
    return {f'{prefix}_{name}': value for name, value in properties.items()}


# The geometry and light of a disk (at 80 Mpc: 0.39 kpc per arcsec)
_GEOMETRY_TRUTH = dict(
    xpos=0.3, ypos=-0.4, posa=40, incl=55, vsys=10, pt_a=1, pt_s=3)
_GEOMETRY_TRIAL = dict(
    xpos=0.3, ypos=-0.4, posa=dict(value=43, min=0, max=90),
    incl=dict(value=50, min=10, max=80), vsys=dict(value=12, min=-50, max=50),
    pt_a=1, pt_s=3)


# --- C1: rotation from the mass, with pressure support -------------------

_C1_GAS = _disk(
    'gas', vptraits=dict(type='mass', pressure_support='jeans'),
    dptraits=[
        dict(type='uniform', direction='rad'),
        dict(type='uniform', direction='tan'),
        dict(type='uniform', direction='ver')])

# (thick: a thin disk pulls infinitely at the edge of the gas)
_C1_MASS = [
    dict(type='disk', name='gas',
         surface=dict(type='component', component='gas'),
         vertical=dict(type='sech2')),
    _NFW]

_MASS_ROTATION_PRESSURE = dict(
    name='mass_rotation_pressure',
    doc="A gas disk that rotates in its mass (a thick disk that follows "
        "its light, and an NFW halo; a direct distance), with the Jeans "
        "pressure support of an anisotropic dispersion (sigma_R, "
        "sigma_phi, sigma_z); moment maps; a derived vcirc at 2.2 scale "
        "lengths, and priors on it and on sigma_phi / sigma_R.",
    galaxies=[dict(components=[_C1_GAS], mass=_C1_MASS, distance=_DISTANCE)],
    observations=[_moments()],
    truth=_prefixed('gas', _GEOMETRY_TRUTH | dict(
        dpt_a=35, dpt1_a=25, dpt2_a=15)) | dict(
        distance=80, mass_gas_upsilon=5e8, mass_gas_hz=0.2,
        mass_halo_log_m=11.5, mass_halo_c=10),
    trial=_prefixed('gas', _GEOMETRY_TRIAL | dict(
        dpt_a=dict(value=30, min=5, max=80), dpt1_a=25, dpt2_a=15)) | dict(
        distance=80, mass_gas_upsilon=dict(value=4e8, min=1e7, max=1e10),
        mass_gas_hz=0.2, mass_halo_log_m=dict(value=11.8, min=10, max=13),
        mass_halo_c=10),
    noises=[dict(type='gauss', rms=0.002)],
    likelihoods=[_MOMENTS_LIKELIHOOD],
    params=dict(
        derived=dict(v22='vcirc(2.2 * gas_pt_s * kpc_per_arcsec)'),
        priors=[
            dict(type='gauss', expr='v22', mean=120, std=10),
            dict(type='gauss', expr='gas_dpt1_a / gas_dpt_a', mean=0.7,
                 std=0.1)]))


# --- C2: vertical equilibrium and rotation at each height, in MOND -------

_C2_GAS = _thick(
    _disk('gas', vptraits=dict(type='mass', pressure_support='jeans')),
    equilibrium='jeans', vhtraits=dict(type='mass', zmax=6))

# (a Sersic bulge, not an exponential disk, whose forces off its plane
# cost some 9 s of tables in each process)
_BULGE = dict(type='spheroid', name='bulge', profile=dict(type='sersic'))

_EQUILIBRIUM_HEIGHTS_MOND = dict(
    name='equilibrium_heights_mond',
    doc="A thick gas disk held up by the Jeans vertical equilibrium, "
        "rotating at each height in the gravity of its mass (a stellar "
        "bulge, and a disk that follows the gas) in MOND (gravity_a0); a "
        "small cube.",
    galaxies=[dict(
        components=[_C2_GAS],
        mass=[_BULGE, dict(
            type='disk', name='gas',
            surface=dict(type='component', component='gas'),
            vertical=dict(type='component', component='gas'))],
        distance=_DISTANCE,
        gravity=dict(type='mond', interpolation='simple'))],
    observations=[_cube([20, 20, 30])],
    truth=_prefixed('gas', _GEOMETRY_TRUTH | dict(ht_s=0.6, dpt_a=10)) | dict(
        distance=80, mass_bulge_log_m=10.0, mass_bulge_re=1.0,
        mass_bulge_n=1.5, mass_gas_upsilon=3e8, gravity_a0=1.2),
    trial=_prefixed('gas', _GEOMETRY_TRIAL | dict(
        ht_s=dict(value=0.8, min=0.1, max=3), dpt_a=10)) | dict(
        distance=80, mass_bulge_log_m=dict(value=10.2, min=9, max=12),
        mass_bulge_re=1.0, mass_bulge_n=1.5, mass_gas_upsilon=3e8,
        gravity_a0=dict(value=1.6, min=0, max=5)),
    noises=[dict(type='gauss', rms=2e-4)])


# --- C3: spiral, bar and wedge flows on the rotation of the mass ---------

_C3_GAS = _disk('gas', vptraits=[
    dict(type='mass'),
    dict(type='spiral', order=3),
    dict(type='bar'),
    dict(type='wedge', direction='rad')])

_SPIRAL_BAR_FLOWS = dict(
    name='spiral_bar_flows',
    doc="A gas disk rotating in its mass (a stellar disk and an NFW "
        "halo), with the flows of a spiral density wave of 3 arms (which "
        "the galaxy stage tabulates from the rotation), along a bar, and "
        "in two wedges; moment maps.",
    galaxies=[dict(
        components=[_C3_GAS], mass=[_STARS, _NFW], distance=_DISTANCE)],
    observations=[_moments()],
    truth=_prefixed('gas', _GEOMETRY_TRUTH | dict(
        dpt_a=20, vpt1_p=40, vpt1_c=60, vpt1_e=0.3, vpt1_o=1,
        vpt2_v=-40, vpt2_p=30, vpt2_s=2, vpt3_vr=25, vpt3_p=70,
        vpt3_s=30)) | dict(
        distance=80, mass_stars_log_m=10.3, mass_stars_rd=1.2,
        mass_halo_log_m=11.5, mass_halo_c=10),
    trial=_prefixed('gas', _GEOMETRY_TRIAL | dict(
        dpt_a=20, vpt1_p=dict(value=50, min=0, max=120), vpt1_c=60,
        vpt1_e=0.3, vpt1_o=1, vpt2_v=dict(value=-30, min=-100, max=0),
        vpt2_p=30, vpt2_s=2, vpt3_vr=25, vpt3_p=70, vpt3_s=30)) | dict(
        distance=80, mass_stars_log_m=dict(value=10.4, min=9, max=12),
        mass_stars_rd=1.2, mass_halo_log_m=11.5, mass_halo_c=10),
    noises=[dict(type='gauss', rms=0.002)],
    likelihoods=[_MOMENTS_LIKELIHOOD])


# --- C4: f(R) gravity, with a contracted halo ----------------------------

_FR_GRAVITY_RESPONSE = dict(
    name='fr_gravity_response',
    doc="A gas disk rotating in Hu-Sawicki f(R) gravity (log_fr0) of a "
        "stellar disk, and an NFW halo contracted by it (gnedin04, "
        "screened); "
        "a profile observation of vcirc and the rotation of the gas at "
        "radii in arcsec, and small moment maps.",
    galaxies=[dict(
        components=[_disk('gas', vptraits=dict(type='mass'))],
        mass=[_STARS, _NFW | dict(
            baryons=['stars'], response=dict(type='gnedin04'))],
        distance=_DISTANCE,
        gravity=dict(type='hu_sawicki', screened=['halo']))],
    observations=[
        dict(type='profile', name='rc', functions=['vcirc', 'rotation_gas'],
             radii=[1.0, 2.5, 4.0, 6.0, 8.0, 10.0], radius_unit='arcsec'),
        _moments(size=16, name='maps')],
    truth=_prefixed('gas', _GEOMETRY_TRUTH | dict(dpt_a=20)) | dict(
        distance=80, mass_stars_log_m=10.3, mass_stars_rd=1.2,
        mass_halo_log_m=11.5, mass_halo_c=10, gravity_log_fr0=-6.0),
    trial=_prefixed('gas', _GEOMETRY_TRIAL | dict(dpt_a=20)) | dict(
        distance=80, mass_stars_log_m=dict(value=10.4, min=9, max=12),
        mass_stars_rd=1.2, mass_halo_log_m=dict(value=11.7, min=10, max=13),
        mass_halo_c=10, gravity_log_fr0=dict(value=-6.5, min=-8, max=-4)),
    noises=[dict(type='gauss', rms=3), dict(type='gauss', rms=0.002)],
    likelihoods=[None, _MOMENTS_LIKELIHOOD])


# --- C5: a galaxy of mass alone, of tables, and its rotation curve -------

# A rotmod file of SPARC, made up: radius (kpc), Vobs, errV, Vgas,
# Vdisk, Vbul (km/s), at 10 Mpc
_ROTMOD = """\
# Distance = 10 Mpc
# Rad Vobs errV Vgas Vdisk Vbul
0.5 40.0 5.0 -3.0 50.0 0.0
1.0 70.0 4.0 5.0 80.0 0.0
1.5 88.0 4.0 9.0 92.0 0.0
2.0 100.0 4.0 12.0 95.0 0.0
3.0 112.0 4.0 17.0 94.0 0.0
4.0 120.0 5.0 20.0 90.0 0.0
5.0 124.0 5.0 22.0 85.0 0.0
6.5 126.0 6.0 23.0 78.0 0.0
"""


def _write_rotmod(folder: pathlib.Path) -> None:
    """Write the rotmod file of C5 into the folder of the case."""
    (folder / 'rotmod.dat').write_text(_ROTMOD)


def _rotmod(**columns: Any) -> dict[str, Any]:
    """The options of columns of the rotmod file, at its distance."""
    return dict(
        file='rotmod.dat', format='ascii.no_header',
        radius=dict(column='col1', unit='kpc'), distance=10) | columns


_PROFILE_TABLE_MASS = dict(
    name='profile_table_mass',
    doc="A galaxy of mass alone: the gas and the disk of a rotmod table "
        "of SPARC (upsilon, and D / D0 of a direct distance) and an NFW "
        "halo; its rotation curve from the table (radii in kpc at 10 Mpc), "
        "deprojected at an inclination that is a parameter; a derived "
        "dark fraction.",
    galaxies=[dict(
        mass=[
            dict(type='table', name='gas')
            | _rotmod(velocity=dict(column='col4')),
            dict(type='table', name='disk')
            | _rotmod(velocity=dict(column='col5')),
            _NFW],
        distance=_DISTANCE)],
    observations=[dict(
        type='profile', name='rc', inclination=60,
        data=_rotmod(vcirc=dict(column='col2', error='col3')))],
    truth=dict(
        distance=10, mass_gas_upsilon=1, mass_disk_upsilon=0.5,
        mass_halo_log_m=11.5, mass_halo_c=10, rc_incl=60),
    trial=dict(
        distance=dict(value=11, min=5, max=20), mass_gas_upsilon=1,
        mass_disk_upsilon=dict(value=0.7, min=0.1, max=2),
        mass_halo_log_m=dict(value=11.2, min=10, max=13), mass_halo_c=10,
        rc_incl=dict(value=55, min=20, max=90)),
    noises=[dict(type='gauss', rms=3)],
    params=dict(derived=dict(fdm='vcirc_halo(2.0) ** 2 / vcirc(2.0) ** 2')),
    prepare=_write_rotmod)


# --- C6: two galaxies ----------------------------------------------------

def _c6_gas() -> dict[str, Any]:
    """A thick gas disk of arctan rotation."""
    return _thick(_disk('gas', vptraits=dict(type='arctan', direction='tan')))


_C6_DUST = dict(
    type='smdisk', name='dust', loose=False, tilted=False,
    rnodes=list(range(0, 11)),
    density=dict(
        ptraits=dict(type='exponential'),
        htraits=dict(type='sech2', zmax=6)),
    opacity=dict(type='dust'))


def _c6_properties(
        geometry: dict[str, Any], vpt_vt: Any) -> dict[str, Any]:
    """
    The properties of the galaxies of C6: the gas of both of the given
    geometry and rotation speed (the one behind elsewhere), and the dust.
    """
    gas = geometry | dict(ht_s=0.5, vpt_rt=2, vpt_vt=vpt_vt, dpt_a=20)
    dust = dict(
        xpos=0, ypos=0, posa=60, incl=50, pt_a=0.4, pt_s=3, ht_s=0.4)
    back = gas | dict(xpos=2.5, ypos=2.0, posa=120, vsys=40)
    return (_prefixed('front_gas', gas) | _prefixed('front_dust', dust)
            | _prefixed('back_gas', back))


_TWO_GALAXIES = dict(
    name='two_galaxies',
    doc="Two named galaxies (their parameters prefixed by their names): "
        "a cube of both through radiative transfer, the dust of the one "
        "in front (section depth_order) dimming the one behind; and "
        "moment maps of the one behind alone (option galaxies).",
    galaxies=[
        dict(name='front', components=[_c6_gas(), _C6_DUST]),
        dict(name='back', components=[_c6_gas()])],
    observations=[
        _cube([20, 20, 30], 'transfer', galaxies=['front', 'back']),
        _moments(size=20, line_of_sight='grid', galaxies=['back'])],
    truth=_c6_properties(_GEOMETRY_TRUTH, vpt_vt=150),
    trial=_c6_properties(
        _GEOMETRY_TRIAL, vpt_vt=dict(value=130, min=50, max=300)),
    noises=[dict(type='gauss', rms=5e-4), dict(type='gauss', rms=0.002)],
    likelihoods=[None, _MOMENTS_LIKELIHOOD],
    sections=dict(depth_order=['front', 'back']))


# --- C7: the features of the parameters ----------------------------------

def _warped(name: str) -> dict[str, Any]:
    """A thin disk of the shared warped geometry 'disk'."""
    return _disk(
        name, vptraits=dict(type='arctan', direction='tan'),
        rings=dict(geometry='disk'))


# The flux of the nucleus, of the extra parameters (lnf_*) of the section
# pdescs
_NUCLEUS_FLUX = 'np.exp(lnf_mu + lnf_sigma * lnf_z)'

_PARAMS_FEATURES = dict(
    name='params_features',
    doc="Two gas disks (cold and warm) of one shared warped geometry "
        "(galaxy option geometries) and a nucleus, in a cube: the position "
        "angles of the warp given as offsets (mode), ties, extra "
        "parameters of the section pdescs (the flux of the nucleus, "
        "exp(mu + sigma z)), derived values, and priors on several "
        "parameters (gauss on an expression, smooth over the warp).",
    galaxies=[dict(
        geometries=[dict(
            name='disk', tilted=True, rnodes=[0, 1.5, 3, 5, 7, 9.5, 12])],
        components=[
            _warped('cold'), _warped('warm'),
            dict(type='point', name='nucleus', emission=dict(type='lines'))])],
    observations=[_cube([24, 24, 30], 'flat')],
    truth=dict(
        disk_xpos=0.3, disk_ypos=-0.4, disk_vsys=10,
        disk_posa=[40, 41, 43, 46, 50, 54, 58], disk_incl=55,
        cold_pt_a=1, cold_pt_s=3, cold_vpt_rt=2, cold_vpt_vt=150,
        cold_dpt_a=15, warm_pt_a=0.5, warm_pt_s=4, warm_vpt_rt=2,
        warm_vpt_vt=120, warm_dpt_a=40, nucleus_xpos=0.3,
        nucleus_ypos=-0.4, nucleus_vsys=10, nucleus_flux=_NUCLEUS_FLUX,
        nucleus_disp=60, lnf_mu=0.5, lnf_sigma=0.3, lnf_z=0.6),
    trial=dict(
        disk_xpos=0.2, disk_ypos=-0.3,
        disk_vsys=dict(value=12, min=-50, max=50),
        # (the first ring, and the offsets of the others from it)
        disk_posa=[dict(value=42, min=0, max=90)] + [
            dict(value=offset, min=-30, max=30)
            for offset in (1, 2, 4, 8, 12, 16)],
        disk_incl=dict(value=50, min=10, max=80),
        cold_pt_a=1, cold_pt_s=3, cold_vpt_rt=2,
        cold_vpt_vt=dict(value=140, min=50, max=300), cold_dpt_a=15,
        warm_pt_a=0.5, warm_pt_s=4, warm_vpt_rt='cold_vpt_rt',
        warm_vpt_vt=dict(value=110, min=50, max=300),
        warm_dpt_a='cold_dpt_a * 2.5', nucleus_xpos='disk_xpos',
        nucleus_ypos='disk_ypos', nucleus_vsys='disk_vsys',
        nucleus_flux=_NUCLEUS_FLUX, nucleus_disp=60, lnf_mu=0.5,
        lnf_sigma=0.3, lnf_z=dict(value=0.0, min=-5, max=5)),
    noises=[dict(type='gauss', rms=3e-4)],
    params=dict(
        modes=dict(disk_posa=dict(type='offsets', origin=0)),
        derived=dict(
            warp='disk_posa[-1] - disk_posa[0]',
            v_ratio='rotation_warm(8) / rotation_cold(8)'),
        priors=[
            dict(type='gauss', expr='warm_vpt_vt / cold_vpt_vt',
                 mean=0.8, std=0.05),
            dict(type='smooth', expr='disk_posa', order=2, std=1,
                 positions='disk_rnodes')]),
    sections=dict(pdescs=dict(
        lnf_mu=dict(type='scalar'), lnf_sigma=dict(type='scalar'),
        lnf_z=dict(type='scalar'))))


CASES: list[dict[str, Any]] = [
    _MASS_ROTATION_PRESSURE,
    _EQUILIBRIUM_HEIGHTS_MOND,
    _SPIRAL_BAR_FLOWS,
    _FR_GRAVITY_RESPONSE,
    _PROFILE_TABLE_MASS,
    _TWO_GALAXIES,
    _PARAMS_FEATURES,
]
