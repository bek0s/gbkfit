"""
The traits of the components made of nodes (disks and cones): the
profiles of their properties, as functions of the radius r and the
azimuth theta on the surface of a component (polar traits), or of the
height above it (height traits).

The surface of a disk is its plane, and its nodes are rings: r is the
radius in the plane, and the height is z, the height above the plane
(above the displaced plane with vertical distortion traits). The surface
of a cone is the mid-surface of its walls, at the angle wall from its
axis, and its nodes are shells, spheres around its apex: r is the
distance from the apex, and the height is d, the distance from the
mid-surface across the wall. The axis of a cone is the spin axis of a
disk of its inclination incl and position angle posa, and theta is the
azimuth around the axis, as in that disk.

The kinds of traits, and how the traits of one kind combine:

- density (rpt, rht): the density per unit area of the surface (of a
  disk, seen face-on): the brightness of a component that emits, the
  optical depth of dust; the sum of the polar traits, each times its
  height trait in a thick component (a probability density of the
  height, which integrates to 1).
- velocity (vpt, vht): the line-of-sight velocity; vsys plus the sum of
  the polar traits, each a speed along its direction projected on the
  line of sight, times its height trait (a factor of the height, 1 at 0
  for most).
- dispersion (dpt, dht): the velocity dispersion; the absolute value of
  the sum of the polar traits, each times its height trait. On a disk,
  the polar traits can each be along one direction (see below).
- vertical distortion (zpt): the sum displaces the plane of a disk (its
  midplane) along z, each within its zmax.
- selection (spt): 1 where a trait selects the component, 0 elsewhere;
  a part of the component is kept if every trait selects it.

The height traits pair with the polar traits of their kind by their
order, and a component has at most four traits of a kind. Each type of
component takes its own list of traits (see _nodal.TraitTable): the
cones, e.g., take no vertical distortion, mixture or map traits.

Coordinates and units: theta is from the major axis on the side of the
position angle posa (the receding side when the rotation is positive),
increasing towards the far side (theta = 90 degrees, on the sky at the
position angle posa + 90 degrees). z is positive away from the viewer
for inclinations below 90 degrees, as is the positive side of the axis
of a cone; the height traits depend on |z| or |d| only. Lengths are in
arcsec, angles in degrees and velocities in km/s. The traits are
evaluated between the first and the last radius of the nodes of a
component only.

A velocity polar trait is a speed v along its direction (its option
direction; the trait mass is tangential), and its speed parameter is
named after it: tangential ('tan', vt, around the axis), radial ('rad',
vr, in the plane of a disk, positive outwards), vertical ('ver', vv,
along z, positive along it), away from the plane ('away', va, along z
on both faces of a thick disk, positive away from its midplane, as
TiRiFiC's VVER), outward ('out', vo, along the radius of the sphere
about the centre of a disk or the apex of a cone, positive away from
it: on a disk, away from its plane on either side, e.g. a wind of its
gas) or the line of sight ('los', vl). On a disk, the speeds are
projected on the line of sight as v cos(theta) sin(incl), v sin(theta)
sin(incl), v cos(incl), v sign(z) cos(incl), v (sin(a) sin(theta)
sin(incl) + cos(a) cos(incl)) and v, a the angle of the point from the
positive side of the axis (sin(a) = R / r and cos(a) = z / r, at the
radius R, the height z and r = sqrt(R^2 + z^2)); on a cone, the
tangential and outward ones as on a disk. The tangential and radial
ones are 0 on the axis of a disk, the tangential one on the axis of a
cone, and the outward ones at the centre of a disk and the apex of a
cone, where their directions are undefined.

A dispersion polar trait is isotropic (its option direction 'iso', the
default: the same dispersion along every direction) or, on a disk, along
one direction: radial ('rad', sigma_R, in the plane), tangential ('tan',
sigma_phi, around the axis) or vertical ('ver', sigma_z, along z). The
dispersion along each direction is then the absolute value of the sum of
the isotropic traits and the traits along it (each times its height
trait), and that on the line of sight is the velocity ellipsoid of the
three seen along it: sqrt(sigma_phi^2 cos^2(theta) sin^2(incl) +
sigma_R^2 sin^2(theta) sin^2(incl) + sigma_z^2 cos^2(incl)), the
dispersion of the isotropic traits alone where all are isotropic. Once
a trait is along a direction, every direction needs a trait: its own or
an isotropic one (see parts.Kinematics). The dispersion on the line of
sight sets only two of the three at each radius, sigma_phi^2
sin^2(incl) + sigma_z^2 cos^2(incl) (on the major axis) and sigma_R^2
sin^2(incl) + sigma_z^2 cos^2(incl) (on the minor axis): a fit needs a
ratio (e.g. of the traits 'rad', 'tan' and 'ver' in this order, sigma_z
= sigma_R / 2 is the parameter expression dpt2_a: 'dpt_a * 0.5') or a
prior.

The planar velocity polar traits of a disk (e.g. the flow along a bar)
are not speeds along one direction, but velocities in its plane, with a
radial and a tangential part: they have no option direction, and their
velocity is projected on the line of sight as v_y sin(incl), v_y its
part along the azimuth 90 degrees (see VPTraitPlanar).

Node-wise parameters (those of the nw_ traits, and of the height traits
with rnodes) have a value for each node: at its radii (sampling
'rnodes'), interpolated to the nodes the component is evaluated on (its
subnodes, see rstep) with the interpolation of the component (interp),
or at those subnodes (sampling 'subrings'); between the subnodes, they
are interpolated linearly.

The mixture traits are sums of elliptical blobs. Blob i is centred at
the radius r_i and the azimuth t_i, has the amplitude a_i, the scale s_i
along its major axis and q_i s_i along its minor axis, and its major
axis at the azimuth t_i + p_i + 90 degrees (p_i = 0: tangential, 90:
radial); rho is the elliptical distance from its centre, sqrt(x^2 + y^2
/ q_i^2) along its axes.

The map trait is a grid of square cells in the plane of the disk,
centred on its centre, x along the azimuth 0 and y along the azimuth 90
degrees, with one density for each cell (a free map). The image trait is
such a grid, whose densities its galaxy makes from an observed image.

The disks of clouds draw their clouds from the density traits: for the
whole disk (the uniform, mixture and map traits), or ring by ring (see
Clouds).

The modules: base (Trait and the bases of the kinds), options (the
options the traits hold, e.g. Direction), density, images (the maps of
the density), heights, velocity, dispersion, warps (vertical
distortion) and selections.
"""

from gbkfit.utils import parseutils
from .options import *
from .base import *
from .density import *
from .images import *
from .heights import *
from .velocity import *
from .dispersion import *
from .warps import *
from .selections import *


# Density polar traits parser
rpt_parser = parseutils.TypedParser(RPTrait, [
    RPTraitUniform,
    RPTraitExponential,
    RPTraitGauss,
    RPTraitGGauss,
    RPTraitLorentz,
    RPTraitMoffat,
    RPTraitSech2,
    RPTraitMixtureExponential,
    RPTraitMixtureGauss,
    RPTraitMixtureGGauss,
    RPTraitMixtureMoffat,
    RPTraitNWUniform,
    RPTraitNWHarmonic,
    RPTraitNWDistortion,
    RPTraitSpiral,
    RPTraitMap,
    RPTraitImage])

# Density height traits parser
rht_parser = parseutils.TypedParser(RHTrait, [
    RHTraitUniform,
    RHTraitExponential,
    RHTraitGauss,
    RHTraitGGauss,
    RHTraitLorentz,
    RHTraitMoffat,
    RHTraitSech2])


# Velocity polar traits parser
vpt_parser = parseutils.TypedParser(VPTrait, [
    VPTraitUniform,
    VPTraitArctan,
    VPTraitBoissier,
    VPTraitEpinat,
    VPTraitLRamp,
    VPTraitTanh,
    VPTraitPolyex,
    VPTraitRix,
    VPTraitCourteau,
    VPTraitBrandt,
    VPTraitIso,
    VPTraitNFW,
    VPTraitNWUniform,
    VPTraitNWHarmonic,
    VPTraitWedge,
    VPTraitBar,
    VPTraitNWBar,
    VPTraitSpiral,
    VPTraitMass])

# Velocity height traits parser
vht_parser = parseutils.TypedParser(VHTrait, [
    VHTraitOne,
    VHTraitLinear,
    VHTraitExponential,
    VHTraitGauss,
    VHTraitMass])

# Dispersion polar traits parser
dpt_parser = parseutils.TypedParser(DPTrait, [
    DPTraitUniform,
    DPTraitExponential,
    DPTraitGauss,
    DPTraitGGauss,
    DPTraitLorentz,
    DPTraitMoffat,
    DPTraitSech2,
    DPTraitMixtureExponential,
    DPTraitMixtureGauss,
    DPTraitMixtureGGauss,
    DPTraitMixtureMoffat,
    DPTraitNWUniform,
    DPTraitNWHarmonic,
    DPTraitNWDistortion])

# Dispersion height traits parser
dht_parser = parseutils.TypedParser(DHTrait, [
    DHTraitOne,
    DHTraitLinear,
    DHTraitExponential,
    DHTraitGauss])

# Vertical polar distortion traits parser
zpt_parser = parseutils.TypedParser(ZPTrait, [
    ZPTraitNWUniform,
    ZPTraitNWHarmonic])

# Selection polar traits parser
spt_parser = parseutils.TypedParser(SPTrait, [
    SPTraitAzimuthalRange,
    SPTraitRadialRange,
    SPTraitNWAzimuthalRange])
