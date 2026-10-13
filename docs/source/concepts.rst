Basic Concepts
==============

The modelling strategy of GBKFIT can be described using a series of concepts
which are explained in the next sections.

Data models
-----------
- Image
- Spectral cube
- Moment maps
- Long-slit spectra

Galaxies
--------
A galaxy is a set of components, with optional mass and distance.

Components
^^^^^^^^^^
- The smooth disk
- The Monte Carlo disk
- The point

Each component emits light (in a band, seen by images, or in lines, seen
by spectral cubes and the data made from them), removes light (dust), or
both. The density of dust is its optical depth at V; an extinction law
(ccm, calzetti or a table) scales it at the rest wavelength of each line,
or of the band of an image (its effective wavelength over 1 + the
redshift of the galaxy), and without one the dust is grey. A component
emits in every band, or in named bands (e.g. sdss_g, sdss_r) with the
flux ratios of the others to the first; an image of a band (band:
sdss_r) sees those that emit in it.

Traits
^^^^^^

Weights
-------
An observation can weight its data by the place of each pixel in the
plane of a disk (with its current parameters): e.g. less weight near the
minor axis (weights: {component: gas, traits: {type: azimuth_power,
power: 1}}), or none within an angle of it (type: azimuth_range).

A named observation can also scale the flux of its model by a parameter
(calibration: true gives the parameter {name}_calibration), e.g. between
the instruments of a joint fit.

Lines of sight
--------------
- Flat: the galaxies are evaluated on the plane of the sky (thin disks
  and points)
- Grid: the galaxies are evaluated on a grid with a depth axis (thick
  disks, thin disks and points at their depth, and dust, which dims the
  light behind it)
- Transfer: radiative transfer along each line of sight. The galaxies
  are evaluated on a grid with a depth axis, as by Grid, and a march
  along each line of sight, front to back, adds the light of each
  channel that reaches the viewer. Without line opacity it gives what
  Grid gives (the clouds of a Monte Carlo disk in each voxel become one
  line of their mean velocity and width). Its components can also absorb
  their own lines (line opacity: e.g. the self-absorption of HI), and
  the option slab evaluates the depth axis a few planes at a time, to
  bound the memory.

A thick disk that emits lines can absorb them (opacity: {type: line,
coefficients: {type: uniform}}): the light j of each voxel at a line has
the optical depth j/S at that line, of the source function S (e.g. the
spin temperature of HI, in the units of the model); the coefficient 1/S
is a set of density traits (parameters op_*, so it can vary in the
disk), or for named lines, a mapping of each line to its traits
(parameters op_{line}_*). The brightness of an optically thick line
then levels off at S.

The march has two resolutions, both of the model. In velocity, a
channel wider than the lines averages their optical depth before it
dims the light, which then dims too little: the option subchannels of
transfer marches each channel at that many sub-channels, whose mean is
the light of the channel. Along the line of sight, a voxel has one
velocity: where the velocity changes by more than the dispersion between
adjacent voxels (e.g. near the centre of an edge-on disk), the depth
axis needs a smaller step (depth: {step: ...}). An evaluation with extra
outputs (e.g. by the task eval) warns when either is too coarse where
the lines are not optically thin. Its extra outputs also give, for each
galaxy, its light without any opacity (total_unabsorbed) and the
optical depth of its lines at each channel (total_line_tau).

Through radiative transfer, an observation can see several galaxies
with opacity, which dim the light of one another (e.g. a galaxy behind
the dust of another, or HI absorbed by HI in front of it): each galaxy
needs a position along the line of sight (depth: true gives the
parameter {galaxy}_depth, in arcsec away from the viewer), and the march
goes through them one after the other, front to back, in the order of
their depths. Each has a depth axis of its own, centred on its depth,
and the axes of the galaxies must not overlap: galaxies that
interpenetrate along the line of sight are not supported yet. The dust
of a galaxy dims the light of another at the wavelength of that light in
its own rest frame. Grid does not order galaxies: it sees several only
when none has opacity.

Fitters
-------
Least-Squares
^^^^^^^^^^^^^
Swarm Intelligence
^^^^^^^^^^^^^^^^^^
Markov Chain Monte Carlo
^^^^^^^^^^^^^^^^^^^^^^^^
Nested Sampling
^^^^^^^^^^^^^^^

Drivers
-------
Host
^^^^
CUDA
^^^^
