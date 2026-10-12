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

Lines of sight
--------------
- Flat: the galaxies are evaluated on the plane of the sky (thin disks
  and points)
- Grid: the galaxies are evaluated on a grid with a depth axis (thick
  disks, thin disks and points at their depth, and dust, which dims the
  light behind it)

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
