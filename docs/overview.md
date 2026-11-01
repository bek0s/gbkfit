# GBKFIT in brief (2026-11-01)

## What it is

GBKFIT is a Python library, with C++/OpenMP and CUDA kernels, for
building forward models of galaxies and fitting them to data. A model
describes:

- the galaxy: its components, their light, their kinematics and their
  mass;
- how a telescope sees it: line of sight, foreground lens, PSF, LSF;
- what the instrument records: cubes, moment maps, images, slits, grism
  images, visibilities.

The model is compared with the data under a noise model, and a fitter
explores the parameters.

- **Data it targets:** HI radio cubes, optical/NIR IFU data at z ~ 0.5-1.7,
  JWST (NIRSpec IFU and MSA, NIRCam/NIRISS grisms), in FITS and ASDF.
- **First science:** clumpy disks at z ~ 1.5 (KMOS/SINFONI/MUSE/NIRSpec
  Halpha: low S/N, strong beam smearing, high sigma/v, clumps that smooth
  models miss). This is the territory of DysmalPy, and GBKFIT must do it
  at least as well.

## The pillars

Every change is judged by three things:

1. **Correctness:** the physics is right, no approximation is on by
   default, and the numerics are judged in km/s against the data errors.
2. **Flexibility:** any combination of components, observations and
   fitters, without special cases.
3. **Good software engineering:**
   - Library first: tasks, the CLI and configs are clients of the
     Python API.
   - Explicit design: no hidden flags, no uber-classes, composition over
     inheritance tricks.
   - Every function is type-hinted and documented (numpydoc).

## The big picture

```
dataset (data + errors + mask, on a grid or in regions)
   |
observation = observable + line of sight + foreground (lens) + instrument
   |                (PSF, LSF, primary beam)
   |   sees one or more
galaxy = components (disk, cone, spheroid, point, particles)
         + mass model + gravity law + distance
   |
objective = observations x likelihood (noise model)  --> fitter
```

- A **config** (YAML/JSON) is just the arguments of the classes'
  constructors. A typed family picks its class by `type`. Errors give
  their path in the config.
- **Drivers** run on the host CPU or a CUDA GPU. Many observations can be
  fitted jointly (e.g. KMOS + SINFONI) with per-observation offsets.

## Features

- **Components:**
  - Thin and thick disks: smooth, or made of clouds.
  - Cones (outflows).
  - Spheroids: a profile, kinematics by traits, isotropic or anisotropic
    Jeans, or Schwarzschild orbits.
  - Points (nuclei), and particles (clouds given by the user).
- **Traits of disks and cones,** stacked in any number:
  - Radial and polar light profiles: exponential, Gauss, Sersic-like,
    Moffat, mixtures of blobs (clumps), spirals, light deprojected from an
    observed image, free values ring by ring.
  - Heights.
  - Rotation curves: arctan, tanh, Courteau, Epinat, NFW, the curve of the
    mass model, ...
  - Radial, vertical and planar flows: bars, wedges, spirals, harmonics.
  - Dispersions along R, phi and z.
  - Warps, selections, pixel weights.
- **Light and dust:**
  - Emission lines from a catalogue (with doublets), bands, stellar
    continuum.
  - Physical units through to the data unit.
  - Dust and line opacity with extinction laws (Calzetti, CCM).
  - Radiative transfer through the depth of the galaxy.
- **Mass and gravity:**
  - Disks, spheroids (NFW, Burkert, Einasto, coreNFW, DC14/Dekel-Zhao,
    Sersic, MGE, ...), points, tables (SPARC curves), user functions.
  - Halo responses: adiabatic contraction (Blumenthal, Gnedin, Cautun),
    SIDM.
  - Gravity: Newton, MOND/QUMOND, f(R) (Hu-Sawicki with screening),
    Palatini.
  - Pressure support, vertical equilibrium, rotation that falls with
    height.
- **Observables:**
  - Spectral cubes, moment maps (including Gauss-Hermite h3/h4), images.
  - Spectra and moments in bins or apertures.
  - Long slits; slitless grism spectra (traces across the field, NIRSpec
    MSA shutters, nods).
  - Interferometric visibilities (NUFFT).
  - Radial profiles, such as rotation-curve tables.
  - Gravitational lensing in front of any of them (SIE, EPL, NFW,
    multipoles, shear, deflection maps).
- **Instrument:**
  - PSFs: Gauss, Moffat, beams, images, sums.
  - LSFs: Gauss, Hanning, Moffat, images.
  - Both can vary along the spectral axis (MUSE, CASA beams).
  - Spectral axes in radio, optical, log or relativistic velocity.
- **Noise and likelihood:**
  - Gaussian, with fitted noise (scale or jitter), upper limits, exact
    whitening of correlated noise, or a user likelihood function.
  - Mock data with Gaussian, Student-t or Poisson noise.
- **Fitting:**
  - Optimizers: lmfit, scipy least squares, CMA-ES, multistart, grid.
  - Samplers: emcee, zeus, nautilus, dynesty, UltraNest, DNEST4
    (DNEST4 handles clumps whose number is unknown).
  - Priors and joint priors (Gaussian, MVN, smoothness).
  - Bootstrap.
  - Parallel workers: threads, processes, dask, MPI, ray.
- **Prep and analysis:**
  - Prep reads cubes in any axis order, makes masks, outlier screens,
    Voronoi bins and moments, subtracts the continuum, and has MUSE and
    NIRSpec readers.
  - Analysis tools: kinemetry, kinematic PA, PV diagrams, asymmetry
    maps, pattern speed, LOO/WAIC, information gain, prior predictive
    checks.

## Compared with other codes

About 40 codes were surveyed in 2026-10: cube fitters, IFU forward
models, velocity-field codes, mass-model codes and dynamical codes. The
parity round closed on 2026-10-24.

- **Fully matched:** 24 codes, including GalPaK3D, 3D Barolo, TiRiFiC,
  KinMS(_fitter), DysmalPy, pyFAT, DiskFit, NIRVANA, discminer, Galactus,
  ROTCUR/ROTMOD, SPAM, Kinemetry and XookSuut.
- **Gaps left on purpose for later:**
  - Gradients (NUTS: DINGO, geko, pyROTMOD, eddy).
  - Linear sub-solves (pyuvkin, IFUL, kl-tools).
  - Hydrostatic disk thickness (galpynamics).
  - Continuum emission in spectral models (kl-tools).
  - The NIRSpec MSA detector model (jwst-msafit).
  - Hierarchical priors for clumps of unknown number (Blobby3D).
- **A different class of code, only partly covered:**
  - Orbit superposition (DYNAMITE, AGAMA/FORSTAND).
  - Discrete-star codes (MAMPOSSt).
  - Full Jeans packages (JamPy, GravSphere).
- **The selling points:**
  - One library covers what those codes do separately: all the data
    kinds, joint fits, mass models and gravity laws, lensing, dust, a GPU
    path, and every major sampler.
  - Every option is checked and documented in a schema.

## State

- The work is on the `develop` branch; `master` holds the old engine
  until a release.
- A full audit (88 items) was finished on 2026-11-01. The full test suite
  passes (4005 tests).
- Old configs are not migrated yet. Migration rules will be written once
  the config format is final; several option names changed in 2026-10/11.

## To-do (none urgent)

- **Error-bar check (#37):** pulls (fit - truth) / error over about 30
  mocks of the first-science cube, for the sampler and bootstrap errors
  (least-squares errors are not meant to be honest). Only if GBKFIT errors
  get quoted.
- **Config migration:** rules for the old formats, once the format is
  final.
- **Parity numbers (#59), last, before the paper:** models stored from
  DysmalPy and KinMS or 3DBarolo, with GBKFIT tested against them at a
  stated km/s tolerance.
- **Fitting, after the code is final:**
  - gradients (NUTS);
  - linear sub-solves;
  - hierarchical priors for DNEST4's clumps.
- **Physics later:**
  - hydrostatic thickness;
  - a continuum emission type;
  - a shared galaxy level for gas and orbit models;
  - Schwarzschild extras: orbit dithering, a black-hole-mass grid.
- **Engineering:**
  - the config layer as a package of its own (#55);
  - ParamSpace taking joint priors in its config;
  - the build flag `nvcc --split-compile=0`.
