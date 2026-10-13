User Guide
==========

The functionality of GBKFIT can be accessed through
a Command Line Interface (CLI, ``gbkfit-cli``), and soon, through
a Graphical User Interface (GUI, ``gbkfit-gui``) as well.

If the functionality provided by those two interfaces is not enough, one can
use GBKFIT's Python API. For more information see the Developer Guide.

The command line interface (``gbkfit-cli``)
-------------------------------------------

The CLI of GBKFIT can be run using the command ``gbkfit-cli`` followed by a
positional argument indicating the task the user wants to execute, which in
turn is followed by a series of positional and optional arguments specific to
that task. There are four available tasks: ``eval``, ``prep``, ``fit``, and
``plot``. The first generates mock models; the second provides a series of
pre-processing functions to prepare data for fitting; the third fits models to
data; and finally, the fourth creates visualisations for the results of the
``fit`` task.

The ``eval`` task
^^^^^^^^^^^^^^^^^

The ``eval`` task can be used to generate models without performing any
fitting. The description of the model must be supplied to GBKFIT in the form of
a text-based configuration file.

To generate a mock model, run:

.. code-block:: console

   gbkfit-cli eval model path/to/configuration/file

(``eval objective`` also compares the model with the data of the
observations.)

After ..., a series of output files created ...

Instructions on how to create a configuration file see link.

The ``prep`` task
^^^^^^^^^^^^^^^^^

The ``fit`` task
^^^^^^^^^^^^^^^^^

The ``plot`` task
^^^^^^^^^^^^^^^^^

The graphical user interface (``gbkfit-gui``)
---------------------------------------------

.. attention::
   The graphical user interface is not available yet.

The configuration file
----------------------

All the configuration files used by the CLI and GUI must be in a format that is
flexible enough to describe the complex and sophisticated modelling strategies
available in GBKFIT. Furthermore, the configuration files must be
human-readable/writable, but also supported by a wide range of programming
languages in case the user wishes to read/write them programmatically.

For this reason we decided to make use of the JSON and YAML file formats. While
the JSON format is enough to describe the modelling strategies of GBKFIT, many
users may find it a bit too verbose. To overcome this issue, GBKFIT also
supports configuration files in the YAML file format. In fact, YAML 1.2 is
a superset of JSON, and as a result, users could also use JSON syntax inside
YAML configuration files if they wish to do so. Throughout this documentation
we will adopt a hybrid approach. While for the most part we will be using YAML,
for some parts we may have to use JSON.

To learn more about the JSON and YAML file formats, see
`JSON <https://json.org>`_ and `YAML <https://yaml.org>`_.

Example
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # where the models are evaluated: host (the default) or cuda; several
   # drivers are declared by name in a section 'drivers', and then each
   # observation names its own (e.g. driver: gpu)
   driver: {type: host}

   # the galaxy: one thin disk, which emits a line
   galaxies:
   - components:
     - type: smdisk
       loose: false
       tilted: false
       rnmin: 0
       rnmax: 50
       rnsep: 10
       density:
         ptraits: {type: gauss}
       kinematics:
         vptraits: {type: tan_arctan}
         dptraits: {type: uniform}
       # one line, at the velocities of the spectral axis; several lines
       # are named, e.g. lines: [halpha, nii6583], from the catalogue of
       # GBKFIT or the section 'lines' of the configuration (e.g.
       # lines: {mgii2796: 2796.35 Angstrom}; rest wavelengths in vacuum;
       # the spectral axis of data in air wavelengths, e.g. MUSE, has the
       # option medium: air, of the data, or of an observable without
       # data; gbkfit-prep writes it in the files it converts, as the
       # header keyword AIRORVAC)
       emission: {type: lines}

   # observations of the galaxy: a spectral cube; the galaxy is evaluated
   # on the plane of the sky (a flat line of sight; a thick disk needs a
   # depth axis: {type: grid}, or radiative transfer along it, for
   # self-absorbing lines: {type: transfer})
   observations:
   - line_of_sight: {type: flat}
     observable:
       type: pixel_spectra
       size: [101, 101, 101]
       step: [1, 1, 5]
     instrument:
       psf: {type: gauss, sigma: 2}
       lsf: {type: gauss, sigma: 20}

   # parameters
   params:
     properties:
       vsys: 0
       xpos: 0
       ypos: 0
       posa: 45
       incl: 45
       pt_a: 1
       pt_s: 30
       vpt_rt: 10
       vpt_vt: 200
       dpt_a: 10

Cookbook
--------

Self-absorbing lines
^^^^^^^^^^^^^^^^^^^^

A thick disk of HI whose line absorbs itself, seen through radiative
transfer along each line of sight. Without line opacity, its brightness
would peak at about 180; the line levels off near the source function
(S = 100), at about 83.

.. code-block:: yaml

   driver: {type: host}

   # a thick disk of HI whose line absorbs itself (line opacity)
   galaxies:
   - components:
     - type: smdisk
       loose: false
       tilted: false
       rnmin: 0
       rnmax: 18
       rnsep: 1
       density:
         ptraits: {type: exponential}
         htraits: {type: sech2}
       kinematics:
         vptraits: {type: tan_arctan}
         dptraits: {type: uniform}
       emission: {type: lines}
       # the optical depth of the light j of a voxel at its line is j / S,
       # S the source function (for HI, the spin temperature, in the units
       # of the model); the coefficient 1/S is a set of density traits
       # (parameters op_*, here one: op_a)
       opacity:
         type: line
         coefficients: {type: uniform}

   # line opacity needs radiative transfer along each line of sight
   observations:
   - line_of_sight:
       type: transfer
       # the march of optically thick lines needs channels about as narrow
       # as the lines (here 2 sub-channels of each channel of 10 km/s), and,
       # in a disk seen almost edge-on, a fine depth axis (here half a
       # pixel); an evaluation with extra outputs warns if either is too
       # coarse
       subchannels: 2
       depth: {step: 0.5}
     observable:
       type: pixel_spectra
       size: [41, 41, 61]
       step: [1, 1, 10]

   params:
     properties:
       xpos: 0
       ypos: 0
       posa: 90
       incl: 80
       vsys: 0
       pt_a: 1000
       pt_s: 5
       ht_s: 0.3
       vpt_rt: 0.5
       vpt_vt: 200
       dpt_a: 10
       # 1/S, of S = 100 (e.g. a spin temperature of 100 K)
       op_a: 0.01

With extra outputs (``gbkfit-cli eval``), each galaxy also gives its
light without any opacity (``total_unabsorbed``) and the optical depth
of its lines at each channel (``total_line_tau``). An evaluation with
extra outputs warns when the march is too coarse where the lines are
optically thick, and names the setting to change: with one sub-channel
and a depth step of one pixel, this model is off by several percent of
its peak, and both warnings say so.

Galaxies along one line of sight
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two galaxies that overlap on the sky, one of them with dust: through
radiative transfer, the dust of the galaxy in front dims the light of
the one behind it, and not the other way round.

.. code-block:: yaml

   driver: {type: host}

   # a galaxy of gas and dust in front of a galaxy of gas, on one line of
   # sight; depth: true gives each a position along it ({name}_depth, in
   # arcsec, away from the viewer), which orders them
   galaxies:
   - name: front
     depth: true
     geometries:
     - {name: disk, loose: false, tilted: false}
     components:
     - type: smdisk
       name: gas
       geometry: disk
       rnmin: 0
       rnmax: 20
       rnsep: 2
       density:
         ptraits: {type: exponential}
         htraits: {type: sech2}
       kinematics:
         vptraits: {type: tan_arctan}
         dptraits: {type: uniform}
       emission: {type: lines}
     - type: smdisk
       name: dust
       geometry: disk
       rnmin: 0
       rnmax: 20
       rnsep: 2
       density:
         ptraits: {type: exponential}
         htraits: {type: sech2}
       opacity: {type: dust}
   - name: back
     depth: true
     components:
     - type: smdisk
       loose: false
       tilted: false
       rnmin: 0
       rnmax: 20
       rnsep: 2
       density:
         ptraits: {type: exponential}
         htraits: {type: sech2}
       kinematics:
         vptraits: {type: tan_arctan}
         dptraits: {type: uniform}
       emission: {type: lines}

   # radiative transfer marches the galaxies front to back, each on a depth
   # axis of its own, centred on its depth; the axes must not overlap (each
   # is as long as twice the radius of its largest disk, here 40 arcsec)
   observations:
   - galaxies: [front, back]
     line_of_sight: {type: transfer}
     observable:
       type: pixel_spectra
       size: [61, 41, 61]
       step: [1, 1, 10]

   params:
     properties:
       front_depth: -50
       back_depth: 50
       front_disk_xpos: -8
       front_disk_ypos: 0
       front_disk_posa: 30
       front_disk_incl: 60
       front_disk_vsys: 0
       front_gas_pt_a: 1
       front_gas_pt_s: 5
       front_gas_ht_s: 1
       front_gas_vpt_rt: 3
       front_gas_vpt_vt: 150
       front_gas_dpt_a: 15
       front_dust_pt_a: 0.5
       front_dust_pt_s: 5
       front_dust_ht_s: 0.5
       back_xpos: 8
       back_ypos: 0
       back_posa: 120
       back_incl: 50
       back_vsys: 50
       back_pt_a: 1
       back_pt_s: 6
       back_ht_s: 1
       back_vpt_rt: 3
       back_vpt_vt: 180
       back_dpt_a: 15

The extra outputs of each galaxy are prefixed by its name (e.g.
``front_total_unabsorbed``). Swapping the two depths puts the dust
behind the other galaxy. A line of sight of type grid does not order
galaxies: it sees several galaxies only when none has opacity.

Data Model descriptions
^^^^^^^^^^^^^^^^^^^^^^^

Galaxy Model descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^

Thin flat disk

Thick flat disk

Fitter descriptions
^^^^^^^^^^^^^^^^^^^

