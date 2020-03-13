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

   gbkfit-cli eval path/to/configuration/file

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

   # data model
   dmodels:
     type: scube
     size: [101, 101, 101]
     step: [1, 1, 5]
     cval: [0, 0, 0]
     scale: [1, 1, 1]
     psf: {type: gauss, sigma: 2}
     lsf: {type: gauss, sigma: 20}

   # galaxy model
   gmodels:
     type: kinematics_2d
     components:
       type: smdisk
       rnodes: [0, 50]
       nrnodes: 10
       rnodes_min: 0
       rnodes_max: 50
       rnodes_sep: 10
       rptraits: {type: gauss}
       vptraits: {type: tan_arctan}
       dptraits: {type: uniform}

   # parameters
   params:
     vsys: 0
     xpos: 0
     ypos: 0
     posa: 45
     incl: 45
     rpt_a: 1
     rpt_s: 30
     rpt_b: 10
     vpt_rt: 10
     vpt_vt: 200
     dpt_a: 10

Cookbook
--------

Data Model descriptions
^^^^^^^^^^^^^^^^^^^^^^^

Galaxy Model descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^

Thin flat disk

Thick flat disk

Fitter descriptions
^^^^^^^^^^^^^^^^^^^

