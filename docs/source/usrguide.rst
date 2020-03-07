User Guide
==========

The functionality of GBKFIT can be accessed through a Command Line Interface
(CLI, ``gbkfit-cli``), and soon, through a Graphical User Interface
(GUI, ``gbkfit-gui``) as well.

If the functionality provided by these two interfaces is not enough, one can
use GBKFIT's Python API. For more information see the Developer Guide.

The command line interface (``gbkfit-cli``)
-------------------------------------------

The graphical user interface (``gbkfit-gui``)
---------------------------------------------

.. attention::
   The graphical user interface is not available yet.

The configuration file
----------------------

`JSON <https://json.org>`_.

`YAML <https://yaml.org>`_.

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
