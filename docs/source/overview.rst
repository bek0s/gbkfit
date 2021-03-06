Overview
========

GBKFIT is a high-performance and extremely flexible software for
modelling galaxy surface brightness and kinematics. Its goal is to
provide a solution for a wide range of galaxy modelling studies.

.. attention::
   This documentation covers the latest version of GBKFIT. If you are
   interested in the old and deprecated version of the software, visit
   `gbkfit-legacy <https://github.com/bek0s/gbkfit-legacy>`_.

.. warning::
   GBKFIT is currently in Alpha development phase and is expected to contain
   incomplete features and bugs.

Features
--------

GBKFIT combines most of the features and techniques available in other
modelling codes, while also offering insane computational performance. The most
important features of the software are listed below.

- Fits models to images, spectral cubes, moment maps, and long-slit data.
- Performs great with low-resolution observations by utilising supersampling
  and supporting a wide variety of point and line spread functions.
- Supports superpositions of various galaxy model geometries, including
  thin-disk, thick-disk, and tilted-ring models.
- Supports a plethora of superimposed surface brightness, velocity,
  velocity dispersion, and optical depth functions.
- Utilises the most popular optimisation and sampling techniques, including
  Least-Squares minimisation, Swarm Intelligence, Markov Chain Monte Carlo,
  and Nested Sampling.
- Takes advantage of multi-core CPUs and CUDA GPU accelerators on workstation,
  distributed, and cloud computing environments.
