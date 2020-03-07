Installation
============

GBFKIT supports the idea that all software should be trivially installable and
require minimal technical expertise from the end user.

Operating system requirements
-----------------------------

The following operating systems are supported:

- **Linux:** All Linux distributions released after year 2010
- **macOS:** 10.14 (Mojave) - 10.15 (Catalina)

.. note:: While the software should, in principle, work on Microsoft Windows,
   there is no official support for this operating system yet. If you are a
   Windows user and you want to use GBKFIT, please give us a shout.

Python environment requirements
-------------------------------

GBKFIT requires Python 3.7 or later.

Dealing with old Python environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In case your Python version is older than the minimum required, and for some
reason you cannot update it, you could try running the commands below.

On Linux:

.. code-block:: bash

   mkdir gbkfit && cd gbkfit
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   chmod +x Miniconda3-latest-Linux-x86_64.sh
   ./Miniconda3-latest-Linux-x86_64.sh -b -s -p ./miniconda
   ./miniconda/bin/python -m venv venv

On macOS:

.. code-block:: bash

   mkdir gbkfit && cd gbkfit
   curl -O -L https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
   chmod +x Miniconda3-latest-MacOSX-x86_64.sh
   ./Miniconda3-latest-MacOSX-x86_64.sh -b -s -p ./miniconda
   ./miniconda/bin/python -m venv venv

The above lines will create an isolated Python environment (*venv*) which
can be used to install and run GBKFIT.

To activate the environment run:

.. code-block:: bash

   . venv/bin/activate

To deactivate the environment run:

.. code-block:: bash

   deactivate

.. attention::
   The above steps are just a suggestion and not required. It is up to the
   user to choose how to update their Python version. Furthermore, the above
   snippets use Minicoda for convenience. Miniconda is not a GBKFIT
   requirement.

Dependencies
------------

The only requirement of GBKFIT is a working Python 3.7+ environment. The rest
of the required dependencies are downloaded automatically during its
installation.

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

GBFKFIT has a series of optional dependencies which can be installed by
the user in order to activate particular features:

- TODO

Install from PyPI
-----------------

Use the following steps to install GBKFIT from the Python Package Index (PyPI):

- Upgrade pip to the latest version (optional step, but highly recommended):

  .. code-block:: bash

     pip install pip -U

- Install GBKFIT:

  .. code-block:: bash

     pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple --no-cache-dir gbkfit

.. attention::
   Because GBKFIT is currently on a testing phase, it is located on the Test
   instance of PyPI. Once the testing phase is complete, the software will be
   moved to the production version of PyPI and will be installable using:
   ``pip install gbkfit``.


Install from Source
-------------------

When installing GBKFIT from source, the following dependencies are required:

- **FFTW3**: This should be available through your OS package manager
  (apt, yum, pacman, Homebrew, MacPorts, etc). Alternatively, it can be
  obtained from `here <http://www.fftw.org/>`_.

- **OpenMP**: This usually comes with your compiler and you don't have to
  install anything. However, this is not always the case. For example, when
  compiling with Apple Clang compiler, you may have to install the
  libomp library.

.. code-block:: bash

   git clone --recurse-submodules --remote-submodules https://github.com/bek0s/gbkfit.git


