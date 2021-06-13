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

GBKFIT requires Python 3.9 or later.

Dealing with old Python environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In case your Python version is older than the minimum required, and for some
reason you cannot update it, you could try running the commands below.

On Linux:

.. code-block:: console

   mkdir gbkfit && cd gbkfit
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   chmod +x Miniconda3-latest-Linux-x86_64.sh
   ./Miniconda3-latest-Linux-x86_64.sh -b -s -p ./miniconda
   ./miniconda/bin/python -m venv venv

On macOS:

.. code-block:: console

   mkdir gbkfit && cd gbkfit
   curl -O -L https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
   chmod +x Miniconda3-latest-MacOSX-x86_64.sh
   ./Miniconda3-latest-MacOSX-x86_64.sh -b -s -p ./miniconda
   ./miniconda/bin/python -m venv venv

The above lines will create an isolated Python environment (*venv*) which will
not affect your system's Python environment and can be used to install and run
GBKFIT.

To activate the environment, run:

.. code-block:: console

   . venv/bin/activate

To deactivate the environment, run:

.. code-block:: console

   deactivate

.. attention::
   The above steps are just a suggestion and not required. It is up to the
   user to choose how to update their Python version. Furthermore, the above
   snippets use Minicoda for convenience. Miniconda is not a GBKFIT
   requirement.

Dependencies
------------

All required run-time dependencies are installed automatically during
GBKFIT's installation.

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

The following optional run-time dependencies can be installed by the user
in order to activate additional functionality:

- TODO

Installing from PyPI
--------------------

Use the steps below to install GBKFIT from the Python Package Index (PyPI).

- Upgrade pip to the latest version (optional step, but highly recommended):

  .. code-block:: console

     pip install pip -U

- Install GBKFIT:

  .. code-block:: console

     pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple --no-cache-dir gbkfit

.. attention::
   Because GBKFIT is currently in Alpha development phase, it is located on the
   Test instance of PyPI. Once the Alpha phase is complete, the software will
   be moved to the production version of PyPI and will be installable using:
   ``pip install gbkfit``.


Installing from source
----------------------

Installing GBKFIT from source requires additional technical expertise and you
should have a good reason for preferring it over installing from PyPI. The two
most common reasons are:

- GBKFIT is not available on PyPI for your platform
- You want to compile GBKFIT with a specific compiler

To install GBKFIT from source you will need:

- A C++ 14 capable compiler.
  Any recent version of GCC, Clang, ICC, or PGI will do.
- The FFTW3 library.
  This should be available through your OS package manager. Alternatively, it
  can be obtained from `here <http://www.fftw.org/>`_.
- An OpenMP library.
  This usually comes with your compiler and you do not have to install
  anything. However, this is not always the case. For example, when compiling
  with Apple Clang compiler, you may have to install the libomp library
  (available through Homebrew and MacPorts).

Once all required dependencies are installed, run:

.. code-block:: console

   git clone --recurse-submodules --remote-submodules https://github.com/bek0s/gbkfit.git

Before compiling the source code, we need to specify what hardware support we want to compile GBKFIT with.

To enable multi-core CPU support, run:

.. code-block:: console

   export GBKFIT_BUILD_HOST=1

To enable CUDA GPU support, run:

.. code-block:: console

   export GBKFIT_BUILD_CUDA=1

.. attention::
   Support for CUDA GPUs is not fully functional yet. Do not enable it.

To compile and install your local copy of GBKFIT, run:

.. code-block:: console

   pip install ./gbkfit

Congratulations! Now it is time to model some galaxies!
