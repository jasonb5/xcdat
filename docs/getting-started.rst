.. highlight:: shell

===============
Getting Started
===============

Prerequisites
-------------

1. Familiarity with ``xarray``, since this package is an extension of it

   - We highly recommend visiting the `xarray tutorial`_ and `xarray documentation`_
     pages if you aren't familiar with ``xarray``.

2. xCDAT is distributed through conda, which is available through Anaconda and Miniconda.

   We recommend following these steps to install Miniconda:

   .. code-block:: console

      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
      bash Miniconda3-latest-Linux-x86_64.sh

   Then follow the instructions for installation. To have conda added to
   your path you will need to type ``yes`` in response to ``"Do you wish the
   installer to initialize Miniconda3 by running conda init?"`` (we recommend
   that you do this). Note that this will modify your shell profile (e.g.,
   ``~/.bashrc``) to add ``conda`` to your path.

   Note: After installation completes you may need to type ``bash`` to
   restart your shell (if you use bash). Alternatively, you can log out and
   log back in.

.. _xarray tutorial: https://tutorial.xarray.dev/intro.html
.. _xarray documentation: https://docs.xarray.dev/en/stable/getting-started-guide/index.html

Installation
------------

1. Create a conda environment from scratch with ``xcdat`` (`conda create`_)

   We recommend using the Conda environment creation procedure to install ``xcdat``.
   The advantage with following this approach is that Conda will attempt to resolve
   dependencies (e.g. ``python >= 3.8``) for compatibility.

   To create an ``xcdat`` conda environment with ``xesmf`` (a recommended dependency),
   run:

   .. code-block:: console

       >>> conda create -n <ENV_NAME> -c conda-forge xcdat xesmf
       >>> conda activate <ENV_NAME>

   Note that ``xesmf`` is an optional dependency, which is required for using ``xesmf``
   based horizontal regridding APIs in ``xcdat``. ``xesmf`` is not currently supported
   on `osx-arm64`_ or `windows`_ because ``esmpy`` is not yet available on these
   platforms. Windows users can try `WSL2`_ as a workaround.

.. _windows: https://github.com/conda-forge/esmf-feedstock/issues/64
.. _osx-arm64: https://github.com/conda-forge/esmf-feedstock/issues/74
.. _WSL2: https://docs.microsoft.com/en-us/windows/wsl/install

2. Install ``xcdat`` in an existing conda environment (`conda install`_)

   You can also install ``xcdat`` in an existing Conda environment, granted that Conda
   is able to resolve the compatible dependencies.

   .. code-block:: console

       >>> conda activate <ENV_NAME>
       >>> conda install -c conda-forge xcdat xesmf

   Note: As above, ``xesmf`` is an optional dependency.

3. [Optional] Some packages that are commonly used with ``xcdat`` can be installed
   either in step 1 or step 2 above:

   - ``jupyterlab``: a web-based interactive development environment for notebooks,
     code, and data. This package also includes ``ipykernel``.
   - ``matplotlib``: a library for creating visualizations in Python.
   - ``cartopy``: an add-on package for ``matplotlib`` and specialized for geospatial data processing.

.. _conda create: https://docs.conda.io/projects/conda/en/latest/commands/create.html?highlight=create
.. _conda install: https://docs.conda.io/projects/conda/en/latest/commands/install.html?highlight=install
