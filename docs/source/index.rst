.. lrux documentation master file, created by
   sphinx-quickstart on Tue Jun 10 16:05:41 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lrux documentation
==================

Fast low-rank update (LRU) of matrix determinants and pfaffians in JAX

Installation
-------------------------------

Requires Python 3.11+ and JAX 0.7.1+. `fermix <https://github.com/ChenAo-Phys/fermix>`_ is installed automatically as a dependency.

.. code-block::

   pip install lrux


.. currentmodule:: lrux


Low-rank update of determinants
-------------------------------

.. autosummary::
   :toctree:

   det_lru
   init_det_carrier
   merge_det_delays
   det_lru_delayed


Low-rank update of pfaffians
-------------------------------

.. autosummary::
   :toctree:

   pf_lru
   init_pf_carrier
   merge_pf_delays
   pf_lru_delayed


Utilities
-------------------------------

The full determinants and pfaffians are computed by
`fermix <https://github.com/ChenAo-Phys/fermix>`_, which provides
``det``, ``slogdet``, ``pf``, and ``slogpf``.

.. autosummary::
   :toctree:

   skew_eye
