.. lrux documentation master file, created by
   sphinx-quickstart on Tue Jun 10 16:05:41 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lrux documentation
==================

Fast low-rank update (LRU) of matrix determinants and pfaffians in JAX

Installation
-------------------------------

Requires Python 3.8+ and JAX 0.4.4+

.. code-block::

   pip install lrux


.. currentmodule:: lrux


Low-rank update of determinants
-------------------------------

.. autosummary::
   :toctree:

   det_lru
   init_det_carrier
   det_lru_delayed


Low-rank update of pfaffians
-------------------------------

.. autosummary::
   :toctree:

   pf_lru
   init_pf_carrier
   pf_lru_delayed

    

Pfaffian functions
-------------------------------

.. autosummary::
   :toctree:

   skew_eye
   pf
   slogpf
