spook package
=============

spook module
-----------------
.. figure:: _static/famtree.png
   :alt: Family tree of spook solver classes
   :width: 80%
   :align: center

.. automodule:: spook.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: spook.quad_program
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: spook.lin_solve
   :members:
   :undoc-members:
   :show-inheritance:


spook.utils module
------------------

.. automodule:: spook.utils
   :members:
   :undoc-members:
   :show-inheritance:

spook.xval module
-----------------

.. automodule:: spook.xval
   :members:
   :undoc-members:
   :show-inheritance:

spook.vmi\_special module
-------------------------

.. automodule:: spook.vmi_special
   :members:
   :undoc-members:
   :show-inheritance:


spook.contraction\_utils module
-------------------------------
.. note::
   Function `adaptive_contraction` in this module serves in the raw mode of the spook solver.
   It is not parallelized, so it is slow for large datasets.
   For TMO users, a good alternative is the outer-product accumulation application in the tmo-preproc repo.

.. automodule:: spook.contraction_utils
   :members:
   :undoc-members:
   :show-inheritance:
