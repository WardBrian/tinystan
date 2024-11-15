C API
=====

----

Installation
------------

Please follow the :doc:`Getting Started guide <../getting-started>` to install
TinyStan's pre-requisites and downloaded a copy of the TinyStan source code.

This API is implemented in C++ and exposed to C using the ``extern "C"`` directive.
It is used to implement all the other high-level interfaces. 


API Reference
-------------

The following are the C functions exposed by the TinyStan library in :file:`tinystan.h`.
These are wrapped in the various high-level interfaces.

.. These functions are implemented in C++, see :doc:`../internals` for more details.

.. autodoxygenfile:: tinystan.h
    :project: tinystan
    :sections: func typedef var enum
