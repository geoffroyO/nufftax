Core Concepts
=============

This page explains the mathematical foundations of the NUFFT and how nufftax implements them.

The Problem: Non-Uniform Data
-----------------------------

The standard FFT computes the Discrete Fourier Transform (DFT):

.. math::

   F[k] = \sum_{j=0}^{N-1} f[j] \cdot e^{-2\pi i \, jk/N}

This assumes your data ``f[j]`` lives on a **uniform grid** with spacing :math:`\Delta x = 1/N`.

But what if your samples are at **arbitrary locations** :math:`x_j`? You need the Non-Uniform DFT:

.. math::

   F[k] = \sum_{j=0}^{M-1} c_j \cdot e^{i k x_j}

where :math:`x_j \in [-\pi, \pi)` are non-uniform sample locations and :math:`c_j` are the values at those locations.

Computing this directly costs :math:`O(MN)` operations. The NUFFT reduces this to :math:`O(M + N \log N)`.

The Three Transform Types
-------------------------

nufftax provides three types of NUFFT, each solving a different problem:

Type 1: Nonuniform → Uniform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Given:** Values :math:`c_j` at scattered points :math:`x_j`

**Compute:** Fourier coefficients :math:`f_k` on a regular frequency grid

.. math::

   f_k = \sum_{j=1}^{M} c_j \cdot e^{i \cdot \text{isign} \cdot k \cdot x_j}

**Use when:** You have scattered measurements and want their frequency content.

.. code-block:: python

   from nufftax import nufft1d1

   # Scattered measurement locations
   x = jnp.array([0.1, 0.5, 1.2, -0.8, 2.3])

   # Values at those locations
   c = jnp.array([1+0j, 2+1j, 0.5-0.5j, 1+1j, 0.3+0j])

   # Get 64 Fourier modes
   f = nufft1d1(x, c, n_modes=64, eps=1e-6)

Type 2: Uniform → Nonuniform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Given:** Fourier coefficients :math:`f_k` on a regular grid

**Compute:** Values :math:`c_j` at arbitrary query points :math:`x_j`

.. math::

   c_j = \sum_{k} f_k \cdot e^{i \cdot \text{isign} \cdot k \cdot x_j}

**Use when:** You have a frequency representation and want to evaluate it at specific points.

.. code-block:: python

   from nufftax import nufft1d2

   # Fourier coefficients (e.g., from a previous analysis)
   f = jnp.zeros(64, dtype=jnp.complex64)
   f = f.at[10].set(1.0)  # Single frequency component

   # Query locations
   x = jnp.linspace(-jnp.pi, jnp.pi, 100)

   # Evaluate the Fourier series at those points
   c = nufft1d2(x, f, eps=1e-6)

Type 3: Nonuniform → Nonuniform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Given:** Values :math:`c_j` at scattered source points :math:`x_j`

**Compute:** Transform at scattered target frequencies :math:`s_k`

.. math::

   f_k = \sum_{j=1}^{M} c_j \cdot e^{i \cdot \text{isign} \cdot s_k \cdot x_j}

**Use when:** Both your sample locations and your target frequencies are irregular.

.. code-block:: python

   from nufftax import nufft1d3, compute_type3_grid_size

   # Source points and values
   x = jnp.array([0.1, 0.5, 1.2, -0.8])
   c = jnp.array([1+0j, 2+1j, 0.5-0.5j, 1+1j])

   # Target frequencies (not on a regular grid)
   s = jnp.array([0.3, 1.7, 5.2, -2.1, 8.0])

   # Pre-compute grid size for JIT (see below)
   n_modes = compute_type3_grid_size(x, s, eps=1e-6)

   # Compute the transform
   f = nufft1d3(x, c, s, n_modes=n_modes, eps=1e-6)

How the Algorithm Works
-----------------------

The NUFFT achieves its speed through a three-step pipeline:

.. code-block:: text

   Nonuniform points → [Spread] → Fine grid → [FFT] → Freq grid → [Deconvolve] → Output modes

1. **Spreading (Type 1) / Interpolation (Type 2)**

   Nonuniform point values are "spread" onto a fine uniform grid using a carefully chosen
   kernel function. nufftax uses the **Exponential of Semicircle (ES)** kernel:

   .. math::

      \phi(z) = e^{\beta(\sqrt{1 - z^2} - 1)}

   This kernel has excellent accuracy properties - it achieves near-optimal error for a given support width.

2. **FFT**

   A standard FFT is applied to the fine grid. The grid is typically 2x oversampled
   (controlled internally) to ensure accuracy.

3. **Deconvolution**

   The FFT result is divided by the Fourier transform of the spreading kernel,
   correcting for the smoothing introduced in step 1.

Precision Control
-----------------

The ``eps`` parameter controls the accuracy/speed tradeoff:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - ``eps``
     - Accuracy
     - Speed
   * - ``1e-2``
     - ~1% relative error
     - Fastest (small kernel)
   * - ``1e-6``
     - ~0.0001% relative error
     - Good balance (default)
   * - ``1e-12``
     - Near machine precision
     - Slower (large kernel)

The kernel width scales roughly as :math:`\lceil \log_{10}(1/\text{eps}) \rceil + 1` grid points.

.. code-block:: python

   # Fast but approximate
   f_fast = nufft1d1(x, c, n_modes=64, eps=1e-2)

   # High precision
   f_precise = nufft1d1(x, c, n_modes=64, eps=1e-12)

Coordinate Conventions
----------------------

**Input range:** Sample locations should be in :math:`[-\pi, \pi)`.

.. code-block:: python

   # Correct: points in [-pi, pi)
   x = jnp.array([-2.5, -0.5, 0.3, 1.8, 2.9])

   # Also works: automatic wrapping is applied
   x = jnp.array([0.0, 1.0, 7.0, -8.0])  # Will be wrapped to [-pi, pi)

**Output modes:** For ``n_modes=N``, the output contains modes :math:`k = -N/2, ..., N/2-1`.

**Sign convention:** The ``isign`` parameter controls the sign of the exponent:

- ``isign=+1`` (default for Type 1): Uses :math:`e^{+ikx}` convention
- ``isign=-1`` (default for Type 2): Uses :math:`e^{-ikx}` convention

Automatic Differentiation
-------------------------

A key feature of nufftax is full differentiability. The gradients are computed efficiently
using the mathematical relationship between Type 1 and Type 2 transforms:

**Key insight:** Type 1 and Type 2 are adjoints of each other.

This means:

- The gradient of Type 1 w.r.t. ``c`` is computed using Type 2
- The gradient of Type 2 w.r.t. ``f`` is computed using Type 1

For gradients w.r.t. the point locations ``x``, the kernel derivative is used.

.. code-block:: python

   import jax

   # Gradient w.r.t. values (uses Type 2 internally)
   def loss_values(c):
       f = nufft1d1(x, c, n_modes=64)
       return jnp.sum(jnp.abs(f) ** 2)

   grad_c = jax.grad(loss_values)(c)

   # Gradient w.r.t. locations (uses kernel derivative)
   def loss_positions(x):
       f = nufft1d1(x, c, n_modes=64)
       return jnp.sum(jnp.abs(f) ** 2)

   grad_x = jax.grad(loss_positions)(x)

Type 3 and JIT Compilation
--------------------------

Type 3 transforms require knowing the output grid size at JIT compile time. Use the helper functions:

.. code-block:: python

   from nufftax import compute_type3_grid_size

   # The grid size depends on the "spread" of your source and target points
   n_modes = compute_type3_grid_size(x, s, eps=1e-6)

   # Now you can JIT the transform
   @jax.jit
   def my_type3(x, c, s):
       return nufft1d3(x, c, s, n_modes=n_modes, eps=1e-6)

   result = my_type3(x, c, s)

For 2D and 3D Type 3 transforms, use ``compute_type3_grid_sizes_2d`` and ``compute_type3_grid_sizes_3d``.

Further Reading
---------------

- `FINUFFT documentation <https://finufft.readthedocs.io/>`_ - The C++ library that inspired this implementation
- `Accelerating the Nonuniform FFT <https://epubs.siam.org/doi/10.1137/S003614450343200X>`_ - The foundational paper by Greengard & Lee
- `Wikipedia: Non-uniform discrete Fourier transform <https://en.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform>`_
