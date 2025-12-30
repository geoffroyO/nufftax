"""
Pytest fixtures and utilities for JAX-FINUFFT tests.

Provides common test fixtures, numerical utilities, and shared configuration.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# ============================================================================
# Configuration
# ============================================================================

# Seed for reproducible tests
DEFAULT_SEED = 42

# Precision levels to test
PRECISION_LEVELS = [1e-3, 1e-6, 1e-9]

# Default problem sizes
DEFAULT_M = 500  # Number of nonuniform points
DEFAULT_N_MODES_1D = 64
DEFAULT_N_MODES_2D = (32, 32)
DEFAULT_N_MODES_3D = (16, 16, 16)


# ============================================================================
# Fixtures for random number generation
# ============================================================================


@pytest.fixture
def rng():
    """Provide a seeded numpy random number generator."""
    return np.random.default_rng(DEFAULT_SEED)


@pytest.fixture
def jax_key():
    """Provide a seeded JAX PRNG key."""
    return jax.random.PRNGKey(DEFAULT_SEED)


# ============================================================================
# Fixtures for test data generation
# ============================================================================


@pytest.fixture
def random_points_1d(rng):
    """Generate random 1D nonuniform points in [0, 2*pi)."""

    def _generate(M=DEFAULT_M, dtype=np.float64):
        return rng.uniform(0, 2 * np.pi, M).astype(dtype)

    return _generate


@pytest.fixture
def random_points_2d(rng):
    """Generate random 2D nonuniform points in [0, 2*pi)^2."""

    def _generate(M=DEFAULT_M, dtype=np.float64):
        x = rng.uniform(0, 2 * np.pi, M).astype(dtype)
        y = rng.uniform(0, 2 * np.pi, M).astype(dtype)
        return x, y

    return _generate


@pytest.fixture
def random_points_3d(rng):
    """Generate random 3D nonuniform points in [0, 2*pi)^3."""

    def _generate(M=DEFAULT_M, dtype=np.float64):
        x = rng.uniform(0, 2 * np.pi, M).astype(dtype)
        y = rng.uniform(0, 2 * np.pi, M).astype(dtype)
        z = rng.uniform(0, 2 * np.pi, M).astype(dtype)
        return x, y, z

    return _generate


@pytest.fixture
def random_strengths(rng):
    """Generate random complex strengths."""

    def _generate(M=DEFAULT_M, dtype=np.complex128):
        real_dtype = np.float64 if dtype == np.complex128 else np.float32
        real = rng.standard_normal(M).astype(real_dtype)
        imag = rng.standard_normal(M).astype(real_dtype)
        return (real + 1j * imag).astype(dtype)

    return _generate


@pytest.fixture
def random_modes_1d(rng):
    """Generate random 1D Fourier modes."""

    def _generate(n_modes=DEFAULT_N_MODES_1D, dtype=np.complex128):
        real_dtype = np.float64 if dtype == np.complex128 else np.float32
        real = rng.standard_normal(n_modes).astype(real_dtype)
        imag = rng.standard_normal(n_modes).astype(real_dtype)
        return (real + 1j * imag).astype(dtype)

    return _generate


@pytest.fixture
def random_modes_2d(rng):
    """Generate random 2D Fourier modes."""

    def _generate(n_modes=DEFAULT_N_MODES_2D, dtype=np.complex128):
        real_dtype = np.float64 if dtype == np.complex128 else np.float32
        shape = n_modes if isinstance(n_modes, tuple) else (n_modes, n_modes)
        real = rng.standard_normal(shape).astype(real_dtype)
        imag = rng.standard_normal(shape).astype(real_dtype)
        return (real + 1j * imag).astype(dtype)

    return _generate


@pytest.fixture
def random_modes_3d(rng):
    """Generate random 3D Fourier modes."""

    def _generate(n_modes=DEFAULT_N_MODES_3D, dtype=np.complex128):
        real_dtype = np.float64 if dtype == np.complex128 else np.float32
        shape = n_modes if isinstance(n_modes, tuple) else (n_modes, n_modes, n_modes)
        real = rng.standard_normal(shape).astype(real_dtype)
        imag = rng.standard_normal(shape).astype(real_dtype)
        return (real + 1j * imag).astype(dtype)

    return _generate


# ============================================================================
# Numerical utilities
# ============================================================================


def numerical_gradient_real(
    f: Callable,
    x: jax.Array,
    h: float = 1e-7,
) -> jax.Array:
    """
    Compute numerical gradient for real-valued input.

    Uses central finite differences: (f(x+h) - f(x-h)) / (2h)

    Args:
        f: Function that takes real array and returns scalar
        x: Point at which to compute gradient
        h: Step size for finite difference

    Returns:
        Numerical gradient, same shape as x
    """
    grad = np.zeros_like(x)
    x_flat = x.flatten()

    for i in range(len(x_flat)):
        x_plus = x_flat.copy()
        x_minus = x_flat.copy()
        x_plus[i] += h
        x_minus[i] -= h

        f_plus = float(f(x_plus.reshape(x.shape)))
        f_minus = float(f(x_minus.reshape(x.shape)))
        grad.flat[i] = (f_plus - f_minus) / (2 * h)

    return jnp.array(grad)


def numerical_gradient_complex(
    f: Callable,
    z: jax.Array,
    h: float = 1e-7,
) -> jax.Array:
    """
    Compute numerical gradient for complex-valued input.

    For complex inputs with a real-valued loss, we compute
    the Wirtinger derivative: df/dz* = df/d(Re z) + i * df/d(Im z)

    Args:
        f: Function that takes complex array and returns real scalar
        z: Complex point at which to compute gradient
        h: Step size for finite difference

    Returns:
        Conjugate gradient (df/dz*), same shape as z
    """
    z_np = np.array(z)
    grad_real = np.zeros_like(z_np.real)
    grad_imag = np.zeros_like(z_np.imag)

    for i in range(z_np.size):
        # Real part perturbation
        z_plus = z_np.copy()
        z_minus = z_np.copy()
        z_plus.flat[i] += h
        z_minus.flat[i] -= h

        f_plus = float(f(jnp.array(z_plus)))
        f_minus = float(f(jnp.array(z_minus)))
        grad_real.flat[i] = (f_plus - f_minus) / (2 * h)

        # Imaginary part perturbation
        z_plus = z_np.copy()
        z_minus = z_np.copy()
        z_plus.flat[i] += 1j * h
        z_minus.flat[i] -= 1j * h

        f_plus = float(f(jnp.array(z_plus)))
        f_minus = float(f(jnp.array(z_minus)))
        grad_imag.flat[i] = (f_plus - f_minus) / (2 * h)

    # Wirtinger gradient: df/dz* = (df/dx + i*df/dy)
    return jnp.array(grad_real + 1j * grad_imag)


def relative_error(computed: jax.Array, reference: jax.Array) -> float:
    """
    Compute relative L2 error between computed and reference arrays.

    Args:
        computed: Computed array
        reference: Reference array

    Returns:
        Relative error ||computed - reference|| / ||reference||
    """
    diff_norm = float(jnp.linalg.norm(computed - reference))
    ref_norm = float(jnp.linalg.norm(reference))
    if ref_norm < 1e-15:
        return diff_norm
    return diff_norm / ref_norm


def assert_allclose_complex(
    actual: jax.Array,
    desired: jax.Array,
    rtol: float = 1e-7,
    atol: float = 0,
    err_msg: str = "",
):
    """
    Assert that two complex arrays are close.

    Args:
        actual: Computed array
        desired: Expected array
        rtol: Relative tolerance
        atol: Absolute tolerance
        err_msg: Error message prefix
    """
    np.testing.assert_allclose(
        np.array(actual.real), np.array(desired.real), rtol=rtol, atol=atol, err_msg=f"{err_msg} (real part)"
    )
    np.testing.assert_allclose(
        np.array(actual.imag), np.array(desired.imag), rtol=rtol, atol=atol, err_msg=f"{err_msg} (imag part)"
    )


# ============================================================================
# Reference implementations using DFT (for small problem sizes)
# ============================================================================


def dft_nufft1d1(x: np.ndarray, c: np.ndarray, n_modes: int) -> np.ndarray:
    """
    Direct DFT implementation of 1D Type 1 NUFFT.

    f[k] = sum_j c[j] * exp(i * k * x[j])  for k = -N/2, ..., N/2-1

    Args:
        x: Nonuniform points, shape (M,)
        c: Complex strengths, shape (M,)
        n_modes: Number of output modes

    Returns:
        Fourier modes, shape (n_modes,)
    """
    k = np.arange(n_modes) - n_modes // 2
    # Compute f[k] = sum_j c[j] * exp(i*k*x[j])
    f = np.zeros(n_modes, dtype=np.complex128)
    for ik, kval in enumerate(k):
        f[ik] = np.sum(c * np.exp(1j * kval * x))
    return f


def dft_nufft1d2(x: np.ndarray, f: np.ndarray, isign: int = 1) -> np.ndarray:
    """
    Direct DFT implementation of 1D Type 2 NUFFT.

    c[j] = sum_k f[k] * exp(isign * i * k * x[j])  for k = -N/2, ..., N/2-1

    For Type 2 to be the adjoint of Type 1 (with isign=1), use isign=-1.

    Args:
        x: Nonuniform points, shape (M,)
        f: Fourier modes, shape (n_modes,)
        isign: Sign of imaginary unit in exponential (default: 1)

    Returns:
        Complex values at nonuniform points, shape (M,)
    """
    n_modes = len(f)
    k = np.arange(n_modes) - n_modes // 2
    # Compute c[j] = sum_k f[k] * exp(isign*i*k*x[j])
    c = np.zeros(len(x), dtype=np.complex128)
    for j, xj in enumerate(x):
        c[j] = np.sum(f * np.exp(isign * 1j * k * xj))
    return c


# ============================================================================
# Test markers
# ============================================================================


def _finufft_available():
    """Check if finufft is available."""
    try:
        import finufft  # noqa: F401

        return True
    except ImportError:
        return False


slow = pytest.mark.slow
requires_finufft = pytest.mark.skipif(not _finufft_available(), reason="finufft not installed")


# ============================================================================
# Parametrize helpers
# ============================================================================

precision_params = pytest.mark.parametrize("eps", PRECISION_LEVELS, ids=[f"eps={e}" for e in PRECISION_LEVELS])

dtype_params = pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])

complex_dtype_params = pytest.mark.parametrize("dtype", [np.complex64, np.complex128], ids=["complex64", "complex128"])
