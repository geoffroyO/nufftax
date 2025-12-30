# JAX-FINUFFT Architecture

## Overview

Pure JAX implementation of FINUFFT (Flatiron Institute Non-Uniform Fast Fourier Transform) focusing on Type 1 and Type 2 NUFFTs with full JAX transformation support.

## Algorithm Structure

### Type 1 NUFFT (Nonuniform → Uniform)
```
f[k] = Σ(j=0 to M-1) c[j] * exp(±i * k · x[j])  for k ∈ K
```

**Steps:**
1. **Spread**: Spread nonuniform point values onto oversampled uniform grid using kernel
2. **FFT**: Apply FFT to the uniform grid
3. **Deconvolve**: Divide by kernel Fourier coefficients and shuffle modes

### Type 2 NUFFT (Uniform → Nonuniform)
```
c[j] = Σ(k ∈ K) f[k] * exp(±i * k · x[j])  for j = 0, ..., M-1
```

**Steps:**
1. **Deconvolve**: Amplify Fourier modes by dividing by kernel Fourier coefficients
2. **iFFT**: Apply inverse FFT
3. **Interpolate**: Interpolate from uniform grid to nonuniform points

## Key Components

### 1. Kernel Module (`jax_finufft/core/kernel.py`)
- ES (Exponential of Semicircle) kernel: `phi(x) = exp(beta * (sqrt(1 - c*x²) - 1))`
- Kaiser-Bessel kernel (optional): `phi(x) = I_0(beta * sqrt(1 - c*x²)) / I_0(beta)`
- Horner polynomial approximation for efficient JIT-compiled evaluation
- Kernel parameter computation based on tolerance

### 2. Spreading/Interpolation (`jax_finufft/core/spread.py`)
- `spread_1d`, `spread_2d`, `spread_3d`: Scatter nonuniform values to uniform grid
- `interp_1d`, `interp_2d`, `interp_3d`: Gather from uniform grid at nonuniform points
- Custom VJP rules for gradient computation
- Efficient batched operations for vmap compatibility

### 3. Deconvolution (`jax_finufft/core/deconvolve.py`)
- Precompute kernel Fourier series coefficients
- Apply correction in frequency domain
- Mode ordering (CMCL vs FFT-style)

### 4. NUFFT Functions (`jax_finufft/transforms/`)
- `nufft1d1`, `nufft1d2`: 1D transforms
- `nufft2d1`, `nufft2d2`: 2D transforms
- `nufft3d1`, `nufft3d2`: 3D transforms
- Plan-based interface for point reuse

## JAX Transformation Support

### JIT Compilation
- All operations use JAX primitives (no Python loops in hot paths)
- Static shapes where possible, dynamic shapes via `jax.lax.dynamic_*`

### Automatic Differentiation
- **grad w.r.t. strengths (c)**: Type 1 adjoint is Type 2; Type 2 adjoint is Type 1
- **grad w.r.t. points (x)**: Requires derivative of kernel, computed analytically
- Custom `vjp` rules for spreading/interpolation operations

### Vectorization (vmap)
- Batched transforms over multiple strength vectors
- Batched over different point sets (with careful padding)

### Forward-mode AD (jvp)
- Supports efficient Jacobian-vector products
- Essential for some optimization algorithms

## Key Design Decisions

### 1. Pure JAX Operations
- No custom CUDA kernels or XLA primitives initially
- Use `jax.lax.scatter_add` for spreading (atomic accumulation)
- Use `jax.lax.gather` for interpolation

### 2. Upsampling Strategy
- Default `upsampfac = 2.0` (100% oversampling)
- Configurable for accuracy/speed tradeoff
- Grid sizes rounded to products of small primes (2, 3, 5) for FFT efficiency

### 3. Kernel Width Selection
- `nspread` (kernel width) determined by tolerance
- Higher tolerance → smaller kernel → faster but less accurate
- Typical: nspread=10 for tol=1e-6

### 4. Memory Layout
- Complex arrays stored as complex64/complex128
- Points stored as float32/float64
- Contiguous memory for cache efficiency

## Development Agents

The `.claude/agents/` directory contains specialized agents for development:

1. **kernel-dev**: Kernel function implementation and optimization
2. **spread-dev**: Spreading/interpolation development
3. **transform-dev**: NUFFT transform implementation
4. **autodiff-dev**: Custom gradient rules
5. **test-runner**: Validation against FINUFFT reference
6. **benchmark**: Performance testing

## Testing Strategy

1. **Correctness Tests**: Compare against `finufft` Python library
2. **Gradient Tests**: Verify gradients via finite differences
3. **Transformation Tests**: Ensure jit/vmap/grad compose correctly
4. **Numerical Tests**: Check precision across tolerance levels

## File Structure

```
jax_finufft/
├── jax_finufft/
│   ├── __init__.py           # Public API
│   ├── core/
│   │   ├── __init__.py
│   │   ├── kernel.py         # Kernel evaluation
│   │   ├── spread.py         # Spreading/interpolation
│   │   ├── deconvolve.py     # Deconvolution
│   │   └── fft.py            # FFT wrappers
│   ├── transforms/
│   │   ├── __init__.py
│   │   ├── nufft1.py         # Type 1 transforms
│   │   ├── nufft2.py         # Type 2 transforms
│   │   └── plan.py           # Plan-based interface
│   └── utils/
│       ├── __init__.py
│       ├── grid.py           # Grid size selection
│       └── params.py         # Parameter computation
├── tests/
│   ├── test_kernel.py
│   ├── test_spread.py
│   ├── test_nufft1.py
│   ├── test_nufft2.py
│   └── test_transforms.py
├── .claude/
│   ├── agents/               # Agent definitions
│   └── hooks/                # Auto-continuation hooks
└── pyproject.toml
```
