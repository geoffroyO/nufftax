# JAX-FINUFFT Development Guide

## Project Goal
Create a pure JAX implementation of FINUFFT (Non-Uniform FFT) focusing on Type 1 and Type 2 transforms with full support for JAX transformations: `jit`, `grad`, `jvp`, `vjp`, `vmap`.

## Reference Implementation
The original FINUFFT is at: `../finufft/`

Key files to reference:
- `../finufft/src/finufft_core.cpp` - Main algorithm
- `../finufft/src/spreadinterp.cpp` - Spreading/interpolation
- `../finufft/include/finufft_common/kernel.h` - Kernel functions
- `../finufft/python/finufft/finufft/_interfaces.py` - Python API

## Development Workflow

### Auto-Continuation System
This project uses an auto-continuation hook. After each agent task:
1. The hook checks implementation progress
2. Suggests the next task to implement
3. Development continues automatically

### Agent Usage
Use specialized agents in `.claude/agents/`:

1. **kernel-dev**: Start here. Implement kernel evaluation.
   ```
   Read .claude/agents/kernel-dev.md for detailed instructions
   ```

2. **spread-dev**: Implement spreading/interpolation.
   ```
   Read .claude/agents/spread-dev.md for detailed instructions
   ```

3. **transform-dev**: Implement full NUFFT transforms.
   ```
   Read .claude/agents/transform-dev.md for detailed instructions
   ```

4. **autodiff-dev**: Add gradient support.
   ```
   Read .claude/agents/autodiff-dev.md for detailed instructions
   ```

5. **test-runner**: Validate against FINUFFT.
   ```
   Read .claude/agents/test-runner.md for detailed instructions
   ```

## Implementation Order

### Phase 1: Core Components
1. `jax_finufft/core/kernel.py` - ES kernel, parameter computation
2. `jax_finufft/utils/params.py` - Kernel parameter selection
3. `jax_finufft/utils/grid.py` - Grid size utilities

### Phase 2: Spreading/Interpolation
4. `jax_finufft/core/spread.py` - spread_1d, spread_2d, spread_3d
5. Add interp_1d, interp_2d, interp_3d to spread.py

### Phase 3: Transforms
6. `jax_finufft/core/deconvolve.py` - Deconvolution functions
7. `jax_finufft/transforms/nufft1.py` - Type 1 transforms
8. `jax_finufft/transforms/nufft2.py` - Type 2 transforms

### Phase 4: Autodiff
9. Add `@jax.custom_vjp` to nufft1, nufft2
10. Add `@jax.custom_jvp` for forward-mode

### Phase 5: Testing
11. `tests/test_kernel.py`
12. `tests/test_spread.py`
13. `tests/test_nufft1.py`, `tests/test_nufft2.py`
14. `tests/test_transforms.py` - JAX transformation tests

## Key Mathematical Formulas

### Type 1 (Nonuniform → Uniform)
```
f[k] = Σ(j=0 to M-1) c[j] * exp(i * sign * k * x[j])
```

### Type 2 (Uniform → Nonuniform)
```
c[j] = Σ(k in K) f[k] * exp(i * sign * k * x[j])
```

### ES Kernel
```
phi(z) = exp(beta * (sqrt(1 - c*z²) - 1))
```

## Testing Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_kernel.py -v

# Test with FINUFFT comparison
python -c "from jax_finufft import nufft1d1; import finufft; ..."
```

## Code Style
- Use type hints
- Follow JAX conventions (lowercase function names)
- Document all public functions with docstrings
- Keep functions pure (no side effects)

## Common Patterns

### JIT-compatible loops
```python
# BAD: Python loop
for i in range(n):
    result += ...

# GOOD: JAX scan or vmap
result = jax.lax.fori_loop(0, n, body_fn, init)
# or
result = jax.vmap(fn)(inputs).sum()
```

### Custom gradients
```python
@jax.custom_vjp
def my_function(x, y):
    return _impl(x, y)

def my_function_fwd(x, y):
    return _impl(x, y), (x, y)

def my_function_bwd(res, g):
    x, y = res
    return (grad_x, grad_y)

my_function.defvjp(my_function_fwd, my_function_bwd)
```

## When Agent Stops

If development is interrupted:
1. Check `.claude/state/development_state.json` for progress
2. Read the suggested next task from hook output
3. Resume with the appropriate agent
4. Or use: `claude code --continue` to resume

## Success Criteria

- [ ] All 1D, 2D, 3D Type 1 transforms match FINUFFT within tolerance
- [ ] All 1D, 2D, 3D Type 2 transforms match FINUFFT within tolerance
- [ ] `jax.jit` works without retracing
- [ ] `jax.grad` gives correct gradients (verified via finite diff)
- [ ] `jax.vmap` works for batched transforms
- [ ] `jax.jvp` works for forward-mode AD
- [ ] Float32 and float64 precision supported
