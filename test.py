"""Quick check that the container works and JAX can use the GPU."""
import importlib
import sys
import time

print(f"Python {sys.version.split()[0]}")

# 1. JAX and GPU
import jax
import jax.numpy as jnp

print(f"JAX {jax.__version__}")
devices = jax.devices()
print(f"Devices: {devices}")
print(f"Backend: {jax.default_backend()}")

gpu_ok = jax.default_backend() == "gpu"
print("GPU found" if gpu_ok else "WARNING: no GPU found, JAX is using the CPU")

# 2. Run a computation on the default device
x = jnp.ones((4000, 4000))
f = jax.jit(lambda a: (a @ a).sum())
f(x).block_until_ready()  # warm-up / compile
t0 = time.time()
result = f(x).block_until_ready()
print(f"4000x4000 matmul: result={float(result):.3e}, time={time.time() - t0:.4f}s")
print(f"Result lives on: {x.devices() if hasattr(x, 'devices') else 'unknown'}")

# 3. Check the project's packages import
packages = ["dLux", "zodiax", "optax", "numpyro", "chainconsumer",
            "numpy", "scipy", "pandas", "skimage", "imageio", "matplotlib", "tqdm"]
failed = []
for name in packages:
    try:
        mod = importlib.import_module(name)
        print(f"  OK   {name} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        failed.append(name)

if failed or not gpu_ok:
    print("\nTest finished with problems.")
    sys.exit(1)
print("\nAll checks passed.")
