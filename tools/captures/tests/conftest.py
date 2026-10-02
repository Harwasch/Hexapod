"""Pins OpenBLAS to one CPU kernel before numpy loads.

The committed tiles are checked byte for byte against a fresh run, and packaging goes
through BLAS (the batched covariance matmul, `eigh` on the merged parents). OpenBLAS picks
its kernels by CPU, and the last bits of those floats differ between them: the yard written
with SkylakeX kernels did not match one written on a CI runner's Haswell/Zen kernels.
Haswell's run on any AVX2 machine, so every checkout and CI writes the same bytes;
regenerate the committed fixtures the same way (synthetic_yard.py's usage).
"""

import os

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
