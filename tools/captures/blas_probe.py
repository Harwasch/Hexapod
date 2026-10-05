"""Temporary: the dgemm shapes kaolin_rkpm runs, under whatever OPENBLAS_CORETYPE is set."""

import os

import numpy as np

rng = np.random.default_rng(0)
for n, k, m in [(2048, 600, 16), (600, 2048, 600), (4000, 600, 600), (2048, 600, 600)]:
    a = rng.random((n, k))
    b = rng.random((k, m))
    c = a @ b
    d = a.T @ a
print("ok", os.environ.get("OPENBLAS_CORETYPE", "native"), float(c.sum() + d.sum()))
