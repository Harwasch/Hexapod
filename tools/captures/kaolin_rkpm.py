# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the Hexapod project, 2026-10-02 (see NOTICE beside this file):
# - ported from PyTorch to NumPy/SciPy, CPU only, float64 throughout;
# - kaolin/physics/simplicits/skinning/rkpm.py (SimplicitsRKPM, RKPM), the bounding-box
#   normalisation of kaolin/physics/simplicits/network.py (SkinningModule) and
#   kaolin/physics/materials/material_utils.py (to_lame) gathered in one module;
# - the generalised eigenproblem is solved with scipy.linalg.eigh (dense, smallest
#   eigenpairs) instead of torch.lobpcg, with a relative jitter on the mass matrix so it is
#   positive definite on thin shapes;
# - farthest point sampling is a deterministic NumPy loop (seeded) instead of Kaolin's CUDA
#   kernel;
# - a relative ridge (MOMENT_RIDGE) on the RKPM moment matrix, so coplanar nodes (a flat
#   object: a lawn, a wall) do not make it singular;
# - kernel evaluations are chunked over query points to bound memory;
# - (2026-10-05) an optional penalty that pins chosen points (`init(..., pinned=...)`), for
#   eigenmodes of a rooted object whose base stays still.
"""FreeForm / Simplicits skinning weights from the Reproducing Kernel Particle Method.

Vendored from NVIDIA Kaolin (master, 2026; Apache-2.0) because the pip release (0.18) does not
ship it yet, and ported to NumPy so the capture tools need no PyTorch. The method
(https://research.nvidia.com/labs/sil/projects/freeform/): place `num_nodes` Gaussian kernels
on the shape by farthest point sampling, correct them to reproduce linear fields (RKPM, first
order), assemble a mass matrix `M = Φᵀ Φ` and an elastic stiffness `H = Jᵀ diag(λ + 4μ) J`
from the kernels' gradients over sample points, and keep the smallest non-constant generalised
eigenvectors `H c = λ M c` as weight fields. The constant field (eigenvalue 0) is the rigid
handle and is appended as a column of ones by `compute_skinning_weights`.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.linalg import eigh
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

__all__ = ["RKPM", "SimplicitsRKPM", "farthest_point_sampling", "to_lame"]

#: Query points per kernel evaluation block.
CHUNK = 2048
#: Relative ridge on the moment matrix M(x) (a Hexapod addition; Kaolin has none).
MOMENT_RIDGE = 1e-9


def to_lame(yms: np.ndarray, prs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Young's modulus and Poisson's ratio to Lamé's `(mu, lambda)`."""
    mus = yms / (2 * (1 + prs))
    lams = yms * prs / ((1 + prs) * (1 - 2 * prs))
    return mus, lams


def farthest_point_sampling(pts: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Indices of `k` points of `pts` (n, 3), each the farthest from those already chosen."""
    rng = np.random.default_rng(seed)
    n = pts.shape[0]
    idx = np.empty(k, np.int64)
    idx[0] = int(rng.integers(n))
    d = np.full(n, np.inf)
    for i in range(1, k):
        d = np.minimum(d, ((pts - pts[idx[i - 1]]) ** 2).sum(1))
        idx[i] = int(d.argmax())
    return idx


class RKPM:
    r"""First-order Reproducing Kernel Particle Method functions on scattered nodes.

    The corrected kernel :math:`\phi_I(x) = C(x)ᵀ P(x_I) \, f_I(x)`, with :math:`f_I` a
    Gaussian of node radius :math:`r_I` and :math:`C(x) = M(x)^{-1} P(x)`,
    :math:`M(x) = \sum_I f_I(x) P(x_I) P(x_I)ᵀ`, reproduces :math:`[1, x, y, z]` exactly.
    """

    def __init__(self, nodes: np.ndarray, radius: np.ndarray, polynomial_degree: int = 1):
        if polynomial_degree != 1:
            raise ValueError("Unknown polynomial degree")
        self.nodes = np.asarray(nodes, np.float64)
        self.radius = np.asarray(radius, np.float64)
        self.num_nodes = self.nodes.shape[0]
        self.num_dims = 3
        self.P = 1 + self.num_dims
        self._pn = self.polynomial(self.nodes)  # (N, P)
        self._pn_pnt = np.einsum("Ni,Nj->Nij", self._pn, self._pn)  # (N, P, P)

    @staticmethod
    def polynomial(x: np.ndarray) -> np.ndarray:
        """`[1, x, y, z]` per point, (n, 4)."""
        return np.concatenate([np.ones((x.shape[0], 1)), x], axis=-1)

    def func_x(self, x: np.ndarray) -> np.ndarray:
        """The uncorrected Gaussian kernels at `x`, (n, N)."""
        d2 = (
            (x * x).sum(1)[:, None]
            - 2.0 * x @ self.nodes.T
            + (self.nodes * self.nodes).sum(1)[None, :]
        )
        r = np.sqrt(np.maximum(d2, 0.0))
        return np.exp(-((r / self.radius[None, :]) ** 2))

    def _phi_block(self, x: np.ndarray, grad: bool):
        f = self.func_x(x)  # (n, N)
        n = x.shape[0]
        pp = self._pn_pnt.reshape(self.num_nodes, self.P * self.P)
        mx = (f @ pp).reshape(n, self.P, self.P)  # (n, P, P)
        # Hexapod: a relative ridge, so coplanar nodes (a flat object) stay solvable.
        mx = mx + np.eye(self.P)[None] * (
            MOMENT_RIDGE * np.trace(mx, axis1=1, axis2=2)[:, None, None] + 1e-300
        )
        px = self.polynomial(x)  # (n, P)
        cx = np.linalg.solve(mx, px[..., None])[..., 0]  # (n, P)
        cx_pnt = cx @ self._pn.T  # (n, N)
        phi = cx_pnt * f
        if not grad:
            return phi, None
        disp = x[:, None, :] - self.nodes[None, :, :]  # (n, N, D)
        dfunc = f[..., None] * (-2.0 / self.radius[None, :, None] ** 2) * disp  # (n, N, D)
        term1 = cx_pnt[..., None] * dfunc
        dpx = np.zeros((n, self.P, self.num_dims))
        dpx[:, 1:, :] = np.eye(self.num_dims)[None]
        # dM/dx (n, D, P, P) as one matrix product over the nodes.
        dmx = (np.swapaxes(dfunc, 1, 2) @ pp).reshape(n, self.num_dims, self.P, self.P)
        dmx_cx = np.einsum("ndij,nj->nid", dmx, cx)  # (n, P, D)
        dcx = np.linalg.solve(mx, dpx - dmx_cx)  # (n, P, D)
        term2 = (self._pn[None] @ dcx) * f[..., None]  # (n, N, D)
        return phi, term1 + term2

    def phi(self, x: np.ndarray) -> np.ndarray:
        """Corrected kernels at `x`, (n, N)."""
        x = np.asarray(x, np.float64)
        return np.concatenate(
            [self._phi_block(x[i : i + CHUNK], False)[0] for i in range(0, len(x), CHUNK)]
            or [np.zeros((0, self.num_nodes))]
        )

    def grad_phi(self, x: np.ndarray) -> np.ndarray:
        """Spatial gradients of the corrected kernels at `x`, (n, N, 3)."""
        x = np.asarray(x, np.float64)
        return np.concatenate(
            [self._phi_block(x[i : i + CHUNK], True)[1] for i in range(0, len(x), CHUNK)]
            or [np.zeros((0, self.num_nodes, 3))]
        )

    def __call__(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """The interpolation of node values `c` (N, C) at `x`, (n, C)."""
        x = np.asarray(x, np.float64)
        return np.concatenate(
            [self._phi_block(x[i : i + CHUNK], False)[0] @ c for i in range(0, len(x), CHUNK)]
            or [np.zeros((0, c.shape[1]))]
        )


class SimplicitsRKPM:
    r"""Simplicits skinning weights from RKPM eigenmodes (Kaolin's ``SimplicitsRKPM``).

    ``num_handles`` counts the constant handle: ``num_handles - 1`` eigenmodes are fitted and
    ``compute_skinning_weights`` appends the constant column. Points are mapped into the unit
    box ``(x - bb_min) / (bb_max - bb_min)`` before anything else, as Kaolin's
    ``SkinningModule`` does; pass a cube (equal extents) to keep the shape's proportions.
    """

    def __init__(
        self,
        num_handles: int,
        num_nodes: int,
        radius_scale: float = 1.0,
        radius_init_kNN: int = 2,
        radius_min: float | str | None = "3x",
        num_points: int | None = None,
        bb_min: np.ndarray | None = None,
        bb_max: np.ndarray | None = None,
        seed: int = 0,
        mass_jitter: float = 1e-10,
    ):
        self.bb_min = np.zeros(3) if bb_min is None else np.asarray(bb_min, np.float64)
        self.bb_max = np.ones(3) if bb_max is None else np.asarray(bb_max, np.float64)
        if not (self.bb_min < self.bb_max).all():
            raise ValueError(f"bb_min must be below bb_max: {self.bb_min}, {self.bb_max}")
        self.num_points = num_points
        self.num_handles = num_handles - 1  # the constant handle is appended, not fitted
        self.num_nodes = num_nodes
        self.radius_scale = radius_scale
        self.radius_init_kNN = radius_init_kNN
        self.radius_min = radius_min
        self.seed = seed
        self.mass_jitter = mass_jitter
        self.rkpm: RKPM | None = None
        self.evecs = np.zeros((num_nodes, self.num_handles))
        self.evals = np.zeros(self.num_handles)

    def _offset_scale(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, np.float64) - self.bb_min) / (self.bb_max - self.bb_min)

    def init(
        self,
        pts: np.ndarray,
        yms: np.ndarray,
        prs: np.ndarray,
        pinned: np.ndarray | None = None,
        pin_stiffness: float = 1e4,
    ) -> None:
        """Nodes by farthest point sampling, radii from node spacing, then the eigenanalysis.

        `pts` (n, 3) in the evaluation frame; `yms`, `prs` (n,) per point. (Kaolin's `init`
        also takes densities and a volume, which cancel out of the eigenvectors.)

        Hexapod: `pinned` (n,) marks points held still (a rooted object's base). A penalty
        `κ Φ_pᵀ Φ_p` over them is added to the Hessian (κ = `pin_stiffness` times the
        Hessian's mean diagonal over the pinned mass's), so every mode nearly vanishes there
        and none is the constant field: all `num_handles - 1` eigenvectors are kept.
        """
        pts = self._offset_scale(pts)
        n = pts.shape[0]
        if n < self.num_nodes:
            logger.warning(
                "num_nodes (%d) exceeds the points (%d): all are nodes", self.num_nodes, n
            )
            self.num_nodes = n
            node_indices = np.arange(n)
        else:
            node_indices = farthest_point_sampling(pts, self.num_nodes, self.seed)
        nodes = pts[node_indices]
        k = min(self.radius_init_kNN + 1, len(nodes))
        dists, _ = cKDTree(nodes).query(nodes, k=k)
        dists = dists.reshape(len(nodes), -1)
        node_radius = dists[:, -1] * self.radius_scale
        if isinstance(self.radius_min, float):
            node_radius = np.maximum(node_radius, self.radius_min)
        elif isinstance(self.radius_min, str):
            if not self.radius_min.endswith("x"):
                raise ValueError("radius_min must end with 'x'")
            factor = float(self.radius_min[:-1])
            pd, _ = cKDTree(pts).query(pts, k=2)
            node_radius = np.maximum(node_radius, pd[:, -1].mean() * factor)
        else:
            raise TypeError("Unknown radius_min")
        node_radius = np.maximum(node_radius, 1e-9)
        self.rkpm = RKPM(nodes, node_radius)

        if self.num_points is None or self.num_points >= n:
            sample = np.arange(n)
        else:
            sample = farthest_point_sampling(pts, self.num_points, self.seed + 1)
        x = pts[sample]
        mass = self.get_mass_matrix(x)
        hess = self.get_hessian_matrix(x, np.asarray(yms)[sample], np.asarray(prs)[sample])
        mass = mass + np.eye(len(mass)) * self.mass_jitter * np.trace(mass) / len(mass)
        if pinned is not None and np.asarray(pinned).any():
            held = pts[np.asarray(pinned, bool)]
            penalty = np.zeros_like(hess)
            for i in range(0, len(held), CHUNK):
                phi = self.rkpm.phi(held[i : i + CHUNK])
                penalty += phi.T @ phi
            kappa = pin_stiffness * np.trace(hess) / max(np.trace(penalty), 1e-300)
            hess = hess + kappa * penalty
            want = min(self.num_handles, len(mass))
            evals, evecs = eigh(hess, mass, subset_by_index=[0, want - 1])
            self.evecs, self.evals = evecs, evals
        else:
            want = min(self.num_handles + 1, len(mass))
            evals, evecs = eigh(hess, mass, subset_by_index=[0, want - 1])
            self.evecs = evecs[:, 1:]  # the first is the constant field
            self.evals = evals[1:]
        self.num_handles = self.evecs.shape[1]

    def get_mass_matrix(self, x: np.ndarray) -> np.ndarray:
        """`M = Φᵀ Φ` over sample points `x` (normalised frame)."""
        if self.rkpm is None:
            raise ValueError("RKPM not initialised")
        mass = np.zeros((self.rkpm.num_nodes, self.rkpm.num_nodes))
        for i in range(0, len(x), CHUNK):
            phi = self.rkpm.phi(x[i : i + CHUNK])
            mass += phi.T @ phi
        return mass

    def get_hessian_matrix(
        self, x: np.ndarray, yms: np.ndarray, prs: np.ndarray, reparameterize_lame: bool = True
    ) -> np.ndarray:
        """`H = Jᵀ diag(λ + 4μ) J`, `J` the kernels' gradients stacked over points and axes."""
        if self.rkpm is None:
            raise ValueError("RKPM not initialised")
        mus, lams = to_lame(np.asarray(yms, np.float64), np.asarray(prs, np.float64))
        coeff = lams + (4 if reparameterize_lame else 3) * mus
        hess = np.zeros((self.rkpm.num_nodes, self.rkpm.num_nodes))
        for i in range(0, len(x), CHUNK):
            g = self.rkpm.grad_phi(x[i : i + CHUNK])  # (n, N, D)
            c = coeff[i : i + CHUNK]
            for d in range(3):
                jd = g[:, :, d]
                hess += jd.T @ (c[:, None] * jd)
        return hess

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Weights at normalised points `x`, (n, num_handles - 1): the eigenmodes only."""
        if self.rkpm is None:
            raise ValueError("RKPM not initialised")
        return self.rkpm(x, self.evecs)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """Gradients of the eigenmode weights at normalised points, (n, C, 3)."""
        if self.rkpm is None:
            raise ValueError("RKPM not initialised")
        return np.swapaxes(np.swapaxes(self.rkpm.grad_phi(x), 1, 2) @ self.evecs, 1, 2)

    def compute_skinning_weights(self, pts: np.ndarray) -> np.ndarray:
        """Weights at `pts` (evaluation frame), with the constant handle last: (n, C + 1)."""
        w = self.forward(self._offset_scale(pts))
        return np.concatenate([w, np.ones((w.shape[0], 1))], axis=1)

    def compute_dwdx(self, pts: np.ndarray) -> np.ndarray:
        """Spatial Jacobian of `compute_skinning_weights` at `pts`, (n, C + 1, 3)."""
        g = self.grad(self._offset_scale(pts)) / (self.bb_max - self.bb_min)[None, None]
        return np.concatenate([g, np.zeros((g.shape[0], 1, 3))], axis=1)
