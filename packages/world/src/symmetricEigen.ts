/**
 * Small dense symmetric linear algebra for the reduced-order models (`skinWind.ts`): a
 * Cholesky factor and the cyclic Jacobi eigen-solver. Matrices are row-major `Float64Array`s of
 * `n × n`, `n` a handful to a few dozen; nothing here is meant for large systems.
 */

/** `a[i][j]` of a row-major `n × n` matrix. */
function at(a: ArrayLike<number>, n: number, i: number, j: number): number {
  return a[i * n + j] ?? 0;
}

/**
 * The lower Cholesky factor `L` of a symmetric positive-definite `a` (`L·Lᵀ = a`), or undefined
 * when a pivot is not positive.
 */
export function cholesky(a: ArrayLike<number>, n: number): Float64Array | undefined {
  const l = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j <= i; j += 1) {
      let sum = at(a, n, i, j);
      for (let k = 0; k < j; k += 1) sum -= at(l, n, i, k) * at(l, n, j, k);
      if (i === j) {
        if (!(sum > 0)) return undefined;
        l[i * n + i] = Math.sqrt(sum);
      } else {
        l[i * n + j] = sum / at(l, n, j, j);
      }
    }
  }
  return l;
}

/** `L⁻¹·b` for a lower-triangular `L` and a row-major `n × cols` `b`. */
export function forwardSubstitute(
  l: ArrayLike<number>,
  n: number,
  b: ArrayLike<number>,
  cols: number,
): Float64Array {
  const x = Float64Array.from({ length: n * cols }, (_, k) => b[k] ?? 0);
  for (let c = 0; c < cols; c += 1) {
    for (let i = 0; i < n; i += 1) {
      let sum = x[i * cols + c] ?? 0;
      for (let k = 0; k < i; k += 1) sum -= at(l, n, i, k) * (x[k * cols + c] ?? 0);
      x[i * cols + c] = sum / at(l, n, i, i);
    }
  }
  return x;
}

/** `L⁻ᵀ·b` for a lower-triangular `L` and a row-major `n × cols` `b`. */
export function backSubstituteTransposed(
  l: ArrayLike<number>,
  n: number,
  b: ArrayLike<number>,
  cols: number,
): Float64Array {
  const x = Float64Array.from({ length: n * cols }, (_, k) => b[k] ?? 0);
  for (let c = 0; c < cols; c += 1) {
    for (let i = n - 1; i >= 0; i -= 1) {
      let sum = x[i * cols + c] ?? 0;
      for (let k = i + 1; k < n; k += 1) sum -= at(l, n, k, i) * (x[k * cols + c] ?? 0);
      x[i * cols + c] = sum / at(l, n, i, i);
    }
  }
  return x;
}

export interface SymmetricEigen {
  /** Ascending. */
  readonly values: Float64Array;
  /** Row-major `n × n`: column `k` is the unit eigenvector of `values[k]`. */
  readonly vectors: Float64Array;
}

/**
 * Eigenvalues and orthonormal eigenvectors of a symmetric `n × n` matrix, by cyclic Jacobi
 * rotations (Golub & Van Loan §8.5): exact to rounding for the sizes used here, and
 * deterministic. Only the upper triangle is read; it is symmetrised first.
 */
export function symmetricEigen(matrix: ArrayLike<number>, n: number): SymmetricEigen {
  const a = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = i; j < n; j += 1) {
      const v = at(matrix, n, i, j);
      a[i * n + j] = v;
      a[j * n + i] = v;
    }
  }
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) v[i * n + i] = 1;
  let scale = 0;
  for (let k = 0; k < n * n; k += 1) scale += (a[k] ?? 0) ** 2;
  const tolerance = 1e-30 * Math.max(scale, 1e-300);
  for (let sweep = 0; sweep < 64; sweep += 1) {
    let off = 0;
    for (let i = 0; i < n; i += 1) for (let j = i + 1; j < n; j += 1) off += at(a, n, i, j) ** 2;
    if (off <= tolerance) break;
    for (let p = 0; p < n; p += 1) {
      for (let q = p + 1; q < n; q += 1) {
        const apq = at(a, n, p, q);
        if (apq === 0) continue;
        const app = at(a, n, p, p);
        const aqq = at(a, n, q, q);
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k += 1) {
          const akp = at(a, n, k, p);
          const akq = at(a, n, k, q);
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k += 1) {
          const apk = at(a, n, p, k);
          const aqk = at(a, n, q, k);
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k += 1) {
          const vkp = at(v, n, k, p);
          const vkq = at(v, n, k, q);
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, k) => k).sort(
    (x, y) => at(a, n, x, x) - at(a, n, y, y),
  );
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  order.forEach((k, col) => {
    values[col] = at(a, n, k, k);
    for (let i = 0; i < n; i += 1) vectors[i * n + col] = at(v, n, i, k);
  });
  return { values, vectors };
}
