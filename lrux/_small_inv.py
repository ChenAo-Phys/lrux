r"""
Determinant / pfaffian *and* inverse of the small k x k matrix of a low-rank update.

With ``return_update=True``, `~lrux.det_lru` and `~lrux.pf_lru` (and their delayed
versions) need both the ratio :math:`\det(R)` or :math:`\mathrm{pf}(R)` and
:math:`R^{-1}` for the inverse update. A determinant followed by a solve factorizes
:math:`R` twice and launches two library calls; here:

- det k <= 2 / pf k <= 4: the explicit polynomial of ``fermix.det`` / ``fermix.pf``
  and its derivative, the adjugate (:math:`\partial \det / \partial R =
  \mathrm{adj}(R)^T`, :math:`\partial \mathrm{pf} / \partial S = \mathrm{pf}(S)
  S^{-T} / 2`), so the inverse needs no factorization at all. (For det k = 3, 4 and pf
  k = 6 the adjugate inverse loses accuracy in float32 / complex64 on ill-conditioned
  R, so those sizes take the paths below.)
- up to k = 32 on a CUDA GPU: one fused Pallas (Triton) kernel, one program per matrix
  holding :math:`R` as a register tile. Gauss-Jordan elimination with partial pivoting
  gives the determinant and :math:`R^{-1}` from the same pivots; for the pfaffian a
  Parlett-Reid pass on the same tile comes first, then Gauss-Jordan for the inverse.
- otherwise (other devices, or k > 32): the determinant and :math:`R^{-1}` from one LU;
  the pfaffian from ``fermix.pf`` and :math:`R^{-1}` from an LU.

A singular matrix (an exactly zero pivot, or an inverse that is not finite) gives the
value 0 and an all-zero inverse. Both functions are differentiable: d det =
det tr(R^-1 dR), d pf = pf tr(S^-1 dS) / 2 and d R^-1 = -R^-1 dR R^-1 are formed from
the returned values. `det_value` / `pf_value` return ``fermix.det`` / ``fermix.pf``
of the small matrix with derivatives of every order from these rules (fermix 0.1.0's
own rules lose the phase term of complex second derivatives above its polynomial
sizes).
"""

import contextlib
import functools
import warnings
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array, lax, tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
from fermix import FermixFallbackWarning, det, pf

# largest matrix of the fused kernel: one 32 x 32 register tile per matrix
SMALL_MAX = 32
# sizes whose inverse comes from the adjugate of fermix's explicit polynomial: up to
# these it is as accurate as LU; det k = 3, 4 and pf k = 6 are not in float32 /
# complex64 (cond(R) = 1e5: median inverse error 3e-3 / 2e-2 / 7e-3 vs 2e-4 / 2e-4 /
# 6e-4 for LU), so they go to the kernel or the LU path
_DET_POLY_MAX = 2
_PF_POLY_MAX = 4


@contextlib.contextmanager
def _quiet():
    """fermix warns that it falls back to its generic XLA path off a CUDA GPU; for the
    tiny matrices of a low-rank update that path is the expected one, so lrux's
    internal calls silence the warning (it is emitted at trace time)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FermixFallbackWarning)
        yield


_KINDS = tuple(
    jnp.dtype(t) for t in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)


# --------------------------------------------------------- values inside the kernel
@tree_util.register_pytree_node_class
class _C:
    """A complex value as two real arrays (Triton has no complex type)."""

    __slots__ = ("re", "im")

    def __init__(self, re, im):
        self.re = re
        self.im = im

    def tree_flatten(self):
        return (self.re, self.im), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)

    def __getitem__(self, idx):
        return _C(self.re[idx], self.im[idx])

    def __neg__(self):
        return _C(-self.re, -self.im)

    def __add__(self, o):
        if isinstance(o, _C):
            return _C(self.re + o.re, self.im + o.im)
        return _C(self.re + o, self.im)

    __radd__ = __add__

    def __sub__(self, o):
        if isinstance(o, _C):
            return _C(self.re - o.re, self.im - o.im)
        return _C(self.re - o, self.im)

    def __rsub__(self, o):
        return _C(o - self.re, -self.im)

    def __mul__(self, o):
        if isinstance(o, _C):
            re = self.re * o.re - self.im * o.im
            im = self.re * o.im + self.im * o.re
            return _C(re, im)
        return _C(self.re * o, self.im * o)

    __rmul__ = __mul__


def _parts(x):
    return (x.re, x.im) if isinstance(x, _C) else (x,)


def _from_parts(p):
    return _C(*p) if len(p) == 2 else p[0]


def _where(c, x, y):
    if isinstance(x, _C) or isinstance(y, _C):
        xr, xi = (x.re, x.im) if isinstance(x, _C) else (x, 0.0)
        yr, yi = (y.re, y.im) if isinstance(y, _C) else (y, 0.0)
        return _C(jnp.where(c, xr, yr), jnp.where(c, xi, yi))
    return jnp.where(c, x, y)


def _vsum(x, axis=None):
    if isinstance(x, _C):
        return _C(jnp.sum(x.re, axis=axis), jnp.sum(x.im, axis=axis))
    return jnp.sum(x, axis=axis)


def _abs1(x):
    """Pivot magnitude: |x|, or |re| + |im| for complex values (LAPACK's cabs1)."""
    if isinstance(x, _C):
        return jnp.abs(x.re) + jnp.abs(x.im)
    return jnp.abs(x)


def _iszero(x):
    if isinstance(x, _C):
        return (x.re == 0.0) & (x.im == 0.0)
    return x == 0.0


def _inv0(x):
    """1 / x with 1 / 0 -> 0 (lax.select: jnp.where on scalars can lower badly)."""
    return lax.select(x == 0.0, jnp.zeros_like(x), 1.0 / x)


def _recip(x):
    """Guarded reciprocal 1 / x (0 -> 0); conj(x) / |x|^2 for complex values."""
    if isinstance(x, _C):
        s = _inv0(x.re * x.re + x.im * x.im)
        return _C(x.re * s, -(x.im * s))
    return _inv0(x)


class _Kind:
    """Static description of the dtype of one call: the real component dtype, the
    number of components and the typed constants the kernels need."""

    def __init__(self, dtype):
        self.dtype = jnp.dtype(dtype)
        self.cplx = bool(jnp.issubdtype(self.dtype, jnp.complexfloating))
        self.real = jnp.dtype(jnp.finfo(self.dtype).dtype)
        self.k = 2 if self.cplx else 1

    def rs(self, x):
        """A real constant of the component dtype (a NumPy scalar, so the in-kernel
        constant does not depend on the x64 flag)."""
        return self.real.type(x)

    def one(self):
        return _C(self.rs(1.0), self.rs(0.0)) if self.cplx else self.rs(1.0)

    def split(self, A):
        return (jnp.real(A), jnp.imag(A)) if self.cplx else (A,)

    def join(self, p):
        return lax.complex(p[0], p[1]) if self.cplx else p[0]


def _ld(refs, idx):
    return _from_parts(tuple(r[idx] for r in refs))


def _mld(refs, idx, mask):
    return _from_parts(tuple(plgpu.load(r.at[idx], mask=mask, other=0.0) for r in refs))


def _mst(refs, idx, val, mask):
    for r, p in zip(refs, _parts(val)):
        plgpu.store(r.at[idx], p, mask=mask)


# ---------------------------------------------------------- in-register algorithms
def _gauss_jordan(M, kind, N):
    """In-place Gauss-Jordan elimination with virtual partial pivoting on the (N, N)
    register tile M. No row ever moves: step j picks the unused row p with the largest
    |M[p, j]|, scales it by 1 / pivot (its entry j becomes 1 / pivot) and eliminates
    column j from every other row (their entry j becomes -factor / pivot), the in-place
    form of [A | I] -> [I | A^-1]. Returns (M, pos, perm, det, singular):
    A^-1[pos[i], perm[j]] = M[i, j], with pos[i] the step at which row i was the pivot
    and perm[j] the pivot row of step j; det = sgn(perm) prod(pivots); singular = an
    exactly zero pivot occurred (its reciprocal is then 0 and det exactly 0)."""
    ib = lax.broadcasted_iota(jnp.int32, (N, N), 0)
    jb = lax.broadcasted_iota(jnp.int32, (N, N), 1)
    ci = lax.broadcasted_iota(jnp.int32, (N,), 0)
    one = kind.rs(1.0)
    minus = kind.rs(-1.0)

    def step(j, carry):
        M, pos, perm, d, unused, nzero = carry
        col = _vsum(_where(jb == j, M, 0.0), axis=1)
        cand = jnp.where(unused, _abs1(col), -1.0)
        p = lax.argmax(cand, 0, jnp.int32)
        isp = ci == p
        piv = _vsum(_where(isp, col, 0.0))
        inv = _recip(piv)
        rowp = _vsum(_where(ib == p, M, 0.0), axis=0)
        rowp_new = _where(ci == j, inv, rowp * inv)
        f = _where(isp, 0.0, col)
        upd = M - f[:, None] * rowp_new[None, :]
        upd = _where(jb == j, -(f * inv)[:, None], upd)
        M = _where(isp[:, None], rowp_new[None, :], upd)
        d = d * piv
        pos = jnp.where(isp, j, pos)
        perm = jnp.where(ci == j, p, perm)
        unused = unused & (~isp)
        nzero = nzero + lax.select(_iszero(piv), jnp.int32(1), jnp.int32(0))
        return M, pos, perm, d, unused, nzero

    idx0 = jnp.zeros((N,), jnp.int32)
    carry = (M, idx0, idx0, kind.one(), ci < N, jnp.int32(0))
    M, pos, perm, d, _, nzero = lax.fori_loop(0, N, step, carry)
    # sgn(perm) = sgn(pos), from the inversion count of the (tiny) tile
    inversions = jnp.sum((pos[:, None] > pos[None, :]) & (ib < jb), dtype=jnp.int32)
    d = d * lax.select((inversions & 1) == 1, minus, one)
    return M, pos, perm, d, nzero > 0


def _parlett_reid(S, kind, N):
    """Pfaffian of the skew-symmetric (N, N) register tile S by Parlett-Reid with
    physical row / column swaps (the pivoting of fermix: pair step s works on columns
    c = 2s and c + 1, the pivot is the entry of largest magnitude below row c in column
    c, swapped into row c + 1). Returns sgn(P) prod(d_s), exactly 0 for a zero pivot."""
    ib = lax.broadcasted_iota(jnp.int32, (N, N), 0)
    jb = lax.broadcasted_iota(jnp.int32, (N, N), 1)
    ci = lax.broadcasted_iota(jnp.int32, (N,), 0)
    one = kind.rs(1.0)
    minus = kind.rs(-1.0)

    def step(s, carry):
        S, p = carry
        c = 2 * s
        col = _vsum(_where(jb == c, S, 0.0), axis=1)
        cand = jnp.where(ci > c, _abs1(col), -1.0)
        kp = lax.argmax(cand, 0, jnp.int32)
        swap = kp != c + 1
        ra = _vsum(_where(ib == c + 1, S, 0.0), axis=0)
        rb = _vsum(_where(ib == kp, S, 0.0), axis=0)
        S = _where(ib == c + 1, rb[None, :], _where(ib == kp, ra[None, :], S))
        ca = _vsum(_where(jb == c + 1, S, 0.0), axis=1)
        cb = _vsum(_where(jb == kp, S, 0.0), axis=1)
        S = _where(jb == c + 1, cb[:, None], _where(jb == kp, ca[:, None], S))
        d = _vsum(_where((ib == c) & (jb == c + 1), S, 0.0))
        inv = _recip(d)
        colc = _vsum(_where(jb == c, S, 0.0), axis=1)
        colc1 = _vsum(_where(jb == c + 1, S, 0.0), axis=1)
        below = ci > c + 1
        tau = _where(below, -(colc * inv), 0.0)
        w = _where(below, colc1, 0.0)
        S = S + tau[:, None] * w[None, :] - w[:, None] * tau[None, :]
        p = p * d * lax.select(swap, minus, one)
        return S, p

    _, p = lax.fori_loop(0, N // 2, step, (S, kind.one()))
    return p


def _load_tile(a_refs, kind, n, N, skew):
    """The n x n matrix as an N x N register tile (N a power of 2 >= n), padded with
    the identity (skew=False) or with [[0, 1], [-1, 0]] blocks (skew=True, pfaffian
    +1): a nonsingular, decoupled padding with the same det / pf, whose inverse holds
    A^-1 in the leading n x n block."""
    ib = lax.broadcasted_iota(jnp.int32, (N, N), 0)
    jb = lax.broadcasted_iota(jnp.int32, (N, N), 1)
    if N == n:
        return _ld(a_refs, (pl.ds(0, N), pl.ds(0, N)))
    M = _mld(a_refs, (pl.ds(0, N), pl.ds(0, N)), (ib < n) & (jb < n))
    if skew:
        even = ((ib - n) & 1) == 0
        upper = (ib >= n) & (jb == ib + 1) & even
        lower = (ib >= n) & (jb == ib - 1) & (~even)
        return _where(upper, 1.0, _where(lower, -1.0, M))
    return _where((ib == jb) & (ib >= n), 1.0, M)


def _store(val_refs, inv_refs, val, M, pos, perm, singular, kind, n, N):
    """value -> val_refs; the inverse -> inv_refs through the pivot-order scatter
    out[pos[i], perm[j]] = M[i, j] (padding rows / columns masked). A singular matrix
    stores 0 and an all-zero inverse (through the identity scatter, so that every
    entry of the output is written)."""
    ci = lax.broadcasted_iota(jnp.int32, (N,), 0)
    val = val * lax.select(singular, kind.rs(0.0), kind.rs(1.0))
    M = _where(singular, 0.0, M)
    pos = jnp.where(singular, ci, pos)
    perm = jnp.where(singular, ci, perm)
    for r, comp in zip(val_refs, _parts(val)):
        r[0] = comp
    valid = (pos[:, None] < n) & (perm[None, :] < n)
    _mst(inv_refs, (pos[:, None], perm[None, :]), M, valid)


def _det_inv_kernel(a_refs, val_refs, inv_refs, *, kind, n, N):
    M = _load_tile(a_refs, kind, n, N, skew=False)
    M, pos, perm, d, singular = _gauss_jordan(M, kind, N)
    _store(val_refs, inv_refs, d, M, pos, perm, singular, kind, n, N)


def _pf_inv_kernel(a_refs, val_refs, inv_refs, *, kind, n, N):
    S = _load_tile(a_refs, kind, n, N, skew=True)
    p = _parlett_reid(S, kind, N)
    # the inverse by Gauss-Jordan on a fresh copy of the tile (the pair steps destroyed
    # the first one)
    S = _load_tile(a_refs, kind, n, N, skew=True)
    M, pos, perm, _, singular = _gauss_jordan(S, kind, N)
    _store(val_refs, inv_refs, p, M, pos, perm, singular, kind, n, N)


def _num_warps(N, kind):
    """Warps per program. Exclusive A100-80GB, B = 4096, interleaved vs one warp
    (2026-10-03): the 8 x 8 / 16 x 16 tiles are fastest on one warp for every dtype
    (more warps only add cross-warp reductions); the 32 x 32 tile on one warp for
    float32 (2 warps 0.97-0.98x) and on two for the wider dtypes (float64 1.02 / 1.03x,
    complex128 1.02 / 1.13x for det / pf)."""
    if N <= 16 or kind.dtype == jnp.float32:
        return 1
    return 2


def _small_call(A, n, kind, kernel):
    """The fused kernel on the (B, n, n) array A: one program per matrix. Returns
    (value (B,), inverse (B, n, n))."""
    B = A.shape[0]
    N = max(8, 1 << (n - 1).bit_length())
    k = kind.k
    mat = pl.BlockSpec((None, n, n), lambda b: (b, 0, 0))
    vec = pl.BlockSpec((None, 1), lambda b: (b, 0))
    body = functools.partial(kernel, kind=kind, n=n, N=N)

    def wrapped(*refs):
        body(refs[:k], refs[k : 2 * k], refs[2 * k :])

    out_shape = [jax.ShapeDtypeStruct((B, 1), kind.real)] * k
    out_shape += [jax.ShapeDtypeStruct((B, n, n), kind.real)] * k
    res = pl.pallas_call(
        wrapped,
        name=kernel.__name__,
        grid=(B,),
        in_specs=[mat] * k,
        out_specs=[vec] * k + [mat] * k,
        out_shape=out_shape,
        compiler_params=plgpu.CompilerParams(
            num_warps=_num_warps(N, kind), num_stages=1
        ),
    )(*kind.split(A))
    return kind.join(res[:k])[:, 0], kind.join(res[k:])


# ------------------------------------------------------------- the other paths
def _safe_recip(x):
    nz = x != 0
    return jnp.where(nz, 1 / jnp.where(nz, x, 1), 0)


def _det_inv_poly(A):
    """(det, A^-1) for n <= _DET_POLY_MAX: fermix.det's polynomial and its derivative
    adj(A)^T, so A^-1 = adj(A) / det (0 for a singular A)."""
    with _quiet():
        d, vjp = jax.vjp(det, A)
    (adjT,) = vjp(jnp.ones_like(d))
    return d, jnp.swapaxes(adjT, -1, -2) * _safe_recip(d)[:, None, None]


def _pf_inv_poly(S):
    """(pf, S^-1) for skew S, n <= _PF_POLY_MAX: fermix.pf's polynomial and its
    derivative G = pf S^-T / 2 = -pf S^-1 / 2, so S^-1 = -2 G / pf (0 for a singular
    S)."""
    with _quiet():
        p, vjp = jax.vjp(lambda s: pf(s, skew_symmetrize=False), S)
    (G,) = vjp(jnp.ones_like(p))
    return p, -2 * G * _safe_recip(p)[:, None, None]


def _prod_pairwise(x):
    """Product over the last axis as an explicit pairwise tree. XLA picks the order of
    a reduction per program, so jnp.prod can differ in the last bit between a batched
    and a vmapped call of the same function; explicit elementwise products are never
    reordered."""
    n = x.shape[-1]
    if n == 0:
        return jnp.ones(x.shape[:-1], x.dtype)
    while n > 1:
        h = (n + 1) // 2
        if 2 * h > n:
            x = jnp.concatenate([x, jnp.ones(x.shape[:-1] + (1,), x.dtype)], axis=-1)
        x = x[..., :h] * x[..., h:]
        n = h
    return x[..., 0]


def _lu_inverse(A):
    """(det, A^-1, bad) from one LU, A = P^T L U; bad flags an exactly zero pivot or an
    inverse that is not finite."""
    n = A.shape[-1]
    lu, piv, perm = lax.linalg.lu(A)
    diag = jnp.diagonal(lu, axis1=-2, axis2=-1)
    swaps = jnp.sum(piv != jnp.arange(n, dtype=piv.dtype), axis=-1)
    sign = jnp.where(swaps % 2 == 1, -1, 1).astype(A.dtype)
    d = sign * _prod_pairwise(diag)
    X = jax.nn.one_hot(perm, n, dtype=A.dtype)  # P
    X = lax.linalg.triangular_solve(
        lu, X, left_side=True, lower=True, unit_diagonal=True
    )
    X = lax.linalg.triangular_solve(lu, X, left_side=True, lower=False)
    bad = jnp.any(diag == 0, axis=-1) | ~jnp.all(jnp.isfinite(X), axis=(-2, -1))
    return d, X, bad


def _zero_bad(val, X, bad):
    """Value 0 and an all-zero inverse where ``bad`` (never NaN, so no NaN reaches the
    tangents either)."""
    return jnp.where(bad, 0, val), jnp.where(bad[:, None, None], 0, X)


def _det_inv_lu(A):
    """(det, A^-1) from one LU (singular: (0, 0))."""
    d, X, bad = _lu_inverse(A)
    return _zero_bad(d, X, bad)


def _pf_inv_lu(S):
    """(pf, S^-1) for skew S: fermix.pf (its Pallas kernels on a GPU) and S^-1 from an
    LU (singular: (0, 0)). The two factorizations may disagree on a numerically
    singular S (an exactly zero LU pivot next to a round-off-sized pfaffian); either
    one flagging it zeroes both outputs, as the fused kernel does."""
    with _quiet():
        p = pf(S, skew_symmetrize=False)
    _, X, bad = _lu_inverse(S)
    return _zero_bad(p, X, bad | (p == 0))


def _cuda_or(A, kernels, generic):
    """``kernels`` when lowering for CUDA, ``generic`` elsewhere (chosen at lowering
    time, so a CPU array never reaches the Triton lowering)."""
    return lax.platform_dependent(A, cuda=kernels, default=generic)


def _inv_tangents(X, dA):
    XdA = jnp.matmul(X, dA, precision=lax.Precision.HIGHEST)
    tr = jnp.trace(XdA, axis1=-2, axis2=-1)
    return tr, -jnp.matmul(XdA, X, precision=lax.Precision.HIGHEST)


@functools.lru_cache(maxsize=None)
def _det_inv_fn(n, dtype):
    kind = _Kind(dtype)

    def value(A):
        if n <= _DET_POLY_MAX:
            return _det_inv_poly(A)
        if n <= SMALL_MAX and kind.dtype in _KINDS:
            kernels = lambda A: _small_call(A, n, kind, _det_inv_kernel)
            return _cuda_or(A, kernels, _det_inv_lu)
        return _det_inv_lu(A)

    @jax.custom_jvp
    def f(A):
        return value(A)

    @f.defjvp
    def f_jvp(primals, tangents):
        (A,), (dA,) = primals, tangents
        # the primal through f itself: a nested jvp applies this rule again, so
        # higher derivatives never differentiate the kernel
        d, X = f(A)
        tr, dX = _inv_tangents(X, dA)
        return (d, X), (d * tr, dX)

    return jax.jit(f)


@functools.lru_cache(maxsize=None)
def _pf_inv_fn(n, dtype):
    kind = _Kind(dtype)

    def value(S):
        if n <= _PF_POLY_MAX:
            return _pf_inv_poly(S)
        if n <= SMALL_MAX and kind.dtype in _KINDS:
            kernels = lambda S: _small_call(S, n, kind, _pf_inv_kernel)
            return _cuda_or(S, kernels, _pf_inv_lu)
        return _pf_inv_lu(S)

    @jax.custom_jvp
    def f(S):
        return value(S)

    @f.defjvp
    def f_jvp(primals, tangents):
        (S,), (dS,) = primals, tangents
        p, X = f(S)
        tr, dX = _inv_tangents(X, dS)
        return (p, X), (0.5 * p * tr, dX)

    return jax.jit(f)


@functools.lru_cache(maxsize=None)
def _det_value_fn(n, dtype):
    inv_fn = _det_inv_fn(n, dtype)

    @jax.custom_jvp
    def f(A):
        with _quiet():
            return det(A)

    @f.defjvp
    def f_jvp(primals, tangents):
        (A,), (dA,) = primals, tangents
        d, X = inv_fn(A)
        tr = jnp.trace(jnp.matmul(X, dA, precision=lax.Precision.HIGHEST), 0, -2, -1)
        return d, d * tr

    return jax.jit(f)


@functools.lru_cache(maxsize=None)
def _pf_value_fn(n, dtype):
    inv_fn = _pf_inv_fn(n, dtype)

    @jax.custom_jvp
    def f(S):
        with _quiet():
            return pf(S, skew_symmetrize=False)

    @f.defjvp
    def f_jvp(primals, tangents):
        (S,), (dS,) = primals, tangents
        p, X = inv_fn(S)
        tr = jnp.trace(jnp.matmul(X, dS, precision=lax.Precision.HIGHEST), 0, -2, -1)
        return p, 0.5 * p * tr

    return jax.jit(f)


def _check_square(a):
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"Expect input shape (..., n, n), got {a.shape}.")


def det_inv(a: Array) -> tuple:
    r"""
    Determinant and inverse of a (batch of) square matrices from one factorization
    (the polynomial adjugate, the fused kernel's Gauss-Jordan pass, or one LU).

    :param a:
        An array of shape (..., n, n).

    :return:
        ``(det, inv)`` with shapes (...) and (..., n, n). An exactly singular matrix
        gives ``det = 0`` and an all-zero ``inv``.
    """
    a = jnp.asarray(a)
    _check_square(a)
    n = a.shape[-1]
    batch = a.shape[:-2]
    flat = a.reshape((int(np.prod(batch)), n, n))
    d, X = _det_inv_fn(n, a.dtype)(flat)
    return d.reshape(batch), X.reshape(a.shape)


def pf_inv(a: Array) -> tuple:
    r"""
    Pfaffian and inverse of a (batch of) skew-symmetric matrices: the polynomial
    adjugate, one fused kernel on a CUDA GPU (Parlett-Reid, then Gauss-Jordan on the
    same register tile), or ``fermix.pf`` plus one LU. The input is skew-symmetrized as
    :math:`S = (A - A^T) / 2`, like ``fermix.pf``.

    :param a:
        An array of shape (..., n, n).

    :return:
        ``(pf, inv)`` with shapes (...) and (..., n, n), ``inv`` = :math:`S^{-1}`. An
        exactly singular :math:`S` (and any odd n) gives ``pf = 0`` and an all-zero
        ``inv``.
    """
    a = jnp.asarray(a)
    _check_square(a)
    n = a.shape[-1]
    batch = a.shape[:-2]
    if n % 2 == 1:
        return jnp.zeros(batch, a.dtype), jnp.zeros(a.shape, a.dtype)
    S = (a - jnp.swapaxes(a, -1, -2)) / 2
    flat = S.reshape((int(np.prod(batch)), n, n))
    p, X = _pf_inv_fn(n, a.dtype)(flat)
    return p.reshape(batch), X.reshape(a.shape)


def det_value(a: Array) -> Array:
    r"""
    ``fermix.det`` of a (batch of) small square matrices, with derivatives of every
    order taken from `det_inv` (d det = det tr(A^{-1} dA)).
    """
    a = jnp.asarray(a)
    _check_square(a)
    n = a.shape[-1]
    batch = a.shape[:-2]
    d = _det_value_fn(n, a.dtype)(a.reshape((int(np.prod(batch)), n, n)))
    return d.reshape(batch)


def pf_value(a: Array) -> Array:
    r"""
    ``fermix.pf`` of a (batch of) small skew-symmetric matrices (skew-symmetrized as
    :math:`(A - A^T) / 2`), with derivatives of every order taken from `pf_inv`
    (d pf = pf tr(S^{-1} dS) / 2). Odd n gives 0.
    """
    a = jnp.asarray(a)
    _check_square(a)
    n = a.shape[-1]
    batch = a.shape[:-2]
    if n % 2 == 1:
        return jnp.zeros(batch, a.dtype)
    S = (a - jnp.swapaxes(a, -1, -2)) / 2
    p = _pf_value_fn(n, a.dtype)(S.reshape((int(np.prod(batch)), n, n)))
    return p.reshape(batch)
