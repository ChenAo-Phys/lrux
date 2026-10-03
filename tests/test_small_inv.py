import os

os.environ["JAX_ENABLE_X64"] = "1"
# CPU unless the caller picks a platform: JAX_PLATFORMS=cuda pytest runs the fused kernel
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from lrux._small_inv import det_inv, det_value, pf_inv, pf_value

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]
IDS = ["f32", "f64", "c64", "c128"]
dtypes = pytest.mark.parametrize("dtype", DTYPES, ids=IDS)


def is_complex(dtype):
    return np.issubdtype(dtype, np.complexfloating)


def ref64(x):
    x = np.asarray(x)
    return x.astype(np.complex128 if is_complex(x.dtype) else np.float64)


def randn(shape, dtype, gen):
    z = gen.standard_normal(shape)
    if is_complex(dtype):
        z = z + 1j * gen.standard_normal(shape)
    return z.astype(dtype)


def skew(A):
    return A - np.swapaxes(A, -1, -2)


def well_conditioned(shape, dtype, gen):
    """(randn / (2 sqrt n) + 1.5 I) with its rows randomly permuted: condition number
    ~3 and |det| inside float32 range, while the dominant entries sit off the diagonal,
    so partial pivoting has to swap (and the determinant's sign matters)."""
    n = shape[-1]
    A = randn(shape, dtype, gen) / (2 * np.sqrt(max(n, 1))) + 1.5 * np.eye(n)
    A = A[..., gen.permutation(n), :]
    return A.astype(dtype)


def well_conditioned_skew(shape, dtype, gen):
    """The skew analogue: P (skew(randn) / (2 sqrt n) + 1.5 J) P^T with a random
    permutation P (J = sum of [[0, 1], [-1, 0]]), so the Parlett-Reid pivoting swaps."""
    n = shape[-1]
    J = np.kron(np.eye(n // 2), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    S = skew(randn(shape, dtype, gen)) / (2 * np.sqrt(n)) + 1.5 * J
    p = gen.permutation(n)
    S = S[..., p, :][..., :, p]
    return S.astype(dtype)


def relerr(x, ref):
    return float(np.max(np.abs(x - ref)) / max(np.max(np.abs(ref)), 1e-30))


def tol(dtype, t32, t64):
    return t32 if np.finfo(dtype).bits == 32 else t64


def rel_tol(dtype):
    return tol(dtype, 1e-4, 1e-10)


def assert_value(v, ref, dtype):
    """det / pf values against the 64-bit reference: magnitude and phase."""
    v, ref = np.asarray(v), np.asarray(ref)
    assert np.all(np.isfinite(v)) and np.all(v != 0)
    assert relerr(v, ref) < tol(dtype, 1e-3, 1e-10)


def assert_same(a, b, dtype):
    """The same computation in two programs (batched vs vmapped, plain vs under jvp):
    bit-identical on a GPU (the kernel / cuSOLVER do the same arithmetic in both), to
    round-off on XLA:CPU (FMA contraction and solve blocking depend on the program)."""
    a, b = np.asarray(a), np.asarray(b)
    if jax.default_backend() == "gpu":
        assert np.array_equal(a, b)
    else:
        assert relerr(a, b) < tol(dtype, 1e-5, 1e-12)


def pf64(S):
    """Pfaffian by Parlett-Reid with partial pivoting in 64-bit NumPy."""
    A = np.array(ref64(S))
    n = A.shape[0]
    val = 1.0 + 0.0j if np.iscomplexobj(A) else 1.0
    for k in range(0, n - 1, 2):
        kp = k + 1 + np.argmax(np.abs(A[k + 1 :, k]))
        if kp != k + 1:
            A[[k + 1, kp], :] = A[[kp, k + 1], :]
            A[:, [k + 1, kp]] = A[:, [kp, k + 1]]
            val = -val
        d = A[k, k + 1]
        if d == 0:
            return 0.0
        val = val * d
        if k + 2 < n:
            tau = -A[k + 2 :, k] / d
            w = A[k + 2 :, k + 1]
            A[k + 2 :, k + 2 :] += np.outer(tau, w) - np.outer(w, tau)
    return val


# polynomial (n <= 4), fused kernel on GPU / one LU on CPU (5..32), one LU (33, 40)
@pytest.mark.parametrize("n", [1, 3, 4, 5, 8, 13, 16, 31, 32, 33, 40])
@dtypes
def test_det_inv(n, dtype):
    gen = np.random.default_rng(100 + n)
    A = well_conditioned((4, n, n), dtype, gen)
    A[1, :, 2 % n] = 0.0  # exactly singular member
    d, X = map(np.asarray, det_inv(jnp.asarray(A)))
    assert d.shape == (4,) and X.shape == (4, n, n)
    assert d.dtype == dtype and X.dtype == dtype
    assert d[1] == 0 and np.all(X[1] == 0)
    reg = [0, 2, 3]
    A64 = ref64(A)[reg]
    assert_value(d[reg], np.linalg.det(A64), dtype)
    assert relerr(X[reg], np.linalg.inv(A64)) < rel_tol(dtype)


# polynomial (n <= 6), fused kernel on GPU / fermix.pf + LU on CPU (8..32), above
@pytest.mark.parametrize("n", [2, 4, 6, 8, 14, 18, 32, 34, 40])
@dtypes
def test_pf_inv(n, dtype):
    gen = np.random.default_rng(200 + n)
    S = well_conditioned_skew((4, n, n), dtype, gen)
    S[1, 3 % n, :] = 0.0  # exactly singular member (rank n - 2)
    S[1, :, 3 % n] = 0.0
    p, X = map(np.asarray, pf_inv(jnp.asarray(S)))
    assert p.shape == (4,) and X.shape == (4, n, n)
    assert p.dtype == dtype and X.dtype == dtype
    assert p[1] == 0 and np.all(X[1] == 0)
    reg = [0, 2, 3]
    S64 = ref64(S)[reg]
    assert_value(p[reg], [pf64(s) for s in S64], dtype)
    assert relerr(X[reg], np.linalg.inv(S64)) < rel_tol(dtype)
    # the input is skew-symmetrized: a non-skew input gives the same result as its
    # skew part
    N = randn(S.shape, dtype, gen)
    p2, X2 = map(np.asarray, pf_inv(jnp.asarray(S + N + np.swapaxes(N, -1, -2))))
    assert relerr(p2[reg], p[reg]) < rel_tol(dtype)
    assert relerr(X2[reg], X[reg]) < rel_tol(dtype)


@pytest.mark.parametrize("n", [3, 8, 40])
@dtypes
def test_jvp_and_grad(n, dtype):
    """d det = det tr(A^-1 dA), d pf = pf tr(S^-1 dS) / 2 (dS the skew part of the
    tangent), d A^-1 = -A^-1 dA A^-1; the primal under jvp equals the plain call; the
    gradient of Re det is the holomorphic adj(A)^T."""
    rt = rel_tol(dtype)
    gen = np.random.default_rng(300 + n)
    A = well_conditioned((3, n, n), dtype, gen)
    dA = randn(A.shape, dtype, gen)
    x, dx = jnp.asarray(A), jnp.asarray(dA)
    (d, X), (dd, dX) = jax.jvp(det_inv, (x,), (dx,))
    for direct, under_jvp in zip(det_inv(x), (d, X)):
        assert_same(direct, under_jvp, dtype)
    A64, dA64 = ref64(A), ref64(dA)
    inv = np.linalg.inv(A64)
    tr = np.einsum("bij,bji->b", inv, dA64)
    assert relerr(np.asarray(dd), np.linalg.det(A64) * tr) < rt
    assert relerr(np.asarray(dX), -inv @ dA64 @ inv) < rt
    g = np.asarray(jax.grad(lambda a: jnp.real(det_inv(a)[0]).sum())(x))
    ref = np.linalg.det(A64)[:, None, None] * np.swapaxes(inv, -1, -2)
    assert relerr(g, ref) < rt

    m = n + n % 2
    S = well_conditioned_skew((3, m, m), dtype, gen)
    dS = randn(S.shape, dtype, gen)
    s, ds = jnp.asarray(S), jnp.asarray(dS)
    (p, Y), (dp, dY) = jax.jvp(pf_inv, (s,), (ds,))
    for direct, under_jvp in zip(pf_inv(s), (p, Y)):
        assert_same(direct, under_jvp, dtype)
    S64, dS64 = ref64(S), 0.5 * skew(ref64(dS))
    Sinv = np.linalg.inv(S64)
    pref = np.array([pf64(t) for t in S64])
    tr = np.einsum("bij,bji->b", Sinv, dS64)
    assert relerr(np.asarray(dp), 0.5 * pref * tr) < rt
    assert relerr(np.asarray(dY), -Sinv @ dS64 @ Sinv) < rt


@pytest.mark.parametrize("n", [8, 40])
@dtypes
def test_vmap_matches_batched(n, dtype):
    """lrux calls these per sample inside jax.vmap: the vmapped results equal the
    batched ones, singular members included; the gradients to round-off."""
    gen = np.random.default_rng(400 + n)
    A = well_conditioned((4, n, n), dtype, gen)
    A[0, :, 3] = 0.0
    S = well_conditioned_skew((4, n, n), dtype, gen)
    S[0, 2, :] = 0.0
    S[0, :, 2] = 0.0
    for f, x in ((det_inv, A), (pf_inv, S)):
        x = jnp.asarray(x)
        for b, v in zip(f(x), jax.vmap(f)(x)):
            assert_same(b, v, dtype)
        loss = lambda a: jnp.real(f(a)[0]) + jnp.sum(jnp.real(f(a)[1]))
        gb = np.asarray(jax.grad(lambda a: jnp.sum(jax.vmap(loss)(a)))(x))
        gv = np.asarray(jax.vmap(jax.grad(loss))(x))
        assert np.all(np.isfinite(gb))
        assert relerr(gv, gb) < tol(dtype, 1e-6, 1e-14)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128], ids=["f64", "c128"])
def test_second_order(dtype):
    """The jvp rules take the primal through the custom_jvp function, so a nested jvp
    applies the rule again instead of differentiating the kernel."""
    n = 8
    gen = np.random.default_rng(500)
    hol = dict(holomorphic=is_complex(dtype))
    A = jnp.asarray(well_conditioned((n, n), dtype, gen))
    h = jax.hessian(lambda a: det_inv(a)[0], **hol)(A)
    h_ref = jax.hessian(jnp.linalg.det, **hol)(A)
    assert relerr(np.asarray(h), np.asarray(h_ref)) < rel_tol(dtype)


def test_shapes_and_edge_cases():
    """Leading batch dims, no batch dim, n = 0 (value 1, empty inverse), odd n for the
    pfaffian (zeros) and the input checks."""
    gen = np.random.default_rng(600)
    A = jnp.asarray(well_conditioned((2, 3, 5, 5), np.float64, gen))
    d, X = det_inv(A)
    assert d.shape == (2, 3) and X.shape == (2, 3, 5, 5)
    assert np.allclose(np.asarray(X), np.linalg.inv(np.asarray(A)))
    d, X = det_inv(A[0, 0])
    assert d.shape == () and X.shape == (5, 5)
    p, Y = pf_inv(A[..., :4, :4])
    assert p.shape == (2, 3) and Y.shape == (2, 3, 4, 4)
    for f in (det_inv, pf_inv):
        v, Z = f(jnp.zeros((3, 0, 0), np.float64))
        assert v.shape == (3,) and np.all(np.asarray(v) == 1) and Z.shape == (3, 0, 0)
        with pytest.raises(ValueError):
            f(jnp.zeros((3, 4), np.float64))
    p, Y = pf_inv(jnp.ones((2, 5, 5), np.float64))
    assert np.all(np.asarray(p) == 0) and np.all(np.asarray(Y) == 0)
    # jit + vmap over a scalar-batch call, the way lrux uses it
    f = jax.jit(jax.vmap(det_inv))
    d, X = f(A[0])
    assert np.allclose(np.asarray(X), np.linalg.inv(np.asarray(A[0])))


@pytest.mark.parametrize("n", [8, 16, 34])
@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_pf_inv_numerically_singular(n, dtype):
    """Rank n - 2 skew matrices U W U^T rounded to the dtype: the factorizations see a
    round-off-sized pivot or an exact zero (and fermix.pf and an LU need not agree).
    The outputs stay finite, and a zero inverse always comes with a zero pfaffian."""
    gen = np.random.default_rng(700 + n)
    big = np.complex128 if is_complex(dtype) else np.float64
    U = randn((64, n, n - 2), big, gen)
    W = skew(randn((64, n - 2, n - 2), big, gen))
    S = (U @ W @ np.swapaxes(U, -1, -2)).astype(dtype)
    p, X = map(np.asarray, pf_inv(jnp.asarray(S)))
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(X))
    zero = np.all(X == 0, axis=(-2, -1))
    assert np.all(p[zero] == 0)
    # the same holds for the gradient of the pfaffian
    g = np.asarray(jax.grad(lambda a: jnp.real(pf_inv(a)[0]).sum())(jnp.asarray(S)))
    assert np.all(np.isfinite(g))


@pytest.mark.parametrize("n", [3, 5, 8])
def test_value_second_derivatives(n):
    """det_value / pf_value return fermix's value, but take every derivative from the
    det_inv / pf_inv rules: second directional derivatives of complex det / pf against
    jnp.linalg.det and against central finite differences (fermix 0.1.0's own rules
    lose the phase term above its polynomial sizes)."""
    dtype = np.complex128
    gen = np.random.default_rng(800 + n)
    A = jnp.asarray(well_conditioned((n, n), dtype, gen))
    dA = jnp.asarray(randn((n, n), dtype, gen))
    d0 = np.asarray(det_value(A))
    assert relerr(d0, np.linalg.det(np.asarray(A))) < 1e-12

    def second(f, x, dx):
        g = lambda t: jax.jvp(lambda s: f(x + s * dx), (t,), (1.0,))[1]
        return jax.jvp(g, (0.0,), (1.0,))[1]

    ours = np.asarray(second(det_value, A, dA))
    ref = np.asarray(second(jnp.linalg.det, A, dA))
    assert relerr(ours, ref) < 1e-10
    m = n + n % 2
    S = jnp.asarray(well_conditioned_skew((m, m), dtype, gen))
    dS = jnp.asarray(skew(randn((m, m), dtype, gen)))
    h = 1e-4
    f = lambda t: np.asarray(pf_value(S + t * dS))
    fd = (f(h) - 2 * f(0.0) + f(-h)) / h**2
    ours = np.asarray(second(pf_value, S, dS))
    assert relerr(ours, fd) < 1e-5
