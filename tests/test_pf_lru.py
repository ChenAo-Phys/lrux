import os

os.environ["JAX_ENABLE_X64"] = "1"
# CPU unless the caller picks a platform: JAX_PLATFORMS=cuda pytest runs the GPU paths
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import random
import jax
import jax.numpy as jnp
import jax.random as jr
from fermix import pf
from lrux import skew_eye, pf_lru, init_pf_carrier, merge_pf_delays, pf_lru_delayed


def _get_key():
    seed = random.randint(0, 2**31 - 1)
    return jr.key(seed)


pf_lru = jax.jit(pf_lru, static_argnums=2)


@pytest.mark.parametrize("n", [2, 10])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_rank_2(n, dtype):
    A = jr.normal(_get_key(), (n, n), dtype)
    A = (A - A.T) / 2  # make it skew-symmetric
    Ainv = jnp.linalg.inv(A)
    pfA = pf(A)

    # general update
    u = jr.normal(_get_key(), (n, 2), dtype)
    J = skew_eye(1, dtype)
    A_ = A - u @ J @ u.T
    ratio = pf_lru(Ainv, u)
    assert jnp.allclose(ratio, pf(A_) / pfA)
    ratio, new_inv = pf_lru(Ainv, u, return_update=True)
    assert jnp.allclose(ratio, pf(A_) / pfA)
    assert jnp.allclose(new_inv, jnp.linalg.inv(A_))

    # row-column update
    x = jr.normal(_get_key(), (n,), dtype)
    e = random.randint(0, n - 1)
    u = (x, e)
    A_ = A.at[e].add(x)
    A_ = A_.at[:, e].add(-x)
    ratio, new_inv = pf_lru(Ainv, u, return_update=True)
    assert jnp.allclose(ratio, pf(A_) / pfA)
    assert jnp.allclose(new_inv, jnp.linalg.inv(A_))


@pytest.mark.parametrize("n", [2, 10])
@pytest.mark.parametrize("kx, ke", [(3, 3), (2, 4), (5, 3), (20, 14)])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_rank_k(n, kx, ke, dtype):
    A = jr.normal(_get_key(), (n, n), dtype)
    A = (A - A.T) / 2  # make it skew-symmetric
    Ainv = jnp.linalg.inv(A)
    pfA = pf(A)

    xu = jr.normal(_get_key(), (n, kx), dtype)
    eu = jr.randint(_get_key(), ke, 0, n)

    eu_arr = jnp.zeros((n, ke), dtype).at[eu, jnp.arange(ke)].set(1)
    u_full = jnp.concatenate((xu, eu_arr), axis=1)
    J = skew_eye(u_full.shape[1] // 2, dtype)
    A_ = A - u_full @ J @ u_full.T

    ratio, new_inv = pf_lru(Ainv, (xu, eu), return_update=True)
    assert jnp.allclose(ratio, pf(A_) / pfA)
    assert jnp.allclose(new_inv, jnp.linalg.inv(A_))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_vmap(dtype):
    batch = 2
    n = 10
    k = 4
    A = jr.normal(_get_key(), (batch, n, n), dtype)
    A = (A - A.transpose(0, 2, 1)) / 2
    Ainv = jnp.linalg.inv(A)
    pfA = pf(A)

    u = jr.normal(_get_key(), (batch, n, k), dtype)
    J = skew_eye(k // 2, dtype)
    A_ = A - jnp.einsum("bnk,kl,bml->bnm", u, J, u)
    vmap_lru = jax.vmap(pf_lru, in_axes=(0, 0, None))
    ratio, new_inv = vmap_lru(Ainv, u, True)
    assert jnp.allclose(ratio, pf(A_) / pfA)
    assert jnp.allclose(new_inv, jnp.linalg.inv(A_))


@pytest.mark.parametrize("k", [2, 4, 8])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_single_delay(k, dtype):
    n = 10
    A = jr.normal(_get_key(), (n, n), dtype)
    A = (A - A.T) / 2
    carrier = init_pf_carrier(A, max_delay=n // 2)
    u = jr.normal(_get_key(), (n, k), dtype)
    J = skew_eye(k // 2, dtype)
    A_ = A - u @ J @ u.T

    ratio = pf_lru_delayed(carrier, u)
    assert jnp.allclose(ratio, pf(A_) / pf(A))


@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_multiple_delayed(k, dtype):
    n = 10
    max_delay = n // 2
    A = jr.normal(_get_key(), (n, n), dtype)
    A = (A - A.T) / 2
    carrier = init_pf_carrier(A, max_delay, k)
    pfA0 = pf(A)

    lru_fn = jax.jit(pf_lru_delayed, static_argnums=(2, 3), donate_argnums=0)
    merge_fn = jax.jit(merge_pf_delays, donate_argnums=0)

    for i in range(20):
        current_delay = i % max_delay
        ki = random.randint(0, k // 2) * 2  # ensure ki is even
        u = jr.normal(_get_key(), (n, ki), dtype)
        ratio, carrier = lru_fn(carrier, u, True, current_delay)

        if current_delay == max_delay - 1:
            carrier = merge_fn(carrier)

        J = skew_eye(ki // 2, dtype)
        A -= u @ J @ u.T
        pfA1 = pf(A)
        assert jnp.allclose(ratio, pfA1 / pfA0)
        pfA0 = pfA1


@pytest.mark.parametrize("k", [2, 8])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_grad(k, dtype):
    n = 10
    A = jr.normal(_get_key(), (n, n), dtype)
    A = (A - A.T) / 2
    Ainv = jnp.linalg.inv(A)
    pfA = pf(A)
    J = skew_eye(k // 2, dtype)
    u = jr.normal(_get_key(), (n, k), dtype)
    du = jr.normal(_get_key(), (n, k), dtype)

    def ratio_lru(u):
        return pf_lru(Ainv, u)

    def ratio_ref(u):
        return pf(A - u @ J @ u.T) / pfA

    def inv_lru(u):
        return pf_lru(Ainv, u, True)[1]

    def inv_ref(u):
        return jnp.linalg.inv(A - u @ J @ u.T)

    for f_lru, f_ref in [(ratio_lru, ratio_ref), (inv_lru, inv_ref)]:
        out_lru, jvp_lru = jax.jvp(f_lru, (u,), (du,))
        out_ref, jvp_ref = jax.jvp(f_ref, (u,), (du,))
        assert jnp.allclose(out_lru, out_ref)
        assert jnp.allclose(jvp_lru, jvp_ref)

    # the ratio is a polynomial in u, so a central difference is an independent check
    eps = 1e-6
    fd = (ratio_ref(u + eps * du) - ratio_ref(u - eps * du)) / (2 * eps)
    assert jnp.allclose(jax.jvp(ratio_lru, (u,), (du,))[1], fd, atol=1e-6)


@pytest.mark.parametrize("k", [8, 10])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_delayed_large_rank(k, dtype):
    """Delayed updates of rank k > 6, where the ratio and R^-1 come from pf_inv's
    kernel (GPU) or fermix.pf + LU path: ratios against the direct pfaffians and the
    merged inverse against the direct inverse."""
    n = 16
    max_delay = 3
    A = jr.normal(_get_key(), (n, n), dtype)
    A = (A - A.T) / 2 + 4 * skew_eye(n // 2, dtype)
    carrier = init_pf_carrier(A, max_delay, k)
    pfA0 = pf(A)
    J = skew_eye(k // 2, dtype)
    for i in range(max_delay):
        u = 0.3 * jr.normal(_get_key(), (n, k), dtype)
        ratio, carrier = pf_lru_delayed(carrier, u, True, i)
        A = A - u @ J @ u.T
        pfA1 = pf(A)
        assert jnp.allclose(ratio, pfA1 / pfA0)
        pfA0 = pfA1
    carrier = merge_pf_delays(carrier)
    assert jnp.allclose(carrier.Ainv, jnp.linalg.inv(A))
