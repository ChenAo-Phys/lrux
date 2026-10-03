import os

os.environ["JAX_ENABLE_X64"] = "1"
# CPU unless the caller picks a platform: JAX_PLATFORMS=cuda pytest runs the GPU paths
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import random
import jax
import jax.numpy as jnp
import jax.random as jr
from lrux import det_lru, det_lru_delayed, merge_det_delays, init_det_carrier


def _get_key():
    seed = random.randint(0, 2**31 - 1)
    return jr.key(seed)


det_lru = jax.jit(det_lru, static_argnums=3)


@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_rank_1(n, dtype):
    A = jr.normal(_get_key(), (n, n), dtype)
    Ainv = jnp.linalg.inv(A)
    detA = jnp.linalg.det(A)

    # general update
    u = jr.normal(_get_key(), (n,), dtype)
    v = jr.normal(_get_key(), (n,), dtype)
    A_ = A + jnp.outer(v, u)
    ratio = det_lru(Ainv, u, v)
    assert jnp.allclose(ratio, jnp.linalg.det(A_) / detA)
    ratio, new_inv = det_lru(Ainv, u, v, return_update=True)
    assert jnp.allclose(ratio, jnp.linalg.det(A_) / detA)
    assert jnp.allclose(new_inv, jnp.linalg.inv(A_))

    # row update
    u = jr.normal(_get_key(), (n,), dtype)
    v = random.randint(0, n - 1)
    A_ = A.at[v].add(u)
    ratio, new_inv = det_lru(Ainv, u, v, return_update=True)
    assert jnp.allclose(ratio, jnp.linalg.det(A_) / detA)
    assert jnp.allclose(new_inv, jnp.linalg.inv(A_))

    # column update
    v = jr.normal(_get_key(), (n,), dtype)
    u = random.randint(0, n - 1)
    A_ = A.at[:, u].add(v)
    ratio, new_inv = det_lru(Ainv, u, v, return_update=True)
    assert jnp.allclose(ratio, jnp.linalg.det(A_) / detA)
    assert jnp.allclose(new_inv, jnp.linalg.inv(A_))


@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.parametrize(
    "kxu, keu, kxv, kev", [(3, 0, 0, 3), (2, 4, 4, 2), (20, 14, 14, 20)]
)
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_rank_k(n, kxu, keu, kxv, kev, dtype):
    A = jr.normal(_get_key(), (n, n), dtype)
    Ainv = jnp.linalg.inv(A)
    detA = jnp.linalg.det(A)

    xu = jr.normal(_get_key(), (n, kxu), dtype)
    eu = jr.randint(_get_key(), keu, 0, n)
    xv = jr.normal(_get_key(), (n, kxv), dtype)
    ev = jr.randint(_get_key(), kev, 0, n)

    eu_arr = jnp.zeros((n, keu), dtype).at[eu, jnp.arange(keu)].set(1)
    u_full = jnp.concatenate((xu, eu_arr), axis=1)
    ev_arr = jnp.zeros((n, kev), dtype).at[ev, jnp.arange(kev)].set(1)
    v_full = jnp.concatenate((ev_arr, xv), axis=1)
    A_ = A + v_full @ u_full.T

    ratio, new_inv = det_lru(Ainv, (xu, eu), (xv, ev), return_update=True)
    assert jnp.allclose(ratio, jnp.linalg.det(A_) / detA)
    assert jnp.allclose(new_inv, jnp.linalg.inv(A_))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_vmap(dtype):
    batch = 2
    n = 10
    k = 5
    A = jr.normal(_get_key(), (batch, n, n), dtype)
    Ainv = jnp.linalg.inv(A)
    detA = jnp.linalg.det(A)

    u = jr.normal(_get_key(), (batch, n, k), dtype)
    v = jr.normal(_get_key(), (batch, n, k), dtype)
    A_ = A + jnp.einsum("bnk,bmk->bnm", v, u)
    vmap_lru = jax.vmap(det_lru, in_axes=(0, 0, 0, None))
    ratio, new_inv = vmap_lru(Ainv, u, v, True)
    assert jnp.allclose(ratio, jnp.linalg.det(A_) / detA)
    assert jnp.allclose(new_inv, jnp.linalg.inv(A_))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_single_delay(dtype):
    n = 10
    A = jr.normal(_get_key(), (n, n), dtype)
    carrier = init_det_carrier(A, max_delay=n // 2)
    u = jr.normal(_get_key(), (n,), dtype)
    v = jr.normal(_get_key(), (n,), dtype)
    A_ = A + jnp.outer(v, u)

    ratio = det_lru_delayed(carrier, u, v)
    assert jnp.allclose(ratio, jnp.linalg.det(A_) / jnp.linalg.det(A))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_multiple_delayed(dtype):
    n = 10
    max_delay = n // 2
    max_rank = 2
    A = jr.normal(_get_key(), (n, n), dtype)
    carrier = init_det_carrier(A, max_delay, max_rank)
    detA0 = jnp.linalg.det(A)

    lru_fn = jax.jit(det_lru_delayed, static_argnums=(3, 4), donate_argnums=0)
    merge_fn = jax.jit(merge_det_delays, donate_argnums=0)

    for i in range(20):
        current_delay = i % max_delay
        k = random.randint(0, max_rank)
        u = jr.normal(_get_key(), (n, k), dtype)
        v = jr.normal(_get_key(), (n, k), dtype)
        ratio, carrier = lru_fn(carrier, u, v, True, current_delay)

        if current_delay == max_delay - 1:
            carrier = merge_fn(carrier)

        A += v @ u.T
        detA1 = jnp.linalg.det(A)
        assert jnp.allclose(ratio, detA1 / detA0)
        detA0 = detA1


@pytest.mark.parametrize("k", [1, 6])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_grad(k, dtype):
    n = 10
    A = jr.normal(_get_key(), (n, n), dtype)
    Ainv = jnp.linalg.inv(A)
    detA = jnp.linalg.det(A)
    u = jr.normal(_get_key(), (n, k), dtype)
    v = jr.normal(_get_key(), (n, k), dtype)
    du = jr.normal(_get_key(), (n, k), dtype)

    def ratio_lru(u):
        return det_lru(Ainv, u, v)

    def ratio_ref(u):
        return jnp.linalg.det(A + v @ u.T) / detA

    def inv_lru(u):
        return det_lru(Ainv, u, v, True)[1]

    def inv_ref(u):
        return jnp.linalg.inv(A + v @ u.T)

    for f_lru, f_ref in [(ratio_lru, ratio_ref), (inv_lru, inv_ref)]:
        out_lru, jvp_lru = jax.jvp(f_lru, (u,), (du,))
        out_ref, jvp_ref = jax.jvp(f_ref, (u,), (du,))
        assert jnp.allclose(out_lru, out_ref)
        assert jnp.allclose(jvp_lru, jvp_ref)

    # the ratio is a polynomial in u, so a central difference is an independent check
    eps = 1e-6
    fd = (ratio_ref(u + eps * du) - ratio_ref(u - eps * du)) / (2 * eps)
    assert jnp.allclose(jax.jvp(ratio_lru, (u,), (du,))[1], fd, atol=1e-6)


@pytest.mark.parametrize("k", [5, 8])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_delayed_large_rank(k, dtype):
    """Delayed updates of rank k > 4, where the ratio and R^-1 come from det_inv's
    kernel (GPU) or LU path: ratios against the direct determinants and the merged
    inverse against the direct inverse."""
    n = 16
    max_delay = 3
    A = jr.normal(_get_key(), (n, n), dtype) + 4 * jnp.eye(n, dtype=dtype)
    carrier = init_det_carrier(A, max_delay, k)
    detA0 = jnp.linalg.det(A)
    for i in range(max_delay):
        u = 0.3 * jr.normal(_get_key(), (n, k), dtype)
        v = 0.3 * jr.normal(_get_key(), (n, k), dtype)
        ratio, carrier = det_lru_delayed(carrier, u, v, True, i)
        A = A + v @ u.T
        detA1 = jnp.linalg.det(A)
        assert jnp.allclose(ratio, detA1 / detA0)
        detA0 = detA1
    carrier = merge_det_delays(carrier)
    assert jnp.allclose(carrier.Ainv, jnp.linalg.inv(A))
