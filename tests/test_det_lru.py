import os

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import pytest
import random
import jax
import jax.numpy as jnp
import jax.random as jr
from lrux import det_lru, det_lru_delayed, init_det_carrier


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
@pytest.mark.parametrize("kxu, keu, kxv, kev", [(3, 0, 0, 3), (2, 4, 4, 2)])
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

    for i in range(20):
        current_delay = i % max_delay
        k = random.randint(0, max_rank)
        u = jr.normal(_get_key(), (n, k), dtype)
        v = jr.normal(_get_key(), (n, k), dtype)
        ratio, carrier = lru_fn(carrier, u, v, True, current_delay)
        A += v @ u.T
        detA1 = jnp.linalg.det(A)
        assert jnp.allclose(ratio, detA1 / detA0)
        detA0 = detA1
