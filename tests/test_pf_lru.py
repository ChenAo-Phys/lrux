import os

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import pytest
import random
import jax
import jax.numpy as jnp
import jax.random as jr
from lrux import pf, skew_eye, pf_lru, init_pf_carrier, pf_lru_delayed


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
@pytest.mark.parametrize("kx, ke", [(3, 3), (2, 4)])
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


@pytest.mark.parametrize("k", [2, 4])
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

    for i in range(20):
        current_delay = i % max_delay
        ki = random.randint(0, k // 2) * 2  # ensure ki is even
        u = jr.normal(_get_key(), (n, ki), dtype)

        ratio, carrier = lru_fn(carrier, u, True, current_delay)
        J = skew_eye(ki // 2, dtype)
        A -= u @ J @ u.T
        pfA1 = pf(A)
        assert jnp.allclose(ratio, pfA1 / pfA0)
        pfA0 = pfA1
