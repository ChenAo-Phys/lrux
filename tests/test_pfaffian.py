import os

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import pytest
import random
import jax.numpy as jnp
import jax.random as jr
from lrux import pf, slogpf


def _get_key():
    seed = random.randint(0, 2**31 - 1)
    return jr.key(seed)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_pf(dtype):
    A = jr.normal(_get_key(), (10, 10), dtype)
    A = A - A.T
    pfA = pf(A)
    detA = jnp.linalg.det(A)
    assert jnp.allclose(pfA**2, detA)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_slogpf(dtype):
    A = jr.normal(_get_key(), (10, 10), dtype)
    A = A - A.T
    slogpfA = slogpf(A)
    slogdetA = jnp.linalg.slogdet(A)
    assert jnp.allclose(slogpfA.sign**2, slogdetA.sign)
    assert jnp.allclose(slogpfA.logabspf * 2, slogdetA.logabsdet)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_low_rank(dtype):
    A = jr.normal(_get_key(), (5, 5), dtype)
    A = A - A.T
    B = jr.normal(_get_key(), (10, 5), dtype)
    M = B @ A @ B.T
    pfM = pf(M)
    assert jnp.allclose(pfM, 0)


def test_schur():
    A = jr.normal(_get_key(), (10, 10), dtype=jnp.float64)
    A = A - A.T
    pf_householder = pf(A, method="householder")
    pf_schur = pf(A, method="schur")
    assert jnp.allclose(pf_householder, pf_schur)
