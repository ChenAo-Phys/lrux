import os

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import pytest
import random
import jax
import jax.numpy as jnp
import jax.random as jr
import lrux
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


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_direct_pf(dtype):
    A = jr.normal(_get_key(), (10, 4, 4), dtype)
    A = A - jnp.swapaxes(A, -1, -2)

    pfA1 = pf(A)  # direct method
    sign, log = jax.vmap(lrux.pfaffian._slogpf_householder)(A)  # householder method
    pfA2 = sign * jnp.exp(log)
    assert jnp.allclose(pfA1, pfA2)


def test_methods():
    A = jr.normal(_get_key(), (10, 10), dtype=jnp.float64)
    A = A - A.T
    pf_householder = pf(A, method="householder")
    pf_householder_for = pf(A, method="householder_for")
    pf_schur = pf(A, method="schur")
    assert jnp.allclose(pf_householder, pf_householder_for)
    assert jnp.allclose(pf_householder, pf_schur)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_grad(dtype):
    A = jr.normal(_get_key(), (4, 4), dtype)
    A = A - A.T

    def pf1(A):
        A = (A - A.T) / 2
        return lrux.pfaffian._pfaffian_direct(A)

    def pf2(A):
        signA, logA = lrux.pfaffian._slogpf_householder(A)
        return signA * jnp.exp(logA)

    v = jr.normal(_get_key(), A.shape, dtype)
    jvp1 = jax.jvp(pf1, (A,), (v,))
    jvp2 = jax.jvp(pf2, (A,), (v,))
    assert jnp.allclose(jvp1[0], jvp2[0])
    assert jnp.allclose(jvp1[1], jvp2[1])

    v = jr.normal(_get_key(), A.shape, dtype)
    jvp1 = jax.jvp(jnp.linalg.det, (A,), (v,))
    jvp2 = jax.jvp(lambda x: pf2(x) ** 2, (A,), (v,))
    assert jnp.allclose(jvp1[0], jvp2[0])
    assert jnp.allclose(jvp1[1], jvp2[1])
