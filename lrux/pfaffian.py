from typing import Tuple, NamedTuple
from jax import Array
import jax
import jax.numpy as jnp


def householder(x: jax.Array):
    x0 = x[0]
    sigma = jnp.vdot(x[1:], x[1:])
    norm_x = jnp.sqrt(x0.conj() * x0 + sigma)

    phase = jnp.where(x0 == 0.0, 1.0, jnp.sign(x0))
    alpha = -phase * norm_x

    v = x.at[0].subtract(alpha)
    v *= jax.lax.rsqrt(jnp.vdot(v, v))

    cond = sigma == 0.0
    v = jnp.where(cond, 0, v)
    tau = jnp.where(cond, 0, 2)
    alpha = jnp.where(cond, x0, alpha)

    return v, tau, alpha


def _single_pfaffian(A: Array) -> Array:
    n = A.shape[0]
    if n % 2 == 1:
        return jnp.array(0, dtype=A.dtype)

    if n == 2:
        return A[0, 1]

    if n == 4:
        a, b, c, d, e, f = A[jnp.triu_indices(n, 1)]
        return a * f - b * e + d * c

    pfaffian_mul = []
    for i in range(n - 2):
        v, tau, alpha = householder(A[1:, 0])
        A = A[1:, 1:]
        w = tau * A @ v.conj()
        vw = jnp.outer(v, w)
        A += vw - vw.T

        pfaffian_mul.append(1 - tau)
        if i % 2 == 0:
            pfaffian_mul.append(-alpha)

    pfaffian_mul.append(A[-2, -1])
    return jnp.prod(jnp.asarray(pfaffian_mul))


@jax.custom_vjp
def pf(A: Array) -> Array:
    """
    Return pfaffian of the input matrix A. A customized vjp is used for faster gradients.
    """
    batch = A.shape[:-2]

    # By convention, pfaffian of 0 particle is 1
    if A.size == 0:
        return jnp.ones(batch, A.dtype)

    A = A.reshape(-1, *A.shape[-2:])
    pfaffian = jax.vmap(_single_pfaffian)(A)
    pfaffian = pfaffian.reshape(batch)
    return pfaffian


def _pf_fwd(A: Array) -> Tuple[Array, Array]:
    pfaA = pf(A)
    Ainv = jnp.linalg.inv(A)
    Ainv = (Ainv - jnp.swapaxes(Ainv, -2, -1)) / 2
    return pfaA, pfaA[..., None, None] * Ainv


def _pf_bwd(res: Array, g: Array) -> Tuple[Array]:
    return (-res * g[..., None, None] / 2,)


pf.defvjp(_pf_fwd, _pf_bwd)


class SlogpfResult(NamedTuple):
    sign: Array
    logabspf: Array


@jax.custom_vjp
def slogpf(A: Array) -> Array:
    """
    Return the log of pfaffian. A customized vjp is used for faster gradients.
    """

    n = A.shape[0]
    if n % 2 == 1:
        return jnp.array(0, dtype=A.dtype)

    def body_fun(i, val):
        A, logpf_val = val
        v, tau, alpha = householder(A[:, i], i + 1)
        w = tau * A @ v.conj()
        A += jnp.outer(v, w) - jnp.outer(w, v)

        new_val = jnp.log((1 - tau) * jnp.where(i % 2 == 0, -alpha, 1.0))
        logpf_val = jnp.logaddexp(logpf_val, new_val)
        return A, logpf_val

    init_val = (A, jnp.array(0, dtype=A.dtype))
    A, pfaffian_val = jax.lax.fori_loop(0, A.shape[0] - 2, body_fun, init_val)
    pfaffian_val *= A[n - 2, n - 1]

    return pfaffian_val


def _slogpf_fwd(A: Array) -> Tuple[Array, Array]:
    logpfA = slogpf(A)
    Ainv = jnp.linalg.inv(A)
    return logpfA, Ainv


def _slogpf_bwd(res: Array, g: Array) -> Tuple[Array]:
    return (-g * res / 2,)


slogpf.defvjp(_slogpf_fwd, _slogpf_bwd)
