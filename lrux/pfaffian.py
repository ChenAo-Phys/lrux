from typing import Tuple, NamedTuple, Union
from jax import Array
from functools import partial
import jax
import jax.numpy as jnp


class SlogpfResult(NamedTuple):
    sign: Array
    logabspf: Array


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


def _pfaffian_direct(A: Array, return_log: bool) -> Union[Array, Tuple[Array, Array]]:
    n = A.shape[0]

    if n % 2 == 1:
        val = jnp.array(0, dtype=A.dtype)

    # By convention, pfaffian of an empty matrix is 1
    elif n == 0:
        val = jnp.array(1, dtype=A.dtype)

    elif n == 2:
        val = A[0, 1]

    elif n == 4:
        a, b, c, d, e, f = A[jnp.triu_indices(n, 1)]
        val = a * f - b * e + d * c

    if return_log:
        return jnp.sign(val), jnp.log(jnp.abs(val))
    else:
        return val


def _pfaffian_schur(A: Array, return_log: bool) -> Union[Array, Tuple[Array, Array]]:
    T, Z = jax.scipy.linalg.schur(A)
    vals = jnp.diag(T, k=1)[::2]
    s, _ = jnp.linalg.slogdet(Z)

    if return_log:
        sign = s * jnp.prod(jnp.sign(vals))
        log = jnp.sum(jnp.log(jnp.abs(vals)))
        return sign, log
    else:
        return s * jnp.prod(vals)


def _pfaffian_householder(
    A: Array, return_log: bool
) -> Union[Array, Tuple[Array, Array]]:
    vals = []
    for i in range(A.shape[0] - 2):
        v, tau, alpha = householder(A[1:, 0])
        A = A[1:, 1:]
        w = tau * A @ v.conj()
        vw = jnp.outer(v, w)
        A += vw - vw.T

        vals.append(1 - tau)
        if i % 2 == 0:
            vals.append(-alpha)
            
    vals.append(A[-2, -1])
    vals = jnp.asarray(vals)

    if return_log:
        sign = jnp.prod(jnp.sign(vals))
        log = jnp.sum(jnp.log(jnp.abs(vals)))
        return sign, log
    else:
        return jnp.prod(vals)


def _single_pfaffian(
    A: Array, return_log: bool, method: str
) -> Union[Array, Tuple[Array, Array]]:
    n = A.shape[0]

    if n <= 4 or n % 2 == 1:
        return _pfaffian_direct(A, return_log)
    elif method == "schur":
        return _pfaffian_schur(A, return_log)
    else:
        return _pfaffian_householder(A, return_log)


def _batched_pfaffian(
    A: Array, return_log: bool, method: str
) -> Union[Array, SlogpfResult]:
    if A.ndim < 2 or A.shape[-2] != A.shape[-1]:
        raise ValueError(
            f"The expected input is a square matrix or a batch of them, got input shape {A.shape}."
        )

    if method not in ("householder", "schur"):
        raise ValueError(
            f"Unknown pfaffian method '{method}'. Supported methods include 'householder' and 'schur'"
        )
    
    if method == "schur" and jnp.issubdtype(A, jnp.complexfloating):
        raise ValueError("The schur method is only available for real dtypes.")

    batch = A.shape[:-2]
    A = A.reshape(-1, *A.shape[-2:])
    batched_fn = jax.vmap(_single_pfaffian, in_axes=(0, None, None))
    outputs = batched_fn(A, return_log, method)

    if return_log:
        outputs = SlogpfResult(outputs[0].reshape(batch), outputs[1].reshape(batch))
    else:
        outputs = outputs.reshape(batch)
    return outputs


@partial(jax.jit, static_argnames=('method',))
def pf(A: Array, *, method: str = "householder") -> Array:
    """
    Return pfaffian of the input matrix A. A customized vjp is used for faster gradients.
    """
    return _batched_pfaffian(A, return_log=False, method=method)


@partial(jax.jit, static_argnames=('method',))
def slogpf(A: Array, *, method: str = "householder") -> SlogpfResult:
    """
    Return the log of pfaffian. A customized vjp is used for faster gradients.
    """
    return _batched_pfaffian(A, return_log=True, method=method)
