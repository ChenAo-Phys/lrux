from typing import Tuple, NamedTuple, Union
from jax import Array
from functools import partial
import jax
import jax.numpy as jnp


def _check_input(A: Array, method: str) -> None:
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

    return (A - jnp.swapaxes(A, -2, -1)) / 2


def _pfaffian_direct(A: Array) -> Array:
    n = A.shape[-1]
    batch = A.shape[:-2]

    if n % 2 == 1:
        return jnp.zeros(batch, dtype=A.dtype)

    # By convention, pfaffian of an empty matrix is 1
    elif n == 0:
        return jnp.ones(batch, dtype=A.dtype)

    elif n == 2:
        return A[..., 0, 1]

    elif n == 4:
        idx = jnp.triu_indices(n, 1)
        A_upper = A[..., idx[0], idx[1]]
        a, b, c, d, e, f = jnp.moveaxis(A_upper, -1, 0)
        return a * f - b * e + d * c


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


@jax.custom_jvp
def _slogpf_householder(A: Array) -> Tuple[Array, Array]:
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

    sign = jnp.prod(jnp.sign(vals))
    log = jnp.sum(jnp.log(jnp.abs(vals)))
    return sign, log


@jax.custom_jvp
def _slogpf_schur(A: Array) -> Tuple[Array, Array]:
    T, Z = jax.scipy.linalg.schur(A)
    vals = jnp.diag(T, k=1)[::2]
    s, _ = jnp.linalg.slogdet(Z)

    sign = s * jnp.prod(jnp.sign(vals))
    log = jnp.sum(jnp.log(jnp.abs(vals)))
    return sign, log


class SlogpfResult(NamedTuple):
    sign: Array
    logabspf: Array


@partial(jax.jit, static_argnames=("method",))
def slogpf(A: Array, *, method: str = "householder") -> SlogpfResult:
    """
    Return the log of pfaffian. A customized vjp is used for faster gradients.
    """
    A = _check_input(A, method)

    n = A.shape[-1]
    if n <= 4 or n % 2 == 1:
        pfA = _pfaffian_direct(A)
        return SlogpfResult(jnp.sign(pfA), jnp.log(jnp.abs(pfA)))
    else:
        batch = A.shape[:-2]
        A = A.reshape(-1, *A.shape[-2:])
        slogpf_fn = _slogpf_householder if method == "householder" else _slogpf_schur
        batched_fn = jax.vmap(slogpf_fn)
        outputs = batched_fn(A)
        return SlogpfResult(outputs[0].reshape(batch), outputs[1].reshape(batch))


@partial(jax.jit, static_argnames=("method",))
def pf(A: Array, *, method: str = "householder") -> Array:
    """
    Return pfaffian of the input matrix A. A customized vjp is used for faster gradients.
    """
    A = _check_input(A, method)

    n = A.shape[-1]
    if n <= 4 or n % 2 == 1:
        return _pfaffian_direct(A)
    else:
        sign, log = slogpf(A, method=method)
        return sign * jnp.exp(log)


def _slogpf_jvp(
    primals: Tuple[Array], tangents: Tuple[Array], method: str
) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
    (A,) = primals
    (dA,) = tangents
    A = (A - A.T) / 2

    slogpf_fn = _slogpf_householder if method == "householder" else _slogpf_schur
    sign, ans = slogpf_fn(A)
    ans_dot = jnp.trace(jnp.linalg.solve(A, dA)) / 2

    if jnp.issubdtype(A.dtype, jnp.complexfloating):
        sign_dot = 1j * sign * ans_dot.imag
        ans_dot = ans_dot.real
    else:
        sign_dot = jnp.zeros_like(sign)

    return (sign, ans), (sign_dot, ans_dot)


_slogpf_householder.defjvp(partial(_slogpf_jvp, method="householder"))
_slogpf_schur.defjvp(partial(_slogpf_jvp, method="schur"))
