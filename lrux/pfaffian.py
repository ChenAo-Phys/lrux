from typing import Tuple, NamedTuple, Optional, Callable
from jax import Array
from functools import partial
import jax
import jax.numpy as jnp


def skew_eye(n: int, dtype: Optional[jnp.dtype] = None) -> Array:
    r"""
    The skew-symmetric identity matrix

    :param n:
        Number of rows in the output divided by 2.

    :param dtype:
        Optional dtype, default to floating point.

    :return:
        The skew-symmetric identity matrix of shape (2n, 2n), defined as
        
        .. math::

            J = \begin{pmatrix}
                0 & I \\ -I & 0
            \end{pmatrix}
    """
    I = jnp.eye(n, dtype=dtype)
    O = jnp.zeros((n, n), dtype=dtype)
    return jnp.block([[O, I], [-I, O]])


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


def _householder(x: Array, n: Optional[int] = None) -> Tuple[Array, Array, Array]:
    if n is None:
        n = 0
        x0 = x[0]
        x = x.at[0].set(0)
    else:
        x0 = x[n]
        x = jnp.where(jnp.arange(x.size) <= n, 0, x)

    sigma = jnp.vdot(x, x)
    norm_x = jnp.sqrt(x0.conj() * x0 + sigma)

    phase = jnp.where(x0 == 0.0, 1.0, jnp.sign(x0))
    alpha = -phase * norm_x

    v = x.at[n].set(x0 - alpha)
    v *= jax.lax.rsqrt(jnp.vdot(v, v))

    cond = sigma == 0.0
    v = jnp.where(cond, 0, v)
    tau = jnp.where(cond, 0, 2)
    alpha = jnp.where(cond, x0, alpha)

    return v, tau, alpha


@jax.custom_jvp
def _slogpf_householder(A: Array) -> Array:
    n = A.shape[0]

    def body_fun(i, val):
        A, sign, log = val
        v, tau, alpha = _householder(A[:, i], i + 1)
        w = tau * A @ v.conj()
        vw = jnp.outer(v, w)
        A += vw - vw.T

        new_val = (1 - tau) * jnp.where(i % 2 == 0, -alpha, 1.0)
        sign *= jnp.sign(new_val)
        log += jnp.log(jnp.abs(new_val))
        return A, sign, log

    sign = jnp.array(1, dtype=A.dtype)
    log = jnp.array(0, dtype=jnp.finfo(A.dtype).dtype)
    init_val = (A, sign, log)
    A, sign, log = jax.lax.fori_loop(0, n - 2, body_fun, init_val)

    sign *= jnp.sign(A[n - 2, n - 1])
    log += jnp.log(jnp.abs(A[n - 2, n - 1]))
    return sign, log


@jax.custom_jvp
def _slogpf_householder_for(A: Array) -> Tuple[Array, Array]:
    vals = []
    for i in range(A.shape[0] - 2):
        v, tau, alpha = _householder(A[1:, 0])
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


def _check_input(
    A: Array, method: str
) -> Tuple[Array, Callable[[Array], Tuple[Array, Array]]]:
    if A.ndim < 2 or A.shape[-2] != A.shape[-1]:
        raise ValueError(
            f"The expected input is a square matrix or a batch of them, got input shape {A.shape}."
        )
    A = (A - jnp.swapaxes(A, -2, -1)) / 2

    if method == "householder":
        slogpf_fn = _slogpf_householder
    elif method == "householder_for":
        slogpf_fn = _slogpf_householder_for
    elif method == "schur":
        if jnp.issubdtype(A, jnp.complexfloating):
            raise ValueError("The schur method is only available for real dtypes.")
        slogpf_fn = _slogpf_schur
    else:
        raise ValueError(
            f"Unknown pfaffian method '{method}'. "
            "Supported methods include 'householder', 'householder_for' and 'schur'."
        )

    return A, slogpf_fn


class SlogpfResult(NamedTuple):
    sign: Array
    logabspf: Array


@partial(jax.jit, static_argnames=("method",))
def slogpf(A: Array, *, method: str = "householder") -> SlogpfResult:
    """
    Compute the sign and (natural) logarithm of the pfaffian of an array.

    :param A:
        An array with shape (..., n, n)

    :param method:
        The method used to compute the pfaffian. Options include

        ``"householder"``:
            Houserholder transformation, internally using ``jax.lax.fori_loop``
            to balance between running and compiling time;

        ``"householder_for"``:
            Houserholder transformation, internally using jitted python for loops
            to reduce the running time at a cost of much longer compiling time;

        ``"schur""``:
            Schur decomposition using 
            `jax.scipy.linalg.schur <https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.schur.html>`_,
            but only available on CPU and real dtypes.

        The default method is ``"householder"``.

    :return:
        A ``NamedTuple`` with the following attributes. Both attribtutes have the same
        batch dimension as the input A.

        ``sign``:
            sign(A). For a real input, it's 1, 0, or -1. For a complex input, it's a
            complex number with absolute value 1, or else 0.
        
        ``logabspf``:
            The natural log of the absolute value of the pfaffian.
    
    .. tip::

        The input A is always skew-symmetrized.

    .. tip::

        This function has ``jax.custom_jvp`` defined and is backward compatible for both real and complex dtypes.
    """
    A, slogpf_fn = _check_input(A, method)

    n = A.shape[-1]
    if n <= 4 or n % 2 == 1:
        pfA = _pfaffian_direct(A)
        return SlogpfResult(jnp.sign(pfA), jnp.log(jnp.abs(pfA)))
    else:
        batch = A.shape[:-2]
        A = A.reshape(-1, n, n)

        outputs = jax.vmap(slogpf_fn)(A)
        return SlogpfResult(outputs[0].reshape(batch), outputs[1].reshape(batch))


@partial(jax.jit, static_argnames=("method",))
def pf(A: Array, *, method: str = "householder") -> Array:
    """
    Compute the pfaffian of an array

    :param A:
        An array with shape (..., n, n)

    :param method:
        The method used to compute the pfaffian. Options include

        ``"householder"``:
            Houserholder transformation, internally using ``jax.lax.fori_loop``
            to balance between running and compiling time;

        ``"householder_for"``:
            Houserholder transformation, internally using jitted python for loops
            to reduce the running time at a cost of much longer compiling time;

        ``"schur""``:
            Schur decomposition using 
            `jax.scipy.linalg.schur <https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.schur.html>`_,
            but only available on CPU and real dtypes.

        The default method is ``"householder"``.

    :return:
        The pfaffian of A with the same batch dimensions.
    
    .. tip::

        The input A is always be skew-symmetrized.

    .. tip::

        This function has ``jax.custom_jvp`` defined and is backward compatible for both real and complex dtypes.
    """
    A, slogpf_fn = _check_input(A, method)

    n = A.shape[-1]
    if n <= 4 or n % 2 == 1:
        return _pfaffian_direct(A)
    else:
        batch = A.shape[:-2]
        A = A.reshape(-1, n, n)

        sign, log = jax.vmap(slogpf_fn)(A)
        pfA = sign * jnp.exp(log)
        return pfA.reshape(batch)


def _slogpf_jvp(
    primals: Tuple[Array], tangents: Tuple[Array], method: str
) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
    (A,) = primals
    (dA,) = tangents
    A, slogpf_fn = _check_input(A, method)

    sign, ans = slogpf_fn(A)
    ans_dot = jnp.trace(jnp.linalg.solve(A, dA)) / 2

    if jnp.issubdtype(A.dtype, jnp.complexfloating):
        sign_dot = 1j * sign * ans_dot.imag
        ans_dot = ans_dot.real
    else:
        sign_dot = jnp.zeros_like(sign)

    return (sign, ans), (sign_dot, ans_dot)


_slogpf_householder.defjvp(partial(_slogpf_jvp, method="householder"))
_slogpf_householder_for.defjvp(partial(_slogpf_jvp, method="householder_for"))
_slogpf_schur.defjvp(partial(_slogpf_jvp, method="schur"))
