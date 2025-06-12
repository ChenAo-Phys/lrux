from typing import Optional, Tuple, Union, NamedTuple
from jax.typing import ArrayLike
from jax import Array
import jax
import jax.numpy as jnp
from .det_lru import _standardize_uv, _update_ab
from .pfaffian import skew_eye, pf


def _check_mat(mat: Array) -> Array:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] % 2 == 1:
        raise ValueError(f"Expect input matrix shape (2n, 2n), got {mat.shape}.")
    return (mat - mat.T) / 2


def _get_R(Ainv: Array, u: Tuple[Array, Array]) -> Array:
    xu_Ainv_xu = jnp.einsum("nk,nm,ml->kl", u[0], Ainv, u[0])
    xu_Ainv_eu = u[0].T @ Ainv[:, u[1]]
    eu_Ainv_eu = Ainv[u[1], :][:, u[1]]
    uT_Ainv_u = jnp.block([[xu_Ainv_xu, xu_Ainv_eu], [-xu_Ainv_eu.T, eu_Ainv_eu]])
    J = skew_eye(uT_Ainv_u.shape[0] // 2, Ainv.dtype)
    R = uT_Ainv_u + J
    return (R - R.T) / 2  # ensure skew-symmetric


def _update_Ainv(Ainv: Array, u: Tuple[Array, Array], R: Array) -> Array:
    Ainv_u = jnp.concatenate((Ainv @ u[0], Ainv[:, u[1]]), axis=1)
    if R.shape[0] == 2:
        Ainv_u1, Ainv_u2 = Ainv_u.T
        outer = jnp.outer(Ainv_u1, Ainv_u2)
        Ainv -= (outer - outer.T) / R[0, 1]
    else:
        Rinv_Ainv_u = jax.scipy.linalg.solve(R, Ainv_u.T)
        Ainv += Ainv_u @ Rinv_Ainv_u
    return (Ainv - Ainv.T) / 2


def pf_lru(
    Ainv: Array,
    u: Union[ArrayLike, Tuple[Array, ArrayLike]],
    return_update: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    r"""
    Low-rank update of pfaffian :math:`\mathrm{pf}(A_1) = \mathrm{pf}(A_0 - u J u^T)`.
    Here :math:`J` is the skew-symmetric identity matrix
        
    .. math::

        J = \begin{pmatrix}
            0 & I \\ -I & 0
        \end{pmatrix}

    as given in `~lrux.skew_eye`.

    :param Ainv:
        Inverse of the original skew-symmetric matrix :math:`A_0^{-1}`, shape (n, n)

    :param u:
        Low-rank update vector(s) :math:`u`, the same as :math:`u` in `lrux.det_lru`.

    :param return_update:
        Whether the new matrix inverse :math:`A_1^{-1}` should be returned,
        defaul to False.

    :return:
        ratio:
            The ratio between two pfaffians

            .. math::

                r = \frac{\mathrm{pf}(A_1)}{\mathrm{pf}(A_0)} = \frac{\mathrm{pf}(R)}{\mathrm{pf}(J)}

            where

            .. math::

                R = J + u^T A_0^{-1} u

        new_Ainv:
            The new matrix inverse

            .. math::

                A_1^{-1} = (A_0 + u J u^T)^{-1} = A_0^{-1} + (A_0^{-1} u) R^{-1} (A_0^{-1} u)^T

            Only returned when ``return_update`` is True.

    .. tip::

        This function is compatible with ``jax.jit`` and ``jax.vmap``, while
        ``return_update`` is a static argument which shouldn't be jitted or vmapped.

        Furthermore, we recommend setting ``donate_argnums=0`` in ``jax.jit`` to reuse 
        the memory of ``Ainv`` if it's no longer needed. This helps to greatly reduce 
        the time and memory cost. For instance,

        .. code-block:: python

            lru_vmap = jax.vmap(pf_lru, in_axes=(0, 0, None))
            lru_jit = jax.jit(lru_vmap, static_argnums=2, donate_argnums=0)

    .. note::

        Here are examples of how to define ``u`` before calling ``pf_lru(Ainv, u)``.
        Keep in mind that the low-rank update we need should be skew-symmetric and takes
        the form

        .. math::

            A_1 - A_0 = -u J u^T

        **Update of 1 row and 1 column**

        .. math::

            A_1 - A_0 = \begin{pmatrix}
                0 & -u_0 & 0 & 0 \\ 
                u_0 & 0 & u_2 & u_3 \\
                0 & -u_2 & 0 & 0 \\ 
                0 & -u_3 & 0 & 0 \\ 
            \end{pmatrix}
            = -\begin{pmatrix}
                u_0 & 0 \\ u_1 & 1 \\ u_2 & 0 \\ u_3 & 0
            \end{pmatrix}
            \begin{pmatrix}
                0 & 1 \\ -1 & 0
            \end{pmatrix}
            \begin{pmatrix}
                u_0 & u_1 & u_2 & u_3 \\
                0 & 1 & 0 & 0\\
            \end{pmatrix}
            
        .. code-block:: python
        
            u = (jnp.array([u0, u1, u2, u3]), 1)

        **Update of 2 rows and 2 columns**

        .. math::

            \begin{split}
                A_1 - A_0 &= \begin{pmatrix}
                    0 & -u_{00} & 0 & -u_{10} \\ 
                    u_{00} & 0 & u_{02} & u_{03} - u_{11} \\
                    0 & -u_{02} & 0 & -u_{12} \\ 
                    u_{10} & u_{11} - u_{03} & u_{12} & 0 \\
                \end{pmatrix} \\
                &= -\begin{pmatrix}
                    u_{00} & u_{10} & 0 & 0 \\
                    u_{01} & u_{11} & 1 & 0 \\
                    u_{02} & u_{12} & 0 & 0 \\
                    u_{03} & u_{13} & 0 & 1 \\
                \end{pmatrix}
                \begin{pmatrix}
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 1 \\
                    -1 & 0 & 0 & 0 \\
                    0 & -1 & 0 & 0 \\
                \end{pmatrix}
                \begin{pmatrix}
                    u_{00} & u_{01} & u_{02} & u_{03} \\
                    u_{10} & u_{11} & u_{12} & u_{13} \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1 \\
                \end{pmatrix}
            \end{split}
            
        .. code-block:: python
        
            x = jnp.array([[u00, u10], [u01, u11], [u02, u12], [u03, u13]])
            e = jnp.array([1, 3])
            u = (x, e)
    """
    Ainv = _check_mat(Ainv)
    u = _standardize_uv(u, Ainv.shape[0], Ainv.dtype)
    k = u[0].shape[1] + u[1].size
    if k % 2 == 1:
        raise ValueError(f"The input u should have even rank, got rank {k}.")

    R = _get_R(Ainv, u)
    pfR = pf(R)
    k_half = k // 2
    ratio = jnp.where((k_half * (k_half - 1) // 2) % 2 == 0, pfR, -pfR)
    if return_update:
        Ainv = _update_Ainv(Ainv, u, R)
        return ratio, Ainv
    else:
        return ratio


class PfCarrier(NamedTuple):
    Ainv: Array
    a: Array
    Rinv: Array


def init_pf_carrier(A: Array, max_delay: int, max_rank: int = 2) -> PfCarrier:
    r"""
    Prepare the data and space for `~lrux.pf_lru_delayed`

    :param A:
        The initial skew-symmetric matrix :math:`A_0` with shape (n, n).

    :param max_delay:
        The maximum iterations T of delayed updates, usually chosen to be ~n/10.

    :param max_rank:
        The maximum rank K in delayed updates, default to 2.

    :return:
        A ``NamedTuple`` with the following attributes.

        Ainv:
            The initial matrix inverse :math:`A_0^{-1}` of shape (n, n).
        a:
            The delayed update vectors of shape (T, n, K), initialized to 0
        Rinv:
            The delayed update matrices :math:`R_t^{-1}` of shape (T, K, K), initialized to 0
    """
    if max_delay <= 0:
        raise ValueError(
            "`max_delay` should be a positive integer. "
            "Otherwise, please use `pf_lru` for non-delayed updates."
        )
    A = _check_mat(A)
    Ainv = jnp.linalg.inv(A)
    Ainv = (Ainv - Ainv.T) / 2  # ensure skew-symmetric
    a = jnp.zeros((max_delay, A.shape[0], max_rank), A.dtype)
    Rinv = jnp.zeros((max_delay, max_rank, max_rank), A.dtype)
    return PfCarrier(Ainv, a, Rinv)


def _get_delayed_updates(Ainv: Array, a: Array, Rinv: Array) -> Array:
    if Rinv.shape[-1] == 2:
        a1 = a[:, :, 0]
        a2 = a[:, :, 1]
        outer = jnp.einsum("tn,t,tm->nm", a1, Rinv[:, 0, 1], a2)
        update = outer - outer.T
    else:
        update = jnp.einsum("tnj,tjk,tmk->nm", a, Rinv, a)

    Ainv += update
    return (Ainv - Ainv.T) / 2  # ensure skew-symmetric


def _get_delayed_output(
    carrier: PfCarrier, u: Tuple[Array, Array], return_update: bool, current_delay: int
) -> Union[Array, Tuple[Array, Array]]:
    Ainv = carrier.Ainv
    a = carrier.a[:current_delay]
    Rinv = carrier.Rinv[:current_delay]
    R0 = _get_R(Ainv, u)

    xT_a = jnp.einsum("nk,tnl->tkl", u[0], a)
    eT_a = a[:, u[1], :]
    uT_a = jnp.concatenate((xT_a, eT_a), axis=1)

    R = R0 + jnp.einsum("tjk,tkl,tml->jm", uT_a, Rinv, uT_a)
    pfR = pf(R)
    k = u[0].shape[1] + u[1].size
    if k % 2 == 1:
        raise ValueError(f"The input u should have even rank, got rank {k}.")
    k_half = k // 2
    ratio = jnp.where((k_half * (k_half - 1) // 2) % 2 == 0, pfR, -pfR)

    if return_update:
        a0 = jnp.concatenate((Ainv @ u[0], Ainv[:, u[1]]), axis=1)
        new_a = a0 + jnp.einsum("tnj,tjk,tlk->nl", a, Rinv, uT_a)
        a = _update_ab(carrier.a, new_a, current_delay)

        if k == 2:
            rinv = -1 / ratio
            new_Rinv = jnp.array([[0, rinv], [-rinv, 0]], dtype=Rinv.dtype)
        else:
            new_Rinv = jnp.linalg.inv(R)
            new_Rinv = (new_Rinv - new_Rinv.T) / 2  # ensure skew-symmetric
        Rinv = carrier.Rinv.at[current_delay, :k, :k].set(new_Rinv)

        if current_delay == a.shape[0] - 1:
            Ainv = _get_delayed_updates(Ainv, a, Rinv)
            carrier = PfCarrier(Ainv, jnp.zeros_like(a), jnp.zeros_like(Rinv))
        else:
            carrier = PfCarrier(Ainv, a, Rinv)
        return ratio, carrier
    else:
        return ratio


def pf_lru_delayed(
    carrier: PfCarrier,
    u: Union[ArrayLike, Tuple[Array, ArrayLike]],
    return_update: bool = False,
    current_delay: Optional[int] = None,
) -> Union[Array, Tuple[Array, PfCarrier]]:
    r"""
    Delayed low-rank update of pfaffian

    :param carrier:
        The existing delayed update quantities, including :math:`A_0^{-1}`, :math:`R_t^{-1}`, and

        .. math::

            a_t = A_{t-1}^{-1} u_t

        with :math:`t` from 1 to :math:`\tau-1`.
        Initially provided by `~lrux.init_pf_carrier`.

    :param u:
        Low-rank update vector(s) :math:`u_\tau`, the same as :math:`u` in `lrux.det_lru`.
        The rank of u shouldn't exceed the maximum allowed rank specified
        in `~lrux.init_pf_carrier`.

    :param return_update:
        Whether the new carrier with updated quantities should be returned,
        defaul to False.

    :param current_delay:
        The current iterations :math:`\tau` of delayed updates. As python starts counting
        from 0, the actual :math:`\tau` should be ``current_delay + 1``.
        It must be specified when ``return_update`` is True.

    :return:
        ratio:
            The ratio between two pfaffians

            .. math::

                r_\tau = \frac{\mathrm{pf}(A_\tau)}{\mathrm{pf}(A_{\tau-1})} = \frac{\mathrm{pf}(R_\tau)}{\mathrm{pf}(J)}

            where

            .. math::

                R_\tau = J + u_\tau^T A_0^{-1} u_\tau + \sum_{t=1}^{\tau-1} (u_\tau^T a_t) (a_t^T u_\tau)

        new_carrier:
            Only returned when ``return_update`` is True. The new carrier contains
            the quantities from the input carrier, and in addition :math:`R_\tau` and

            .. math::

                a_\tau = A_{\tau-1}^{-1} u_\tau = A_0^{-1} u_\tau + \sum_{t=1}^{\tau-1} a_t R_t^{-1} (a_t^T u_\tau)

            When :math:`\tau` reaches the maximum delayed iterations :math:`T`
            specified in `~lrux.init_pf_carrier`, i.e. ``current_delay == max_delay - 1``,
            the current :math:`A_\tau` will be set as the new :math:`A_0`,
            whose inverse is given by

            .. math::

                A_\tau^{-1} = A_0^{-1} + \sum_{t=1}^\tau a_t R_t^{-1} a_t^T

            The ``Ainv`` in ``new_carrier`` will be replaced by :math:`A_\tau^{-1}`, and
            ``a`` and ``Rinv`` will be set to 0.

    .. warning::

        This function is only recommended for heavy users who understand why and when 
        to use delayed updates. Otherwise, please choose `~\lrux.pf_lru`.

    .. tip::

        Similar to `~lrux.det_lru_delayed` and `~lrux.pf_lru`, this function is compatible
        with ``jax.jit`` and ``jax.vmap``, while ``return_update`` and ``current_delay``
        are static arguments which shouldn't be jitted or vmapped.

        We still recommend setting ``donate_argnums=0`` in ``jax.jit`` to reuse
        the memory of ``carrier`` if it's no longer needed. For instance,

        .. code-block:: python

            lru_vmap = jax.vmap(pf_lru_delayed, in_axes=(0, 0, None, None))
            lru_jit = jax.jit(lru_vmap, static_argnums=(2, 3), donate_argnums=0)

    Here is a complete example of delayed updates.

    .. code-block:: python

        import os
        os.environ["JAX_ENABLE_X64"] = "1"

        import random
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        from lrux import skew_eye, pf, init_pf_carrier, pf_lru_delayed

        def _get_key():
            seed = random.randint(0, 2**31 - 1)
            return jr.key(seed)

        dtype = jnp.float64
        n = 10
        k = 2
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
    """
    max_delay = carrier.a.shape[0]
    if current_delay is None:
        if return_update:
            raise ValueError("`current_delay` must be specified to return updates.")
        current_delay = max_delay - 1

    elif current_delay < 0 or current_delay >= max_delay:
        raise ValueError(
            f"`current_delay` should be in range [0, {max_delay}), got {current_delay}."
        )

    Ainv = carrier.Ainv
    u = _standardize_uv(u, Ainv.shape[0], Ainv.dtype)

    return _get_delayed_output(carrier, u, return_update, current_delay)
