from typing import Optional, Tuple, Union, Sequence, NamedTuple
from jax import Array
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
from jax._src.numpy import reductions


_LowRankVecInput = Union[Array, int, Tuple[Array, Array], Tuple[Array, int]]


def _check_mat(mat: Array) -> None:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expect input matrix shape (n, n), got {mat.shape}.")


def _check_u_shape_dtype(
    u: Union[Array, int], n: int, dtype: Optional[jnp.dtype] = None
) -> Array:
    u = jnp.asarray(u)
    if dtype is not None and not jnp.issubdtype(u.dtype, dtype):
        raise ValueError(f"Expect input u to be of dtype {dtype}, but got {u.dtype}.")

    if jnp.issubdtype(u.dtype, jnp.integer):
        if u.ndim > 1:
            raise ValueError(
                f"Expect a 0-D or 1-D integer array, but got {u.ndim}-D array."
            )
        u = jnp.atleast_1d(u)
    else:
        if u.shape[0] != n or u.ndim > 2:
            raise ValueError(
                f"Expect input u shape (n,) or (n, k), got {u.shape} with n={n}."
            )
        u = u.reshape(n, -1)

    return u


def _standardize_uv(
    u: _LowRankVecInput, n: int, dtype: jnp.dtype
) -> Tuple[Array, Array]:
    if isinstance(u, ArrayLike):
        u = jnp.asarray(u)
        if jnp.issubdtype(u.dtype, jnp.integer):
            u = _check_u_shape_dtype(u, n)
            u = (jnp.empty((n, 0), dtype), u)
        else:
            u = _check_u_shape_dtype(u, n, dtype)
            u = (u, jnp.array([], dtype=jnp.int32))
    elif isinstance(u, Sequence):
        if len(u) != 2:
            raise ValueError(f"Expect input u to be a tuple of two arrays, got {u}.")
        x = _check_u_shape_dtype(u[0], n, dtype)
        e = _check_u_shape_dtype(u[1], n, jnp.integer)
        u = (x, e)
    else:
        raise ValueError(f"Got unsupported u or v data type {type(u)}.")
    return u


def _check_uv(u: Tuple[Array, Array], v: Tuple[Array, Array]) -> None:
    rank_u = u[0].shape[1] + u[1].size
    rank_v = v[0].shape[1] + v[1].size
    if rank_u != rank_v:
        raise ValueError(
            f"The input u and v should have matched rank, got {rank_u} and {rank_v}."
        )


def _get_R(Ainv: Array, u: Tuple[Array, Array], v: Tuple[Array, Array]) -> Array:
    if u[0].shape[1] <= v[0].shape[1]:
        xu_Ainv = u[0].T @ Ainv
        xu_Ainv_xv = xu_Ainv @ v[0]
        xu_Ainv_ev = xu_Ainv[:, v[1]]
        eu_Ainv = Ainv[u[1], :]
        eu_Ainv_xv = eu_Ainv @ v[0]
        eu_Ainv_ev = eu_Ainv[:, v[1]]
    else:
        Ainv_xv = Ainv @ v[0]
        xu_Ainv_xv = u[0].T @ Ainv_xv
        eu_Ainv_xv = Ainv_xv[u[1], :]
        Ainv_ev = Ainv[:, v[1]]
        xu_Ainv_ev = u[0].T @ Ainv_ev
        eu_Ainv_ev = Ainv_ev[u[1], :]

    uT_Ainv_v = jnp.block([[xu_Ainv_ev, xu_Ainv_xv], [eu_Ainv_ev, eu_Ainv_xv]])
    I = jnp.eye(uT_Ainv_v.shape[0], dtype=Ainv.dtype)
    return I + uT_Ainv_v


def _det_and_lufac(R: Array) -> Tuple[Array, Tuple[Array, Array]]:
    lu, pivot = jax.scipy.linalg.lu_factor(R)
    iota = jnp.arange(pivot.size, dtype=pivot.dtype)
    parity = reductions.count_nonzero(pivot != iota, axis=-1)
    sign = jnp.array(-2 * (parity % 2) + 1, dtype=lu.dtype)
    det = sign * jnp.prod(jnp.diag(lu))
    return det, (lu, pivot)


def _det_lru_no_update(
    Ainv: Array, u: Tuple[Array, Array], v: Tuple[Array, Array]
) -> Array:
    R = _get_R(Ainv, u, v)
    ratio = jnp.linalg.det(R)
    return ratio


def _det_lru_update(
    Ainv: Array, u: Tuple[Array, Array], v: Tuple[Array, Array]
) -> Tuple[Array, Array]:
    xu_Ainv = u[0].T @ Ainv
    eu_Ainv = Ainv[u[1], :]
    Ainv_xv = Ainv @ v[0]
    Ainv_ev = Ainv[:, v[1]]

    xu_Ainv_xv = xu_Ainv @ v[0]
    xu_Ainv_ev = xu_Ainv[:, v[1]]
    eu_Ainv_xv = Ainv_xv[u[1], :]
    eu_Ainv_ev = eu_Ainv[:, v[1]]
    uT_Ainv_v = jnp.block([[xu_Ainv_ev, xu_Ainv_xv], [eu_Ainv_ev, eu_Ainv_xv]])
    I = jnp.eye(uT_Ainv_v.shape[0], dtype=Ainv.dtype)
    R = I + uT_Ainv_v
    ratio, lufac = _det_and_lufac(R)

    uT_Ainv = jnp.concatenate((xu_Ainv, eu_Ainv), axis=0)
    Rinv_uT_Ainv = jax.scipy.linalg.lu_solve(lufac, uT_Ainv)
    Ainv_v = jnp.concatenate((Ainv_ev, Ainv_xv), axis=1)
    Ainv -= Ainv_v @ Rinv_uT_Ainv
    return ratio, Ainv


def det_lru(
    Ainv: Array,
    u: _LowRankVecInput,
    v: _LowRankVecInput,
    return_update: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    r"""
    Low-rank update of determinant :math:`\det(A_1) = \det(A_0 + vu^T)`.

    :param Ainv:
        Inverse of the original matrix :math:`A_0^{-1}` with shape (n, n)

    :param u:
        Low-rank update vector(s) :math:`u`. See the note below for details.

    :param v:
        Low-rank update vector(s) :math:`v`. See the note below for details.

    :param return_update:
        Whether the new matrix inverse :math:`A_1^{-1}` should be returned,
        defaul to False.

    :return:
        ratio:
            The ratio between two determinants

            .. math::

                r = \frac{\det(A_1)}{\det(A_0)} = \det(R)

            where

            .. math::

                R = I + u^T A_0^{-1} v

        new_Ainv:
            The new matrix inverse

            .. math::

                A_1^{-1} = (A_0 + vu^T)^{-1} = A_0^{-1} - A_0^{-1} v R^{-1} u^T A_0^{-1}

            Only returned when ``return_update`` is True.

    .. tip::

        This function is compatible with ``jax.jit`` and ``jax.vmap``, while
        ``return_update`` is a static argument which shouldn't be jitted or vmapped.

        Furthermore, we recommend setting ``donate_argnums=0`` in ``jax.jit`` to reuse 
        the memory of ``Ainv`` if it's no longer needed. This helps to greatly reduce 
        the time and memory cost. For instance,

        .. code-block:: python

            lru_vmap = jax.vmap(det_lru, in_axes=(0, 0, 0, None))
            lru_jit = jax.jit(lru_vmap, static_argnums=3, donate_argnums=0)
    
    .. note::

        We often need to define ``u`` and ``v`` as one-hot vectors, for instance,
        ``v = jnp.array([0, 1, 0, 0])``. In this case, the matrix product like  ``Ainv @ v`` 
        can be alternatively performed by ``Ainv[:, 1]`` to achieve great acceleration.

        Therefore, we allow ``u`` and ``v`` to be provided not only by dense arrays,
        but also by one-hot indeces. The acceptable inputs are listed below.

        An array with shape (n,) or (n, k):
            Dense array of ``u`` or ``v``.

        An integer or array of integers with size k:
            One-hot indices. The full array is defined as
            ``u_full = jnp.zeros((n, k)).at[u, jnp.arange(k)].set(1)`` and similarly for ``v``.
            For example, when you need a full matrix

            .. code-block:: python

                u = jnp.array([
                    [0, 0],
                    [1, 0], 
                    [0, 0], 
                    [0, 1],
                ])
            
            you can alternatively specify 

            .. code-block:: python
            
                u = jnp.array([1, 3])

        A tuple of two arrays with respective shapes (n, k0) and (k1,):
            A concatenation of the previous two. For example, when you need a full matrix

            .. code-block:: python

                u = jnp.array([
                    [x00, x01, 0, 0],
                    [x10, x11, 1, 0], 
                    [x20, x21, 0, 0], 
                    [x30, x31, 0, 1],
                ])

            you can alternatively specify

            .. code-block:: python
            
                x = jnp.array([
                    [x00, x01],
                    [x10, x11], 
                    [x20, x21], 
                    [x30, x31],
                ])
                e = jnp.array([1, 3])
                u = (x, e)

            As discussed in the note below, we usually need the one-hot vectors of ``v``
            to be on the left of dense vectors. Therefore, when we similarly define
            ``v = (x, e)``, it actually represents a different array

            .. code-block:: python

                v = jnp.array([
                    [0, 0, x00, x01],
                    [1, 0, x10, x11], 
                    [0, 0, x20, x21], 
                    [0, 1, x30, x31],
                ])

    .. admonition:: Example

        Here are examples of how to define ``u`` and ``v`` before calling ``det_lru(Ainv, u, v)``.
        Keep in mind that the low-rank update we need takes the form

        .. math::

            A_1 - A_0 = vu^T

        **Rank-1 row update**

        .. math::

            A_1 - A_0 = \begin{pmatrix}
                0 & 0 & 0 & 0 \\ 
                u_0 & u_1 & u_2 & u_3 \\
                0 & 0 & 0 & 0 \\ 
                0 & 0 & 0 & 0 \\ 
            \end{pmatrix}
            = \begin{pmatrix}
                0 \\ 1 \\ 0 \\ 0
            \end{pmatrix}
            (u_0, u_1, u_2, u_3)
            
        .. code-block:: python
        
            u = jnp.array([u0, u1, u2, u3])
            v = 1

        **Rank-1 column update**

        .. math::

            A_1 - A_0 = \begin{pmatrix}
                0 & 0 & v_0 & 0 \\ 
                0 & 0 & v_1 & 0 \\
                0 & 0 & v_2 & 0 \\ 
                0 & 0 & v_3 & 0 \\ 
            \end{pmatrix}
            = \begin{pmatrix}
                v_0 \\ v_1 \\ v_2 \\ v_3
            \end{pmatrix}
            (0, 0, 1, 0)
            
        .. code-block:: python
        
            u = 2
            v = jnp.array([v0, v1, v2, v3])

        **Rank-2 row update**

        .. math::

            A_1 - A_0 = \begin{pmatrix}
                0 & 0 & 0 & 0 \\ 
                u_{00} & u_{01} & u_{02} & u_{03} \\
                0 & 0 & 0 & 0 \\ 
                u_{10} & u_{11} & u_{12} & u_{13} \\
            \end{pmatrix}
            = \begin{pmatrix}
                0 & 0 \\ 1 & 0 \\ 0 & 0 \\ 0 & 1
            \end{pmatrix}
            \begin{pmatrix}
                u_{00} & u_{01} & u_{02} & u_{03} \\
                u_{10} & u_{11} & u_{12} & u_{13} \\
            \end{pmatrix}
            
        .. code-block:: python
        
            u = jnp.array([[u00, u10], [u01, u11], [u02, u12], [u03, u13]])
            v = jnp.array([1, 3])

        **Simultaneous update of row and column**

        .. math::

            A_1 - A_0 = \begin{pmatrix}
                0 & 0 & v_0 & 0 \\ 
                u_0 & u_1 & u_2 + v_1 & u_3 \\
                0 & 0 & v_2 & 0 \\ 
                0 & 0 & v_3 & 0 \\ 
            \end{pmatrix}
            = \begin{pmatrix}
                0 & v_0 \\ 1 & v_1 \\ 0 & v_2 \\ 0 & v_3
            \end{pmatrix}
            \begin{pmatrix}
                u_0 & u_1 & u_2 & u_3 \\
                0 & 0 & 1 & 0 \\
            \end{pmatrix}
            
        .. code-block:: python
        
            u = (jnp.array([u0, u1, u2, u3]), 2)
            v = (jnp.array([v0, v1, v2, v3]), 1)
    """
    _check_mat(Ainv)
    u = _standardize_uv(u, Ainv.shape[0], Ainv.dtype)
    v = _standardize_uv(v, Ainv.shape[0], Ainv.dtype)
    _check_uv(u, v)

    if return_update:
        return _det_lru_update(Ainv, u, v)
    else:
        return _det_lru_no_update(Ainv, u, v)


class DetCarrier(NamedTuple):
    Ainv: Array
    a: Array
    b: Array


def init_det_carrier(A: Array, max_delay: int, max_rank: int = 1) -> DetCarrier:
    r"""
    Prepare the data and space for `~lrux.det_lru_delayed`.

    :param A:
        The initial matrix :math:`A_0` with shape (n, n).

    :param max_delay:
        The maximum iterations T of delayed updates, usually chosen to be ~n/10.

    :param max_rank:
        The maximum rank K in delayed updates, default to 1.

    :return:
        A ``NamedTuple`` with the following attributes.

        Ainv:
            The initial matrix inverse :math:`A_0^{-1}` of shape (n, n).
        a:
            The delayed update vectors of shape (T, n, K), initialized to 0
        b:
            The delayed update vectors of shape (T, n, K), initialized to 0
    """

    if max_delay <= 0:
        raise ValueError(
            "`max_delay` should be a positive integer. "
            "Otherwise, please use `det_lru` for non-delayed updates."
        )
    _check_mat(A)
    Ainv = jnp.linalg.inv(A)
    a = jnp.zeros((max_delay, A.shape[0], max_rank), A.dtype)
    b = jnp.zeros_like(a)
    return DetCarrier(Ainv, a, b)


def merge_det_delays(carrier: DetCarrier) -> DetCarrier:
    r"""
    Merge the delayed updates in the carrier.

    When :math:`\tau` reaches the maximum delayed iterations :math:`T`
    specified in `~lrux.init_det_carrier`, i.e. ``current_delay == max_delay - 1``,
    the current :math:`A_\tau` should be set as the new :math:`A_0`,
    whose inverse is given by

    .. math::

        A_\tau^{-1} = A_0^{-1} - \sum_{t=1}^\tau a_t b_t^T

    The ``Ainv`` in ``new_carrier`` will be replaced by :math:`A_\tau^{-1}`, and
    ``a`` and ``b`` will be set to 0. See the example in `~lrux.det_lru_delayed`
    for details.

    .. tip::

        This function is compatible with ``jax.jit`` and ``jax.vmap``. 
        We recommend setting ``donate_argnums=0`` in ``jax.jit`` to reuse 
        the memory of ``carrier`` if it's no longer needed. This helps to greatly reduce 
        the time and memory cost. For instance,

        .. code-block:: python

            merge_vmap = jax.vmap(merge_det_delays)
            merge_jit = jax.jit(merge_vmap, donate_argnums=0)
    """
    Ainv = carrier.Ainv - jnp.einsum("tnk,tmk->nm", carrier.a, carrier.b)
    return DetCarrier(Ainv, jnp.zeros_like(carrier.a), jnp.zeros_like(carrier.b))


def _update_ab(a: Array, new_a: Array, current_delay: int) -> Array:
    k = new_a.shape[-1]
    if k > a.shape[-1]:
        raise ValueError(
            "The rank of update exceeds max_rank specified in `init_det_carrier`."
        )
    return a.at[current_delay, :, :k].set(new_a)


def _det_lru_delayed_no_update(
    carrier: DetCarrier,
    u: Tuple[Array, Array],
    v: Tuple[Array, Array],
    current_delay: int,
) -> jax.Array:
    Ainv = carrier.Ainv
    a = carrier.a[:current_delay]
    b = carrier.b[:current_delay]
    R = _get_R(Ainv, u, v)

    xuT_a = jnp.einsum("nk,tnl->tkl", u[0], a)
    euT_a = a[:, u[1], :]
    uT_a = jnp.concatenate((xuT_a, euT_a), axis=1)
    xvT_b = jnp.einsum("nk,tnl->tkl", v[0], b)
    evT_b = b[:, v[1], :]
    vT_b = jnp.concatenate((evT_b, xvT_b), axis=1)

    R -= jnp.einsum("tkl,tml->km", uT_a, vT_b)
    ratio = jnp.linalg.det(R)
    return ratio


def _det_lru_delayed_update(
    carrier: DetCarrier,
    u: Tuple[Array, Array],
    v: Tuple[Array, Array],
    current_delay: int,
) -> Tuple[Array, DetCarrier]:
    Ainv = carrier.Ainv
    a = carrier.a[:current_delay]
    b = carrier.b[:current_delay]

    xu_Ainv = u[0].T @ Ainv
    eu_Ainv = Ainv[u[1], :]
    Ainv_xv = Ainv @ v[0]
    Ainv_ev = Ainv[:, v[1]]

    xu_Ainv_xv = xu_Ainv @ v[0]
    xu_Ainv_ev = xu_Ainv[:, v[1]]
    eu_Ainv_xv = Ainv_xv[u[1], :]
    eu_Ainv_ev = eu_Ainv[:, v[1]]
    uT_Ainv_v = jnp.block([[xu_Ainv_ev, xu_Ainv_xv], [eu_Ainv_ev, eu_Ainv_xv]])
    I = jnp.eye(uT_Ainv_v.shape[0], dtype=Ainv.dtype)
    R0 = I + uT_Ainv_v

    xuT_a = jnp.einsum("nk,tnl->tkl", u[0], a)
    euT_a = a[:, u[1], :]
    uT_a = jnp.concatenate((xuT_a, euT_a), axis=1)
    xvT_b = jnp.einsum("nk,tnl->tkl", v[0], b)
    evT_b = b[:, v[1], :]
    vT_b = jnp.concatenate((evT_b, xvT_b), axis=1)
    R = R0 - jnp.einsum("tkl,tml->km", uT_a, vT_b)
    ratio, lufac = _det_and_lufac(R)

    a0 = jnp.concatenate((Ainv_ev, Ainv_xv), axis=1)
    new_a = a0 - jnp.einsum("tnk,tlk->nl", a, vT_b)
    bT0 = jnp.concatenate((xu_Ainv, eu_Ainv), axis=0)
    new_bT = bT0 - jnp.einsum("tkl,tnl->kn", uT_a, b)
    new_bT = jax.scipy.linalg.lu_solve(lufac, new_bT)

    a = _update_ab(carrier.a, new_a, current_delay)
    b = _update_ab(carrier.b, new_bT.T, current_delay)

    carrier = DetCarrier(Ainv, a, b)
    return ratio, carrier


def det_lru_delayed(
    carrier: DetCarrier,
    u: _LowRankVecInput,
    v: _LowRankVecInput,
    return_update: bool = False,
    current_delay: Optional[int] = None,
) -> Union[Array, Tuple[Array, DetCarrier]]:
    r"""
    Delayed low-rank update of determinant.

    :param carrier:
        The existing delayed update quantities, including :math:`A_0^{-1}`, and

        .. math::

            a_t = A_{t-1}^{-1} v_t

        .. math::

            b_t = (A_{t-1}^{-1})^T u_t

        with :math:`t` from 1 to :math:`\tau-1`.
        Initially provided by `~lrux.init_det_carrier`.

    :param u:
        Low-rank update vector(s) :math:`u_\tau`, the same as :math:`u` in `lrux.det_lru`.
        The rank of u shouldn't exceed the maximum allowed rank specified
        in `~lrux.init_det_carrier`.

    :param v:
        Low-rank update vector(s) :math:`v_\tau`, the same as :math:`v` in `lrux.det_lru`.
        The rank of v shouldn't exceed the maximum allowed rank specified
        in `~lrux.init_det_carrier`.

    :param return_update:
        Whether the new carrier with updated quantities should be returned,
        defaul to False.

    :param current_delay:
        The current iterations :math:`\tau` of delayed updates,
        must be specified when ``return_update`` is True.
        As python starts counting at 0, the actual :math:`\tau` value is given by
        ``current_delay + 1``.

    :return:
        ratio:
            The ratio between two determinants

            .. math::

                r_\tau = \frac{\det(A_\tau)}{\det(A_{\tau-1})} = \det(R_\tau)

            where

            .. math::

                R_\tau = I + u_\tau^T A_0^{-1} v_\tau - \sum_{t=1}^{\tau-1} (u_\tau^T a_t) (b_t^T v_\tau)

        new_carrier:
            Only returned when ``return_update`` is True. The new carrier contains
            the quantities from the input carrier, and in addition

            .. math::

                a_\tau = A_{\tau-1}^{-1} v_\tau = A_0^{-1} v_\tau - \sum_{t=1}^{\tau-1} a_t (b_t^T v_\tau)

            .. math::

                b_\tau = (A_{\tau-1}^{-1})^T u_\tau = (A_0^{-1})^T u_\tau - \sum_{t=1}^{\tau-1} b_t (a_t^T u_\tau)

    .. warning::

        This function is only recommended for heavy users who understand why and when
        to use delayed updates. Otherwise, please choose `~\lrux.det_lru`.

    .. warning::

        When ``current_delay`` reaches the maximum delayed iteration, i.e.
        ``current_delay == max_delay - 1``, one should call `~lrux.merge_det_delays`
        to merge the delayed updates in the carrier, and reset the carrier for the next round.
        See the example below for details.

    .. tip::

        Similar to `~lrux.det_lru`, this function is compatible with ``jax.jit`` and
        ``jax.vmap``, while ``return_update`` and ``current_delay`` are static arguments
        which shouldn't be jitted or vmapped.

        We still recommend setting ``donate_argnums=0`` in ``jax.jit`` to reuse
        the memory of ``carrier`` if it's no longer needed. For instance,

        .. code-block:: python

            lru_vmap = jax.vmap(det_lru_delayed, in_axes=(0, 0, 0, None, None))
            lru_jit = jax.jit(lru_vmap, static_argnums=(3, 4), donate_argnums=0)

    Here is a complete example of delayed updates.

    .. code-block:: python

        import os
        os.environ["JAX_ENABLE_X64"] = "1"

        import random
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        from lrux import init_det_carrier, merge_det_delays, det_lru_delayed

        def _get_key():
            seed = random.randint(0, 2**31 - 1)
            return jr.key(seed)

        dtype = jnp.float64
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

            # verify the low-rank update result
            A += v @ u.T
            detA1 = jnp.linalg.det(A)
            assert jnp.allclose(ratio, detA1 / detA0)
            detA0 = detA1
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
    v = _standardize_uv(v, Ainv.shape[0], Ainv.dtype)
    _check_uv(u, v)

    if return_update:
        return _det_lru_delayed_update(carrier, u, v, current_delay)
    else:
        return _det_lru_delayed_no_update(carrier, u, v, current_delay)
