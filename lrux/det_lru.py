from typing import Optional, Tuple, Union, Sequence
from jax import Array
from jax.typing import ArrayLike
from jax.core import Tracer
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax._src.numpy import reductions


def _check_mat(mat: Array) -> None:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expect input matrix shape (n, n), got {mat.shape}.")


def _standardize_uv(
    u: Union[int, Array], n: int, dtype: jnp.dtype
) -> Tuple[Array, Array]:
    if isinstance(u, ArrayLike):
        u = jnp.asarray(u)
        if jnp.issubdtype(u.dtype, jnp.integer):
            u = (jnp.empty((n, 0), dtype), u.flatten())
        else:
            u = (u.reshape(n, -1), jnp.array([], dtype=jnp.int32))
    elif isinstance(u, Sequence):
        u = tuple(u)
    else:
        raise ValueError(f"Got unsupported u or v data type {type(u)}.")
    return u


def _check_uv(u: Union[int, Array], v: Union[int, Array]) -> None:
    rank_u = u[0].shape[1] + u[1].size
    rank_v = v[0].shape[1] + v[1].size
    if rank_u != rank_v:
        raise ValueError(
            f"The input u and v should have matched rank, got {rank_u} and {rank_v}."
        )


def _get_R(Ainv: Array, u: Tuple[Array, Array], v: Tuple[Array, Array]) -> Array:
    xu_Ainv_xv = jnp.einsum("nk,nm,ml->kl", u[0], Ainv, v[0])
    eu_Ainv_xv = Ainv[u[1]] @ v[0]
    xu_Ainv_ev = u[0].T @ Ainv[:, v[1]]
    eu_Ainv_ev = Ainv[u[1], :][:, v[1]]
    uT_Ainv_v = jnp.block([[xu_Ainv_ev, xu_Ainv_xv], [eu_Ainv_ev, eu_Ainv_xv]])
    return uT_Ainv_v.at[jnp.diag_indices_from(uT_Ainv_v)].add(1)


def _det_and_lufac(R: Array) -> Tuple[Array, Tuple[Array, Array]]:
    lu, pivot = jax.scipy.linalg.lu_factor(R)
    iota = jnp.arange(pivot.size, dtype=pivot.dtype)
    parity = reductions.count_nonzero(pivot != iota, axis=-1)
    sign = jnp.array(-2 * (parity % 2) + 1, dtype=lu.dtype)
    det = sign * jnp.prod(jnp.diag(lu))
    return det, (lu, pivot)


def _update_Ainv(
    Ainv: Array,
    u: Tuple[Array, Array],
    v: Tuple[Array, Array],
    lu_and_piv: Tuple[Array, Array],
) -> Array:
    uT_Ainv = jnp.concatenate((u[0].T @ Ainv, Ainv[u[1], :]), axis=0)
    Rinv_uT_Ainv = jax.scipy.linalg.lu_solve(lu_and_piv, uT_Ainv)
    Ainv_v = jnp.concatenate((Ainv[:, v[1]], Ainv @ v[0]), axis=1)
    return Ainv - Ainv_v @ Rinv_uT_Ainv


def det_lru(
    Ainv: Array,
    u: Union[int, Array, Tuple[Array, Array]],
    v: Union[int, Array, Tuple[Array, Array]],
    return_update: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """
    Low-rank update of determinant
    """
    _check_mat(Ainv)
    u = _standardize_uv(u, Ainv.shape[0], Ainv.dtype)
    v = _standardize_uv(v, Ainv.shape[0], Ainv.dtype)
    _check_uv(u, v)

    R = _get_R(Ainv, u, v)
    ratio, lufac = _det_and_lufac(R)
    if return_update:
        Ainv = _update_Ainv(Ainv, u, v, lufac)
        return ratio, Ainv
    else:
        return ratio


@jtu.register_pytree_node_class
class DetCarrier:
    """
    The pytree carrying intermediate information for low-rank updates.
    """

    def __init__(self, Ainv: Array, a: Optional[Array], b: Optional[Array]):
        self.Ainv = Ainv
        self.a = a
        self.b = b

    def tree_flatten(self) -> Tuple:
        children = (self.Ainv, self.a, self.b)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __repr__(self):
        return f"DetCarrier(Ainv: {self.Ainv}\na: {self.a}\nb: {self.b})"

    def get_current_Ainv(self) -> Array:
        if self.a is None:
            return self.Ainv
        else:
            return self.Ainv - jnp.einsum("tnk,tmk->nm", self.a, self.b)


def init_det_carrier(A: Array, max_delay: int, max_rank: int = 1) -> DetCarrier:
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


def _update_ab(a: Array, new_a: Array, current_delay: int) -> Array:
    k = new_a.shape[-1]
    if k > a.shape[-1]:
        raise ValueError(
            "The rank of update exceeds max_rank specified in `init_det_carrier`."
        )
    return a.at[current_delay % a.shape[0], :, :k].set(new_a)


def _get_delayed_output(
    carrier: DetCarrier,
    u: Tuple[Array, Array],
    v: Tuple[Array, Array],
    return_update: bool,
    current_delay: Optional[int],
    tau: int,
) -> Tuple[Array, Array]:
    Ainv = carrier.Ainv
    a = carrier.a[:tau]
    b = carrier.b[:tau]
    R0 = _get_R(Ainv, u, v)

    xuT_a = jnp.einsum("nk,tnl->tkl", u[0], a)
    euT_a = a[:, u[1], :]
    uT_a = jnp.concatenate((xuT_a, euT_a), axis=1)

    xvT_b = jnp.einsum("nk,tnl->tkl", v[0], b)
    evT_b = b[:, v[1], :]
    vT_b = jnp.concatenate((evT_b, xvT_b), axis=1)

    R = R0 - jnp.einsum("tkl,tml->km", uT_a, vT_b)
    ratio, lufac = _det_and_lufac(R)

    if return_update:
        a0 = jnp.concatenate((Ainv[:, v[1]], Ainv @ v[0]), axis=1)
        new_a = a0 - jnp.einsum("tnk,tlk->nl", a, vT_b)
        bT0 = jnp.concatenate((u[0].T @ Ainv, Ainv[u[1], :]), axis=0)
        new_bT = bT0 - jnp.einsum("tkl,tnl->kn", uT_a, b)
        new_bT = jax.scipy.linalg.lu_solve(lufac, new_bT)

        a = _update_ab(carrier.a, new_a, current_delay)
        b = _update_ab(carrier.b, new_bT.T, current_delay)
        carrier = DetCarrier(Ainv, a, b)
        return ratio, carrier
    else:
        return ratio


def _push_Ainv_output(
    carrier: DetCarrier,
    u: Tuple[Array, Array],
    v: Tuple[Array, Array],
    current_delay: int,
) -> Tuple[Array, Array]:
    Ainv = carrier.get_current_Ainv()
    R = _get_R(Ainv, u, v)
    ratio, lufac = _det_and_lufac(R)

    new_a = jnp.concatenate((Ainv[:, v[1]], Ainv @ v[0]), axis=1)
    new_bT = jnp.concatenate((u[0].T @ Ainv, Ainv[u[1], :]), axis=0)
    new_bT = jax.scipy.linalg.lu_solve(lufac, new_bT)
    zeros = jnp.zeros_like(carrier.a)
    a = _update_ab(zeros, new_a, current_delay)
    b = _update_ab(zeros, new_bT.T, current_delay)
    carrier = DetCarrier(Ainv, a, b)
    return ratio, carrier


def det_lru_delayed(
    carrier: DetCarrier,
    u: Union[int, Array, Tuple[Array, Array]],
    v: Union[int, Array, Tuple[Array, Array]],
    return_update: bool = False,
    current_delay: Optional[int] = None,
) -> Union[Array, Tuple[Array, DetCarrier]]:
    """
    Low-rank update of determinant with delayed updates
    """
    Ainv = carrier.Ainv
    u = _standardize_uv(u, Ainv.shape[0], Ainv.dtype)
    v = _standardize_uv(v, Ainv.shape[0], Ainv.dtype)
    _check_uv(u, v)

    max_delay = carrier.a.shape[0]
    if isinstance(current_delay, Tracer) or current_delay is None:
        tau = max_delay
    else:
        # slice a and b if current_delay is static_arg
        tau = current_delay % max_delay

    if return_update:
        if current_delay is None:
            raise ValueError("`current_delay` must be specified to return updates.")

        cond1 = current_delay % max_delay == 0
        cond2 = current_delay > 0
        is_pushing_Ainv = jnp.logical_and(cond1, cond2)
        args = (carrier, u, v, current_delay)
        delayed_output = lambda c, u, v, i: _get_delayed_output(c, u, v, True, i, tau)
        return jax.lax.cond(is_pushing_Ainv, _push_Ainv_output, delayed_output, *args)
    else:
        return _get_delayed_output(carrier, u, v, False, current_delay, tau)
