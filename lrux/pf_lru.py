from typing import Optional, Tuple, Union, NamedTuple
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
    u: Union[Array, Tuple[Array, Array]],
    return_update: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """
    Low-rank update of pfaffian
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
    if max_delay <= 0:
        raise ValueError(
            "`max_delay` should be a positive integer. "
            "Otherwise, please use `det_lru` for non-delayed updates."
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
    u: Union[int, Array, Tuple[Array, Array]],
    return_update: bool = False,
    current_delay: Optional[int] = None,
) -> Union[Array, Tuple[Array, PfCarrier]]:
    """
    Low-rank update of pfaffian with delayed updates
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
