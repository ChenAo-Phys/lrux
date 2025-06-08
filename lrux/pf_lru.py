from typing import Optional, Tuple, Union
from jax import Array
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from .det_lru import _check_mat, _standardize_uv
from .pfaffian import pf


def pf_lru(
    Ainv: Array,
    u: Union[Array, Tuple[Array, Array]],
    return_update: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """
    Low-rank update of determinant
    """
    _check_mat(Ainv)
    u = _standardize_uv(u, Ainv.shape[0], Ainv.dtype)
    pass


@jtu.register_pytree_node_class
class PfCarrier:
    """
    The pytree carrying intermediate information for low-rank updates.
    """

    def __init__(self, Ainv: jax.Array, a: jax.Array, Rinv: jax.Array):
        self.inv = Ainv
        self.a = a
        self.Rinv = Rinv

    def tree_flatten(self) -> Tuple:
        children = (self.inv, self.a, self.Rinv)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def init_pf_carrier(A: Array, max_delay: int, max_rank: int = 1) -> PfCarrier:
    if max_delay <= 0:
        raise ValueError(
            "`max_delay` should be a positive integer. "
            "Otherwise, please use `det_lru` for non-delayed updates."
        )
    _check_mat(A)
    Ainv = jnp.linalg.inv(A)
    a = jnp.zeros((max_delay, A.shape[0], max_rank), A.dtype)
    Rinv = jnp.zeros((max_delay, max_rank, max_rank), A.dtype)
    return PfCarrier(Ainv, a, Rinv)


def pf_lru_delayed(
    carrier: PfCarrier,
    u: Union[int, Array, Tuple[Array, Array]],
    return_update: bool = False,
    current_delay: Optional[int] = None,
) -> Union[Array, Tuple[Array, PfCarrier]]:
    """
    Low-rank update of pfaffian with delayed updates
    """
    pass
