from typing import Optional, Tuple, Union
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from .pfaffian import pf


@jtu.register_pytree_node_class
class PfCarrier:
    """
    The pytree carrying intermediate information for low-rank updates.
    """

    def __init__(
        self,
        inv: jax.Array,
        a: Optional[jax.Array] = None,
        b: Optional[jax.Array] = None,
        current_delay: Optional[jax.Array] = None,
    ):
        self.inv = inv
        self.a = a
        self.b = b
        self.current_delay = current_delay

    def tree_flatten(self) -> Tuple:
        children = (
            self.inv,
            self.a,
            self.b,
            self.current_delay,
        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    

def init_pf_carrier():pass


def pf_ratio(
    carrier: Union[jax.Array, PfCarrier],
    u: jax.Array,
    e: jax.Array,
    return_update: bool = False,
) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
    pass
