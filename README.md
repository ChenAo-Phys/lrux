<h1 align='center'>lrux</h1>

Fast low-rank update (LRU) of matrix determinants and pfaffians in [JAX](https://github.com/jax-ml/jax)

[📖 Documentation](https://chenao-phys.github.io/lrux/)

## What's low-rank update?

In quantum physics and many other fields, it often happens that we have computed $\det(\mathbf{A}_0)$ and need to compute $\det(\mathbf{A}_1)$, where $\mathbf{A}_0$ and $\mathbf{A}_1$ are nearly identical. Due to the similarity of $\mathbf{A}_0$ and $\mathbf{A}_1$, we can well expect that in many cases $\det(\mathbf{A}_1)$ doesn't have to be recomputed from scratch.

Consider a special case that $\mathbf{A}_0$ and $\mathbf{A}_1$ are only different by one row. We can express their difference as
<!-- $$
    \mathbf{A}_1 - \mathbf{A}_0 = \begin{pmatrix}
        0 & ... & 0 \\ 
        \vdots && \vdots \\ 
        0 & ... & 0 \\ 
        u_1 & ... & u_n \\
        0 & ... & 0 \\ 
        \vdots && \vdots \\ 
        0 & ... & 0
    \end{pmatrix}
    = \begin{pmatrix}
        0 \\ \vdots \\ 0 \\ 1 \\ 0 \\ \vdots \\ 0
    \end{pmatrix}
    (u_1, ..., u_n)
    = \mathbf{vu}^T.
$$ -->
<div align="center">
  <img src="./images/A1_A0.svg"/>
</div>

Then $\mathbf{A}_1$ can be viewed as a low-rank update to $\mathbf{A}_0$, and the [matrix determinant lemma](https://en.wikipedia.org/wiki/Matrix_determinant_lemma) tells us
<!-- $$
    \det(\mathbf{A}_1) = \det (\mathbf{A}_0 + \mathbf{vu}^T)
    = \det (\mathbf{A}_0) (1 + \mathbf{u}^T \mathbf{A}_0^{-1} \mathbf{v}).
$$ -->
<div align="center">
  <img src="./images/detA1.svg"/>
</div>

If $\mathbf{A}_0^{-1}$ and $\det(\mathbf{A}_0)$ has been computed and stored earlier, one can immediately obtain $\det(\mathbf{A}_1)$ with $\mathcal{O}(n^2)$ complexity for any general $\mathbf{u}$ and $\mathbf{v}$, instead of the original determinant complexity $\mathcal{O}(n^3)$. The following code shows how this is done with lrux, where `lrux.det_lru` returns the ratio 
<!-- $$
r = \frac{\det(\mathbf{A}_1)}{\det (\mathbf{A}_0)} = 1 + \mathbf{u}^T \mathbf{A}_0^{-1} \mathbf{v}.
$$ -->
<div align="center">
  <img src="./images/ratio.svg"/>
</div>

```python
import jax

# 64-bit recommended for numerical precision
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from lrux import det_lru

n = 10
A0 = jr.normal(jr.key(0), (n, n))
u = jr.normal(jr.key(1), (n,))
v = 5  # one-hot vector index
detA0 = jnp.linalg.det(A0)
Ainv = jnp.linalg.inv(A0)

ratio = det_lru(Ainv, u, v)
detA1_lru = detA0 * ratio

A1 = A0.at[v, :].add(u)
assert jnp.isclose(detA1_lru, jnp.linalg.det(A1))
```


## Consecutive updates

Sometimes we need to keep computing $\det(\mathbf{A}_2)$ by a low-rank update of $\mathbf{A}_1$, in which case $\mathbf{A}_1^{-1}$ is required. Utilizing the [Sherman–Morrison formula](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula), one can also obtain the low-rank update of matrix inverse as
<!-- $$
\mathbf{A}_1^{-1} = (\mathbf{A}_0 + \mathbf{vu}^T)^{-1} = \mathbf{A}_0^{-1} - \mathbf{A}_0^{-1} \mathbf{v} r^{-1} \mathbf{u}^T \mathbf{A}_0^{-1},
$$ -->
<div align="center">
  <img src="./images/A1inv.svg"/>
</div>

where the complexity is again $\mathcal{O}(n^2)$ instead of $\mathcal{O}(n^3)$. Following the previous example code, one can add a few lines below to perform consecutive low-rank updates.

```python
ratio, Ainv = det_lru(Ainv, u, v, return_update=True)
assert jnp.allclose(Ainv, jnp.linalg.inv(A1))

u_new = jr.normal(jr.key(2), (n,))
v_new = 6

ratio, Ainv = det_lru(Ainv, u_new, v_new, return_update=True)
detA2_lru = detA1_lru * ratio

A2 = A1.at[v_new, :].add(u_new)
assert jnp.isclose(detA2_lru, jnp.linalg.det(A2))
assert jnp.allclose(Ainv, jnp.linalg.inv(A2))
```


## What does lrux provide?

The main functions of lrux include `det_lru`, `det_lru_delayed`, `pf_lru`, and `pf_lru_delayed`. They provide:

- Row and column updates
- General rank-k updates
- Delayed updates
- `jit` and `vmap` compatibility

As the [pfaffian](https://en.wikipedia.org/wiki/Pfaffian) is not directly supported in JAX, we also provide backward-compatible functions `pf` and `slogpf` for pfaffian computations.


## Installation

Requires Python 3.8+ and JAX 0.4.4+

```
git clone https://github.com/ChenAo-Phys/lrux.git
pip install ./lrux
```
