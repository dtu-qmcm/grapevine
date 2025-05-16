"""Functions for guessing the next solution."""

import jax
from jax import numpy as jnp


@jax.jit
def guess_previous(guess_info, *args):
    """Use the previous solution as the next guess."""
    return guess_info[0]


def guess_implicit(guess_info, params, target_function):
    """Guess the next solution using the implicit function theorem."""
    old_x, old_p, *_ = guess_info
    delta_p = jax.tree.map(lambda old, new: new - old, old_p, params)
    _, jvpp = jax.jvp(lambda p: target_function(old_x, p), (old_p,), (delta_p,))
    jacx = jax.jacfwd(target_function, argnums=0)(old_x, old_p)
    u = -(jnp.linalg.inv(jacx))
    return old_x + u @ jvpp


def guess_implicit_cg(guess_info, params, target_function):
    """Guess the next solution implicitly without materialising jacobians."""
    old_x, old_p, *_ = guess_info
    delta_p = jax.tree.map(lambda old, new: new - old, old_p, params)
    _, jvpp = jax.jvp(lambda p: target_function(old_x, p), (old_p,), (delta_p,))

    def matvec(v):
        "Compute Jx @ v for any vector v"
        return jax.jvp(lambda x: target_function(x, old_p), (old_x,), (v,))[1]

    dx = -jax.scipy.sparse.linalg.cg(matvec, jvpp)[0]
    return old_x + dx
