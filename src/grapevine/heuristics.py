"""Functions for guessing the next solution."""

import jax
from jax import numpy as jnp


def guess_previous(guess_inputs, position):
    """Use the previous solution as the next guess."""
    del position
    return guess_inputs.solution


def guess_default(guess_inputs, position, default_guess):
    """Always guess the default, i.e. do not use the grapevine method."""
    del guess_inputs, position
    return default_guess


def guess_implicit(guess_inputs, position, target_function):
    """Guess the next solution using the implicit function theorem."""
    old_x, old_p = guess_inputs.solution, guess_inputs.position
    delta_p = jax.tree.map(lambda old, new: new - old, old_p, position)
    _, jvpp = jax.jvp(lambda p: target_function(old_x, p), (old_p,), (delta_p,))
    jacx = jax.jacfwd(target_function, argnums=0)(old_x, old_p)
    return old_x - jnp.linalg.inv(jacx) @ jvpp


def guess_implicit_cg(guess_inputs, position, target_function):
    """Guess the next solution implicitly without materialising jacobians."""
    old_x, old_p = guess_inputs.solution, guess_inputs.position
    delta_p = jax.tree.map(lambda old, new: new - old, old_p, position)
    _, jvpp = jax.jvp(lambda p: target_function(old_x, p), (old_p,), (delta_p,))

    def matvec(v):
        "Compute Jx @ v for any vector v"
        return jax.jvp(lambda x: target_function(x, old_p), (old_x,), (v,))[1]

    return old_x - jax.scipy.sparse.linalg.cg(matvec, jvpp)[0]
