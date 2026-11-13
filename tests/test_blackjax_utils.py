import jax
import pytest

from blackjax_utils import patch_for_shard_map, run_sampler
from blackjax_utils.shard_map_patch import get_patched_functions
from jax import numpy as jnp

from grapevine import grapenuts
from tests.simple_example_problem import default_guess, posterior_logdensity

SEED = 12345
N_CHAIN = 2
N_SAMPLE = 20


@pytest.fixture
def shard_map_patch(monkeypatch):
    for module, function_name in get_patched_functions():
        monkeypatch.setattr(
            module, function_name, getattr(module, function_name)
        )
    patch_for_shard_map()


@pytest.mark.parametrize("chain_map", [None, jax.vmap])
def test_run_sampler_runs_grapenuts(chain_map, shard_map_patch):
    assert jax.device_count() >= N_CHAIN
    states, info = run_sampler(
        key=jax.random.key(SEED),
        log_posterior=posterior_logdensity,
        init_params=jnp.array(0.0),
        init_sd=0.01,
        n_chain=N_CHAIN,
        n_warmup=20,
        n_sample=N_SAMPLE,
        warmup_options=dict(initial_step_size=0.01),
        sampler=grapenuts(default_guess=default_guess),
        chain_map=chain_map,
    )
    assert states.position.shape == (N_CHAIN, N_SAMPLE)
    assert states.guess.shape == (N_CHAIN, N_SAMPLE)
    assert jnp.isfinite(states.logdensity).all()
    assert info.is_divergent.shape == (N_CHAIN, N_SAMPLE)
