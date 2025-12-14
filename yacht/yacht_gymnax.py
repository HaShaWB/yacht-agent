# yacht/yacht_gymnax.py

import jax
import jax.numpy as jnp
from jax import jit
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
import yacht_env


class YachtEnv(environment.Environment):
    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> environment.EnvParams:
        return environment.EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: yacht_env.YachtState, action: int, params: environment.EnvParams
    ) -> Tuple[chex.Array, yacht_env.YachtState, float, bool, dict]:
        next_state, reward, done = yacht_env.step(state, action)
        return self.get_obs(next_state), next_state, reward, done, {}

    def reset_env(
        self, key: chex.PRNGKey, params: environment.EnvParams
    ) -> Tuple[chex.Array, yacht_env.YachtState]:
        state = yacht_env.reset(key)
        return self.get_obs(state), state

    def get_obs(self, state: yacht_env.YachtState) -> chex.Array:
        """
        Constructs the observation vector.
        - Dices: 5
        - Category Scores: 12
        - Rolls Left: 1
        - Turn: 1
        Total: 19
        """
        return jnp.concatenate([
            state.dices,
            state.category_scores,
            jnp.array([state.rolls_left]),
            jnp.array([state.turn])
        ])

    @property
    def name(self) -> str:
        return "Yacht-v0"

    @property
    def num_actions(self) -> int:
        return 44  # 32 (reroll combinations) + 12 (categories)

    def action_space(self, params: environment.EnvParams) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: environment.EnvParams) -> spaces.Box:
        return spaces.Box(low=-1, high=50, shape=(19,), dtype=jnp.float32)
