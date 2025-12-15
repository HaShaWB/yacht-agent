# yacht/yacht_reinforce.py

import csv
from datetime import datetime

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from pathlib import Path
import orbax.checkpoint as ocp

from yacht_gymnax import YachtEnv


# --- Hyperparameters ---
NUM_BATCHES = int(1e6)
LEARNING_RATE = 1e-3
BATCH_SIZE = int(1e4)

DIMS = (128, 64, 64)

MAX_STEPS = 50
PRINT_INTERVAL = 100
SAVE_INTERVAL = 10000


# --- Network Definition ---
class PolicyNetwork(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        for dim in DIMS:
            x = nn.Dense(dim)(x)
            x = nn.leaky_relu(x)
        x = nn.Dense(self.num_actions)(x)
        return x


# --- Helper: Action Masking ---
def get_valid_actions_mask(state):
    """
    Generate a mask indicating valid actions in the current state
    Returns:
        (44,) boolean array (True: valid, False: invalid)
    """
    # 1. Reroll actions (0~31): Possible only if rolls_left > 0
    can_reroll = state.rolls_left > 0
    reroll_mask = jnp.full((32,), can_reroll, dtype=jnp.bool_)

    # 2. Category actions (32~43): Possible only for categories not yet filled (-1)
    # state.category_scores is a (12,) array
    category_mask = state.category_scores == -1

    return jnp.concatenate([reroll_mask, category_mask])


# --- Agent & Training Logic ---

def create_train_state(rng, input_dim, num_actions):
    """Initialize TrainState"""
    model = PolicyNetwork(num_actions=num_actions)
    params = model.init(rng, jnp.ones((1, input_dim)))['params']

    lr_schedule = optax.piecewise_constant_schedule(
        init_value=LEARNING_RATE
    )
    tx = optax.adam(lr_schedule)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def get_rollout_fn(env):
    """Create episode rollout function"""

    def rollout_episode(train_state, key):
        env_params = env.default_params
        obs, state = env.reset_env(key, env_params)

        def step_fn(carry, _):
            obs, state, done, rng = carry
            rng, action_key = jax.random.split(rng)

            # 1. Run Policy Network (Calculate Logits)
            logits = train_state.apply_fn({'params': train_state.params}, obs)

            # 2. Apply Action Masking
            # Set logits of invalid actions to a very small value (-1e9) to make selection probability 0
            valid_mask = get_valid_actions_mask(state)
            masked_logits = jnp.where(valid_mask, logits, -1e9)

            # 3. Action Sampling
            action = jax.random.categorical(action_key, masked_logits)
            log_prob = jax.nn.log_softmax(masked_logits)[action]

            # 4. Step Environment
            next_obs, next_state, reward, next_done, _ = env.step_env(action_key, state, action, env_params)

            return (next_obs, next_state, next_done, rng), (log_prob, reward, next_done)

        # Proceed with episode using Scan
        (_, _, final_done, _), (log_probs, rewards, dones) = jax.lax.scan(
            step_fn,
            (obs, state, False, key),
            None,
            length=MAX_STEPS
        )

        # --- Return Calculation ---
        # Since masking was applied, -100 penalty situation does not occur (theoretically)
        finish_idx = jnp.argmax(dones)
        final_score = rewards[finish_idx]

        # If done did not occur until the end, use the last score
        final_score = jnp.where(jnp.any(dones), final_score, rewards[-1])

        # --- Calculate log_prob sum (for REINFORCE with baseline) ---
        # Exclude steps after done from training
        mask = jnp.arange(MAX_STEPS) <= finish_idx
        log_prob_sum = jnp.sum(log_probs * mask)

        return log_prob_sum, final_score

    return rollout_episode


def train():
    # 1. Environment and Network Setup
    env = YachtEnv()
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    obs_dim = env.observation_space(env.default_params).shape[0]
    num_actions = env.action_space(env.default_params).n

    train_state = create_train_state(init_rng, obs_dim, num_actions)
    rollout_episode = get_rollout_fn(env)

    # 2. Define Update Step
    @jax.jit
    def update_step(state, rng):
        batch_keys = jax.random.split(rng, BATCH_SIZE)

        def batch_loss_fn(params):
            temp_state = state.replace(params=params)
            log_prob_sums, scores = jax.vmap(rollout_episode, in_axes=(None, 0))(temp_state, batch_keys)
            
            # REINFORCE with baseline: advantage = score - baseline
            baseline = jnp.mean(scores)
            advantages = scores - baseline
            
            # Loss = - sum(log_prob * advantage)
            losses = -log_prob_sums * advantages
            return jnp.mean(losses), jnp.mean(scores)

        (loss, avg_score), grads = jax.value_and_grad(batch_loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, avg_score

    # 3. Training Loop
    print(f"Starting training for {NUM_BATCHES} batches ({BATCH_SIZE} parallel envs)...")
    print(f"{'Batch':<10} | {'Avg Score':<15} | {'Loss':<15}")
    print("-" * 45)

    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # CSV Logging Setup
    log_dir = Path(__file__).parent.parent / "models"
    log_dir.mkdir(exist_ok=True)
    csv_path = log_dir / f"REINFORCE_training_log.csv"
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['batch_idx', 'avg_score', 'loss'])

    # Setup Orbax CheckpointManager (new API)
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=5)
    ckpt_manager = ocp.CheckpointManager(checkpoint_dir, options=ckpt_options)

    start_step = 1
    if ckpt_manager.latest_step() is not None:
        latest_step = ckpt_manager.latest_step()
        print(f"Resuming from checkpoint step: {latest_step}")
        train_state = ckpt_manager.restore(latest_step, args=ocp.args.StandardRestore(train_state))
        start_step = int(train_state.step) + 1

    try:
        # Iterate for the configured NUM_BATCHES
        for batch_idx in range(start_step, NUM_BATCHES + 1):
            rng, step_rng = jax.random.split(rng)
            train_state, loss, avg_score = update_step(train_state, step_rng)

            if batch_idx % PRINT_INTERVAL == 0:
                print(f"{batch_idx:<10} | {avg_score:.2f}{' ':<11} | {loss:.2f}")
                csv_writer.writerow([batch_idx, float(avg_score), float(loss)])
                csv_file.flush()

            if batch_idx % SAVE_INTERVAL == 0:
                ckpt_manager.save(batch_idx, args=ocp.args.StandardSave(train_state))

        print("Training finished.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        csv_file.close()
        print(f"Training log saved to: {csv_path}")
        ckpt_manager.wait_until_finished()
        ckpt_manager.close()


if __name__ == "__main__":
    train()