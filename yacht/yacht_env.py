# yacht/yacht_env.py

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import jit


class YachtState(NamedTuple):
    dices: jnp.ndarray              # (int: 1~6) x 5
    category_scores: jnp.ndarray    # (int: >= -1) x 12
    rolls_left: int                 # int: 0~2
    turn: int                       # int: 0~12
    key: jax.random.PRNGKey


MAX_CATEGORY_SCORE = jnp.array([
    5, 10, 15, 20, 25, 30, # upper
    30, # choice
    30, # 4 of a kind
    30, # full house
    15, # small straight
    30, # large straight
    50, # yacht
])

@jit
def reset(key: jax.random.PRNGKey) -> YachtState:
    """
    Initializes the game state.

    Args:
        key: PRNGKey for random number generation

    Returns:
        YachtState: Initial game state
    """
    key, sub_key = jax.random.split(key)

    return YachtState(
        dices=jax.random.randint(sub_key, (5, ), 1, 7),
        category_scores=jnp.full(12, -1, dtype=jnp.int32),
        rolls_left=2,
        turn=0,
        key=sub_key
    )


@jit
def roll_dice(state: YachtState, keep_mask: jnp.ndarray) -> YachtState:
    """
    Rolls the dice. Re-rolls dice where keep_mask is False.
    Decreases the remaining roll count (rolls_left).

    Args:
        state: Current game state
        keep_mask: (5,) Boolean mask (True to keep, False to roll)

    Returns:
        YachtState: Updated game state
    """
    key, sub_key = jax.random.split(state.key)
    new_dices = jax.random.randint(sub_key, (5, ), 1, 7)
    dice = jnp.where(keep_mask, state.dices, new_dices)
    
    return state._replace(
        dices=dice,
        rolls_left=state.rolls_left - 1,
        key=key
    )


@jit
def calculate_scores(dice: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the scores for each category based on the dice combination.

    Args:
        dice: (5,) Dice values (1~6)

    Returns:
        (12,) Scores for each category
    """
    # Counts of each die value (1~6)
    counts = jnp.bincount(dice, length=7)[1:]

    # 1. Upper Section (Aces, Deuces, Threes, Fours, Fives, Sixes)
    # Sum of dice with the specific number
    upper_scores = counts * jnp.arange(1, 7)

    # 2. Lower Section
    # Choice: Sum of all dice
    choice_score = jnp.sum(dice)

    # 4 of a Kind: Sum of all dice if at least 4 dice are the same
    four_of_a_kind_score = jnp.where(jnp.any(counts >= 4), jnp.sum(dice), 0)

    # Full House: Sum of all dice if 3 of one number, 2 of another
    is_full_house = (jnp.any(counts == 3) & jnp.any(counts == 2))
    full_house_score = jnp.where(is_full_house, jnp.sum(dice), 0)

    # Mask for existence of each die value
    dice_exists = counts >= 1

    # S. Straight: 15 points if 4 sequential dice
    # Combinations: 1-2-3-4, 2-3-4-5, 3-4-5-6
    has_ss1 = jnp.all(dice_exists[0:4])
    has_ss2 = jnp.all(dice_exists[1:5])
    has_ss3 = jnp.all(dice_exists[2:6])
    small_straight_score = jnp.where(has_ss1 | has_ss2 | has_ss3, 15, 0)

    # L. Straight: 30 points if 5 sequential dice
    # Combinations: 1-2-3-4-5, 2-3-4-5-6
    has_ls1 = jnp.all(dice_exists[0:5])
    has_ls2 = jnp.all(dice_exists[1:6])
    large_straight_score = jnp.where(has_ls1 | has_ls2, 30, 0)

    # Yacht: 50 points if all 5 dice are the same
    yacht_score = jnp.where(jnp.any(counts == 5), 50, 0)

    # Combine all scores
    lower_scores = jnp.array([
        choice_score,
        four_of_a_kind_score,
        full_house_score,
        small_straight_score,
        large_straight_score,
        yacht_score
    ])

    return jnp.concatenate([upper_scores, lower_scores])


@jit
def calculate_total_score(category_scores: jnp.ndarray) -> int:
    """
    Calculates the total score including bonus.

    Args:
        category_scores: (12,) Scores for each category

    Returns:
        int: Total score
    """
    # Treat -1 (empty) as 0 for sum calculation
    valid_scores = jnp.maximum(category_scores, 0)
    
    # Bonus check: Sum of Upper Section (0~5) >= 63
    upper_sum = jnp.sum(valid_scores[:6])
    bonus = jnp.where(upper_sum >= 63, 35, 0)
    
    return jnp.sum(valid_scores) + bonus


@jit
def step(state: YachtState, action: int) -> Tuple[YachtState, int, bool]:
    """
    Executes one step in the environment.

    Args:
        state: Current game state
        action: Action to take
            - 0~31: Reroll dice (bitmask of dice to keep)
            - 32~43: Select category (action - 32)

    Returns:
        Tuple[YachtState, int, bool]:
            - Next game state
            - Total score (reward)
            - Whether the episode is done
    """

    key, sub_key = jax.random.split(state.key)

    def reroll_dices() -> Tuple[YachtState, int, bool]:
        keep_mask = ((action >> jnp.arange(5)) & 1).astype(jnp.bool_)
        return roll_dice(state, keep_mask), 0, False

    def select_category() -> Tuple[YachtState, int, bool]:
        category_score = calculate_scores(state.dices)[action - 32]
        new_category_scores = state.category_scores.at[action - 32].set(category_score)
        
        new_state = state._replace(
            dices=jax.random.randint(sub_key, (5, ), 1, 7),
            category_scores=new_category_scores,
            turn=state.turn + 1,
            rolls_left=2,
            key=key
        )
        return new_state, calculate_total_score(new_category_scores), new_state.turn >= 12

    def invalid_action() -> Tuple[YachtState, int, bool]:
        # Penalty for invalid action and end game
        return state._replace(turn=12), -100, True

    def try_reroll() -> Tuple[YachtState, int, bool]:
        return jax.lax.cond(
            state.rolls_left > 0,
            reroll_dices,
            invalid_action
        )

    def try_select_category() -> Tuple[YachtState, int, bool]:
        # Check if category is already used
        is_used = state.category_scores[action - 32] != -1
        return jax.lax.cond(
            is_used,
            invalid_action,
            select_category
        )

    return jax.lax.cond(
        action < 32,
        try_reroll,
        try_select_category
    )
