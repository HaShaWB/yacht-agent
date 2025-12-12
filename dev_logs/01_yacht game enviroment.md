# Yacht Game Environment

**Source Code**: [`yacht/yacht_env.py`](../yacht/yacht_env.py)

### 1. `YachtState` Definition
Designed the state structure suitable for the JAX environment.
- **`done`**: Omitted. Replaced by checking `turn >= 12`.
- **`category_used`**: Omitted. Unused slots are represented by `category_scores = -1`.

### 2. `reset`
Initializes `YachtState`.

### 3. `roll_dice`
Rolls the dice.
- Allows selecting dice to keep via the `keep_mask` parameter.

### 4. `calculate_scores`
Returns a score vector obtainable for each score category based on the dice combination.
- **Optimization**: Calculates all scores at once using matrix operations instead of a specific `category` parameter.
- **JAX Style**: Actively utilizes `jnp.where` instead of Python's conditional `if/else`.

### 5. `calculate_total_score`
Calculates the total score based on `category_scores`.

### 6. `step`
Executes one step in the environment, returning `(next_state, reward, done)`.

- **Phases**: Distinguishes between `reroll_dices` and `select_category`.
- **Bitwise Operations**: Uses `(action >> jnp.arange(5)) & 1` to convert integer action to binary vector.
- **Validation**:
  - Validates invalid cases (e.g., `action < 32` when `rolls_left` is 0).
  - Uses `try_reroll` and `try_select_category`.
  - Invokes `invalid_action` (-100 penalty) on failure.
