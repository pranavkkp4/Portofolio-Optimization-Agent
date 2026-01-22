"""
Training script for the News‑Aware Portfolio Optimization agent.

This script uses a simple Q‑learning algorithm to train a policy that decides
whether to invest in an asset based on past returns and sentiment signals. The
training data is synthetic: asset prices are generated via a random walk with
a small upward drift, and sentiment scores are sampled from a discrete
distribution. Despite its simplicity, the resulting Q‑table captures the
intuition that positive returns and positive sentiment should encourage
investment, while negative signals should discourage it.

The trained Q‑table is saved to ``model.pkl`` in the project root. Running
``python train.py`` produces this file. If you wish to tweak the training
parameters (number of episodes, learning rate, discount factor) you can do so
via module constants defined below.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np

from env import TradingEnv

# Training parameters
EPISODES = 200  # Number of training episodes
STEPS_PER_EPISODE = 200  # Length of each synthetic price series
ALPHA = 0.1  # Learning rate
GAMMA = 0.95  # Discount factor
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.05  # Final exploration rate
EPSILON_DECAY = 0.995  # Decay per episode


def create_synthetic_data(steps: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic price and sentiment time series for training.

    The price series is generated as a cumulative sum of Gaussian noise with a
    small positive drift, resulting in an upward trend. Sentiment scores are
    sampled from {-1, 0, 1} with equal probability.

    Parameters
    ----------
    steps : int
        The length of the time series.
    seed : int | None
        Seed for reproducibility.

    Returns
    -------
    price_series : np.ndarray
        Synthetic asset prices.
    sentiment_series : np.ndarray
        Synthetic sentiment scores.
    """
    rng = np.random.default_rng(seed)
    # Generate returns with slight upward drift
    returns = rng.normal(loc=0.0005, scale=0.01, size=steps)
    # Convert returns to prices starting at 100
    price_series = np.cumsum(returns) + 100.0
    # Ensure prices are positive
    price_series = price_series - price_series.min() + 100.0
    # Sample sentiment uniformly from {-1, 0, 1}
    sentiment_series = rng.integers(-1, 2, size=steps)
    return price_series, sentiment_series


def q_learning_train() -> np.ndarray:
    """Train a Q‑table using the Q‑learning algorithm.

    Returns
    -------
    np.ndarray
        The learned Q‑table of shape (9, 2). Rows correspond to discretized
        states and columns correspond to actions (0: cash, 1: invest).
    """
    n_states = 9
    n_actions = 2
    Q = np.zeros((n_states, n_actions), dtype=float)
    epsilon = EPSILON_START
    rng = np.random.default_rng()
    for ep in range(EPISODES):
        # Generate synthetic series for this episode
        price_series, sentiment_series = create_synthetic_data(STEPS_PER_EPISODE, seed=ep)
        env = TradingEnv(price_series, sentiment_series)
        state = env.reset()
        done = False
        while not done:
            # Epsilon‑greedy action selection
            if rng.random() < epsilon:
                action = rng.integers(0, n_actions)
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, done = env.step(action)

            # Q‑learning update
            best_future = np.max(Q[next_state])
            td_target = reward + GAMMA * best_future
            td_error = td_target - Q[state, action]
            Q[state, action] += ALPHA * td_error

            state = next_state

        # Decay exploration rate after each episode
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    return Q


def main() -> None:
    """Entry point for training.

    Trains the Q‑table and saves it to ``model.pkl`` in the project root.
    """
    Q = q_learning_train()
    model_path = Path(__file__).resolve().parent / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(Q, f)
    print(f"Training complete. Q‑table saved to {model_path}")


if __name__ == "__main__":
    main()