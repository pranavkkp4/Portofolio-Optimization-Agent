"""
Trading environment and helper functions for the News‑Aware Portfolio Optimization agent.

This module defines a simple trading environment along with utility functions used
throughout the project. The environment simulates a single‑asset market where
an agent decides whether to invest all capital into the asset or stay in cash
based on past returns and sentiment signals extracted from news headlines.

The code is intentionally lightweight so that it can run without external
dependencies (e.g. deep reinforcement learning libraries). It provides a
discretized state space and uses a Q‑table for policy representation.

Functions:
    compute_sentiment(text: str) -> int
        Compute a coarse sentiment score for a text by counting positive and
        negative keywords.

    discretize_state(return_value: float, sentiment_score: int) -> int
        Map continuous return and sentiment values into a discrete state ID.

Classes:
    TradingEnv
        A simple environment used to simulate trading decisions. It exposes
        ``reset`` and ``step`` methods typical of RL environments.

Note: The discretization thresholds are configurable via module constants
``RETURN_POS_THRESHOLD`` and ``RETURN_NEG_THRESHOLD``. The positive and
negative words lists used in ``compute_sentiment`` are also defined as
module‑level constants so that they can be extended if desired.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# Thresholds used to categorize returns into up, flat and down buckets.
RETURN_POS_THRESHOLD = 0.002  # >0.2% considered positive
RETURN_NEG_THRESHOLD = -0.002  # <-0.2% considered negative

# Simple lists of positive and negative keywords for sentiment scoring.
POSITIVE_KEYWORDS = [
    "growth", "profit", "gain", "increase", "up", "positive", "bull", "rise",
    "surge", "record"
]
NEGATIVE_KEYWORDS = [
    "loss", "decline", "down", "decrease", "negative", "bear", "drop",
    "fall", "crash", "plunge"
]

def compute_sentiment(text: str) -> int:
    """Compute a sentiment score from a piece of text.

    The scoring is coarse: if the number of positive keywords exceeds the
    negatives the score is +1, if negatives exceed positives the score is -1,
    otherwise the score is 0.

    Parameters
    ----------
    text : str
        Input text describing news or events.

    Returns
    -------
    int
        +1 for positive sentiment, -1 for negative sentiment, 0 for neutral.
    """
    if not text:
        return 0
    text_lower = text.lower()
    pos_count = sum(1 for word in POSITIVE_KEYWORDS if word in text_lower)
    neg_count = sum(1 for word in NEGATIVE_KEYWORDS if word in text_lower)
    if pos_count > neg_count:
        return 1
    if neg_count > pos_count:
        return -1
    return 0


def discretize_state(return_value: float, sentiment_score: int) -> int:
    """Convert continuous return and sentiment into a discrete state ID.

    The return dimension is divided into three buckets: negative, neutral and
    positive. The sentiment dimension is also divided into three buckets: -1,
    0 and 1. The state ID is computed as:

        state_id = return_bucket * 3 + sentiment_bucket

    so that there are nine possible states (0–8).

    Parameters
    ----------
    return_value : float
        The most recent return (e.g. price change divided by previous price).
    sentiment_score : int
        The coarse sentiment score in the range {-1, 0, 1}.

    Returns
    -------
    int
        An integer in [0, 8] representing the discretized state.
    """
    # Determine return bucket
    if return_value > RETURN_POS_THRESHOLD:
        return_bucket = 2  # up
    elif return_value < RETURN_NEG_THRESHOLD:
        return_bucket = 0  # down
    else:
        return_bucket = 1  # flat

    # Map sentiment_score (-1, 0, 1) into 0, 1, 2
    sentiment_bucket = sentiment_score + 1

    return return_bucket * 3 + sentiment_bucket


class TradingEnv:
    """A simple one‑asset trading environment.

    The environment simulates a sequence of prices and sentiment signals. At
    each step the agent observes the discretized state derived from the most
    recent return and sentiment. The agent chooses one of two actions:

        0: stay in cash (do not invest)
        1: invest fully in the asset

    The reward at time ``t`` is the product of the chosen action and the
    asset's return between ``t`` and ``t+1``. If the agent stays in cash the
    reward is zero; if the agent invests and the price goes up (down), the
    reward is positive (negative).

    Parameters
    ----------
    price_series : np.ndarray
        A sequence of prices for the asset.
    sentiment_series : np.ndarray
        A sequence of sentiment scores corresponding to each time step.
    """

    def __init__(self, price_series: np.ndarray, sentiment_series: np.ndarray) -> None:
        assert len(price_series) == len(sentiment_series), (
            "Price and sentiment series must have the same length"
        )
        self.price_series = np.asarray(price_series, dtype=float)
        self.sentiment_series = np.asarray(sentiment_series, dtype=float)
        self.length = len(self.price_series)
        self.index = 0

    def reset(self) -> int:
        """Reset the environment and return the initial state ID.

        Returns
        -------
        int
            The discretized state ID at the start of a new episode.
        """
        self.index = 1  # start at 1 so that a previous price exists
        return self._get_state()

    def step(self, action: int) -> Tuple[int, float, bool]:
        """Advance the environment by one time step.

        Parameters
        ----------
        action : int
            The action taken by the agent (0 for cash, 1 for invest).

        Returns
        -------
        state : int
            The next discretized state ID.
        reward : float
            The reward received after taking the action.
        done : bool
            Whether the episode has terminated.
        """
        # If we reach the end of the series, terminate the episode
        if self.index >= self.length - 1:
            return self._get_state(), 0.0, True

        prev_price = self.price_series[self.index - 1]
        current_price = self.price_series[self.index]
        # Avoid division by zero
        daily_return = (current_price - prev_price) / prev_price if prev_price != 0 else 0.0

        reward = daily_return * float(action)

        # Advance to next time step
        self.index += 1
        done = self.index >= self.length - 1
        next_state = self._get_state()
        return next_state, reward, done

    def _get_state(self) -> int:
        """Compute the current discretized state ID.

        Returns
        -------
        int
            The discretized state ID at the current index.
        """
        # When index == 0 there is no prior return; treat as zero return
        if self.index <= 0:
            ret = 0.0
        else:
            prev_price = self.price_series[self.index - 1]
            price = self.price_series[self.index]
            ret = (price - prev_price) / prev_price if prev_price != 0 else 0.0
        sentiment_score = int(self.sentiment_series[self.index])
        return discretize_state(ret, sentiment_score)