"""Deterministic speaker-independent dataset split helpers for RAVDESS."""

import numpy as np


def actor_independent_split(groups, seed: int = 42):
    """Return train/validation/test indices with no actor shared across splits.

    For the 24-actor RAVDESS dataset this assigns 17 actors to training,
    3 to validation, and 4 to testing, approximately matching a 70/15/15
    sample split while keeping speakers disjoint.
    """
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("Actor groups must be a non-empty one-dimensional array.")

    actors = np.unique(groups)
    if len(actors) < 3:
        raise ValueError("At least three distinct actors are required for the split.")

    rng = np.random.default_rng(seed)
    actors = rng.permutation(actors)
    test_count = max(1, int(np.ceil(len(actors) * 0.15)))
    validation_count = max(1, int(np.floor(len(actors) * 0.15)))
    if test_count + validation_count >= len(actors):
        raise ValueError("Not enough actors to create train, validation, and test splits.")

    test_actors = actors[:test_count]
    validation_actors = actors[test_count:test_count + validation_count]
    train_actors = actors[test_count + validation_count:]

    train_indices = np.flatnonzero(np.isin(groups, train_actors))
    validation_indices = np.flatnonzero(np.isin(groups, validation_actors))
    test_indices = np.flatnonzero(np.isin(groups, test_actors))
    return (
        train_indices,
        validation_indices,
        test_indices,
        train_actors,
        validation_actors,
        test_actors,
    )
