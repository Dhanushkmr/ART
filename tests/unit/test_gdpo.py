"""Tests for GDPO (Group reward-Decoupled normalization Policy Optimization)."""

import math

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
import pytest
from transformers import AutoTokenizer

from art.preprocessing.tokenize import tokenize_trajectory_groups
from art.trajectories import Trajectory, TrajectoryGroup


@pytest.fixture
def tokenizer():
    """Create a tokenizer for testing."""
    return AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct")


def test_gdpo_multi_reward_vs_grpo_single_reward(tokenizer):
    """Test that GDPO correctly handles multi-reward scenarios differently from GRPO."""

    # Create two trajectories with same total reward but different components
    # Trajectory A: High correctness, low format
    trajectory_a = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "What is 2+2?"},
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="4",
                    refusal=None,
                ),
            ),
        ],
        reward=1.8,  # Total reward
        rewards={
            "correctness": 1.0,
            "format": 0.8,
        },
    )

    # Trajectory B: Low correctness, high format
    trajectory_b = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "What is 2+2?"},
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="5",
                    refusal=None,
                ),
            ),
        ],
        reward=1.8,  # Same total reward
        rewards={
            "correctness": 0.8,
            "format": 1.0,
        },
    )

    # Create group for GDPO (multi-reward)
    gdpo_group = TrajectoryGroup([trajectory_a, trajectory_b])

    # Tokenize with GDPO
    gdpo_results = list(
        tokenize_trajectory_groups(
            tokenizer,
            [gdpo_group],
            allow_training_without_logprobs=True,
            scale_rewards=False,
        )
    )

    # With GDPO, both trajectories should have advantage of 0
    # because each component is normalized separately:
    # - correctness: mean=0.9, A gets +0.1, B gets -0.1
    # - format: mean=0.9, A gets -0.1, B gets +0.1
    # Total advantage: A: 0.1-0.1=0, B: -0.1+0.1=0
    # So both get skipped (advantage == 0)
    assert len(gdpo_results) == 0, (
        "Both trajectories should be skipped with 0 advantage"
    )


def test_gdpo_preserves_component_differences(tokenizer):
    """Test that GDPO preserves differences in reward components."""

    # Create trajectories with different reward patterns
    trajectory_high_correct = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "Test"},
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Response 1",
                    refusal=None,
                ),
            ),
        ],
        reward=2.0,
        rewards={
            "correctness": 1.0,
            "format": 0.5,
            "length": 0.5,
        },
    )

    trajectory_balanced = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "Test"},
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Response 2",
                    refusal=None,
                ),
            ),
        ],
        reward=1.5,
        rewards={
            "correctness": 0.5,
            "format": 0.5,
            "length": 0.5,
        },
    )

    trajectory_low = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "Test"},
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Response 3",
                    refusal=None,
                ),
            ),
        ],
        reward=1.0,
        rewards={
            "correctness": 0.0,
            "format": 0.5,
            "length": 0.5,
        },
    )

    group = TrajectoryGroup(
        [trajectory_high_correct, trajectory_balanced, trajectory_low]
    )

    results = list(
        tokenize_trajectory_groups(
            tokenizer,
            [group],
            allow_training_without_logprobs=True,
            scale_rewards=False,
        )
    )

    # Should have 2 trajectories (one with 0 advantage gets skipped)
    assert len(results) > 0, "Should have trajectories with non-zero advantages"


def test_grpo_single_reward_backward_compatibility(tokenizer):
    """Test that GRPO still works with single reward values."""

    trajectory_high = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "Test"},
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Good response",
                    refusal=None,
                ),
            ),
        ],
        reward=1.0,
        # No rewards dict - should use GRPO
    )

    trajectory_low = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "Test"},
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Bad response",
                    refusal=None,
                ),
            ),
        ],
        reward=0.0,
    )

    group = TrajectoryGroup([trajectory_high, trajectory_low])

    results = list(
        tokenize_trajectory_groups(
            tokenizer,
            [group],
            allow_training_without_logprobs=True,
            scale_rewards=False,
        )
    )

    # With GRPO, we should get 2 results with different advantages
    assert len(results) == 2, "Should have 2 trajectories with GRPO"

    # Check that advantages are opposite (group normalization)
    advantages = [r.advantage for r in results]
    assert math.isclose(sum(advantages), 0.0, abs_tol=1e-6), (
        "Advantages should sum to ~0"
    )
    assert advantages[0] != advantages[1], "Advantages should be different"


def test_gdpo_with_reward_scaling(tokenizer):
    """Test that reward scaling works with GDPO."""

    trajectory_a = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "Test"},
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="A",
                    refusal=None,
                ),
            ),
        ],
        reward=2.0,
        rewards={
            "correctness": 1.0,
            "format": 1.0,
        },
    )

    trajectory_b = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "Test"},
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="B",
                    refusal=None,
                ),
            ),
        ],
        reward=0.0,
        rewards={
            "correctness": 0.0,
            "format": 0.0,
        },
    )

    group = TrajectoryGroup([trajectory_a, trajectory_b])

    # Test with scale_rewards=True
    results = list(
        tokenize_trajectory_groups(
            tokenizer,
            [group],
            allow_training_without_logprobs=True,
            scale_rewards=True,
        )
    )

    # Should have 2 results
    assert len(results) == 2, "Should have 2 trajectories"

    # With scaling, each component is normalized by its std
    # Component means: correctness=0.5, format=0.5
    # Component stds: correctness=0.5, format=0.5
    # Trajectory A: (1.0-0.5)/0.5 + (1.0-0.5)/0.5 = 1.0 + 1.0 = 2.0
    # Trajectory B: (0.0-0.5)/0.5 + (0.0-0.5)/0.5 = -1.0 + -1.0 = -2.0
    advantages = sorted([r.advantage for r in results])
    assert math.isclose(advantages[0], -2.0, abs_tol=0.01), (
        "Low trajectory should have advantage ~-2.0"
    )
    assert math.isclose(advantages[1], 2.0, abs_tol=0.01), (
        "High trajectory should have advantage ~2.0"
    )


def test_gdpo_requires_consistent_components(tokenizer):
    """Test that all trajectories must have the same reward components."""

    trajectory_a = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "Test"},
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="A",
                    refusal=None,
                ),
            ),
        ],
        reward=1.0,
        rewards={"correctness": 1.0},
    )

    trajectory_b = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "Test"},
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="B",
                    refusal=None,
                ),
            ),
        ],
        reward=0.5,
        # Missing rewards dict - inconsistent with trajectory_a
    )

    group = TrajectoryGroup([trajectory_a, trajectory_b])

    # Should fall back to GRPO because not all trajectories have rewards dict
    results = list(
        tokenize_trajectory_groups(
            tokenizer,
            [group],
            allow_training_without_logprobs=True,
            scale_rewards=False,
        )
    )

    # Should still work with GRPO fallback
    assert len(results) == 2, "Should fall back to GRPO"
