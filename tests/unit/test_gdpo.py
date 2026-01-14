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


def test_gdpo_reward_weights(tokenizer):
    """Test that reward weights are correctly applied in GDPO."""

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

    # Test without weights (equal weighting)
    results_no_weights = list(
        tokenize_trajectory_groups(
            tokenizer,
            [group],
            allow_training_without_logprobs=True,
            scale_rewards=True,
            reward_weights=None,
        )
    )

    # Test with weights that emphasize correctness (2.0) over format (0.5)
    results_weighted = list(
        tokenize_trajectory_groups(
            tokenizer,
            [group],
            allow_training_without_logprobs=True,
            scale_rewards=True,
            reward_weights={"correctness": 2.0, "format": 0.5},
        )
    )

    # Both should have 2 results
    assert len(results_no_weights) == 2
    assert len(results_weighted) == 2

    # Without weights: each component contributes equally
    # correctness: (1.0-0.5)/0.5 = 1.0 for A, -1.0 for B
    # format: (1.0-0.5)/0.5 = 1.0 for A, -1.0 for B
    # Total: A=2.0, B=-2.0
    no_weight_advantages = sorted([r.advantage for r in results_no_weights])
    assert math.isclose(no_weight_advantages[0], -2.0, abs_tol=0.01)
    assert math.isclose(no_weight_advantages[1], 2.0, abs_tol=0.01)

    # With weights: correctness*2.0 + format*0.5
    # correctness: (1.0-0.5)/0.5 * 2.0 = 2.0 for A, -2.0 for B
    # format: (1.0-0.5)/0.5 * 0.5 = 0.5 for A, -0.5 for B
    # Total: A=2.5, B=-2.5
    weighted_advantages = sorted([r.advantage for r in results_weighted])
    assert math.isclose(weighted_advantages[0], -2.5, abs_tol=0.01)
    assert math.isclose(weighted_advantages[1], 2.5, abs_tol=0.01)


def test_gdpo_sets_is_gdpo_flag(tokenizer):
    """Test that GDPO results have is_gdpo=True."""

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
        reward=0.0,
        rewards={"correctness": 0.0},
    )

    group = TrajectoryGroup([trajectory_a, trajectory_b])

    # GDPO mode
    gdpo_results = list(
        tokenize_trajectory_groups(
            tokenizer,
            [group],
            allow_training_without_logprobs=True,
            scale_rewards=False,
        )
    )

    # All GDPO results should have is_gdpo=True
    for result in gdpo_results:
        assert result.is_gdpo is True, "GDPO results should have is_gdpo=True"


def test_grpo_sets_is_gdpo_flag_false(tokenizer):
    """Test that GRPO results have is_gdpo=False."""

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
        # No rewards dict - GRPO mode
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
    )

    group = TrajectoryGroup([trajectory_a, trajectory_b])

    # GRPO mode
    grpo_results = list(
        tokenize_trajectory_groups(
            tokenizer,
            [group],
            allow_training_without_logprobs=True,
            scale_rewards=False,
        )
    )

    # All GRPO results should have is_gdpo=False
    for result in grpo_results:
        assert result.is_gdpo is False, "GRPO results should have is_gdpo=False"


def test_packed_tensors_preserves_is_gdpo():
    """Test that is_gdpo flag is preserved through packing."""
    from art.preprocessing.pack import packed_tensors_from_tokenized_results
    from art.preprocessing.tokenize import TokenizedResult

    # Create mock tokenized results with is_gdpo=True
    mock_results = [
        TokenizedResult(
            advantage=1.0,
            chat="test",
            tokens=["a", "b", "c"],
            token_ids=[1, 2, 3],
            input_pos=[0, 1, 2],
            assistant_mask=[0, 1, 1],
            logprobs=[float("nan"), 0.1, 0.2],
            pixel_values=None,
            image_grid_thw=None,
            weight=1.0,
            prompt_id=1,
            prompt_length=1,
            is_gdpo=True,
        ),
        TokenizedResult(
            advantage=-1.0,
            chat="test",
            tokens=["a", "b", "c"],
            token_ids=[1, 2, 3],
            input_pos=[0, 1, 2],
            assistant_mask=[0, 1, 1],
            logprobs=[float("nan"), 0.1, 0.2],
            pixel_values=None,
            image_grid_thw=None,
            weight=1.0,
            prompt_id=2,
            prompt_length=1,
            is_gdpo=True,
        ),
    ]

    packed = packed_tensors_from_tokenized_results(
        mock_results,
        seq_len=8,
        pad_token_id=-100,
    )

    assert packed["is_gdpo"] is True, (
        "is_gdpo should be True when any result has is_gdpo=True"
    )


def test_packed_tensors_is_gdpo_false_for_grpo():
    """Test that is_gdpo is False when all results are GRPO."""
    from art.preprocessing.pack import packed_tensors_from_tokenized_results
    from art.preprocessing.tokenize import TokenizedResult

    # Create mock tokenized results with is_gdpo=False (GRPO)
    mock_results = [
        TokenizedResult(
            advantage=1.0,
            chat="test",
            tokens=["a", "b", "c"],
            token_ids=[1, 2, 3],
            input_pos=[0, 1, 2],
            assistant_mask=[0, 1, 1],
            logprobs=[float("nan"), 0.1, 0.2],
            pixel_values=None,
            image_grid_thw=None,
            weight=1.0,
            prompt_id=1,
            prompt_length=1,
            is_gdpo=False,
        ),
    ]

    packed = packed_tensors_from_tokenized_results(
        mock_results,
        seq_len=8,
        pad_token_id=-100,
    )

    assert packed["is_gdpo"] is False, "is_gdpo should be False for GRPO results"
