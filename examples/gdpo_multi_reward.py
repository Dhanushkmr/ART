"""
GDPO Multi-Reward Example

This example demonstrates GDPO (Group reward-Decoupled normalization Policy Optimization)
by training a model to generate responses optimized for multiple objectives:
- Correctness: Does it answer the question correctly?
- Conciseness: Is it the appropriate length?
- Formatting: Does it follow the required format?

With GDPO, each reward component is normalized separately to prevent reward collapse.
"""

import asyncio
import re

from dotenv import load_dotenv
from openai import AsyncOpenAI

import art
from art.local import LocalBackend

load_dotenv()

MODEL_NAME = "gdpo-demo"
BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
TRAINING_STEPS = 100

# Sample questions with expected answers
QA_PAIRS = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "topic": "geography",
    },
    {
        "question": "What is 15 + 27?",
        "answer": "42",
        "topic": "math",
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "answer": "William Shakespeare",
        "topic": "literature",
    },
    {
        "question": "What is the chemical symbol for gold?",
        "answer": "Au",
        "topic": "chemistry",
    },
]


def evaluate_correctness(response: str, expected_answer: str) -> float:
    """Check if the response contains the correct answer."""
    response_lower = response.lower()
    answer_lower = expected_answer.lower()
    if answer_lower in response_lower:
        return 1.0
    # Partial credit for partial matches
    words = answer_lower.split()
    matches = sum(1 for word in words if word in response_lower)
    return matches / len(words) if words else 0.0


def evaluate_conciseness(response: str) -> float:
    """Reward responses that are concise (20-100 characters)."""
    length = len(response)
    if 20 <= length <= 100:
        return 1.0
    elif length < 20:
        return length / 20.0
    else:  # length > 100
        penalty = (length - 100) / 100.0
        return max(0.0, 1.0 - penalty * 0.5)


def evaluate_formatting(response: str, topic: str) -> float:
    """Reward responses that follow the format: 'Answer: <answer> (Topic: <topic>)'"""
    score = 0.0
    # Check for "Answer:" prefix
    if response.strip().startswith("Answer:"):
        score += 0.5
    # Check for topic mention in parentheses
    if re.search(rf"\(Topic:\s*{topic}\)", response, re.IGNORECASE):
        score += 0.5
    return score


async def generate_trajectory(
    client: AsyncOpenAI,
    model_name: str,
    qa_pair: dict,
) -> art.Trajectory:
    """Generate a trajectory with multi-reward evaluation."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions concisely in the format: 'Answer: <your answer> (Topic: <topic>)'",
        },
        {
            "role": "user",
            "content": qa_pair["question"],
        },
    ]

    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=100,
    )

    response_text = response.choices[0].message.content or ""

    # Evaluate multiple reward components
    correctness = evaluate_correctness(response_text, qa_pair["answer"])
    conciseness = evaluate_conciseness(response_text)
    formatting = evaluate_formatting(response_text, qa_pair["topic"])

    # Total reward is sum of components
    total_reward = correctness + conciseness + formatting

    return art.Trajectory(
        messages_and_choices=[*messages, response.choices[0]],
        reward=total_reward,
        rewards={
            "correctness": correctness,
            "conciseness": conciseness,
            "formatting": formatting,
        },
        metrics={
            "correctness": correctness,
            "conciseness": conciseness,
            "formatting": formatting,
            "response_length": len(response_text),
        },
    )


async def main():
    # Initialize model
    model = art.TrainableModel(
        name=MODEL_NAME,
        project="gdpo-multi-reward",
        base_model=BASE_MODEL,
    )
    await model.register(LocalBackend())
    client = model.openai_client()

    print("Starting GDPO multi-reward training...")
    print(f"Training for {TRAINING_STEPS} steps")
    print("\nReward components:")
    print("  - Correctness: Answer matches expected")
    print("  - Conciseness: Response is 20-100 characters")
    print("  - Formatting: Follows 'Answer: <answer> (Topic: <topic>)' format")
    print(
        "\nGDPO will normalize each component separately to preserve reward signals.\n"
    )

    for step in range(await model.get_step(), TRAINING_STEPS):
        # Generate trajectories for each question
        trajectory_groups = []
        for qa_pair in QA_PAIRS:
            # Generate multiple trajectories per question for group comparison
            trajectories = await asyncio.gather(
                *[generate_trajectory(client, MODEL_NAME, qa_pair) for _ in range(4)]
            )
            trajectory_groups.append(art.TrajectoryGroup(trajectories))

        # Train with GDPO (automatically activated by rewards dict)
        await model.train(
            trajectory_groups,
            config=art.TrainConfig(
                learning_rate=5e-5,
                scale_rewards=True,  # Scale each reward component by its std
            ),
        )

        if step % 10 == 0:
            print(f"Step {step}/{TRAINING_STEPS} completed")

            # Sample a trajectory to show progress
            sample_trajectory = await generate_trajectory(
                client, MODEL_NAME, QA_PAIRS[0]
            )
            print(f"Sample response: {sample_trajectory.messages()[2]['content']}")
            print(f"Rewards: {sample_trajectory.rewards}")
            print()

    print("Training completed!")

    # Final evaluation
    print("\nFinal Evaluation:")
    for qa_pair in QA_PAIRS:
        trajectory = await generate_trajectory(client, MODEL_NAME, qa_pair)
        response = trajectory.messages()[2]["content"]
        print(f"\nQ: {qa_pair['question']}")
        print(f"A: {response}")
        print(f"Rewards: {trajectory.rewards}")
        print(f"Total: {trajectory.reward:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
