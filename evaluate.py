
import numpy as np

from simulation.traffic_env import TrafficEnv
from dqn_agent import DQNAgent


MAX_STEPS = 600
FIXED_GREEN_DURATION = 15
TEST_SEEDS = [101, 102, 103, 104, 105,
              106, 107, 108, 109, 110]


def run_episode(env, controller, seed):

    state, info = env.reset(seed=seed)

    total_reward = 0

    for _ in range(MAX_STEPS):

        # Select action based on controller
        if controller == "fixed":

            # Fixed-time control:
            # Switch after 15 seconds of green.
            if env.yellow_remaining > 0:
                action = 0

            elif env.green_elapsed >= FIXED_GREEN_DURATION:
                action = 1

            else:
                action = 0

        elif controller == "dqn":

            # Trained agent chooses the action
            action = agent.select_action(state)

        else:
            raise ValueError("Unknown controller")

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward
        state = next_state

        if terminated or truncated:
            break

    # Collect metrics
    average_queue = env.queue_sum / env.time

    average_waiting_time = (
        env.total_waiting_time / env.total_departures
        if env.total_departures > 0
        else 0
    )

    return {
        "reward": total_reward,
        "average_queue": average_queue,
        "average_waiting_time": average_waiting_time,
        "throughput": env.total_departures,
        "max_queue": max(
            (
                # The environment's queue_sum is cumulative.
                # Maximum total queue is tracked separately below.
                0,
            ),
            default=0
        ),
        "remaining_vehicles": env.get_total_queue()
    }


def run_episode_with_metrics(env, controller, seed):

    state, info = env.reset(seed=seed)

    total_reward = 0
    maximum_queue = 0

    for _ in range(MAX_STEPS):

        if controller == "fixed":

            if env.yellow_remaining > 0:
                action = 0

            elif env.green_elapsed >= FIXED_GREEN_DURATION:
                action = 1

            else:
                action = 0

        else:

            action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward

        maximum_queue = max(
            maximum_queue,
            info["total_queue"]
        )

        state = next_state

        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "average_queue": env.queue_sum / env.time,
        "average_waiting_time": (
            env.total_waiting_time / env.total_departures
            if env.total_departures > 0
            else 0
        ),
        "throughput": env.total_departures,
        "max_queue": maximum_queue,
        "remaining_vehicles": env.get_total_queue()
    }


def evaluate():

    global agent

    env = TrafficEnv(max_steps=MAX_STEPS)

    agent = DQNAgent()

    agent.load("models/dqn_traffic.pth")

    # Disable exploration during evaluation.
    # We want the agent to use its learned policy.
    agent.epsilon = 0

    results = {
        "Fixed-time": [],
        "DQN": []
    }

    print("\n========== FLOWSYNC AI EVALUATION ==========\n")

    for seed in TEST_SEEDS:

        fixed_result = run_episode_with_metrics(
            env,
            "fixed",
            seed
        )

        dqn_result = run_episode_with_metrics(
            env,
            "dqn",
            seed
        )

        results["Fixed-time"].append(fixed_result)
        results["DQN"].append(dqn_result)

        print(f"Scenario seed: {seed}")
        print(
            f"  Fixed-time average queue: "
            f"{fixed_result['average_queue']:.2f}"
        )
        print(
            f"  DQN average queue: "
            f"{dqn_result['average_queue']:.2f}"
        )

    print("\n========== FINAL COMPARISON ==========")

    metrics = [
        ("average_queue", "Average queue"),
        ("average_waiting_time", "Average waiting time"),
        ("throughput", "Throughput"),
        ("max_queue", "Maximum queue"),
        ("reward", "Total reward")
    ]

    for key, label in metrics:

        fixed_avg = np.mean([
            result[key]
            for result in results["Fixed-time"]
        ])

        dqn_avg = np.mean([
            result[key]
            for result in results["DQN"]
        ])

        print(f"\n{label}")
        print(f"  Fixed-time: {fixed_avg:.2f}")
        print(f"  DQN:        {dqn_avg:.2f}")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    evaluate()
