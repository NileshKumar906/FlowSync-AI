
import numpy as np
import torch

from simulation.traffic_env import TrafficEnv
from dqn_agent import DQNAgent


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODELS = {
    "DQN v1": {
        "path": "models/dqn_traffic_emergency.pth",
        "double_dqn": False,
    },
    "DQN v2": {
        "path": "models/dqn_traffic_emergency_v2.pth",
        "double_dqn": False,
    },
    "Double DQN": {
        "path": "models/double_dqn_traffic_emergency.pth",
        "double_dqn": True,
    },
}

SEED = 2026
MAX_STEPS = 600
EMERGENCY_PROBABILITY = 0.015


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

def load_models():
    agents = {}

    for name, config in MODELS.items():

        agent = DQNAgent(
            state_size=13,
            action_size=2,
            double_dqn=config["double_dqn"],
        )

        agent.load(config["path"])

        # Disable exploration.
        agent.epsilon = 0.0

        agents[name] = agent

        print(f"{name} loaded: {config['path']}")

    return agents


# --------------------------------------------------
# SELECT ACTION COMPATIBLY
# --------------------------------------------------

def select_action(agent, state):
    """
    Detect the action-selection method supported
    by the existing DQNAgent implementation.
    """

    possible_methods = [
        "choose_action",
        "select_action",
        "act",
        "get_action",
    ]

    action_method = None

    for method_name in possible_methods:
        if callable(getattr(agent, method_name, None)):
            action_method = getattr(agent, method_name)
            break

    if action_method is None:
        raise AttributeError(
            "No supported action-selection method found in DQNAgent. "
            "Expected one of: choose_action, select_action, act, get_action."
        )

    # Try NumPy input first.
    action = action_method(state)

    # Convert tensor output to Python integer if necessary.
    if isinstance(action, torch.Tensor):
        action = action.item()

    return int(action)


# --------------------------------------------------
# COMPARE POLICIES
# --------------------------------------------------

def compare_policies():

    print("\nLoading trained models...")

    agents = load_models()

    print("\nCreating traffic environment...")

    env = TrafficEnv(
        max_steps=MAX_STEPS,
        emergency_arrival_probability=EMERGENCY_PROBABILITY,
    )

    state, info = env.reset(seed=SEED)

    action_counts = {
        name: {0: 0, 1: 0}
        for name in agents
    }

    comparisons = [
        ("DQN v1", "DQN v2"),
        ("DQN v2", "Double DQN"),
        ("DQN v1", "Double DQN"),
    ]

    disagreements = {
        f"{first} vs {second}": 0
        for first, second in comparisons
    }

    steps = 0

    print("\nRunning policy comparison...")

    while True:

        actions = {}

        # Each model receives exactly the same state.
        for name, agent in agents.items():

            action = select_action(agent, state)

            actions[name] = action

            action_counts[name][action] += 1

        # Compare decisions.
        for first, second in comparisons:

            if actions[first] != actions[second]:

                disagreements[
                    f"{first} vs {second}"
                ] += 1

        # Advance environment using DQN v1.
        state, reward, terminated, truncated, info = env.step(
            actions["DQN v1"]
        )

        steps += 1

        if terminated or truncated:
            break

    env.close()

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("POLICY ACTION DISTRIBUTION")
    print("=" * 55)

    for name, counts in action_counts.items():

        total = counts[0] + counts[1]

        print(f"\n{name}")

        print(
            f"Action 0: {counts[0]} "
            f"({counts[0] / total * 100:.2f}%)"
        )

        print(
            f"Action 1: {counts[1]} "
            f"({counts[1] / total * 100:.2f}%)"
        )

    print("\n" + "=" * 55)
    print("POLICY DISAGREEMENT")
    print("=" * 55)

    for comparison, count in disagreements.items():

        percentage = count / steps * 100

        print(
            f"{comparison}: "
            f"{count}/{steps} "
            f"({percentage:.2f}%)"
        )

    print("\nTotal evaluated steps:", steps)

    print("\nDiagnostic completed.")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    compare_policies()
