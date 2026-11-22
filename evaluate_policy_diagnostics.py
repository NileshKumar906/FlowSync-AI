
import os
import csv
from collections import Counter

import numpy as np

from simulation.traffic_env import TrafficEnv
from dqn_agent import DQNAgent


# ============================================================
# CONFIGURATION
# ============================================================

EPISODES = 50
MAX_STEPS = 600
BASE_SEED = 1000
EMERGENCY_PROBABILITY = 0.015

MODEL_CONFIGS = {
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

OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL LOADING
# ============================================================

def load_agents():

    agents = {}

    for name, config in MODEL_CONFIGS.items():

        agent = DQNAgent(
            state_size=13,
            action_size=2,
            double_dqn=config["double_dqn"],
        )

        agent.load(config["path"])
        agent.epsilon = 0.0

        agents[name] = agent

        print(f"Loaded {name}: {config['path']}")

    return agents


# ============================================================
# ACTION SELECTION
# ============================================================

def select_action(agent, state):

    method_names = [
        "choose_action",
        "select_action",
        "act",
        "get_action",
    ]

    for method_name in method_names:

        method = getattr(agent, method_name, None)

        if callable(method):

            action = method(state)

            if hasattr(action, "item"):
                action = action.item()

            return int(action)

    raise AttributeError(
        "Could not find an action-selection method "
        "in DQNAgent. Checked: "
        + ", ".join(method_names)
    )


# ============================================================
# METRIC HELPERS
# ============================================================

def get_ordinary_waiting_times(env):

    all_waits = Counter(env.departure_waits)

    emergency_waits = Counter(
        env.emergency_departure_waits
    )

    ordinary_waits = all_waits - emergency_waits

    result = []

    for waiting_time, count in ordinary_waits.items():
        result.extend([waiting_time] * count)

    return result


def get_all_emergency_waits(env):

    departed = list(env.emergency_departure_waits)

    remaining = env.get_all_emergency_waiting_times()

    return departed + remaining


# ============================================================
# RUN ONE INDEPENDENT EPISODE
# ============================================================

def run_episode(agent, seed):

    env = TrafficEnv(
        max_steps=MAX_STEPS,
        emergency_arrival_probability=EMERGENCY_PROBABILITY,
    )

    state, _ = env.reset(seed=seed)

    action_counts = {
        0: 0,
        1: 0,
    }

    # Action 1 requests that the current phase switch.
    switch_requests_eligible = 0

    # Requests made before minimum green time.
    blocked_by_min_green = 0

    # Actions received while yellow is active.
    actions_during_yellow = 0

    # Actual completed phase transitions.
    actual_phase_switches = 0

    forced_switches = 0

    maximum_queue = 0

    action_history = []
    phase_history = []

    while True:

        action = select_action(agent, state)

        action_counts[action] += 1

        action_history.append(action)

        previous_phase = env.phase
        previous_yellow = env.yellow_remaining
        previous_green_elapsed = env.green_elapsed

        # Record how the action is handled by the simulator.
        if previous_yellow > 0:

            actions_during_yellow += 1

        elif action == 1:

            if previous_green_elapsed >= env.min_green:

                switch_requests_eligible += 1

            else:

                blocked_by_min_green += 1

        # Record whether a forced switch is due.
        if (
            previous_yellow == 0
            and previous_green_elapsed >= env.max_green
        ):

            forced_switches += 1

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        # Count completed phase changes, not just requests.
        if env.phase != previous_phase:

            actual_phase_switches += 1

        phase_history.append(env.phase)

        maximum_queue = max(
            maximum_queue,
            info["total_queue"],
        )

        state = next_state

        if terminated or truncated:
            break

    # --------------------------------------------------------
    # Calculate episode metrics
    # --------------------------------------------------------

    ordinary_waits = get_ordinary_waiting_times(env)

    emergency_waits = get_all_emergency_waits(env)

    average_ordinary_wait = (
        float(np.mean(ordinary_waits))
        if ordinary_waits else 0.0
    )

    average_emergency_wait = (
        float(np.mean(emergency_waits))
        if emergency_waits else 0.0
    )

    average_emergency_departed_wait = (
        float(np.mean(env.emergency_departure_waits))
        if env.emergency_departure_waits else 0.0
    )

    average_queue = env.queue_sum / max(env.time, 1)

    emergency_service_rate = (
        env.total_emergency_departures
        / max(env.total_emergency_arrivals, 1)
        * 100
    )

    result = {

        "seed": seed,

        "average_queue": average_queue,

        "maximum_queue": maximum_queue,

        "average_ordinary_waiting_time":
            average_ordinary_wait,

        "average_emergency_wait_all":
            average_emergency_wait,

        "average_emergency_departed_wait":
            average_emergency_departed_wait,

        "throughput": env.total_departures,

        "total_arrivals": env.total_arrivals,

        "total_departures": env.total_departures,

        "remaining_vehicles": env.get_total_queue(),

        "emergency_arrivals":
            env.total_emergency_arrivals,

        "emergency_departures":
            env.total_emergency_departures,

        "emergency_remaining":
            env.get_total_emergency_queue(),

        "emergency_service_rate":
            emergency_service_rate,

        "action_0_count": action_counts[0],

        "action_1_count": action_counts[1],

        "action_0_percentage":
            action_counts[0] / max(env.time, 1) * 100,

        "action_1_percentage":
            action_counts[1] / max(env.time, 1) * 100,

        "switch_requests_eligible":
            switch_requests_eligible,

        "blocked_by_min_green":
            blocked_by_min_green,

        "actions_during_yellow":
            actions_during_yellow,

        "actual_phase_switches":
            actual_phase_switches,

        "forced_switches":
            forced_switches,

        "ordinary_arrival_history":
            list(env.ordinary_arrival_history),

        "emergency_opportunity_history":
            list(env.emergency_opportunity_history),

        "emergency_arrival_history":
            list(env.emergency_arrival_history),

        "action_history":
            action_history,

        "phase_history":
            phase_history,
    }

    env.close()

    return result


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_all_models(agents):

    all_results = {}

    reference_schedules = {}

    for name, agent in agents.items():

        print("\n" + "=" * 65)
        print(f"EVALUATING {name}")
        print("=" * 65)

        model_results = []

        for episode in range(EPISODES):

            seed = BASE_SEED + episode

            result = run_episode(agent, seed)

            # Check that every model receives identical
            # ordinary and emergency traffic schedules.
            schedule = (
                result["ordinary_arrival_history"],
                result["emergency_opportunity_history"],
                result["emergency_arrival_history"],
            )

            if seed not in reference_schedules:

                reference_schedules[seed] = schedule

            else:

                if schedule != reference_schedules[seed]:

                    raise AssertionError(
                        f"Traffic schedule mismatch for seed {seed}, "
                        f"controller {name}"
                    )

            # Remove detailed histories from the metric row.
            metric_result = {
                key: value
                for key, value in result.items()
                if key not in (
                    "ordinary_arrival_history",
                    "emergency_opportunity_history",
                    "emergency_arrival_history",
                    "action_history",
                    "phase_history",
                )
            }

            metric_result["controller"] = name

            model_results.append(metric_result)

            print(
                f"Episode {episode + 1:02d}/{EPISODES} | "
                f"Queue: {result['average_queue']:.2f} | "
                f"Throughput: {result['throughput']} | "
                f"Switches: {result['actual_phase_switches']}"
            )

        all_results[name] = model_results

    print("\nTraffic schedule verification passed.")

    return all_results


# ============================================================
# SAVE PER-EPISODE RESULTS
# ============================================================

def save_episode_results(all_results):

    output_path = os.path.join(
        OUTPUT_DIR,
        "independent_policy_episode_results.csv",
    )

    rows = []

    for model_results in all_results.values():

        rows.extend(model_results)

    fieldnames = list(rows[0].keys())

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved episode results: {output_path}")


# ============================================================
# SAVE ACTION AND SWITCH DIAGNOSTICS
# ============================================================

def save_action_summary(all_results):

    diagnostic_metrics = [

        "action_0_percentage",
        "action_1_percentage",

        "switch_requests_eligible",
        "blocked_by_min_green",
        "actions_during_yellow",

        "actual_phase_switches",
        "forced_switches",
    ]

    output_path = os.path.join(
        OUTPUT_DIR,
        "independent_policy_action_summary.csv",
    )

    rows = []

    for name, model_results in all_results.items():

        row = {
            "controller": name,
        }

        for metric in diagnostic_metrics:

            values = [
                result[metric]
                for result in model_results
            ]

            row[f"{metric}_mean"] = float(np.mean(values))

            row[f"{metric}_std"] = float(np.std(values, ddof=1))

        rows.append(row)

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved action summary: {output_path}")


# ============================================================
# PRINT PERFORMANCE SUMMARY
# ============================================================

def print_performance_summary(all_results):

    metrics = [
        "average_queue",
        "maximum_queue",
        "average_ordinary_waiting_time",
        "average_emergency_wait_all",
        "throughput",
        "emergency_service_rate",
        "actual_phase_switches",
    ]

    print("\n" + "=" * 85)
    print("INDEPENDENT POLICY EVALUATION SUMMARY")
    print("=" * 85)

    for name, model_results in all_results.items():

        print(f"\n{name}")

        for metric in metrics:

            values = [
                result[metric]
                for result in model_results
            ]

            print(
                f"{metric:35s}: "
                f"{np.mean(values):.3f} "
                f"+/- {np.std(values, ddof=1):.3f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading trained agents...")

    agents = load_agents()

    print("\nStarting independent evaluation...")

    all_results = evaluate_all_models(agents)

    save_episode_results(all_results)

    save_action_summary(all_results)

    print_performance_summary(all_results)

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
