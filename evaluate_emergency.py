
import os
import csv
from collections import Counter

import numpy as np

from simulation.traffic_env import TrafficEnv
from dqn_agent import DQNAgent


# ============================================================
# 1. EVALUATION CONFIGURATION
# ============================================================

MODEL_V1_PATH = "models/dqn_traffic_emergency.pth"
MODEL_V2_PATH = "models/dqn_traffic_emergency_v2.pth"
MODEL_DOUBLE_DQN_PATH = "models/double_dqn_traffic_emergency.pth"
MODEL_ADAPTIVE_PATH = "models/dqn_adaptive_v1.pth"

RESULTS_DIR = "evaluation_results"

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "emergency_evaluation_adaptive.csv"
)

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "controller_summary_adaptive.csv"
)

COMPARISON_PATH = os.path.join(
    RESULTS_DIR,
    "paired_comparison_adaptive.csv"
)

EPISODES = 50
MAX_STEPS = 600

BASE_SEED = 1000

FIXED_GREEN_DURATION = 20

EMERGENCY_ARRIVAL_PROBABILITY = 0.015

BOOTSTRAP_ITERATIONS = 10000

STATE_SIZE = 13
ACTION_SIZE = 2

SWITCH_PENALTY = 0.20

DQN_CONTROLLER_NAMES = {
    "DQN v1",
    "DQN v2",
    "Double DQN",
    "Adaptive DQN"
}


# ============================================================
# 2. TRAFFIC SCENARIOS
# ============================================================

TRAFFIC_SCENARIOS = {
    "balanced": {
        "North": 0.22,
        "South": 0.22,
        "East": 0.22,
        "West": 0.22
    },

    "north_south_heavy": {
        "North": 0.40,
        "South": 0.35,
        "East": 0.10,
        "West": 0.10
    },

    "east_west_heavy": {
        "North": 0.10,
        "South": 0.10,
        "East": 0.40,
        "West": 0.35
    },

    "uneven": {
        "North": 0.35,
        "South": 0.15,
        "East": 0.25,
        "West": 0.10
    }
}


# ============================================================
# 3. CONTROLLER 1: FIXED-TIME
# ============================================================

class FixedTimeController:

    def __init__(self, green_duration=20):
        self.green_duration = green_duration

    def select_action(self, env):

        if env.yellow_remaining > 0:
            return 0

        if env.green_elapsed >= self.green_duration:
            return 1

        return 0


# ============================================================
# 4. CONTROLLER 2: VEHICLE-ACTUATED
# ============================================================

class VehicleActuatedController:

    def __init__(self, switching_threshold=1.0):
        self.switching_threshold = switching_threshold

    def select_action(self, env):

        if env.yellow_remaining > 0:
            return 0

        if env.green_elapsed < env.min_green:
            return 0

        green_queue = sum(
            len(env.queues[direction])
            for direction in env.get_green_directions()
        )

        red_queue = sum(
            len(env.queues[direction])
            for direction in env.get_red_directions()
        )

        if red_queue > green_queue * self.switching_threshold:
            return 1

        return 0


# ============================================================
# 5. LOAD DQN MODELS
# ============================================================

def load_dqn_agent(model_path, double_dqn=False):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\nTrained model not found: {model_path}\n"
            "Train the model before running evaluation."
        )

    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        epsilon=0.0,
        double_dqn=double_dqn
    )

    agent.load(model_path)

    # Disable exploration during evaluation.
    agent.epsilon = 0.0

    agent.policy_net.eval()

    return agent


# ============================================================
# 6. ORDINARY VEHICLE WAITING TIMES
# ============================================================

def get_ordinary_departure_waits(env):

    all_waits = Counter(env.departure_waits)

    emergency_waits = Counter(
        env.emergency_departure_waits
    )

    ordinary_waits = all_waits - emergency_waits

    return list(ordinary_waits.elements())


# ============================================================
# 7. RUN ONE SIMULATION
# ============================================================

def run_episode(
    controller_name,
    controller,
    seed,
    scenario_name,
    arrival_rates
):

    env = TrafficEnv(
        max_steps=MAX_STEPS,
        emergency_arrival_probability=(
            EMERGENCY_ARRIVAL_PROBABILITY
        ),
        switch_penalty=SWITCH_PENALTY
    )

    # Apply the same traffic distribution to every
    # controller evaluated in this scenario.
    env.arrival_rates = arrival_rates.copy()

    try:

        state, info = env.reset(seed=seed)

        max_queue = 0

        # ------------------------------------------------
        # SIMULATION LOOP
        # ------------------------------------------------

        for step in range(MAX_STEPS):

            if controller_name in DQN_CONTROLLER_NAMES:
                action = controller.select_action(state)

            else:
                action = controller.select_action(env)

            (
                next_state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            state = next_state

            max_queue = max(
                max_queue,
                info["total_queue"]
            )

            if terminated or truncated:
                break

        # =================================================
        # CALCULATE PERFORMANCE METRICS
        # =================================================

        total_steps = max(env.time, 1)

        # ------------------------------------------------
        # Average queue
        # ------------------------------------------------

        average_queue = env.queue_sum / total_steps

        # ------------------------------------------------
        # Ordinary vehicle waiting time
        # ------------------------------------------------

        ordinary_waits = get_ordinary_departure_waits(env)

        average_ordinary_wait = (
            float(np.mean(ordinary_waits))
            if ordinary_waits
            else 0.0
        )

        # ------------------------------------------------
        # Emergency vehicle waiting time
        # ------------------------------------------------

        emergency_departed_waits = list(
            env.emergency_departure_waits
        )

        remaining_emergency_waits = list(
            env.get_all_emergency_waiting_times()
        )

        average_emergency_departed_wait = (
            float(np.mean(emergency_departed_waits))
            if emergency_departed_waits
            else 0.0
        )

        # Include departed and still-waiting emergencies.
        all_emergency_waits = (
            emergency_departed_waits
            + remaining_emergency_waits
        )

        average_emergency_wait_all = (
            float(np.mean(all_emergency_waits))
            if all_emergency_waits
            else 0.0
        )

        # ------------------------------------------------
        # Emergency service rate
        # ------------------------------------------------

        emergency_arrivals = env.total_emergency_arrivals

        emergency_departures = env.total_emergency_departures

        emergency_service_rate = (
            emergency_departures
            / emergency_arrivals
            * 100.0
            if emergency_arrivals > 0
            else 0.0
        )

        # ------------------------------------------------
        # Remaining vehicles
        # ------------------------------------------------

        remaining_emergency = env.get_total_emergency_queue()

        remaining_vehicles = env.get_total_queue()

        # =================================================
        # RETURN RESULTS
        # =================================================

        return {

            "controller": controller_name,

            "seed": seed,

            "traffic_scenario": scenario_name,

            "average_queue": average_queue,

            "maximum_queue": max_queue,

            "average_ordinary_waiting_time": (
                average_ordinary_wait
            ),

            "average_emergency_departed_wait": (
                average_emergency_departed_wait
            ),

            "average_emergency_wait_all": (
                average_emergency_wait_all
            ),

            "throughput": env.total_departures,

            "total_arrivals": env.total_arrivals,

            "total_departures": env.total_departures,

            "remaining_vehicles": remaining_vehicles,

            "emergency_arrivals": emergency_arrivals,

            "emergency_departures": emergency_departures,

            "emergency_remaining": remaining_emergency,

            "emergency_service_rate": emergency_service_rate,

            "remaining_emergency_average_wait": (
                float(np.mean(remaining_emergency_waits))
                if remaining_emergency_waits
                else 0.0
            ),

            "total_emergency_wait_across_arrivals": (
                float(sum(all_emergency_waits))
            ),

            # Internal histories for fairness verification.

            "_ordinary_history": list(
                env.ordinary_arrival_history
            ),

            "_emergency_opportunity_history": list(
                env.emergency_opportunity_history
            ),

            "_emergency_arrival_history": list(
                env.emergency_arrival_history
            )
        }

    finally:
        env.close()


# ============================================================
# 8. VERIFY IDENTICAL TRAFFIC SCENARIOS
# ============================================================

def verify_fairness(scenario_results, seed):

    reference = scenario_results[0]

    for result in scenario_results[1:]:

        if result["traffic_scenario"] != reference["traffic_scenario"]:
            raise AssertionError(
                f"Traffic scenario mismatch for seed {seed}"
            )

        # Ordinary traffic schedules.
        if (
            result["_ordinary_history"]
            != reference["_ordinary_history"]
        ):
            raise AssertionError(
                f"Ordinary traffic schedules differ "
                f"for seed {seed}"
            )

        # Emergency opportunities.
        if (
            result["_emergency_opportunity_history"]
            != reference["_emergency_opportunity_history"]
        ):
            raise AssertionError(
                f"Emergency opportunity schedules differ "
                f"for seed {seed}"
            )

        # Actual emergency arrivals.
        if (
            result["_emergency_arrival_history"]
            != reference["_emergency_arrival_history"]
        ):
            raise AssertionError(
                f"Emergency arrival schedules differ "
                f"for seed {seed}"
            )

    print(
        "Fairness check: PASSED "
        "(identical traffic schedules)"
    )


# ============================================================
# 9. PAIRED STATISTICAL COMPARISON
# ============================================================

def calculate_paired_comparison(results):

    metrics = {
        "average_queue": "lower",
        "average_ordinary_waiting_time": "lower",
        "average_emergency_wait_all": "lower",
        "throughput": "higher"
    }

    comparisons_to_run = [

        (
            "DQN-v1-vs-Actuated",
            "DQN v1",
            "Vehicle-Actuated"
        ),

        (
            "DQN-v2-vs-Actuated",
            "DQN v2",
            "Vehicle-Actuated"
        ),

        (
            "Double-DQN-vs-Actuated",
            "Double DQN",
            "Vehicle-Actuated"
        ),

        (
            "Adaptive-DQN-vs-Actuated",
            "Adaptive DQN",
            "Vehicle-Actuated"
        ),

        (
            "Adaptive-DQN-vs-Double-DQN",
            "Adaptive DQN",
            "Double DQN"
        )
    ]

    rng = np.random.default_rng(42)

    comparisons = []

    # Index results by controller and seed.
    result_lookup = {
        (result["controller"], result["seed"]): result
        for result in results
    }

    expected_seeds = [
        BASE_SEED + episode
        for episode in range(1, EPISODES + 1)
    ]

    for comparison_name, first_name, second_name in comparisons_to_run:

        for metric, direction in metrics.items():

            differences = []

            # --------------------------------------------
            # Paired differences
            # --------------------------------------------

            for seed in expected_seeds:

                first = result_lookup[
                    (first_name, seed)
                ]

                second = result_lookup[
                    (second_name, seed)
                ]

                differences.append(
                    first[metric] - second[metric]
                )

            differences = np.asarray(
                differences,
                dtype=float
            )

            # --------------------------------------------
            # Wins and losses
            # --------------------------------------------

            if direction == "higher":

                wins = np.sum(differences > 0)
                losses = np.sum(differences < 0)

            else:

                wins = np.sum(differences < 0)
                losses = np.sum(differences > 0)

            ties = len(differences) - wins - losses

            # --------------------------------------------
            # Paired bootstrap confidence interval
            # --------------------------------------------

            bootstrap_means = np.empty(
                BOOTSTRAP_ITERATIONS
            )

            for i in range(BOOTSTRAP_ITERATIONS):

                sample = rng.choice(
                    differences,
                    size=len(differences),
                    replace=True
                )

                bootstrap_means[i] = np.mean(sample)

            ci_lower, ci_upper = np.percentile(
                bootstrap_means,
                [2.5, 97.5]
            )

            # --------------------------------------------
            # Store comparison
            # --------------------------------------------

            comparisons.append({

                "comparison": comparison_name,

                "first_controller": first_name,

                "second_controller": second_name,

                "metric": metric,

                "mean_difference": float(
                    np.mean(differences)
                ),

                "standard_deviation": float(
                    np.std(differences, ddof=1)
                ),

                "ci_95_lower": float(ci_lower),

                "ci_95_upper": float(ci_upper),

                "first_controller_wins": int(wins),

                "second_controller_wins": int(losses),

                "ties": int(ties),

                "scenarios": len(differences)

            })

    return comparisons


# ============================================================
# 10. AGGREGATE RESULTS
# ============================================================

def calculate_summary(results, controller_names):

    metric_names = [

        "average_queue",

        "maximum_queue",

        "average_ordinary_waiting_time",

        "average_emergency_departed_wait",

        "average_emergency_wait_all",

        "throughput",

        "total_arrivals",

        "total_departures",

        "remaining_vehicles",

        "emergency_arrivals",

        "emergency_departures",

        "emergency_remaining",

        "remaining_emergency_average_wait",

        "total_emergency_wait_across_arrivals"

    ]

    summary = {}

    for name in controller_names:

        controller_results = [
            result
            for result in results
            if result["controller"] == name
        ]

        summary[name] = {}

        for metric in metric_names:

            values = [
                result[metric]
                for result in controller_results
            ]

            summary[name][metric] = float(
                np.mean(values)
            )

            summary[name][metric + "_std"] = float(
                np.std(values, ddof=1)
            )

        # Pooled emergency service rate.

        total_emergency_arrivals = sum(
            result["emergency_arrivals"]
            for result in controller_results
        )

        total_emergency_departures = sum(
            result["emergency_departures"]
            for result in controller_results
        )

        summary[name]["emergency_service_rate"] = (
            total_emergency_departures
            / total_emergency_arrivals
            * 100.0
            if total_emergency_arrivals > 0
            else 0.0
        )

    return summary


# ============================================================
# 11. SCENARIO-WISE SUMMARY
# ============================================================

def calculate_scenario_summary(results, controller_names):

    metrics = [
        "average_queue",
        "average_ordinary_waiting_time",
        "average_emergency_wait_all",
        "throughput",
        "emergency_service_rate"
    ]

    scenario_summary = []

    for scenario_name in TRAFFIC_SCENARIOS:

        for controller_name in controller_names:

            selected = [
                result
                for result in results
                if (
                    result["traffic_scenario"] == scenario_name
                    and result["controller"] == controller_name
                )
            ]

            if not selected:
                continue

            row = {
                "traffic_scenario": scenario_name,
                "controller": controller_name,
                "scenario_count": len(selected)
            }

            for metric in metrics:

                values = [
                    result[metric]
                    for result in selected
                ]

                row[metric + "_mean"] = float(
                    np.mean(values)
                )

                row[metric + "_std"] = float(
                    np.std(values, ddof=1)
                ) if len(values) > 1 else 0.0

            scenario_summary.append(row)

    return scenario_summary


# ============================================================
# 12. SAVE RESULTS
# ============================================================

def write_csv(path, rows):

    if not rows:
        print(f"Warning: No data available for {path}")
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


def save_results(
    results,
    summary,
    comparisons,
    scenario_summary
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # Detailed evaluation CSV.

    public_results = [
        {
            key: value
            for key, value in result.items()
            if not key.startswith("_")
        }
        for result in results
    ]

    write_csv(CSV_PATH, public_results)

    # Controller summary CSV.

    summary_rows = []

    for name, metrics in summary.items():

        summary_rows.append({
            "controller": name,
            **metrics
        })

    write_csv(SUMMARY_PATH, summary_rows)

    # Paired comparison CSV.

    write_csv(COMPARISON_PATH, comparisons)

    # Scenario-wise summary CSV.

    scenario_path = os.path.join(
        RESULTS_DIR,
        "scenario_wise_summary_adaptive.csv"
    )

    write_csv(scenario_path, scenario_summary)

    return scenario_path


# ============================================================
# 13. PRINT FINAL COMPARISON
# ============================================================

def print_summary(summary, controller_names):

    print("\n" + "=" * 110)

    print("FINAL CONTROLLER COMPARISON")

    print("=" * 110)

    print(
        f"{'Controller':18}"
        f"{'Avg Queue':>12}"
        f"{'Ord. Wait':>12}"
        f"{'Throughput':>12}"
        f"{'Emerg. Wait':>14}"
        f"{'Service Rate':>15}"
    )

    print("-" * 110)

    for name in controller_names:

        data = summary[name]

        print(
            f"{name:18}"
            f"{data['average_queue']:12.2f}"
            f"{data['average_ordinary_waiting_time']:12.2f}"
            f"{data['throughput']:12.2f}"
            f"{data['average_emergency_wait_all']:14.2f}"
            f"{data['emergency_service_rate']:14.2f}%"
        )

    print("=" * 110)


def print_statistical_comparison(comparisons):

    print("\n" + "=" * 100)

    print("PAIRED STATISTICAL COMPARISONS")

    print("=" * 100)

    for item in comparisons:

        print(
            f"\n{item['comparison']} | {item['metric']}"
        )

        print(
            f"Mean difference: "
            f"{item['mean_difference']:.4f}"
        )

        print(
            f"Standard deviation: "
            f"{item['standard_deviation']:.4f}"
        )

        print(
            f"95% Bootstrap CI: "
            f"[{item['ci_95_lower']:.4f}, "
            f"{item['ci_95_upper']:.4f}]"
        )

        print(
            f"{item['first_controller']} wins: "
            f"{item['first_controller_wins']} | "
            f"{item['second_controller']} wins: "
            f"{item['second_controller_wins']} | "
            f"Ties: {item['ties']}"
        )

    print("\n" + "=" * 100)


def print_scenario_summary(scenario_summary):

    print("\n" + "=" * 100)
    print("TRAFFIC SCENARIO-WISE RESULTS")
    print("=" * 100)

    for row in scenario_summary:

        print(
            f"\n{row['traffic_scenario']} | "
            f"{row['controller']}"
        )

        print(
            f"Average Queue: "
            f"{row['average_queue_mean']:.2f}"
        )

        print(
            f"Ordinary Wait: "
            f"{row['average_ordinary_waiting_time_mean']:.2f}"
        )

        print(
            f"Emergency Wait: "
            f"{row['average_emergency_wait_all_mean']:.2f}"
        )

        print(
            f"Throughput: "
            f"{row['throughput_mean']:.2f}"
        )

    print("\n" + "=" * 100)


# ============================================================
# 14. MAIN EVALUATION
# ============================================================

def evaluate():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    print("\n" + "=" * 80)

    print("             FLOWSYNC AI - EVALUATION")

    print("             SIX CONTROLLER COMPARISON")

    print("=" * 80)

    print(f"Scenarios: {EPISODES}")

    print(f"Steps per scenario: {MAX_STEPS}")

    print(
        "Emergency arrival probability:",
        EMERGENCY_ARRIVAL_PROBABILITY
    )

    print("Traffic patterns:", len(TRAFFIC_SCENARIOS))

    print("Switch penalty:", SWITCH_PENALTY)

    # --------------------------------------------------------
    # Load DQN v1
    # --------------------------------------------------------

    print("\nLoading DQN v1...")

    dqn_v1 = load_dqn_agent(
        MODEL_V1_PATH,
        double_dqn=False
    )

    print("DQN v1 loaded successfully.")

    # --------------------------------------------------------
    # Load DQN v2
    # --------------------------------------------------------

    print("\nLoading DQN v2...")

    dqn_v2 = load_dqn_agent(
        MODEL_V2_PATH,
        double_dqn=False
    )

    print("DQN v2 loaded successfully.")

    # --------------------------------------------------------
    # Load Double DQN
    # --------------------------------------------------------

    print("\nLoading Double DQN...")

    double_dqn_agent = load_dqn_agent(
        MODEL_DOUBLE_DQN_PATH,
        double_dqn=True
    )

    print("Double DQN loaded successfully.")

    # --------------------------------------------------------
    # Load Adaptive DQN
    # --------------------------------------------------------

    print("\nLoading Adaptive DQN...")

    adaptive_dqn_agent = load_dqn_agent(
        MODEL_ADAPTIVE_PATH,
        double_dqn=False
    )

    print("Adaptive DQN loaded successfully.")

    # --------------------------------------------------------
    # Initialize controllers
    # --------------------------------------------------------

    controllers = {

        "Fixed-Time": FixedTimeController(
            green_duration=FIXED_GREEN_DURATION
        ),

        "Vehicle-Actuated": VehicleActuatedController(
            switching_threshold=1.0
        ),

        "DQN v1": dqn_v1,

        "DQN v2": dqn_v2,

        "Double DQN": double_dqn_agent,

        "Adaptive DQN": adaptive_dqn_agent

    }

    controller_names = list(controllers.keys())

    results = []

    scenario_names = list(TRAFFIC_SCENARIOS.keys())

    # ========================================================
    # RUN ALL SCENARIOS
    # ========================================================

    for episode in range(1, EPISODES + 1):

        seed = BASE_SEED + episode

        scenario_name = scenario_names[
            (episode - 1) % len(scenario_names)
        ]

        arrival_rates = TRAFFIC_SCENARIOS[scenario_name]

        print(
            f"\n{'=' * 25} "
            f"SCENARIO {episode}/{EPISODES} "
            f"| Seed: {seed} "
            f"| Traffic: {scenario_name} "
            f"{'=' * 25}"
        )

        scenario_results = []

        for controller_name, controller in controllers.items():

            result = run_episode(
                controller_name,
                controller,
                seed,
                scenario_name,
                arrival_rates
            )

            scenario_results.append(result)

            print(

                f"{controller_name:18} | "

                f"Avg Queue: "
                f"{result['average_queue']:6.2f} | "

                f"Ordinary Wait: "
                f"{result['average_ordinary_waiting_time']:6.2f}s | "

                f"Throughput: "
                f"{result['throughput']:5d} | "

                f"Emergency: "
                f"{result['emergency_departures']}/"
                f"{result['emergency_arrivals']} | "

                f"Emergency Wait: "
                f"{result['average_emergency_wait_all']:6.2f}s"

            )

        # ----------------------------------------------------
        # Verify identical traffic schedules
        # ----------------------------------------------------

        verify_fairness(
            scenario_results,
            seed
        )

        results.extend(scenario_results)

    # ========================================================
    # CALCULATE SUMMARY
    # ========================================================

    print("\nCalculating aggregate results...")

    summary = calculate_summary(
        results,
        controller_names
    )

    # ========================================================
    # SCENARIO-WISE SUMMARY
    # ========================================================

    print("\nCalculating traffic scenario-wise results...")

    scenario_summary = calculate_scenario_summary(
        results,
        controller_names
    )

    # ========================================================
    # STATISTICAL COMPARISON
    # ========================================================

    print("\nRunning paired bootstrap analysis...")

    comparisons = calculate_paired_comparison(
        results
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print_summary(
        summary,
        controller_names
    )

    print_statistical_comparison(
        comparisons
    )

    print_scenario_summary(
        scenario_summary
    )

    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    scenario_path = save_results(
        results,
        summary,
        comparisons,
        scenario_summary
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n" + "=" * 80)

    print("EVALUATION COMPLETED SUCCESSFULLY!")

    print("=" * 80)

    print("Fairness checks: PASSED")

    print("Controllers evaluated:", len(controllers))

    print("Scenarios evaluated:", EPISODES)

    print("\nGenerated files:")

    print("Detailed results:", CSV_PATH)

    print("Controller summary:", SUMMARY_PATH)

    print("Paired comparisons:", COMPARISON_PATH)

    print("Scenario-wise summary:", scenario_path)

    print("=" * 80)


# ============================================================
# 15. EXECUTION
# ============================================================

if __name__ == "__main__":

    evaluate()
