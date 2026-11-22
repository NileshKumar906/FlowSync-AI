
import csv
import random
from pathlib import Path

import numpy as np
import torch

from simulation.traffic_env import TrafficEnv
from dqn_agent import DQNAgent


# ======================================================
# 1. REPRODUCIBILITY
# ======================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ======================================================
# 2. TRAINING CONFIGURATION
# ======================================================

EPISODES = 200
MAX_STEPS = 600

WARMUP_STEPS = 500
TRAIN_FREQUENCY = 4
TARGET_UPDATE_FREQUENCY = 500

EMERGENCY_ARRIVAL_PROBABILITY = 0.015

SWITCH_PENALTY = 0.20

MODEL_PATH = "models/dqn_adaptive_v1.pth"

CHECKPOINT_DIR = Path(
    "models/adaptive_checkpoints"
)

CSV_PATH = (
    "training_results/adaptive_training.csv"
)

Path("models").mkdir(exist_ok=True)
Path("training_results").mkdir(exist_ok=True)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================
# 3. TRAFFIC DEMAND SCENARIOS
# ======================================================

TRAFFIC_SCENARIOS = [

    {
        "name": "balanced",
        "North": 0.22,
        "South": 0.22,
        "East": 0.22,
        "West": 0.22
    },

    {
        "name": "north_south_heavy",
        "North": 0.40,
        "South": 0.35,
        "East": 0.10,
        "West": 0.10
    },

    {
        "name": "east_west_heavy",
        "North": 0.10,
        "South": 0.10,
        "East": 0.40,
        "West": 0.35
    },

    {
        "name": "uneven",
        "North": 0.35,
        "South": 0.15,
        "East": 0.25,
        "West": 0.10
    }

]


# ======================================================
# 4. ENVIRONMENT
# ======================================================

env = TrafficEnv(

    max_steps=MAX_STEPS,

    emergency_arrival_probability=(
        EMERGENCY_ARRIVAL_PROBABILITY
    ),

    emergency_wait_penalty=2.0,

    switch_penalty=SWITCH_PENALTY

)


# ======================================================
# 5. DQN AGENT
# ======================================================

agent = DQNAgent(

    state_size=13,

    action_size=2,

    learning_rate=0.0005,

    gamma=0.99,

    epsilon=1.0,

    epsilon_min=0.05,

    epsilon_decay=0.985,

    memory_size=20000,

    batch_size=64

)


# ======================================================
# 6. TRAINING TRACKING
# ======================================================

global_step = 0

training_losses = []

episode_results = []

print("\nStarting Adaptive DQN Traffic Signal Training")

print("=" * 70)

print(f"Episodes: {EPISODES}")

print(f"Steps per episode: {MAX_STEPS}")

print(f"Device: {agent.device}")

print(
    "Emergency arrival probability:",
    EMERGENCY_ARRIVAL_PROBABILITY
)

print("Switch penalty:", SWITCH_PENALTY)

print("Traffic scenarios:", len(TRAFFIC_SCENARIOS))

print("Model output:", MODEL_PATH)

print("=" * 70)


# ======================================================
# 7. TRAINING LOOP
# ======================================================

for episode in range(1, EPISODES + 1):

    # ----------------------------------------------
    # SELECT TRAFFIC SCENARIO
    # ----------------------------------------------

    scenario = TRAFFIC_SCENARIOS[
        (episode - 1) % len(TRAFFIC_SCENARIOS)
    ]

    env.arrival_rates = {

        direction: scenario[direction]

        for direction in [
            "North",
            "South",
            "East",
            "West"
        ]

    }

    # ----------------------------------------------
    # RESET ENVIRONMENT
    # ----------------------------------------------

    state, info = env.reset(
        seed=SEED + episode
    )

    episode_reward = 0.0

    episode_queue_sum = 0.0

    episode_steps = 0

    episode_losses = []

    episode_action_0 = 0

    episode_action_1 = 0

    episode_switches = 0

    episode_forced_switches = 0

    # ----------------------------------------------
    # EPISODE LOOP
    # ----------------------------------------------

    while True:

        # ACTION SELECTION

        action = agent.select_action(state)

        if action == 0:
            episode_action_0 += 1
        else:
            episode_action_1 += 1

        # ENVIRONMENT STEP

        (
            next_state,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        # STORE EXPERIENCE

        # Keep terminated only in replay memory.
        # Truncation is a time limit, not necessarily
        # a true terminal state.

        agent.remember(
            state,
            action,
            reward,
            next_state,
            terminated
        )

        # UPDATE TRACKING

        episode_reward += reward

        episode_queue_sum += env.get_total_queue()

        episode_steps += 1

        global_step += 1

        if info["switch_occurred"]:
            episode_switches += 1

        if info["forced_switch"]:
            episode_forced_switches += 1

        # LEARN

        if (
            global_step >= WARMUP_STEPS
            and global_step % TRAIN_FREQUENCY == 0
        ):

            loss = agent.learn()

            if loss is not None:

                episode_losses.append(loss)

                training_losses.append(loss)

        # TARGET NETWORK UPDATE

        if global_step % TARGET_UPDATE_FREQUENCY == 0:

            agent.update_target_network()

        state = next_state

        # END EPISODE

        if terminated or truncated:
            break

    # ==================================================
    # 8. EPISODE METRICS
    # ==================================================

    average_queue = (

        episode_queue_sum / episode_steps

        if episode_steps > 0

        else 0.0

    )

    emergency_arrivals = len(
        env.emergency_arrival_history
    )

    emergency_departures = len(
        env.emergency_departure_waits
    )

    average_emergency_wait = (

        float(
            np.mean(env.emergency_departure_waits)
        )

        if emergency_departures > 0

        else 0.0

    )

    # Include vehicles that are still waiting.

    remaining_emergency_waits = (
        env.get_all_emergency_waiting_times()
    )

    remaining_emergency_average_wait = (

        float(np.mean(remaining_emergency_waits))

        if remaining_emergency_waits

        else 0.0

    )

    average_loss = (

        float(np.mean(episode_losses))

        if episode_losses

        else 0.0

    )

    # EPSILON DECAY

    agent.decay_epsilon()

    # SAVE RESULTS

    result = {

        "episode": episode,

        "traffic_scenario": scenario["name"],

        "reward": episode_reward,

        "average_queue": average_queue,

        "steps": episode_steps,

        "epsilon": agent.epsilon,

        "average_loss": average_loss,

        "emergency_arrivals": emergency_arrivals,

        "emergency_departures": emergency_departures,

        "average_emergency_wait": average_emergency_wait,

        "remaining_emergency_count": len(
            remaining_emergency_waits
        ),

        "remaining_emergency_average_wait": (
            remaining_emergency_average_wait
        ),

        "action_0_count": episode_action_0,

        "action_1_count": episode_action_1,

        "requested_switches": episode_switches,

        "forced_switches": episode_forced_switches

    }

    episode_results.append(result)

    # ==================================================
    # 9. PRINT PROGRESS
    # ==================================================

    if episode % 10 == 0:

        recent_results = episode_results[-10:]

        recent_reward = np.mean([
            item["reward"]
            for item in recent_results
        ])

        recent_queue = np.mean([
            item["average_queue"]
            for item in recent_results
        ])

        recent_emergency_wait = np.mean([
            item["average_emergency_wait"]
            for item in recent_results
        ])

        recent_switches = np.mean([
            item["requested_switches"]
            for item in recent_results
        ])

        print(

            f"Episode {episode:3d}/{EPISODES} | "

            f"Reward: {recent_reward:9.2f} | "

            f"Queue: {recent_queue:6.2f} | "

            f"Emergency Wait: {recent_emergency_wait:6.2f} | "

            f"Switches: {recent_switches:6.2f} | "

            f"Epsilon: {agent.epsilon:.3f} | "

            f"Loss: {average_loss:.4f}"

        )

    # ==================================================
    # 10. CHECKPOINT
    # ==================================================

    if episode % 50 == 0:

        checkpoint_path = (

            CHECKPOINT_DIR

            / f"adaptive_dqn_episode_{episode}.pth"

        )

        agent.save(
            str(checkpoint_path)
        )


# ======================================================
# 11. SAVE FINAL MODEL
# ======================================================

agent.save(MODEL_PATH)


# ======================================================
# 12. SAVE TRAINING RESULTS
# ======================================================

fieldnames = [

    "episode",

    "traffic_scenario",

    "reward",

    "average_queue",

    "steps",

    "epsilon",

    "average_loss",

    "emergency_arrivals",

    "emergency_departures",

    "average_emergency_wait",

    "remaining_emergency_count",

    "remaining_emergency_average_wait",

    "action_0_count",

    "action_1_count",

    "requested_switches",

    "forced_switches"

]

with open(
    CSV_PATH,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(episode_results)


# ======================================================
# 13. FINAL SUMMARY
# ======================================================

print("\n" + "=" * 70)

print("ADAPTIVE DQN TRAINING COMPLETED")

print("=" * 70)

print("Total episodes:", EPISODES)

print("Total environment steps:", global_step)

print("Total gradient updates:", agent.training_steps)

print("Final epsilon:", round(agent.epsilon, 4))

print(

    "Average reward (last 10):",

    round(
        np.mean([
            item["reward"]
            for item in episode_results[-10:]
        ]),
        3
    )

)

print(

    "Average queue (last 10):",

    round(
        np.mean([
            item["average_queue"]
            for item in episode_results[-10:]
        ]),
        3
    )

)

print(

    "Average requested switches (last 10):",

    round(
        np.mean([
            item["requested_switches"]
            for item in episode_results[-10:]
        ]),
        2
    )

)

print("\nModel saved:", MODEL_PATH)

print("Training CSV saved:", CSV_PATH)

print("=" * 70)

env.close()
