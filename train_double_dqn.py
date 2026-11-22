
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

# Model output
MODEL_PATH = "models/double_dqn_traffic_emergency.pth"

# Checkpoint directory
CHECKPOINT_DIR = Path(
    "models/double_dqn_checkpoints"
)

# Training results
CSV_PATH = (
    "training_results/double_dqn_training.csv"
)


# ======================================================
# 3. CREATE OUTPUT DIRECTORIES
# ======================================================

Path("models").mkdir(
    parents=True,
    exist_ok=True
)

Path("training_results").mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================
# 4. CREATE TRAFFIC ENVIRONMENT
# ======================================================

env = TrafficEnv(

    max_steps=MAX_STEPS,

    emergency_arrival_probability=(
        EMERGENCY_ARRIVAL_PROBABILITY
    ),

    emergency_wait_penalty=2.0

)


# ======================================================
# 5. CREATE DOUBLE DQN AGENT
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

    batch_size=64,

    double_dqn=True

)


# ======================================================
# 6. TRAINING TRACKING
# ======================================================

global_step = 0

episode_results = []

training_losses = []


# ======================================================
# 7. DISPLAY CONFIGURATION
# ======================================================

print("\n")
print("=" * 75)

print("             FLOWSYNC AI")
print("          DOUBLE DQN TRAINING")

print("=" * 75)

print(f"Algorithm: {agent.algorithm_name}")

print(f"Episodes: {EPISODES}")

print(f"Maximum steps per episode: {MAX_STEPS}")

print(f"Total planned environment steps: {EPISODES * MAX_STEPS}")

print(f"Learning rate: 0.0005")

print(f"Discount factor: 0.99")

print(f"Batch size: {agent.batch_size}")

print(f"Initial epsilon: {agent.epsilon}")

print(f"Minimum epsilon: {agent.epsilon_min}")

print(f"Device: {agent.device}")

print(
    f"Emergency arrival probability: "
    f"{EMERGENCY_ARRIVAL_PROBABILITY}"
)

print(f"Model output: {MODEL_PATH}")

print("=" * 75)


# ======================================================
# 8. MAIN TRAINING LOOP
# ======================================================

for episode in range(1, EPISODES + 1):

    # ----------------------------------------------
    # RESET ENVIRONMENT
    # ----------------------------------------------

    state, info = env.reset(
        seed=SEED + episode
    )

    episode_reward = 0.0

    episode_steps = 0

    episode_losses = []

    # ----------------------------------------------
    # EPISODE LOOP
    # ----------------------------------------------

    while True:

        # ==========================================
        # 8.1 SELECT ACTION
        # ==========================================

        action = agent.select_action(state)

        # ==========================================
        # 8.2 EXECUTE ACTION
        # ==========================================

        (
            next_state,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        # ==========================================
        # 8.3 STORE EXPERIENCE
        # ==========================================

        # Store terminated only.
        #
        # A time-limit truncation is not necessarily
        # a terminal state. DQN should normally
        # bootstrap from its next observation.

        agent.remember(

            state,

            action,

            reward,

            next_state,

            terminated

        )

        # ==========================================
        # 8.4 UPDATE METRICS
        # ==========================================

        episode_reward += reward

        episode_steps += 1

        global_step += 1

        # ==========================================
        # 8.5 TRAIN DOUBLE DQN
        # ==========================================

        if (

            global_step >= WARMUP_STEPS

            and global_step % TRAIN_FREQUENCY == 0

        ):

            loss = agent.learn()

            if loss is not None:

                episode_losses.append(loss)

                training_losses.append(loss)

        # ==========================================
        # 8.6 UPDATE TARGET NETWORK
        # ==========================================

        if global_step % TARGET_UPDATE_FREQUENCY == 0:

            agent.update_target_network()

        # ==========================================
        # 8.7 UPDATE STATE
        # ==========================================

        state = next_state

        # ==========================================
        # 8.8 END EPISODE
        # ==========================================

        if terminated or truncated:

            break

    # ==================================================
    # 9. CALCULATE EPISODE METRICS
    # ==================================================

    # Average queue length per simulation step.
    average_queue = (

        env.queue_sum / episode_steps

        if episode_steps > 0

        else 0.0

    )

    # Emergency arrivals.
    emergency_arrivals = len(
        env.emergency_arrival_history
    )

    # Emergency departures.
    emergency_departures = len(
        env.emergency_departure_waits
    )

    # Average waiting time of departed emergencies.
    average_emergency_wait = (

        float(
            np.mean(env.emergency_departure_waits)
        )

        if emergency_departures > 0

        else 0.0

    )

    # Average training loss.
    average_loss = (

        float(np.mean(episode_losses))

        if episode_losses

        else 0.0

    )

    # ==================================================
    # 10. EPSILON DECAY
    # ==================================================

    agent.decay_epsilon()

    # ==================================================
    # 11. SAVE EPISODE RESULTS
    # ==================================================

    result = {

        "episode": episode,

        "reward": episode_reward,

        "average_queue": average_queue,

        "steps": episode_steps,

        "epsilon": agent.epsilon,

        "average_loss": average_loss,

        "emergency_arrivals": emergency_arrivals,

        "emergency_departures": emergency_departures,

        "average_emergency_wait": average_emergency_wait

    }

    episode_results.append(result)

    # ==================================================
    # 12. PRINT TRAINING PROGRESS
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

        recent_emergency_arrivals = sum([

            item["emergency_arrivals"]

            for item in recent_results

        ])

        recent_emergency_departures = sum([

            item["emergency_departures"]

            for item in recent_results

        ])

        print(

            f"\nEpisode {episode:3d}/{EPISODES}"

            f" | Reward: {recent_reward:9.2f}"

            f" | Queue: {recent_queue:6.2f}"

            f" | Emergency Wait: {recent_emergency_wait:6.2f}"

            f" | Emergency Served: "

            f"{recent_emergency_departures}/"

            f"{recent_emergency_arrivals}"

            f" | Epsilon: {agent.epsilon:.3f}"

            f" | Loss: {average_loss:.4f}"

        )

    # ==================================================
    # 13. SAVE CHECKPOINTS
    # ==================================================

    if episode % 50 == 0:

        checkpoint_path = (

            CHECKPOINT_DIR

            / f"double_dqn_episode_{episode}.pth"

        )

        agent.save(
            str(checkpoint_path)
        )

        print(
            f"\nCheckpoint saved: {checkpoint_path}"
        )


# ======================================================
# 14. SAVE FINAL DOUBLE DQN MODEL
# ======================================================

agent.save(MODEL_PATH)


# ======================================================
# 15. SAVE TRAINING CSV
# ======================================================

fieldnames = [

    "episode",

    "reward",

    "average_queue",

    "steps",

    "epsilon",

    "average_loss",

    "emergency_arrivals",

    "emergency_departures",

    "average_emergency_wait"

]

with open(

    CSV_PATH,

    "w",

    newline="",

    encoding="utf-8"

) as file:

    writer = csv.DictWriter(

        file,

        fieldnames=fieldnames

    )

    writer.writeheader()

    writer.writerows(episode_results)


# ======================================================
# 16. FINAL TRAINING SUMMARY
# ======================================================

last_10 = episode_results[-10:]

last_10_reward = np.mean([

    item["reward"]

    for item in last_10

])

last_10_queue = np.mean([

    item["average_queue"]

    for item in last_10

])

last_10_emergency_wait = np.mean([

    item["average_emergency_wait"]

    for item in last_10

])

print("\n")

print("=" * 75)

print("             DOUBLE DQN TRAINING COMPLETED")

print("=" * 75)

print(f"Algorithm: {agent.algorithm_name}")

print(f"Completed episodes: {len(episode_results)}")

print(f"Total environment steps: {global_step}")

print(f"Total gradient updates: {agent.training_steps}")

print(f"Final epsilon: {agent.epsilon:.4f}")

print(f"Average reward (last 10): {last_10_reward:.3f}")

print(f"Average queue (last 10): {last_10_queue:.3f}")

print(
    f"Average emergency wait (last 10): "
    f"{last_10_emergency_wait:.3f}"
)

print("\nGenerated files:")

print(f"Model: {MODEL_PATH}")

print(f"Training CSV: {CSV_PATH}")

print(f"Checkpoints: {CHECKPOINT_DIR}")

print("=" * 75)


# ======================================================
# 17. CLOSE ENVIRONMENT
# ======================================================

env.close()
