
from traffic_env import TrafficEnv


# Create environment
env = TrafficEnv(max_steps=100)

# Start one episode
observation, info = env.reset(seed=42)

print("Initial observation:", observation)
print("Observation shape:", observation.shape)
print("Initial information:", info)

total_reward = 0

# Test with random actions
for step in range(100):

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = (
        env.step(action)
    )

    total_reward += reward

    assert env.observation_space.contains(observation)

    if (step + 1) % 10 == 0:

        print(
            f"Step: {step + 1:3d} | "
            f"Action: {action} | "
            f"Reward: {reward:.2f} | "
            f"Queue: {info['total_queue']} | "
            f"Phase: {info['phase']} | "
            f"Yellow: {info['yellow_remaining']}"
        )

    if terminated or truncated:
        break

print("\n========== TEST RESULTS ==========")
print("Total reward:", round(total_reward, 2))
print("Total arrivals:", info["total_arrivals"])
print("Total departures:", info["total_departures"])
print("Environment test completed!")
