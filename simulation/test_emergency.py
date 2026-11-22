
from simulation.traffic_env import TrafficEnv


env = TrafficEnv(
    max_steps=600,
    emergency_arrival_probability=0.03
)

state, info = env.reset(seed=42)

print("Initial observation:", state)
print("Observation shape:", state.shape)

print("\nStarting emergency traffic test...\n")


for step in range(600):

    # Random actions: testing the environment, not training.
    action = env.action_space.sample()

    state, reward, terminated, truncated, info = env.step(action)

    emergency = info["emergency_vehicle"]

    # Display an active emergency vehicle.
    if emergency is not None:

        print(
            f"Time: {info['time']} | "
            f"Emergency: {emergency['vehicle_type']} | "
            f"Direction: {emergency['direction']} | "
            f"Waiting: {emergency['waiting_time']}s | "
            f"On Green: {emergency['is_on_green']}"
        )

    # IMPORTANT:
    # This is a per-step dictionary, not a cumulative counter.
    emergency_departures = info[
        "emergency_departures_by_direction"
    ]

    if sum(emergency_departures.values()) > 0:

        for direction, count in emergency_departures.items():

            if count > 0:

                print(
                    f"EMERGENCY DEPARTED | "
                    f"Direction: {direction} | "
                    f"Time: {info['time']}"
                )

    if terminated or truncated:
        break


# ------------------------------------------
# FINAL RESULTS
# ------------------------------------------

print("\n========== FINAL RESULTS ==========")

print("Total arrivals:", env.total_arrivals)

print("Total departures:", env.total_departures)

print(
    "Emergency arrivals:",
    env.total_emergency_arrivals
)

print(
    "Emergency departures:",
    env.total_emergency_departures
)

print(
    "Emergency vehicles remaining:",
    env.get_total_emergency_queue()
)

average_emergency_wait = (
    sum(env.emergency_departure_waits)
    / len(env.emergency_departure_waits)
    if env.emergency_departure_waits
    else 0
)

print(
    "Average emergency waiting time:",
    round(average_emergency_wait, 2),
    "seconds"
)

print(
    "Emergency waiting time total:",
    env.total_emergency_waiting_time
)

print("\nEnvironment test completed!")
