import numpy as np

from simulation.traffic_env import TrafficEnv


# ======================================================
# TEST CONFIGURATION
# ======================================================

SEED = 12345
DIFFERENT_SEED = 54321

STEPS = 600

EMERGENCY_PROBABILITY = 0.03


# ======================================================
# HELPER: RUN ONE ENVIRONMENT
# ======================================================

def run_environment(seed, controller_pattern):

    env = TrafficEnv(
        max_steps=STEPS,
        emergency_arrival_probability=EMERGENCY_PROBABILITY
    )

    observation, info = env.reset(seed=seed)

    assert observation.shape == (13,)

    assert env.observation_space.contains(observation)

    for step in range(STEPS):

        if controller_pattern == "always_keep":

            action = 0

        elif controller_pattern == "periodic_switch":

            action = 1 if step % 12 == 0 else 0

        elif controller_pattern == "alternating":

            action = step % 2

        else:

            raise ValueError(
                f"Unknown controller pattern: {controller_pattern}"
            )

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        # Observation validation.

        assert observation.shape == (13,)

        assert env.observation_space.contains(observation)

        # Queue and emergency flag alignment.

        for direction in env.DIRECTIONS:

            assert len(env.queues[direction]) == len(
                env.emergency_flags[direction]
            ), (
                f"Queue/flag mismatch in {direction} "
                f"at step {step + 1}"
            )

        # Every emergency queue flag must be boolean.

        for direction in env.DIRECTIONS:

            assert all(
                isinstance(flag, (bool, np_bool))
                for flag in env.emergency_flags[direction]
            )

        if terminated or truncated:
            break

    return env


# ======================================================
# TEST 1: SAME SEED, DIFFERENT CONTROLLERS
# ======================================================

def test_same_seed_different_controllers():

    print(
        "\nTEST 1: Same seed, different controller actions"
    )

    env_a = run_environment(
        SEED,
        "always_keep"
    )

    env_b = run_environment(
        SEED,
        "periodic_switch"
    )

    # Ordinary arrival histories must be identical.

    assert (
        env_a.ordinary_arrival_history
        == env_b.ordinary_arrival_history
    ), (
        "Ordinary traffic arrival schedules differ!"
    )

    print(
        "PASS: Ordinary arrival schedules are identical."
    )

    # Potential emergency opportunities must be identical.

    assert (
        env_a.emergency_opportunity_history
        == env_b.emergency_opportunity_history
    ), (
        "Emergency opportunity schedules differ!"
    )

    print(
        "PASS: Potential emergency schedules are identical."
    )

    # Actual admitted emergency arrivals must also match.
    # Every potential emergency event is now admitted.

    assert (
        env_a.emergency_arrival_history
        == env_b.emergency_arrival_history
    ), (
        "Actual emergency arrival histories differ!"
    )

    print(
        "PASS: Actual emergency arrival histories are identical."
    )

    print(
        "Emergency arrivals:",
        env_a.total_emergency_arrivals,
        "in both environments."
    )

    # Controller actions can cause different departures.

    print(
        "Emergency departures:",
        env_a.total_emergency_departures,
        "vs",
        env_b.total_emergency_departures
    )


# ======================================================
# TEST 2: NO EMERGENCY ARRIVALS ARE DROPPED
# ======================================================

def test_no_emergency_arrivals_dropped():

    print(
        "\nTEST 2: Emergency arrival admission"
    )

    env = TrafficEnv(
        max_steps=STEPS,
        emergency_arrival_probability=0.05
    )

    env.reset(seed=SEED)

    for step in range(STEPS):

        env.step(0)

    # Count all scheduled emergency opportunities.

    scheduled_emergencies = sum(
        1
        for time, occurs, direction
        in env.emergency_opportunity_history
        if occurs
    )

    admitted_emergencies = len(
        env.emergency_arrival_history
    )

    assert (
        scheduled_emergencies == admitted_emergencies
    ), (
        f"Emergency vehicles were dropped: "
        f"scheduled={scheduled_emergencies}, "
        f"admitted={admitted_emergencies}"
    )

    assert (
        env.total_emergency_arrivals == admitted_emergencies
    )

    print(
        "Scheduled emergency arrivals:",
        scheduled_emergencies
    )

    print(
        "Admitted emergency arrivals:",
        admitted_emergencies
    )

    print(
        "PASS: No emergency arrivals were discarded."
    )


# ======================================================
# TEST 3: MULTIPLE EMERGENCY VEHICLES
# ======================================================

def test_multiple_emergency_vehicles():

    print(
        "\nTEST 3: Multiple emergency vehicles"
    )

    env = TrafficEnv(
        max_steps=100,
        min_green=10,
        max_green=40,
        yellow_duration=3,
        emergency_arrival_probability=1.0
    )

    env.reset(seed=SEED)

    maximum_emergency_queue = 0

    for step in range(100):

        observation, reward, terminated, truncated, info = (
            env.step(0)
        )

        current_emergency_queue = (
            env.get_total_emergency_queue()
        )

        maximum_emergency_queue = max(
            maximum_emergency_queue,
            current_emergency_queue
        )

        if terminated or truncated:
            break

    # One emergency arrival is generated per simulation
    # second because the probability is 1.0.

    assert env.total_emergency_arrivals == env.time

    assert maximum_emergency_queue > 0

    # The compatibility property returns the oldest
    # currently waiting emergency vehicle, if one exists.

    oldest_emergency = env.active_emergency

    if env.get_total_emergency_queue() > 0:

        assert oldest_emergency is not None

        assert "direction" in oldest_emergency

        assert "arrival_time" in oldest_emergency

    print(
        "Emergency arrivals:",
        env.total_emergency_arrivals
    )

    print(
        "Emergency departures:",
        env.total_emergency_departures
    )

    print(
        "Maximum simultaneous emergency queue:",
        maximum_emergency_queue
    )

    print(
        "PASS: Multiple emergency arrivals are supported."
    )


# ======================================================
# TEST 4: SAME SEED REPRODUCIBILITY
# ======================================================

def test_reproducibility():

    print(
        "\nTEST 4: Same seed, repeated simulation"
    )

    env_a = run_environment(
        SEED,
        "alternating"
    )

    env_b = run_environment(
        SEED,
        "alternating"
    )

    assert (
        env_a.ordinary_arrival_history
        == env_b.ordinary_arrival_history
    )

    assert (
        env_a.emergency_opportunity_history
        == env_b.emergency_opportunity_history
    )

    assert (
        env_a.emergency_arrival_history
        == env_b.emergency_arrival_history
    )

    assert (
        env_a.emergency_departure_waits
        == env_b.emergency_departure_waits
    )

    assert (
        env_a.total_departures
        == env_b.total_departures
    )

    assert (
        env_a.total_emergency_departures
        == env_b.total_emergency_departures
    )

    print(
        "PASS: Repeated runs produce identical results."
    )


# ======================================================
# TEST 5: DIFFERENT SEEDS
# ======================================================

def test_different_seeds():

    print(
        "\nTEST 5: Different random seeds"
    )

    env_a = run_environment(
        SEED,
        "always_keep"
    )

    env_b = run_environment(
        DIFFERENT_SEED,
        "always_keep"
    )

    assert (
        env_a.ordinary_arrival_history
        != env_b.ordinary_arrival_history
    )

    assert (
        env_a.emergency_opportunity_history
        != env_b.emergency_opportunity_history
    )

    print(
        "PASS: Different seeds produce different traffic schedules."
    )


# ======================================================
# TEST 6: ENVIRONMENT STRUCTURE
# ======================================================

def test_environment_structure():

    print(
        "\nTEST 6: Environment structure"
    )

    env = TrafficEnv(
        max_steps=100,
        emergency_arrival_probability=0.03
    )

    observation, info = env.reset(
        seed=SEED
    )

    assert env.action_space.n == 2

    assert env.observation_space.shape == (13,)

    assert env.observation_space.contains(observation)

    assert set(env.queues.keys()) == set(env.DIRECTIONS)

    assert set(
        env.emergency_flags.keys()
    ) == set(env.DIRECTIONS)

    assert env.ordinary_rng is not None

    assert env.emergency_rng is not None

    assert env.emergency_direction_rng is not None

    print(
        "PASS: Action space is valid."
    )

    print(
        "PASS: Observation contains 13 features."
    )

    print(
        "PASS: Four directional queues exist."
    )

    print(
        "PASS: Emergency queues are initialized."
    )

    print(
        "PASS: Independent random generators initialized."
    )


# ======================================================
# TEST 7: GYMNASIUM RESET AND STEP API
# ======================================================

def test_gymnasium_api():

    print(
        "\nTEST 7: Gymnasium API compatibility"
    )

    env = TrafficEnv(
        max_steps=10
    )

    observation, info = env.reset(
        seed=SEED
    )

    assert isinstance(observation, np.ndarray)

    assert isinstance(info, dict)

    result = env.step(0)

    assert len(result) == 5

    observation, reward, terminated, truncated, info = result

    assert isinstance(reward, float)

    assert isinstance(terminated, bool)

    assert isinstance(truncated, bool)

    assert isinstance(info, dict)

    print(
        "PASS: Gymnasium reset() and step() work."
    )


# ======================================================
# NUMPY BOOLEAN TYPE
# ======================================================

np_bool = type(__import__("numpy").bool_(True))


# ======================================================
# RUN ALL TESTS
# ======================================================

if __name__ == "__main__":

    print("\n==============================================")

    print("       FLOWSYNC AI - FAIRNESS TESTS")

    print("==============================================")

    test_environment_structure()

    test_gymnasium_api()

    test_same_seed_different_controllers()

    test_no_emergency_arrivals_dropped()

    test_multiple_emergency_vehicles()

    test_reproducibility()

    test_different_seeds()

    print("\n==============================================")

    print("       ALL FAIRNESS TESTS PASSED")

    print("==============================================")
