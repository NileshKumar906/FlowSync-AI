
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from collections import deque


class TrafficEnv(gym.Env):

    metadata = {"render_modes": []}

    DIRECTIONS = ["North", "South", "East", "West"]

    def __init__(
        self,
        max_steps=600,
        min_green=5,
        max_green=30,
        yellow_duration=3,
        emergency_arrival_probability=0.008,
        emergency_wait_penalty=2.0,
        switch_penalty=0.20
    ):
        super().__init__()

        # ==================================================
        # SIMULATION CONFIGURATION
        # ==================================================

        self.max_steps = max_steps
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_duration = yellow_duration

        self.emergency_arrival_probability = float(
            np.clip(emergency_arrival_probability, 0.0, 1.0)
        )

        self.emergency_wait_penalty = float(
            emergency_wait_penalty
        )

        self.switch_penalty = float(switch_penalty)

        # Default traffic demand.
        # Training can replace these values before reset().

        self.arrival_rates = {
            "North": 0.22,
            "South": 0.22,
            "East": 0.22,
            "West": 0.22
        }

        # ==================================================
        # VEHICLE QUEUES
        # ==================================================

        self.queues = {
            direction: deque()
            for direction in self.DIRECTIONS
        }

        self.emergency_flags = {
            direction: deque()
            for direction in self.DIRECTIONS
        }

        # ==================================================
        # RANDOM GENERATORS
        # ==================================================

        self.ordinary_rng = None
        self.emergency_rng = None
        self.emergency_direction_rng = None

        # ==================================================
        # ACTION AND OBSERVATION SPACES
        # ==================================================

        # Action 0: Keep current green.
        # Action 1: Request a phase switch.

        self.action_space = spaces.Discrete(2)

        # 0-3: Directional queue lengths.
        # 4-5: Signal phase.
        # 6: Green duration.
        # 7: Yellow duration.
        # 8-11: Emergency queue lengths.
        # 12: Average emergency waiting time.

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(13,),
            dtype=np.float32
        )

        # ==================================================
        # SIGNAL STATE
        # ==================================================

        self.time = 0
        self.phase = 0
        self.green_elapsed = 0
        self.yellow_remaining = 0

        self._reset_statistics()

    # ======================================================
    # STATISTICS RESET
    # ======================================================

    def _reset_statistics(self):

        self.total_arrivals = 0
        self.total_departures = 0
        self.total_waiting_time = 0

        self.queue_sum = 0
        self.departure_waits = []

        self.last_departures = {
            direction: 0
            for direction in self.DIRECTIONS
        }

        # Emergency statistics.

        self.total_emergency_arrivals = 0
        self.total_emergency_departures = 0
        self.total_emergency_waiting_time = 0

        self.emergency_departure_waits = []

        self.emergency_wait_sum = 0
        self.emergency_queue_sum = 0

        self.last_emergency_departures = {
            direction: 0
            for direction in self.DIRECTIONS
        }

        self.last_departure_details = []

        # Arrival histories.

        self.ordinary_arrival_history = []
        self.emergency_opportunity_history = []
        self.emergency_arrival_history = []

        # Signal statistics.

        self.requested_switches = 0
        self.forced_switches = 0

    # ======================================================
    # RANDOM GENERATORS
    # ======================================================

    def _initialize_random_generators(self):

        child_seeds = self.np_random.integers(
            0,
            2**63,
            size=3,
            dtype=np.int64
        )

        self.ordinary_rng = np.random.default_rng(
            child_seeds[0]
        )

        self.emergency_rng = np.random.default_rng(
            child_seeds[1]
        )

        self.emergency_direction_rng = np.random.default_rng(
            child_seeds[2]
        )

    # ======================================================
    # EMERGENCY COMPATIBILITY PROPERTY
    # ======================================================

    @property
    def active_emergency(self):

        oldest = None

        for direction in self.DIRECTIONS:

            for arrival_time, is_emergency in zip(
                self.queues[direction],
                self.emergency_flags[direction]
            ):

                if not is_emergency:
                    continue

                if (
                    oldest is None
                    or arrival_time < oldest["arrival_time"]
                ):

                    oldest = {
                        "direction": direction,
                        "arrival_time": arrival_time,
                        "vehicle_type": "Ambulance"
                    }

        return oldest

    # ======================================================
    # SIGNAL MANAGEMENT
    # ======================================================

    def get_green_directions(self):

        if self.phase == 0:
            return ["North", "South"]

        return ["East", "West"]

    def get_red_directions(self):

        green_directions = self.get_green_directions()

        return [
            direction
            for direction in self.DIRECTIONS
            if direction not in green_directions
        ]

    # ======================================================
    # QUEUE MANAGEMENT
    # ======================================================

    def get_total_queue(self):

        return sum(
            len(self.queues[direction])
            for direction in self.DIRECTIONS
        )

    def get_queue_lengths(self):

        return {
            direction: len(self.queues[direction])
            for direction in self.DIRECTIONS
        }

    def get_emergency_queue_lengths(self):

        return {
            direction: sum(
                1 for flag in self.emergency_flags[direction]
                if flag
            )
            for direction in self.DIRECTIONS
        }

    def get_total_emergency_queue(self):

        return sum(
            self.get_emergency_queue_lengths().values()
        )

    # ======================================================
    # EMERGENCY INFORMATION
    # ======================================================

    def get_emergency_direction(self):

        emergency = self.active_emergency

        if emergency is None:
            return None

        return emergency["direction"]

    def get_emergency_waiting_time(self):

        emergency = self.active_emergency

        if emergency is None:
            return 0

        return max(
            0,
            self.time - emergency["arrival_time"]
        )

    def get_all_emergency_waiting_times(self):

        waiting_times = []

        for direction in self.DIRECTIONS:

            for arrival_time, is_emergency in zip(
                self.queues[direction],
                self.emergency_flags[direction]
            ):

                if is_emergency:

                    waiting_times.append(
                        max(0, self.time - arrival_time)
                    )

        return waiting_times

    def get_average_emergency_waiting_time(self):

        waiting_times = self.get_all_emergency_waiting_times()

        if not waiting_times:
            return 0.0

        return float(np.mean(waiting_times))

    def is_emergency_on_green(self):

        emergency_directions = self.get_emergency_queue_lengths()

        return any(
            emergency_directions[direction] > 0
            for direction in self.get_green_directions()
        )

    # ======================================================
    # ORDINARY VEHICLE GENERATION
    # ======================================================

    def generate_ordinary_vehicles(self):

        for direction in self.DIRECTIONS:

            probability = float(
                np.clip(
                    self.arrival_rates[direction],
                    0.0,
                    1.0
                )
            )

            if self.ordinary_rng.random() < probability:

                self.queues[direction].append(self.time)

                self.emergency_flags[direction].append(False)

                self.total_arrivals += 1

                self.ordinary_arrival_history.append(
                    (self.time, direction)
                )

    # ======================================================
    # EMERGENCY VEHICLE GENERATION
    # ======================================================

    def generate_emergency_vehicle(self):

        # Always consume one occurrence draw and one direction
        # draw per timestep. This keeps emergency schedules
        # independent of controller actions.

        emergency_occurs = (
            self.emergency_rng.random()
            < self.emergency_arrival_probability
        )

        direction_index = self.emergency_direction_rng.integers(
            0,
            len(self.DIRECTIONS)
        )

        direction = self.DIRECTIONS[direction_index]

        self.emergency_opportunity_history.append(
            (
                self.time,
                bool(emergency_occurs),
                direction
            )
        )

        if not emergency_occurs:
            return

        self.queues[direction].append(self.time)
        self.emergency_flags[direction].append(True)

        self.total_arrivals += 1
        self.total_emergency_arrivals += 1

        self.emergency_arrival_history.append(
            (self.time, direction)
        )

    # ======================================================
    # VEHICLE GENERATION
    # ======================================================

    def generate_vehicles(self):

        self.generate_ordinary_vehicles()
        self.generate_emergency_vehicle()

    # ======================================================
    # EMERGENCY WAITING STATISTICS
    # ======================================================

    def update_emergency_waiting_statistics(self):

        waiting_times = self.get_all_emergency_waiting_times()

        self.emergency_queue_sum += len(waiting_times)
        self.emergency_wait_sum += sum(waiting_times)

    # ======================================================
    # VEHICLE MOVEMENT
    # ======================================================

    def move_vehicles(self):

        departed = 0

        self.last_departures = {
            direction: 0
            for direction in self.DIRECTIONS
        }

        self.last_emergency_departures = {
            direction: 0
            for direction in self.DIRECTIONS
        }

        self.last_departure_details = []

        # One vehicle per green direction per second.

        for direction in self.get_green_directions():

            queue = self.queues[direction]
            emergency_flags = self.emergency_flags[direction]

            if not queue:
                continue

            arrival_time = queue.popleft()
            is_emergency = emergency_flags.popleft()

            waiting_time = max(
                0,
                self.time - arrival_time
            )

            self.total_waiting_time += waiting_time
            self.departure_waits.append(waiting_time)

            self.total_departures += 1
            self.last_departures[direction] += 1

            departed += 1

            departure_detail = {
                "direction": direction,
                "arrival_time": arrival_time,
                "waiting_time": waiting_time,
                "is_emergency": is_emergency
            }

            self.last_departure_details.append(
                departure_detail
            )

            if is_emergency:

                self.total_emergency_departures += 1

                self.total_emergency_waiting_time += waiting_time

                self.emergency_departure_waits.append(
                    waiting_time
                )

                self.last_emergency_departures[direction] += 1

        return departed

    # ======================================================
    # OBSERVATION GENERATION
    # ======================================================

    def get_observation(self):

        # Directional queues.

        queue_values = [
            min(
                len(self.queues[direction]) / 30.0,
                1.0
            )
            for direction in self.DIRECTIONS
        ]

        # Signal phase.

        phase_values = [
            float(self.phase == 0),
            float(self.phase == 1)
        ]

        # Green duration.

        green_value = min(
            self.green_elapsed / self.max_green,
            1.0
        )

        # Yellow duration.

        yellow_value = (
            min(
                self.yellow_remaining / self.yellow_duration,
                1.0
            )
            if self.yellow_duration > 0
            else 0.0
        )

        # Emergency queues.

        emergency_queue_lengths = (
            self.get_emergency_queue_lengths()
        )

        emergency_values = [
            min(
                emergency_queue_lengths[direction] / 10.0,
                1.0
            )
            for direction in self.DIRECTIONS
        ]

        # Average emergency waiting time.

        average_emergency_wait = (
            self.get_average_emergency_waiting_time()
        )

        emergency_wait_value = min(
            average_emergency_wait / 120.0,
            1.0
        )

        observation = (
            queue_values
            + phase_values
            + [green_value]
            + [yellow_value]
            + emergency_values
            + [emergency_wait_value]
        )

        return np.asarray(
            observation,
            dtype=np.float32
        )

    # ======================================================
    # REWARD FUNCTION
    # ======================================================

    def calculate_reward(
        self,
        total_queue,
        switch_occurred=False
    ):

        normal_traffic_penalty = total_queue / 10.0

        emergency_count = self.get_total_emergency_queue()

        emergency_queue_penalty = (
            emergency_count * self.emergency_wait_penalty
        )

        emergency_waiting_times = (
            self.get_all_emergency_waiting_times()
        )

        total_emergency_wait = sum(
            emergency_waiting_times
        )

        emergency_delay_penalty = (
            total_emergency_wait * 0.05
        )

        switching_penalty = (
            self.switch_penalty
            if switch_occurred
            else 0.0
        )

        reward = -(
            normal_traffic_penalty
            + emergency_queue_penalty
            + emergency_delay_penalty
            + switching_penalty
        )

        return float(reward)

    # ======================================================
    # RESET ENVIRONMENT
    # ======================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self._initialize_random_generators()

        self.queues = {
            direction: deque()
            for direction in self.DIRECTIONS
        }

        self.emergency_flags = {
            direction: deque()
            for direction in self.DIRECTIONS
        }

        self.time = 0
        self.phase = 0
        self.green_elapsed = 0
        self.yellow_remaining = 0

        self._reset_statistics()

        observation = self.get_observation()

        info = {
            "time": self.time,
            "total_queue": 0,
            "phase": self.phase,
            "queue_lengths": self.get_queue_lengths(),
            "emergency_queue_lengths": (
                self.get_emergency_queue_lengths()
            ),
            "total_emergency_queue": 0,
            "emergency_vehicle": None,
            "emergency_waiting_time": 0,
            "departures_by_direction": (
                self.last_departures.copy()
            ),
            "emergency_departures_by_direction": (
                self.last_emergency_departures.copy()
            )
        }

        return observation, info

    # ======================================================
    # STEP: EXECUTE ONE SIMULATION SECOND
    # ======================================================

    def step(self, action):

        if not self.action_space.contains(action):

            raise ValueError(
                f"Invalid action: {action}"
            )

        self.time += 1

        self.last_departures = {
            direction: 0
            for direction in self.DIRECTIONS
        }

        self.last_emergency_departures = {
            direction: 0
            for direction in self.DIRECTIONS
        }

        self.last_departure_details = []

        # Generate vehicles.

        self.generate_vehicles()

        departed = 0

        switch_occurred = False
        forced_switch = False

        # Handle an ongoing yellow transition.

        if self.yellow_remaining > 0:

            self.yellow_remaining -= 1

            if self.yellow_remaining == 0:

                self.phase = 1 - self.phase
                self.green_elapsed = 0

        else:

            can_switch = (
                self.green_elapsed >= self.min_green
            )

            forced_switch = (
                self.green_elapsed >= self.max_green
            )

            requested_switch = (
                action == 1 and can_switch
            )

            if requested_switch or forced_switch:

                # Only agent-requested switches receive the
                # switching penalty. Forced switches are
                # imposed by the environment.

                if requested_switch:

                    switch_occurred = True
                    self.requested_switches += 1

                if forced_switch:

                    self.forced_switches += 1

                self.yellow_remaining = self.yellow_duration

                # Consume first yellow second.

                self.yellow_remaining -= 1

                if self.yellow_remaining == 0:

                    self.phase = 1 - self.phase
                    self.green_elapsed = 0

            else:

                departed = self.move_vehicles()
                self.green_elapsed += 1

        # Update emergency statistics.

        self.update_emergency_waiting_statistics()

        # Calculate reward.

        total_queue = self.get_total_queue()

        reward = self.calculate_reward(
            total_queue,
            switch_occurred=switch_occurred
        )

        self.queue_sum += total_queue

        # Episode completion.

        terminated = False

        truncated = (
            self.time >= self.max_steps
        )

        observation = self.get_observation()

        # Emergency information.

        emergency_information = None

        oldest_emergency = self.active_emergency

        if oldest_emergency is not None:

            emergency_information = {
                "direction": oldest_emergency["direction"],
                "vehicle_type": oldest_emergency["vehicle_type"],
                "arrival_time": oldest_emergency["arrival_time"],
                "waiting_time": self.get_emergency_waiting_time(),
                "is_on_green": self.is_emergency_on_green()
            }

        # Detailed information.

        info = {
            "time": self.time,

            "total_queue": total_queue,

            "queue_lengths": self.get_queue_lengths(),

            "emergency_queue_lengths": (
                self.get_emergency_queue_lengths()
            ),

            "total_emergency_queue": (
                self.get_total_emergency_queue()
            ),

            "departed": departed,

            "departures_by_direction": (
                self.last_departures.copy()
            ),

            "departure_details": (
                self.last_departure_details.copy()
            ),

            "emergency_departures_by_direction": (
                self.last_emergency_departures.copy()
            ),

            "total_arrivals": self.total_arrivals,

            "total_departures": self.total_departures,

            "total_emergency_arrivals": (
                self.total_emergency_arrivals
            ),

            "total_emergency_departures": (
                self.total_emergency_departures
            ),

            "total_emergency_waiting_time": (
                self.total_emergency_waiting_time
            ),

            "emergency_vehicle": emergency_information,

            "emergency_waiting_time": (
                self.get_emergency_waiting_time()
            ),

            "average_emergency_waiting_time": (
                self.get_average_emergency_waiting_time()
            ),

            "emergency_on_green": (
                self.is_emergency_on_green()
            ),

            "phase": self.phase,

            "green_elapsed": self.green_elapsed,

            "yellow_remaining": self.yellow_remaining,

            "switch_occurred": switch_occurred,

            "forced_switch": forced_switch,

            "requested_switches": self.requested_switches,

            "forced_switches": self.forced_switches,

            "reward": reward
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )
