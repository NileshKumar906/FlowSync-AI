
import random
from collections import deque


class TrafficSimulation:

    def __init__(self):
        # Each queue stores the arrival time of every waiting vehicle.
        self.queues = {
            "North": deque(),
            "South": deque(),
            "East": deque(),
            "West": deque()
        }

        # Probability of a vehicle arriving from each direction per second.
        self.arrival_rates={
            "North" : 0.40,
            "South": 0.15,
            "East": 0.30,
            "West": 0.10,
        }

        # Signal cycle:
        # North-South green -> yellow -> East-West green -> yellow
        self.phases = [
            "NS_GREEN",
            "NS_YELLOW",
            "EW_GREEN",
            "EW_YELLOW"
        ]

        self.phase_durations = {
            "NS_GREEN": 25,
            "NS_YELLOW": 3,
            "EW_GREEN": 10,
            "EW_YELLOW": 3
        }

        self.current_phase_index = 0
        self.phase_elapsed = 0

        # Statistics
        self.total_arrivals = 0
        self.total_departures = 0
        self.total_waiting_time = 0
        self.queue_sum = 0
        self.max_total_queue = 0
        self.departure_waits = []

    @property
    def current_phase(self):
        return self.phases[self.current_phase_index]

    def get_green_directions(self):
        if self.current_phase == "NS_GREEN":
            return ["North", "South"]

        if self.current_phase == "EW_GREEN":
            return ["East", "West"]

        return []

    def generate_vehicles(self, current_time):
        for direction in self.queues:

            if random.random() < self.arrival_rates[direction]:
                self.queues[direction].append(current_time)
                self.total_arrivals += 1

    def move_vehicles(self, current_time):
        green_directions = self.get_green_directions()

        for direction in green_directions:

            # One vehicle can cross per direction per second.
            if self.queues[direction]:
                arrival_time = self.queues[direction].popleft()

                waiting_time = current_time - arrival_time

                self.total_departures += 1
                self.total_waiting_time += waiting_time
                self.departure_waits.append(waiting_time)

    def update_signal(self):
        self.phase_elapsed += 1

        if self.phase_elapsed >= self.phase_durations[self.current_phase]:
            self.current_phase_index = (
                self.current_phase_index + 1
            ) % len(self.phases)

            self.phase_elapsed = 0

    def get_queue_lengths(self):
        return {
            direction: len(queue)
            for direction, queue in self.queues.items()
        }

    def display_status(self, current_time):
        queues = self.get_queue_lengths()

        total_queue = sum(queues.values())

        print(f"\nTime: {current_time} seconds")
        print(f"Signal: {self.current_phase}")

        for direction, count in queues.items():
            print(f"{direction:7}: {count} vehicles")

        print(f"Total waiting vehicles: {total_queue}")

    def run(self, total_seconds=120, report_interval=10):

        print("Starting FlowSync AI Traffic Simulation")
        print("----------------------------------------")

        for current_time in range(1, total_seconds + 1):

            # 1. Move vehicles whose signals are currently green.
            self.move_vehicles(current_time)

            # 2. Generate new arriving vehicles.
            self.generate_vehicles(current_time)

            # 3. Collect queue statistics.
            total_queue = sum(len(q) for q in self.queues.values())

            self.queue_sum += total_queue
            self.max_total_queue = max(
                self.max_total_queue,
                total_queue
            )

            # 4. Display periodic updates.
            if current_time % report_interval == 0:
                self.display_status(current_time)

            # 5. Update traffic signals.
            self.update_signal()

        self.display_results(total_seconds)

    def display_results(self, total_seconds):

        average_queue = self.queue_sum / total_seconds

        average_waiting_time = (
            self.total_waiting_time / self.total_departures
            if self.total_departures > 0
            else 0
        )

        remaining_vehicles = sum(
            len(queue) for queue in self.queues.values()
        )

        print("\n========== SIMULATION RESULTS ==========")

        print("Simulation duration:", total_seconds, "seconds")
        print("Total arrivals:", self.total_arrivals)
        print("Total departures:", self.total_departures)
        print("Vehicles remaining:", remaining_vehicles)

        print(f"Average queue length: {average_queue:.2f}")
        print(f"Maximum total queue: {self.max_total_queue}")
        print(f"Average waiting time of departed vehicles: "
              f"{average_waiting_time:.2f} seconds")

        print("========================================")


if __name__ == "__main__":

    random.seed(42)

    simulation = TrafficSimulation()

    simulation.run(
        total_seconds=120,
        report_interval=10
    )
