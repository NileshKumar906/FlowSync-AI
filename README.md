# FlowSync AI 🚦
### Emergency-Aware Adaptive Traffic Signal Control Using Deep Reinforcement Learning

FlowSync AI is a software-based traffic signal simulation project that explores intelligent traffic signal control using Deep Reinforcement Learning (DQN). It simulates a four-direction intersection and compares different traffic management strategies under simulated traffic conditions.

## Overview

Traditional traffic signals often use fixed timing schedules that may not respond efficiently to changing traffic conditions. FlowSync AI explores adaptive control strategies to manage traffic queues and improve intersection performance through simulation.

The project implements and compares Fixed-Time, Vehicle-Actuated, and Deep Reinforcement Learning-based traffic controllers.

## Features

- **Multiple Traffic Controllers:** Fixed-Time, Vehicle-Actuated, and DQN-based control.
- **Controller Comparison:** Compare controller performance under simulation scenarios.
- **Performance Metrics:** Average queue, waiting time, throughput, and maximum queue.
- **Traffic Visualization:** Animated four-way intersection.
- **Queue Analysis:** Visualize overall and directional queue behavior over time.
- **Scenario-Based Evaluation:** Evaluate controllers across multiple simulated traffic scenarios.
- **Interactive Dashboard:** Configure and run simulations using Streamlit.

## Controllers

- **Fixed-Time Controller:** Uses predefined signal timing.
- **Vehicle-Actuated Controller:** Adjusts signal behavior according to traffic conditions and opposing queues.
- **Deep Q-Network (DQN):** Uses a trained reinforcement learning model to select traffic signal actions.
- **Double DQN:** An additional reinforcement learning variant evaluated experimentally.

## Technology Stack

- Python
- PyTorch
- Streamlit
- NumPy
- Pandas
- Plotly
- Deep Reinforcement Learning (DQN / Double DQN)

## Evaluation Results

The following results were obtained from a 50-scenario evaluation:

| Controller | Average Queue | Ordinary Waiting Time | Throughput | Emergency Waiting Time | Emergency Service Rate |
|---|---:|---:|---:|---:|---:|
| Fixed-Time | 10.62 | 11.48 | 539.16 | 10.42 | 98.87% |
| Vehicle-Actuated | 6.57 | 7.18 | 546.86 | 7.03 | 99.77% |
| DQN | 28.89 | 29.05 | 499.22 | 22.83 | 95.03% |
| DQN v2 | 28.89 | 29.05 | 499.22 | 22.83 | 95.03% |
| Double DQN | 28.79 | 28.96 | 499.80 | 22.61 | 95.26% |

<sub>These are results from the project's simulated evaluation scenarios and not real-world traffic measurements.</sub>

## Dashboard

The dashboard provides:

- Controller performance metrics
- Controller comparison visualizations
- Traffic queue trends
- Animated traffic intersection
- Directional queue history
- Simulation details

## Project Structure

```text
FlowSync-AI/
├── models/
├── scripts/
├── assets/
├── evaluation_results/
├── dashboard.py
├── requirements.txt
├── README.md
└── ...
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/NileshKumar906/FlowSync-AI.git
cd FlowSync-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the dashboard

```bash
streamlit run dashboard.py
```

## Limitations

- Traffic conditions are generated within a software simulation.
- The project does not use real-world traffic sensor data.
- The simulation does not model a complete real-world road network.
- The evaluated reinforcement learning controllers did not outperform the Vehicle-Actuated baseline in the current evaluation.

## Future Scope

- Improve DQN reward design and training.
- Explore more advanced reinforcement learning algorithms.
- Incorporate more realistic traffic dynamics.
- Extend the simulation to multiple intersections.
- Explore integration with real-world traffic data.

---

**FlowSync AI** — Exploring intelligent and adaptive traffic signal management through reinforcement learning.
