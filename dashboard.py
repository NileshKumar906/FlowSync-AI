
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from simulation.traffic_env import TrafficEnv
from dqn_agent import DQNAgent


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FlowSync AI",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 FlowSync AI")
st.caption("Adaptive Traffic Signal Management Using Deep Reinforcement Learning")


# =========================================================
# LOAD DQN
# =========================================================

@st.cache_resource
def load_agent():

    model_path = "models/dqn_adaptive_v1.pth"

    if not os.path.exists(model_path):
        return None, f"Model not found: {model_path}"

    try:
        agent = DQNAgent(state_size=13, action_size=2)
        agent.load(model_path)
        agent.epsilon = 0

        print(f"Vanilla DQN model loaded: {model_path}")

        return agent, None

    except Exception as e:
        return None, str(e)


# =========================================================
# CONTROLLER ACTION
# =========================================================

def select_controller_action(controller, env, state, agent=None):

    if controller == "Fixed-time":

        if env.yellow_remaining > 0:
            return 0

        if env.green_elapsed >= 15:
            return 1

        return 0

    if controller == "Vehicle-actuated":

        if env.yellow_remaining > 0:
            return 0

        green_directions = env.get_green_directions()

        red_directions = [
            d for d in env.queues
            if d not in green_directions
        ]

        current_queue = sum(
            len(env.queues[d])
            for d in green_directions
        )

        opposing_queue = sum(
            len(env.queues[d])
            for d in red_directions
        )

        if env.green_elapsed < env.min_green:
            return 0

        if opposing_queue > current_queue * 1.2 + 2:
            return 1

        return 0

    if controller == "DQN":

        if agent is None:
            raise ValueError("DQN model unavailable.")

        return agent.select_action(state)

    raise ValueError(f"Unknown controller: {controller}")


# =========================================================
# RUN SIMULATION
# =========================================================

def run_simulation(
    controller,
    seed,
    max_steps,
    demand_multiplier,
    agent=None
):

    env = TrafficEnv(max_steps=max_steps)

    for direction in env.arrival_rates:
        env.arrival_rates[direction] *= demand_multiplier

    state, info = env.reset(seed=seed)

    total_reward = 0
    max_queue = 0

    queue_history = []

    direction_history = {
        "North": [],
        "South": [],
        "East": [],
        "West": []
    }

    phase_history = []

    for step in range(max_steps):

        action = select_controller_action(
            controller,
            env,
            state,
            agent
        )

        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        current_queues = env.get_queue_lengths()

        current_total_queue = sum(current_queues.values())

        queue_history.append(current_total_queue)

        for direction in direction_history:

            direction_history[direction].append(
                current_queues[direction]
            )

        if env.yellow_remaining > 0:

            current_phase = "Yellow"

        elif env.phase == 0:

            current_phase = "North-South Green"

        else:

            current_phase = "East-West Green"

        phase_history.append(current_phase)

        max_queue = max(max_queue, current_total_queue)

        state = next_state

        if terminated or truncated:
            break

    average_queue = (
        env.queue_sum / env.time
        if env.time > 0
        else 0
    )

    average_waiting_time = (
        env.total_waiting_time / env.total_departures
        if env.total_departures > 0
        else 0
    )

    return {

        "controller": controller,
        "seed": seed,

        "average_queue": average_queue,
        "average_waiting_time": average_waiting_time,

        "throughput": env.total_departures,
        "max_queue": max_queue,

        "remaining_vehicles": env.get_total_queue(),

        "total_arrivals": env.total_arrivals,
        "total_departures": env.total_departures,

        "total_reward": total_reward,

        "queue_history": queue_history,

        "direction_history": direction_history,

        "phase_history": phase_history,

        "final_queues": env.get_queue_lengths(),

        "final_phase": (
            phase_history[-1]
            if phase_history
            else "Unknown"
        )
    }


# =========================================================
# CORRECTED ANIMATED INTERSECTION
# =========================================================

def create_intersection_animation(episode, controller_name):

    directions = ["North", "South", "East", "West"]

    direction_history = episode["direction_history"]
    phase_history = episode["phase_history"]

    total_frames = len(phase_history)

    if total_frames == 0:
        return None

    ROAD_LIMIT = 20
    ROAD_WIDTH = 4
    LANE_OFFSET = 0.75

    # Number of vehicles displayed per direction
    MAX_CARS = 12

    # Distance between cars
    CAR_SPACING = 2.5

    # Stop-line distance from center
    STOP_LINE = 3.0


    # -----------------------------------------------------
    # VEHICLE POSITIONS
    # -----------------------------------------------------

    def get_vehicle_positions(direction, queue, frame, green):

        positions = []

        queue = max(0, min(int(queue), MAX_CARS))

        # Movement animation cycles gradually.
        # The movement is restricted to the incoming road.
        movement = 0.0

        if green:
            movement = (frame * 0.08) % 2.0

        for car_index in range(queue):

            # Cars are positioned progressively farther
            # away from the intersection.
            base_distance = STOP_LINE + car_index * CAR_SPACING

            # Movement only affects cars approaching the stop line.
            distance = base_distance - movement

            # Keep cars inside their incoming road
            distance = max(STOP_LINE, distance)

            distance = min(
                ROAD_LIMIT - 1,
                distance
            )

            if direction == "North":

                x = -LANE_OFFSET
                y = distance

            elif direction == "South":

                x = LANE_OFFSET
                y = -distance

            elif direction == "East":

                x = distance
                y = -LANE_OFFSET

            elif direction == "West":

                x = -distance
                y = LANE_OFFSET

            else:

                continue

            positions.append((x, y))

        return positions


    # -----------------------------------------------------
    # FRAME DATA
    # -----------------------------------------------------

    def build_frame(frame_index):

        xs = []
        ys = []
        colors = []

        current_phase = phase_history[frame_index]

        ns_green = current_phase == "North-South Green"
        ew_green = current_phase == "East-West Green"

        for direction in directions:

            queue = direction_history[direction][frame_index]

            green = (
                ns_green
                if direction in ["North", "South"]
                else ew_green
            )

            positions = get_vehicle_positions(
                direction,
                queue,
                frame_index,
                green
            )

            for x, y in positions:

                xs.append(x)
                ys.append(y)

                if current_phase == "Yellow":
                    colors.append("#FFB347")

                elif green:
                    colors.append("#39FF88")

                else:
                    colors.append("#FF5252")

        return xs, ys, colors


    # -----------------------------------------------------
    # FIGURE
    # -----------------------------------------------------

    x0, y0, c0 = build_frame(0)

    fig = go.Figure()


    # -----------------------------------------------------
    # ROADS
    # -----------------------------------------------------

    # Vertical road
    fig.add_shape(

        type="rect",

        x0=-ROAD_WIDTH / 2,
        x1=ROAD_WIDTH / 2,

        y0=-ROAD_LIMIT,
        y1=ROAD_LIMIT,

        fillcolor="#30343B",

        line=dict(width=0),

        layer="below"

    )


    # Horizontal road
    fig.add_shape(

        type="rect",

        x0=-ROAD_LIMIT,
        x1=ROAD_LIMIT,

        y0=-ROAD_WIDTH / 2,
        y1=ROAD_WIDTH / 2,

        fillcolor="#30343B",

        line=dict(width=0),

        layer="below"

    )


    # -----------------------------------------------------
    # ROAD MARKINGS
    # -----------------------------------------------------

    fig.add_shape(

        type="line",

        x0=0,
        x1=0,

        y0=-ROAD_LIMIT,
        y1=ROAD_LIMIT,

        line=dict(

            color="white",
            width=2,
            dash="dash"

        ),

        layer="below"

    )

    fig.add_shape(

        type="line",

        x0=-ROAD_LIMIT,
        x1=ROAD_LIMIT,

        y0=0,
        y1=0,

        line=dict(

            color="white",
            width=2,
            dash="dash"

        ),

        layer="below"

    )


    # -----------------------------------------------------
    # INTERSECTION
    # -----------------------------------------------------

    fig.add_shape(

        type="rect",

        x0=-ROAD_WIDTH / 2,
        x1=ROAD_WIDTH / 2,

        y0=-ROAD_WIDTH / 2,
        y1=ROAD_WIDTH / 2,

        fillcolor="#454B54",

        line=dict(

            color="white",
            width=1

        ),

        layer="below"

    )


    # -----------------------------------------------------
    # VEHICLES
    # -----------------------------------------------------

    fig.add_trace(

        go.Scatter(

            x=x0,
            y=y0,

            mode="markers",

            marker=dict(

                size=18,

                color=c0,

                symbol="square",

                line=dict(

                    color="white",
                    width=1

                )

            ),

            name="Vehicles"

        )

    )


    # -----------------------------------------------------
    # SIGNAL LABELS
    # -----------------------------------------------------

    signal_labels = [

        (-3, 3, "N/S"),

        (3, 3, "E/W"),

        (-3, -3, "E/W"),

        (3, -3, "N/S")

    ]

    for sx, sy, label in signal_labels:

        fig.add_annotation(

            x=sx,
            y=sy,

            text=label,

            showarrow=False,

            font=dict(

                color="white",
                size=13

            ),

            bgcolor="#20242B",

            bordercolor="white",

            borderwidth=1,

            borderpad=5

        )


    # -----------------------------------------------------
    # ANIMATION FRAMES
    # -----------------------------------------------------

    frames = []

    for frame_index in range(total_frames):

        xf, yf, cf = build_frame(frame_index)

        frames.append(

            go.Frame(

                data=[

                    go.Scatter(

                        x=xf,
                        y=yf,

                        mode="markers",

                        marker=dict(

                            size=18,

                            color=cf,

                            symbol="square",

                            line=dict(

                                color="white",
                                width=1

                            )

                        )

                    )

                ],

                name=str(frame_index)

            )

        )

    fig.frames = frames


    # -----------------------------------------------------
    # PLAY / PAUSE / SLIDER
    # -----------------------------------------------------

    fig.update_layout(

        title=f"🚦 Animated Intersection — {controller_name}",

        height=750,

        xaxis=dict(

            range=[-ROAD_LIMIT, ROAD_LIMIT],

            visible=False

        ),

        yaxis=dict(

            range=[-ROAD_LIMIT, ROAD_LIMIT],

            visible=False,

            scaleanchor="x"

        ),

        updatemenus=[

            {

                "type": "buttons",

                "direction": "left",

                "x": 0.05,

                "y": 1.15,

                "showactive": False,

                "buttons": [

                    {

                        "label": "▶ Play",

                        "method": "animate",

                        "args": [

                            None,

                            {

                                "frame": {

                                    "duration": 150,

                                    "redraw": True

                                },

                                "transition": {

                                    "duration": 0

                                },

                                "fromcurrent": True,

                                "mode": "immediate"

                            }

                        ]

                    },

                    {

                        "label": "⏸ Pause",

                        "method": "animate",

                        "args": [

                            [None],

                            {

                                "frame": {

                                    "duration": 0,

                                    "redraw": False

                                },

                                "transition": {

                                    "duration": 0

                                }

                            }

                        ]

                    }

                ]

            }

        ],

        sliders=[

            {

                "active": 0,

                "currentvalue": {

                    "prefix": "Time Step: "

                },

                "steps": [

                    {

                        "label": str(i),

                        "method": "animate",

                        "args": [

                            [str(i)],

                            {

                                "frame": {

                                    "duration": 0,

                                    "redraw": True

                                },

                                "mode": "immediate",

                                "transition": {

                                    "duration": 0

                                }

                            }

                        ]

                    }

                    for i in range(total_frames)

                ]

            }

        ],

        showlegend=False

    )

    return fig


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Simulation Settings")

controller_mode = st.sidebar.selectbox(

    "Select Controller",

    [

        "Compare all",

        "Compare both",

        "Fixed-time",

        "Vehicle-actuated",

        "DQN"

    ]

)


simulation_duration = st.sidebar.slider(

    "Simulation Duration (seconds)",

    min_value=100,

    max_value=1200,

    value=600,

    step=100

)


demand_multiplier = st.sidebar.slider(

    "Traffic Demand Multiplier",

    min_value=0.5,

    max_value=2.0,

    value=1.0,

    step=0.1

)


number_of_scenarios = st.sidebar.slider(

    "Number of Test Scenarios",

    min_value=1,

    max_value=10,

    value=5

)


st.sidebar.divider()

st.sidebar.caption(

    "Comparison mode uses identical traffic seeds "
    "for selected controllers to ensure fair evaluation."

)


# =========================================================
# RUN SIMULATION
# =========================================================

if st.button("▶ Run Traffic Simulation", type="primary"):

    agent = None
    model_error = None

    if controller_mode in ["DQN", "Compare both", "Compare all"]:

        with st.spinner("Loading DQN model..."):

            agent, model_error = load_agent()

        if controller_mode == "DQN" and agent is None:

            st.error("DQN model could not be loaded.")
            st.code(model_error)
            st.stop()

        elif agent is None:

            st.warning(
                "DQN model is unavailable. "
                "Non-DQN controllers will run."
            )

            st.code(model_error)


    # Controller list
    if controller_mode == "Compare all":

        controllers = [
            "Fixed-time",
            "Vehicle-actuated"
        ]

        if agent is not None:
            controllers.append("DQN")

    elif controller_mode == "Compare both":

        controllers = ["Fixed-time"]

        if agent is not None:
            controllers.append("DQN")

    else:

        controllers = [controller_mode]


    all_results = {
        controller: []
        for controller in controllers
    }

    progress = st.progress(0)

    status = st.empty()

    total_runs = len(controllers) * number_of_scenarios

    completed_runs = 0


    # Same seeds across controllers
    for seed in range(101, 101 + number_of_scenarios):

        for controller in controllers:

            status.write(
                f"Running {controller} on scenario {seed}..."
            )

            result = run_simulation(

                controller=controller,

                seed=seed,

                max_steps=simulation_duration,

                demand_multiplier=demand_multiplier,

                agent=agent

            )

            all_results[controller].append(result)

            completed_runs += 1

            progress.progress(
                completed_runs / total_runs
            )


    status.success("All simulations completed successfully!")

    st.session_state["results"] = all_results

    st.session_state["settings"] = {

        "duration": simulation_duration,

        "demand": demand_multiplier,

        "scenarios": number_of_scenarios,

        "controller": controller_mode

    }


# =========================================================
# RESULTS
# =========================================================

if "results" in st.session_state:

    results = st.session_state["results"]

    settings = st.session_state["settings"]

    st.divider()

    st.header("📊 Simulation Performance")

    st.caption(

        f"Duration: {settings['duration']} seconds | "
        f"Traffic demand: {settings['demand']}x | "
        f"Test scenarios: {settings['scenarios']}"

    )


    # -----------------------------------------------------
    # SUMMARY TABLE
    # -----------------------------------------------------

    summary = []

    for controller, episodes in results.items():

        if not episodes:
            continue

        summary.append({

            "Controller": controller,

            "Average Queue": np.mean(
                [x["average_queue"] for x in episodes]
            ),

            "Average Waiting Time": np.mean(
                [x["average_waiting_time"] for x in episodes]
            ),

            "Throughput": np.mean(
                [x["throughput"] for x in episodes]
            ),

            "Maximum Queue": np.mean(
                [x["max_queue"] for x in episodes]
            ),

            "Remaining Vehicles": np.mean(
                [x["remaining_vehicles"] for x in episodes]
            ),

            "Total Arrivals": np.mean(
                [x["total_arrivals"] for x in episodes]
            ),

            "Average Reward": np.mean(
                [x["total_reward"] for x in episodes]
            )

        })


    summary_df = pd.DataFrame(summary)


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    st.subheader("Controller Performance")

    for _, row in summary_df.iterrows():

        st.markdown(
            f"### {row['Controller']} Controller"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Average Queue",
            f"{row['Average Queue']:.2f}"
        )

        c2.metric(
            "Average Waiting Time",
            f"{row['Average Waiting Time']:.2f} s"
        )

        c3.metric(
            "Throughput",
            f"{row['Throughput']:.1f}"
        )

        c4.metric(
            "Maximum Queue",
            f"{row['Maximum Queue']:.1f}"
        )


    # -----------------------------------------------------
    # COMPARISON TABLE
    # -----------------------------------------------------

    st.divider()

    st.subheader("📋 Detailed Controller Comparison")

    st.dataframe(
        summary_df.round(2),
        hide_index=True
    )


    # -----------------------------------------------------
    # COMPARISON CHART
    # -----------------------------------------------------

    if len(summary_df) >= 2:

        st.divider()

        st.subheader("📈 Controller Performance Comparison")

        chart_metrics = [

            "Average Queue",

            "Average Waiting Time",

            "Throughput",

            "Maximum Queue"

        ]

        chart_df = summary_df.melt(

            id_vars="Controller",

            value_vars=chart_metrics,

            var_name="Metric",

            value_name="Value"

        )

        fig = px.bar(

            chart_df,

            x="Metric",

            y="Value",

            color="Controller",

            barmode="group",

            title="Traffic Controller Performance",

            text_auto=".2f"

        )

        fig.update_layout(

            xaxis_title="Performance Metric",

            yaxis_title="Value",

            legend_title="Controller"

        )

        st.plotly_chart(fig)


    # -----------------------------------------------------
    # QUEUE OVER TIME
    # -----------------------------------------------------

    st.divider()

    st.subheader("🚗 Traffic Queue Over Time")

    queue_fig = go.Figure()

    for controller, episodes in results.items():

        if not episodes:
            continue

        average_queues = np.mean(

            [

                episode["queue_history"]

                for episode in episodes

            ],

            axis=0

        )

        queue_fig.add_trace(

            go.Scatter(

                x=list(
                    range(1, len(average_queues) + 1)
                ),

                y=average_queues,

                mode="lines",

                name=controller

            )

        )

    queue_fig.update_layout(

        title="Average Total Queue Length",

        xaxis_title="Simulation Time (seconds)",

        yaxis_title="Waiting Vehicles",

        hovermode="x unified"

    )

    st.plotly_chart(queue_fig)


    # -----------------------------------------------------
    # ANIMATION
    # -----------------------------------------------------

    st.divider()

    st.header("🚦 Animated Traffic Intersection")

    st.write(
        "Explore traffic queue states recorded during the simulation."
    )

    animation_controller = st.selectbox(

        "Select controller for animation",

        list(results.keys()),

        key="animation_controller"

    )

    animation_scenario = st.selectbox(

        "Select simulation scenario",

        list(range(len(results[animation_controller]))),

        format_func=lambda x: f"Scenario {x + 1}",

        key="animation_scenario"

    )

    animation_episode = results[
        animation_controller
    ][animation_scenario]

    animation_fig = create_intersection_animation(

        animation_episode,

        animation_controller

    )

    if animation_fig is not None:

        st.plotly_chart(

            animation_fig,

            key="intersection_animation"

        )


    # -----------------------------------------------------
    # FINAL STATUS
    # -----------------------------------------------------

    st.divider()

    st.subheader("🚦 Intersection Status")

    selected_controller = st.selectbox(

        "Choose Controller to Inspect",

        list(results.keys()),

        key="status_controller"

    )

    selected_episode = results[selected_controller][-1]

    st.caption(
        f"Inspecting final state of scenario "
        f"{selected_episode['seed']}"
    )

    st.write(
        "**Final Traffic Signal:**",
        selected_episode["final_phase"]
    )

    directions_columns = st.columns(4)

    for column, direction in zip(

        directions_columns,

        ["North", "South", "East", "West"]

    ):

        column.metric(

            f"{direction} Queue",

            selected_episode["final_queues"][direction]

        )


    # -----------------------------------------------------
    # DIRECTIONAL QUEUES
    # -----------------------------------------------------

    st.subheader("Directional Queue History")

    direction_df = pd.DataFrame(
        selected_episode["direction_history"]
    )

    direction_df.index = np.arange(
        1,
        len(direction_df) + 1
    )

    direction_fig = px.line(

        direction_df,

        labels={

            "index": "Simulation Time (seconds)",

            "value": "Waiting Vehicles",

            "variable": "Direction"

        },

        title="Queue Length by Direction"

    )

    st.plotly_chart(direction_fig)


    # -----------------------------------------------------
    # DETAILS
    # -----------------------------------------------------

    st.divider()

    st.subheader("Simulation Details")

    details = {

        "Controller": selected_episode["controller"],

        "Scenario Seed": selected_episode["seed"],

        "Total Arrivals": selected_episode["total_arrivals"],

        "Total Departures": selected_episode["total_departures"],

        "Remaining Vehicles": selected_episode["remaining_vehicles"],

        "Total Reward": round(
            selected_episode["total_reward"],
            2
        )

    }

    st.json(details)

    st.caption(
        "Results are produced by a software traffic simulation "
        "and do not represent live road traffic measurements."
    )


# =========================================================
# EMPTY STATE
# =========================================================

else:

    st.info(
        "Configure the simulation using the sidebar "
        "and click 'Run Traffic Simulation' to begin."
    )

    st.markdown(

        """
        ### Available Controllers

        - **Fixed-time:** Uses fixed signal timing.
        - **Vehicle-actuated:** Responds to opposing traffic queues.
        - **DQN:** Uses a trained Deep Q-Network.
        - **Compare both:** Compares Fixed-time and DQN.
        - **Compare all:** Compares all available controllers.
        """

    )
