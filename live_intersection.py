
import os
import json
import random

import streamlit as st
import streamlit.components.v1 as components

from simulation.traffic_env import TrafficEnv
from dqn_agent import DQNAgent


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SPEED = 1.0
DEFAULT_CROSSING_DURATION = 5.0

DIRECTIONS = ["North", "South", "East", "West"]

ROUTE_OPTIONS = ["straight", "left", "right"]


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="FlowSync AI | Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(14,165,233,.09), transparent 35%),
        radial-gradient(ellipse at 90% 20%, rgba(139,92,246,.07), transparent 35%),
        #080d17;
    color: #e5edf8;
}

section[data-testid="stSidebar"] {
    background: #101827;
    border-right: 1px solid #26354a;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #f8fafc;
    letter-spacing: -.5px;
}

h1 {
    font-weight: 900;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg,#152238,#101a2b);
    border: 1px solid #2a3c55;
    padding: 16px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #94a9c5;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc;
    font-weight: 800;
}

div.stButton > button {
    border-radius: 10px;
    font-weight: 700;
}

hr {
    border-color: #293b52;
}

</style>
""", unsafe_allow_html=True)


# Native Streamlit header — no raw HTML.
st.caption("INTELLIGENT URBAN MOBILITY SYSTEM")
st.title("🚦 FlowSync AI")
st.markdown(
    "Adaptive traffic signal management powered by "
    "**Deep Reinforcement Learning**."
)


# ============================================================
# LOAD DQN MODEL
# ============================================================

@st.cache_resource
def load_agent():

    model_path = "models/dqn_traffic.pth"

    if not os.path.exists(model_path):
        return None

    agent = DQNAgent()
    agent.load(model_path)

    if hasattr(agent, "epsilon"):
        agent.epsilon = 0

    return agent


# ============================================================
# CONTROLLER LOGIC
# ============================================================

def select_action(controller, env, state, agent=None):

    if controller == "Fixed-time":

        if env.yellow_remaining > 0:
            return 0

        return int(env.green_elapsed >= 15)

    if controller == "Vehicle-actuated":

        if env.yellow_remaining > 0:
            return 0

        if env.green_elapsed < env.min_green:
            return 0

        green_directions = env.get_green_directions()

        red_directions = [
            d for d in DIRECTIONS
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

        return int(
            opposing_queue > current_queue * 1.2 + 2
        )

    if controller == "DQN":

        if agent is None:
            raise RuntimeError(
                "DQN model is unavailable. "
                "Train the model before selecting DQN."
            )

        return int(agent.select_action(state))

    raise ValueError(
        f"Unknown controller: {controller}"
    )


# ============================================================
# QUEUE HELPERS
# ============================================================

def get_queue_lengths(env, info=None):

    if (
        info
        and isinstance(info.get("queue_lengths"), dict)
    ):

        return {
            d: int(info["queue_lengths"].get(d, 0))
            for d in DIRECTIONS
        }

    if hasattr(env, "get_queue_lengths"):

        return {
            d: int(env.get_queue_lengths().get(d, 0))
            for d in DIRECTIONS
        }

    return {
        d: len(env.queues[d])
        for d in DIRECTIONS
    }


def get_departures_by_direction(
    env,
    before,
    after,
    current_time
):

    result = {
        d: 0 for d in DIRECTIONS
    }

    # Use a directional departure dictionary if your
    # environment provides one.
    supplied = getattr(
        env,
        "last_departures_by_direction",
        None
    )

    if isinstance(supplied, dict):

        return {
            d: max(0, int(supplied.get(d, 0)))
            for d in DIRECTIONS
        }

    # Otherwise infer departures using queue timestamps.
    for direction in DIRECTIONS:

        new_arrivals = sum(
            1
            for arrival_time in env.queues[direction]
            if arrival_time == current_time
        )

        departed = (
            before[direction]
            + new_arrivals
            - after[direction]
        )

        result[direction] = max(0, departed)

    return result


# ============================================================
# RUN SIMULATION
# ============================================================

def run_simulation(
    controller,
    duration,
    demand,
    seed,
    route_weights
):

    agent = None

    if controller == "DQN":

        agent = load_agent()

        if agent is None:
            raise FileNotFoundError(
                "models/dqn_traffic.pth was not found."
            )

    env = TrafficEnv(
        max_steps=duration
    )

    for direction in env.arrival_rates:
        env.arrival_rates[direction] *= demand

    state, _ = env.reset(seed=seed)

    route_rng = random.Random(seed + 1000)

    frames = []
    events = []

    total_reward = 0.0
    maximum_queue = 0

    for _ in range(duration):

        before = get_queue_lengths(env)

        action = select_action(
            controller,
            env,
            state,
            agent
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        state = next_state

        total_reward += float(reward)

        after = get_queue_lengths(
            env,
            info
        )

        departures = get_departures_by_direction(
            env,
            before,
            after,
            env.time
        )

        # Create animation events for departed vehicles.
        for direction, count in departures.items():

            for _ in range(count):

                route = route_rng.choices(
                    ROUTE_OPTIONS,
                    weights=route_weights,
                    k=1
                )[0]

                events.append({
                    "time": int(env.time),
                    "direction": direction,
                    "route": route,
                    "id": len(events)
                })

        total_queue = int(
            info.get(
                "total_queue",
                sum(after.values())
            )
        )

        maximum_queue = max(
            maximum_queue,
            total_queue
        )

        frames.append({

            "time": int(env.time),

            "queues": after,

            "phase": int(env.phase),

            "yellow": int(env.yellow_remaining),

            "green_elapsed": int(env.green_elapsed),

            "arrivals": int(env.total_arrivals),

            "departures": int(env.total_departures),

            "total_queue": total_queue,

            "reward": float(reward)

        })

        if terminated or truncated:
            break

    results = {

        "duration": int(env.time),

        "arrivals": int(env.total_arrivals),

        "departures": int(env.total_departures),

        "remaining": int(env.get_total_queue()),

        "average_queue": (
            env.queue_sum / env.time
            if env.time > 0 else 0
        ),

        "average_waiting": (
            env.total_waiting_time / env.total_departures
            if env.total_departures > 0 else 0
        ),

        "maximum_queue": int(maximum_queue),

        "reward": float(total_reward)

    }

    return frames, events, results


# ============================================================
# ANIMATION HTML
# ============================================================

def build_animation_html(
    frames,
    events,
    controller,
    crossing_duration
):

    payload = json.dumps({
        "frames": frames,
        "events": events,
        "controller": controller,
        "crossingDuration": crossing_duration
    })

    # Prevent accidental HTML/script termination.
    payload = payload.replace("</", "<\\/")

    html = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1">

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    color: #e5edf8;
    background: #080d17;
    font-family: Inter, Arial, sans-serif;
}

.app {
    width: 100%;
    overflow: hidden;
    border: 1px solid #263a53;
    border-radius: 17px;
    background: #0d1727;
    box-shadow: 0 15px 50px rgba(0,0,0,.3);
}

/* HEADER */

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    padding: 17px 21px;
    border-bottom: 1px solid #26364a;
    background: linear-gradient(110deg,#152238,#101a2a);
}

.brand {
    font-size: 18px;
    font-weight: 900;
    letter-spacing: 1px;
}

.brand span {
    color: #38bdf8;
}

.live-status {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #4ade80;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.live-dot {
    width: 8px;
    height: 8px;
    background: #4ade80;
    border-radius: 50%;
    box-shadow: 0 0 12px #4ade80;
}

/* METRICS */

.metrics {
    display: grid;
    grid-template-columns: repeat(4,minmax(0,1fr));
    gap: 10px;
    padding: 14px;
}

.metric {
    min-width: 0;
    padding: 14px;
    border: 1px solid #273a52;
    border-radius: 12px;
    background: linear-gradient(145deg,#16243a,#111d2e);
}

.metric-label {
    color: #8fa5c1;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.metric-value {
    margin-top: 7px;
    color: #f8fafc;
    font-size: clamp(19px,3vw,27px);
    font-weight: 900;
    font-variant-numeric: tabular-nums;
}

/* SCENE */

.scene-wrapper {
    padding: 12px;
    background:
        radial-gradient(ellipse at center,
            rgba(14,165,233,.07),
            transparent 65%),
        #0a1220;
}

.scene-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    padding: 6px 8px 14px;
}

.scene-heading h3 {
    margin: 0;
    color: #f1f5f9;
    font-size: 14px;
    font-weight: 800;
}

.scene-heading span {
    color: #8095b1;
    font-size: 10px;
}

.canvas-container {
    width: 100%;
    overflow: hidden;
    border: 1px solid #283a52;
    border-radius: 14px;
    background: #0b1422;
}

canvas {
    display: block;
    width: 100%;
    height: auto;
}

/* SIGNAL STATUS */

.signal-panel {
    display: grid;
    grid-template-columns: repeat(2,minmax(0,1fr));
    gap: 10px;
    padding: 14px;
}

.signal-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding: 12px 14px;
    border: 1px solid #293a50;
    border-radius: 12px;
    background: #121f31;
}

.signal-title {
    color: #a8bad1;
    font-size: 11px;
    font-weight: 700;
}

.signal-state {
    font-size: 11px;
    font-weight: 900;
}

.green-text { color: #4ade80; }
.red-text { color: #fb7185; }
.yellow-text { color: #facc15; }

/* QUEUES */

.queue-section {
    padding: 12px 15px 18px;
}

.section-title {
    margin: 4px 0 12px;
    color: #e2e8f0;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .8px;
    text-transform: uppercase;
}

.queue-grid {
    display: grid;
    grid-template-columns: repeat(4,minmax(0,1fr));
    gap: 9px;
}

.queue-card {
    padding: 12px;
    border: 1px solid #27394f;
    border-radius: 11px;
    background: #121f31;
}

.queue-name {
    color: #91a5c0;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
}

.queue-count {
    margin: 6px 0;
    color: #f8fafc;
    font-size: 22px;
    font-weight: 900;
}

.queue-track {
    height: 5px;
    overflow: hidden;
    border-radius: 10px;
    background: #26364a;
}

.queue-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg,#06b6d4,#3b82f6);
}

/* LEGEND */

.legend {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
    gap: 17px;
    padding: 14px;
    border-top: 1px solid #26364a;
    color: #9aacc4;
    font-size: 10px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 7px;
}

.legend-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
}

/* CONTROLS */

.playback {
    padding: 17px;
    border-top: 1px solid #293b52;
    background: #111d2e;
}

.playback-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 14px;
}

.playback-title strong {
    color: #eaf2ff;
    font-size: 12px;
}

.playback-title span {
    color: #8da3bf;
    font-size: 10px;
}

.controls {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

button, select {
    border: 1px solid #344b66;
    border-radius: 9px;
    padding: 10px 13px;
    color: #e5edf8;
    background: #1a2a40;
    font-size: 11px;
    font-weight: 800;
    cursor: pointer;
}

button:hover {
    background: #263d58;
    border-color: #38bdf8;
}

button:focus-visible,
select:focus-visible,
input:focus-visible {
    outline: 2px solid #38bdf8;
    outline-offset: 3px;
}

#playButton {
    min-width: 105px;
    color: #06111d;
    border: none;
    background: linear-gradient(110deg,#22d3ee,#3b82f6);
}

.speed-select {
    min-width: 120px;
}

.timeline-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 16px;
}

.timeline-row input {
    flex: 1;
    min-width: 60px;
    accent-color: #22d3ee;
    cursor: pointer;
}

.time-readout {
    min-width: 90px;
    color: #a9bdd3;
    font-size: 11px;
    font-weight: 700;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

.playback-note {
    margin-top: 12px;
    color: #7187a3;
    font-size: 10px;
    line-height: 1.5;
}

/* FOOTER */

.footer {
    padding: 13px 18px;
    border-top: 1px solid #26364a;
    color: #71839b;
    font-size: 10px;
    line-height: 1.6;
}

/* RESPONSIVE */

@media(max-width:650px) {

    .metrics {
        grid-template-columns: repeat(2,minmax(0,1fr));
    }

    .queue-grid {
        grid-template-columns: repeat(2,minmax(0,1fr));
    }

    .signal-panel {
        grid-template-columns: 1fr;
    }

    .scene-wrapper {
        padding: 5px;
    }

    .controls button {
        padding: 9px 10px;
    }

}

@media(prefers-reduced-motion: reduce) {

    button {
        transition: none;
    }

}

</style>

</head>

<body>

<div class="app">

    <div class="topbar">

        <div class="brand">
            FLOW<span>SYNC</span> AI
        </div>

        <div class="live-status">
            <div class="live-dot"></div>
            SIMULATION PLAYBACK
        </div>

    </div>


    <div class="metrics">

        <div class="metric">
            <div class="metric-label">Simulation Time</div>
            <div class="metric-value" id="timeValue">0s</div>
        </div>

        <div class="metric">
            <div class="metric-label">Total Queue</div>
            <div class="metric-value" id="queueValue">0</div>
        </div>

        <div class="metric">
            <div class="metric-label">Vehicles Arrived</div>
            <div class="metric-value" id="arrivalValue">0</div>
        </div>

        <div class="metric">
            <div class="metric-label">Vehicles Departed</div>
            <div class="metric-value" id="departureValue">0</div>
        </div>

    </div>


    <div class="scene-wrapper">

        <div class="scene-heading">
            <h3>LIVE INTERSECTION SIMULATOR</h3>
            <span id="sceneTime">INITIALIZING</span>
        </div>

        <div class="canvas-container">

            <canvas
                id="trafficCanvas"
                width="1100"
                height="850">
            </canvas>

        </div>

    </div>


    <div class="signal-panel">

        <div class="signal-card">
            <div class="signal-title">
                NORTH / SOUTH SIGNAL
            </div>
            <div class="signal-state green-text" id="nsState">
                GREEN
            </div>
        </div>

        <div class="signal-card">
            <div class="signal-title">
                EAST / WEST SIGNAL
            </div>
            <div class="signal-state red-text" id="ewState">
                RED
            </div>
        </div>

    </div>


    <div class="queue-section">

        <div class="section-title">
            Directional Queue Monitor
        </div>

        <div class="queue-grid">

            <div class="queue-card">
                <div class="queue-name">North</div>
                <div class="queue-count" id="northQueue">0</div>
                <div class="queue-track">
                    <div class="queue-fill" id="northFill"></div>
                </div>
            </div>

            <div class="queue-card">
                <div class="queue-name">South</div>
                <div class="queue-count" id="southQueue">0</div>
                <div class="queue-track">
                    <div class="queue-fill" id="southFill"></div>
                </div>
            </div>

            <div class="queue-card">
                <div class="queue-name">East</div>
                <div class="queue-count" id="eastQueue">0</div>
                <div class="queue-track">
                    <div class="queue-fill" id="eastFill"></div>
                </div>
            </div>

            <div class="queue-card">
                <div class="queue-name">West</div>
                <div class="queue-count" id="westQueue">0</div>
                <div class="queue-track">
                    <div class="queue-fill" id="westFill"></div>
                </div>
            </div>

        </div>

    </div>


    <div class="legend">

        <div class="legend-item">
            <span class="legend-dot" style="background:#38bdf8"></span>
            Straight
        </div>

        <div class="legend-item">
            <span class="legend-dot" style="background:#a78bfa"></span>
            Left turn
        </div>

        <div class="legend-item">
            <span class="legend-dot" style="background:#fbbf24"></span>
            Right turn
        </div>

        <div class="legend-item">
            <span class="legend-dot" style="background:#22c55e"></span>
            Green
        </div>

        <div class="legend-item">
            <span class="legend-dot" style="background:#ef4444"></span>
            Red
        </div>

    </div>


    <div class="playback">

        <div class="playback-title">
            <strong>SIMULATION PLAYBACK CONTROL</strong>
            <span id="playbackStatus">READY</span>
        </div>

        <div class="controls">

            <button id="playButton">▶ PLAY</button>

            <button id="stepBack">◀</button>

            <button id="stepForward">▶</button>

            <button id="replayButton">↻ REPLAY</button>

            <select id="speedSelect" class="speed-select">

                <option value="0.5">Slow — 0.5x</option>

                <option value="1.0" selected>Normal — 1x</option>

                <option value="1.5">Fast — 1.5x</option>

                <option value="2.0">Faster — 2x</option>

                <option value="3.0">Very fast — 3x</option>

            </select>

        </div>


        <div class="timeline-row">

            <input
                type="range"
                id="timeline"
                min="0"
                max="100"
                value="0"
                step="0.01">

            <div class="time-readout" id="timeLabel">
                0 / 0 sec
            </div>

        </div>

        <div class="playback-note">
            Default playback: 1 simulation second per real second.
            Vehicle crossing animation: 5 simulation seconds.
            Pause, step through the simulation, or inspect any point
            on the timeline.
        </div>

    </div>


    <div class="footer">
        FLOWSYNC AI • SOFTWARE-BASED SIMULATION • NO LIVE ROAD DATA
        <br>
        Vehicle turning paths are illustrative. Collision avoidance,
        pedestrian behavior and real-world vehicle dynamics are not
        implemented.
    </div>

</div>


<script>

const DATA = __DATA__;

const frames = DATA.frames || [];
const events = DATA.events || [];

const CROSS_DURATION = Number(DATA.crossingDuration || 5);

const canvas = document.getElementById("trafficCanvas");
const ctx = canvas.getContext("2d");

const W = canvas.width;
const H = canvas.height;

const CX = W/2;
const CY = H/2;

const SCALE = 32;
const ROAD_HALF = 3.5;

const playButton = document.getElementById("playButton");
const stepBackButton = document.getElementById("stepBack");
const stepForwardButton = document.getElementById("stepForward");
const replayButton = document.getElementById("replayButton");

const timeline = document.getElementById("timeline");
const speedSelect = document.getElementById("speedSelect");

const playbackStatus = document.getElementById("playbackStatus");

const timeValue = document.getElementById("timeValue");
const queueValue = document.getElementById("queueValue");
const arrivalValue = document.getElementById("arrivalValue");
const departureValue = document.getElementById("departureValue");

const sceneTime = document.getElementById("sceneTime");
const timeLabel = document.getElementById("timeLabel");

const nsState = document.getElementById("nsState");
const ewState = document.getElementById("ewState");

const TOTAL = Math.max(0,frames.length-1);

timeline.max = TOTAL;

let currentTime = 0;
let playing = false;
let lastTimestamp = null;
let lastMetricIndex = -1;
let animationHandle = null;


// ============================================================
// DRAWING HELPERS
// ============================================================

function sx(x) {
    return CX+x*SCALE;
}

function sy(y) {
    return CY-y*SCALE;
}

function roundedRect(x,y,w,h,r,color) {

    ctx.beginPath();
    ctx.roundRect(x,y,w,h,r);

    ctx.fillStyle = color;
    ctx.fill();

}

function drawLine(x1,y1,x2,y2,color,width=1) {

    ctx.beginPath();
    ctx.moveTo(x1,y1);
    ctx.lineTo(x2,y2);

    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();

}


// ============================================================
// BACKGROUND
// ============================================================

function drawBackground() {

    ctx.fillStyle = "#0a1220";
    ctx.fillRect(0,0,W,H);

    ctx.strokeStyle = "rgba(148,163,184,.045)";
    ctx.lineWidth = 1;

    for(let x=0;x<W;x+=40) {
        drawLine(x,0,x,H,ctx.strokeStyle);
    }

    for(let y=0;y<H;y+=40) {
        drawLine(0,y,W,y,ctx.strokeStyle);
    }

}


// ============================================================
// ROAD
// ============================================================

function drawRoad() {

    drawBackground();

    const rw = ROAD_HALF*SCALE;

    ctx.shadowColor = "rgba(0,0,0,.4)";
    ctx.shadowBlur = 25;

    ctx.fillStyle = "#344256";

    ctx.fillRect(CX-rw,0,rw*2,H);
    ctx.fillRect(0,CY-rw,W,rw*2);

    ctx.shadowBlur = 0;

    ctx.fillStyle = "#303d50";

    ctx.fillRect(
        CX-rw,
        CY-rw,
        rw*2,
        rw*2
    );

    ctx.setLineDash([17,18]);

    drawLine(
        CX,0,CX,H,
        "rgba(226,232,240,.5)",2
    );

    drawLine(
        0,CY,W,CY,
        "rgba(226,232,240,.5)",2
    );

    ctx.setLineDash([]);

    // Stop lines

    drawLine(
        CX-rw,CY-rw-12,
        CX+rw,CY-rw-12,
        "#e2e8f0",5
    );

    drawLine(
        CX-rw,CY+rw+12,
        CX+rw,CY+rw+12,
        "#e2e8f0",5
    );

    drawLine(
        CX+rw+12,CY-rw,
        CX+rw+12,CY+rw,
        "#e2e8f0",5
    );

    drawLine(
        CX-rw-12,CY-rw,
        CX-rw-12,CY+rw,
        "#e2e8f0",5
    );

    // Direction labels

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "800 17px Inter,Arial";
    ctx.fillStyle = "#d6e3f4";

    ctx.fillText("NORTH",CX,65);
    ctx.fillText("SOUTH",CX,H-55);
    ctx.fillText("WEST",70,CY);
    ctx.fillText("EAST",W-70,CY);

}


// ============================================================
// VEHICLES
// ============================================================

function drawCar(x,y,direction,color,heading=null,scale=1) {

    ctx.save();

    ctx.translate(sx(x),sy(y));

    if(heading!==null) {
        ctx.rotate(heading+Math.PI/2);
    } else if(direction==="East" || direction==="West") {
        ctx.rotate(Math.PI/2);
    }

    ctx.scale(scale,scale);

    const cw=20;
    const ch=36;

    ctx.shadowColor="rgba(0,0,0,.5)";
    ctx.shadowBlur=9;
    ctx.shadowOffsetY=3;

    roundedRect(
        -cw/2,-ch/2,cw,ch,6,color
    );

    ctx.shadowBlur=0;
    ctx.shadowOffsetY=0;

    roundedRect(
        -cw/2+3,-ch/2+8,cw-6,18,4,
        "#c8e7f7"
    );

    roundedRect(
        -cw/2+4,-ch/2+9,cw-8,6,2,
        "#eaf8ff"
    );

    roundedRect(
        -cw/2+4,4,cw-8,7,2,
        "#19324a"
    );

    ctx.fillStyle="#fff7d6";
    ctx.fillRect(-7,-ch/2,4,3);
    ctx.fillRect(3,-ch/2,4,3);

    ctx.fillStyle="#fb7185";
    ctx.fillRect(-7,ch/2-3,4,3);
    ctx.fillRect(3,ch/2-3,4,3);

    ctx.restore();

}


// ============================================================
// QUEUED VEHICLES
// ============================================================

function drawQueues(frame) {

    const q=frame.queues || {};

    const spacing=.88;

    const configs=[

        {
            direction:"North",
            x:-1,
            y:4.25,
            dx:0,
            dy:spacing
        },

        {
            direction:"South",
            x:1,
            y:-4.25,
            dx:0,
            dy:-spacing
        },

        {
            direction:"East",
            x:4.25,
            y:1,
            dx:spacing,
            dy:0
        },

        {
            direction:"West",
            x:-4.25,
            y:-1,
            dx:-spacing,
            dy:0
        }

    ];

    for(const c of configs) {

        const count=Math.min(
            Number(q[c.direction]||0),
            10
        );

        for(let i=0;i<count;i++) {

            drawCar(
                c.x+c.dx*i,
                c.y+c.dy*i,
                c.direction,
                "#38bdf8",
                null,
                .88
            );

        }

    }

}


// ============================================================
// ROUTE GEOMETRY
// ============================================================

const ROUTES={

    North:{

        straight:[[-1,3.8],[-1,1.5],[-1,-1.5],[-1,-3.8]],

        left:[[-1,3.8],[-1,1.1],[1,1],[3.8,1]],

        right:[[-1,3.8],[-1,1.1],[-2,-1],[-3.8,-1]]

    },

    South:{

        straight:[[1,-3.8],[1,-1.5],[1,1.5],[1,3.8]],

        left:[[1,-3.8],[1,-1.1],[-1,-1],[-3.8,-1]],

        right:[[1,-3.8],[1,-1.1],[2,1],[3.8,1]]

    },

    East:{

        straight:[[3.8,1],[1.5,1],[-1.5,1],[-3.8,1]],

        left:[[3.8,1],[1.1,1],[1,-1],[1,-3.8]],

        right:[[3.8,1],[1.1,1],[-1,2],[-1,3.8]]

    },

    West:{

        straight:[[-3.8,-1],[-1.5,-1],[1.5,-1],[3.8,-1]],

        left:[[-3.8,-1],[-1.1,-1],[-1,1],[-1,3.8]],

        right:[[-3.8,-1],[-1.1,-1],[1,-2],[1,-3.8]]

    }

};


const ROUTE_COLORS={

    straight:"#38bdf8",

    left:"#a78bfa",

    right:"#fbbf24"

};


function bezierPoint(p,t) {

    const [p0,p1,p2,p3]=p;

    const u=1-t;

    return [

        u*u*u*p0[0]+
        3*u*u*t*p1[0]+
        3*u*t*t*p2[0]+
        t*t*t*p3[0],

        u*u*u*p0[1]+
        3*u*u*t*p1[1]+
        3*u*t*t*p2[1]+
        t*t*t*p3[1]

    ];

}


function bezierHeading(p,t) {

    const [p0,p1,p2,p3]=p;

    const u=1-t;

    const dx=
        3*u*u*(p1[0]-p0[0])+
        6*u*t*(p2[0]-p1[0])+
        3*t*t*(p3[0]-p2[0]);

    const dy=
        3*u*u*(p1[1]-p0[1])+
        6*u*t*(p2[1]-p1[1])+
        3*t*t*(p3[1]-p2[1]);

    return Math.atan2(-dy,dx);

}


// ============================================================
// MOVING VEHICLES
// ============================================================

function drawMovingVehicles(simTime) {

    for(const event of events) {

        const elapsed=simTime-event.time;

        if(elapsed<0 || elapsed>CROSS_DURATION) {
            continue;
        }

        const progress=Math.max(
            0,
            Math.min(1,elapsed/CROSS_DURATION)
        );

        const direction=event.direction;
        const route=event.route || "straight";

        if(!ROUTES[direction]) continue;

        const path=ROUTES[direction][route];

        if(!path) continue;

        const [x,y]=bezierPoint(path,progress);

        const heading=bezierHeading(path,progress);

        drawCar(
            x,
            y,
            direction,
            ROUTE_COLORS[route],
            heading,
            1
        );

    }

}


// ============================================================
// TRAFFIC SIGNALS
// ============================================================

function getSignalStates(frame) {

    let ns="green";
    let ew="red";

    if(frame.yellow>0) {

        if(frame.phase===0) {
            ns="yellow";
        } else {
            ew="yellow";
        }

    } else if(frame.phase===1) {

        ns="red";
        ew="green";

    }

    return {ns,ew};

}


function drawSignal(x,y,status,label) {

    const colors={

        red:"#ef4444",
        yellow:"#facc15",
        green:"#22c55e"

    };

    const px=sx(x);
    const py=sy(y);

    ctx.save();

    roundedRect(
        px-17,py-29,34,58,13,
        "#111827"
    );

    const lights=[
        ["red",-17],
        ["yellow",0],
        ["green",17]
    ];

    for(const [name,offset] of lights) {

        const active=name===status;

        ctx.beginPath();

        ctx.arc(
            px,
            py+offset,
            6,
            0,
            Math.PI*2
        );

        ctx.fillStyle=active
            ? colors[name]
            : "#344256";

        if(active) {
            ctx.shadowColor=colors[name];
            ctx.shadowBlur=16;
        }

        ctx.fill();

        ctx.shadowBlur=0;

    }

    ctx.restore();

    ctx.textAlign="center";
    ctx.font="bold 11px Inter,Arial";
    ctx.fillStyle="#cbd5e1";

    ctx.fillText(
        label,
        px,
        py+47
    );

}


function drawSignals(frame) {

    const states=getSignalStates(frame);

    drawSignal(-2.7,4.0,states.ns,"N/S");
    drawSignal(2.7,-4.0,states.ns,"S/N");

    drawSignal(4.0,2.7,states.ew,"E/W");
    drawSignal(-4.0,-2.7,states.ew,"W/E");

}


// ============================================================
// CENTER STATUS
// ============================================================

function drawCenterPanel(frame) {

    let heading;

    if(frame.yellow>0) {

        heading="YELLOW TRANSITION";

    } else if(frame.phase===0) {

        heading="NORTH / SOUTH GREEN";

    } else {

        heading="EAST / WEST GREEN";

    }

    ctx.save();

    roundedRect(
        CX-145,CY-45,290,90,14,
        "rgba(8,15,28,.95)"
    );

    ctx.strokeStyle="rgba(34,211,238,.7)";
    ctx.lineWidth=2;

    ctx.beginPath();
    ctx.roundRect(CX-145,CY-45,290,90,14);
    ctx.stroke();

    ctx.textAlign="center";

    ctx.font="900 15px Inter,Arial";
    ctx.fillStyle="#67e8f9";

    ctx.fillText(heading,CX,CY-8);

    ctx.font="bold 11px Inter,Arial";
    ctx.fillStyle="#d4dfef";

    ctx.fillText(
        DATA.controller.toUpperCase()+" CONTROLLER",
        CX,
        CY+17
    );

    ctx.font="10px Inter,Arial";
    ctx.fillStyle="#8da3bf";

    ctx.fillText(
        "TIME "+frame.time+"s",
        CX,
        CY+34
    );

    ctx.restore();

}


// ============================================================
// RENDER SCENE
// ============================================================

function drawScene(simTime) {

    if(!frames.length) return;

    const index=Math.min(
        Math.floor(simTime),
        frames.length-1
    );

    const frame=frames[index];

    drawRoad();

    drawQueues(frame);

    drawMovingVehicles(simTime);

    drawSignals(frame);

    drawCenterPanel(frame);

}


// ============================================================
// UPDATE DASHBOARD METRICS
// ============================================================

function updateQueues(queues) {

    const directions=[
        "North","South","East","West"
    ];

    for(const direction of directions) {

        const value=Number(queues[direction]||0);

        const key=direction.toLowerCase();

        document.getElementById(
            key+"Queue"
        ).textContent=value;

        document.getElementById(
            key+"Fill"
        ).style.width=Math.min(100,value*7)+"%";

    }

}


function updateSignalStatus(frame) {

    const states=getSignalStates(frame);

    nsState.textContent=states.ns.toUpperCase();
    ewState.textContent=states.ew.toUpperCase();

    nsState.className="signal-state "+states.ns+"-text";
    ewState.className="signal-state "+states.ew+"-text";

}


function updateMetrics(simTime) {

    const index=Math.min(
        Math.floor(simTime),
        frames.length-1
    );

    if(index===lastMetricIndex) return;

    lastMetricIndex=index;

    const frame=frames[index];

    timeValue.textContent=frame.time+"s";

    queueValue.textContent=frame.total_queue;

    arrivalValue.textContent=frame.arrivals;

    departureValue.textContent=frame.departures;

    sceneTime.textContent="SIMULATION SECOND "+frame.time;

    timeLabel.textContent=frame.time+" / "+TOTAL+" sec";

    timeline.value=simTime;

    updateQueues(frame.queues);

    updateSignalStatus(frame);

}


function render() {

    drawScene(currentTime);

    updateMetrics(currentTime);

}


// ============================================================
// PLAYBACK ENGINE
// ============================================================

function setStatus(status) {

    playbackStatus.textContent=status;

}


function pausePlayback() {

    playing=false;

    lastTimestamp=null;

    playButton.textContent="▶ PLAY";

    setStatus("PAUSED");

    if(animationHandle!==null) {

        cancelAnimationFrame(animationHandle);

        animationHandle=null;

    }

}


function animate(timestamp) {

    if(!playing) return;

    if(lastTimestamp===null) {
        lastTimestamp=timestamp;
    }

    // Avoid large jumps if the browser tab was inactive.
    const realDelta=Math.min(
        (timestamp-lastTimestamp)/1000,
        0.1
    );

    lastTimestamp=timestamp;

    const speed=Number(speedSelect.value);

    currentTime+=realDelta*speed;

    if(currentTime>=TOTAL) {

        currentTime=TOTAL;

        playing=false;

        playButton.textContent="↻ REPLAY";

        setStatus("COMPLETED");

    }

    render();

    if(playing) {

        animationHandle=requestAnimationFrame(animate);

    } else {

        animationHandle=null;

    }

}


function startPlayback() {

    if(TOTAL<=0) return;

    if(currentTime>=TOTAL) {
        currentTime=0;
    }

    playing=true;

    lastTimestamp=null;

    playButton.textContent="Ⅱ PAUSE";

    setStatus("PLAYING");

    animationHandle=requestAnimationFrame(animate);

}


// ============================================================
// CONTROLS
// ============================================================

function stepSimulation(amount) {

    pausePlayback();

    currentTime=Math.max(
        0,
        Math.min(
            TOTAL,
            Math.floor(currentTime)+amount
        )
    );

    render();

    setStatus("MANUAL STEP");

}


playButton.addEventListener("click",()=>{

    if(playing) {

        pausePlayback();

    } else {

        startPlayback();

    }

});


stepBackButton.addEventListener("click",()=>{

    stepSimulation(-1);

});


stepForwardButton.addEventListener("click",()=>{

    stepSimulation(1);

});


replayButton.addEventListener("click",()=>{

    pausePlayback();

    currentTime=0;

    render();

    startPlayback();

});


timeline.addEventListener("input",()=>{

    pausePlayback();

    currentTime=Number(timeline.value);

    render();

    setStatus("TIMELINE SCRUBBING");

});


speedSelect.addEventListener("change",()=>{

    if(playing) {
        setStatus("PLAYING AT "+speedSelect.value+"x");
    }

});


// ============================================================
// INITIALIZATION
// ============================================================

if(frames.length>0) {

    currentTime=0;

    render();

    startPlayback();

} else {

    setStatus("NO SIMULATION DATA");

}

</script>

</body>
</html>
"""

    return html.replace("__DATA__", payload)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Simulation Settings")

controller = st.sidebar.selectbox(
    "Traffic controller",
    [
        "DQN",
        "Vehicle-actuated",
        "Fixed-time"
    ]
)

duration = st.sidebar.slider(
    "Simulation duration (seconds)",
    100,
    1200,
    300,
    100
)

demand = st.sidebar.slider(
    "Traffic demand multiplier",
    0.5,
    2.0,
    1.0,
    0.1
)

seed = st.sidebar.number_input(
    "Scenario seed",
    min_value=1,
    max_value=99999,
    value=42
)

st.sidebar.divider()

st.sidebar.subheader("🚗 Vehicle Animation")

crossing_duration = st.sidebar.slider(
    "Vehicle crossing duration",
    min_value=3.0,
    max_value=10.0,
    value=DEFAULT_CROSSING_DURATION,
    step=0.5
)

st.sidebar.caption(
    "This controls visual vehicle movement, not simulation physics."
)

st.sidebar.divider()

st.sidebar.subheader("🎨 Route Distribution")

straight_percentage = st.sidebar.slider(
    "Straight movement",
    0,
    100,
    60
)

left_percentage = st.sidebar.slider(
    "Left turns",
    0,
    100,
    20
)

right_percentage = st.sidebar.slider(
    "Right turns",
    0,
    100,
    20
)

route_total = (
    straight_percentage
    + left_percentage
    + right_percentage
)

st.sidebar.caption(
    f"Total: {route_total}%"
)

if route_total != 100:

    st.sidebar.warning(
        "Route percentages must total 100%."
    )

st.sidebar.divider()

st.sidebar.subheader("Playback defaults")

st.sidebar.metric(
    "Default playback speed",
    f"{DEFAULT_SPEED:.1f}x"
)

st.sidebar.metric(
    "Vehicle crossing time",
    f"{crossing_duration:.1f}s"
)

st.sidebar.caption(
    "Change playback speed directly inside the animation."
)


# ============================================================
# MAIN SIMULATION
# ============================================================

st.divider()

st.subheader("Simulation Workspace")

st.caption(
    "Configure your traffic conditions and generate an "
    "interactive traffic simulation."
)

run_button = st.button(
    "🚦 Generate & Run Simulation",
    type="primary",
    width="stretch"
)


if run_button:

    if route_total != 100:

        st.error(
            "The route percentages must add up to 100%."
        )

        st.stop()

    route_weights = [
        straight_percentage,
        left_percentage,
        right_percentage
    ]

    if sum(route_weights) == 0:

        st.error(
            "At least one route must have a non-zero percentage."
        )

        st.stop()

    try:

        with st.spinner(
            "Generating traffic simulation..."
        ):

            frames, events, results = run_simulation(
                controller=controller,
                duration=duration,
                demand=demand,
                seed=int(seed),
                route_weights=route_weights
            )

        st.success(
            "Simulation generated successfully."
        )

        st.subheader("🚦 Live Intersection")

        animation_html = build_animation_html(
            frames=frames,
            events=events,
            controller=controller,
            crossing_duration=crossing_duration
        )

        components.html(
            animation_html,
            height=1050,
            scrolling=False
        )

        # RESULTS

        st.divider()

        st.header("📊 Simulation Results")

        col1,col2,col3,col4 = st.columns(4)

        col1.metric(
            "Average Queue",
            f"{results['average_queue']:.2f}"
        )

        col2.metric(
            "Average Waiting Time",
            f"{results['average_waiting']:.2f}s"
        )

        col3.metric(
            "Total Throughput",
            results["departures"]
        )

        col4.metric(
            "Maximum Queue",
            results["maximum_queue"]
        )

        col1,col2,col3 = st.columns(3)

        col1.metric(
            "Total Arrivals",
            results["arrivals"]
        )

        col2.metric(
            "Remaining Vehicles",
            results["remaining"]
        )

        col3.metric(
            "Total Reward",
            f"{results['reward']:.2f}"
        )

        # ROUTE SUMMARY

        st.divider()

        st.subheader("🚘 Route Distribution")

        route_counts = {
            "straight":0,
            "left":0,
            "right":0
        }

        for event in events:

            route_counts[event["route"]] += 1

        total_routed = sum(route_counts.values())

        route_columns = st.columns(3)

        for column,route,label in zip(
            route_columns,
            ["straight","left","right"],
            ["Straight","Left Turns","Right Turns"]
        ):

            percentage = (
                route_counts[route]/total_routed*100
                if total_routed > 0 else 0
            )

            column.metric(
                label,
                route_counts[route],
                f"{percentage:.1f}%"
            )

        st.caption(
            "Route choices are generated for visualization. "
            "The current DQN agent controls signal phases, "
            "not individual vehicle routes."
        )

        # DIRECTIONAL DEPARTURES

        st.divider()

        st.subheader("Directional Departure Summary")

        directional_counts = {
            direction:0 for direction in DIRECTIONS
        }

        for event in events:

            direction = event["direction"]

            if direction in directional_counts:
                directional_counts[direction] += 1

        departure_columns = st.columns(4)

        for column,direction in zip(
            departure_columns,
            DIRECTIONS
        ):

            column.metric(
                f"{direction} departures",
                directional_counts[direction]
            )

        # TECHNICAL DETAILS

        with st.expander("Simulation details"):

            st.json({

                "Controller":controller,

                "Scenario seed":int(seed),

                "Simulation duration":results["duration"],

                "Total arrivals":results["arrivals"],

                "Total departures":results["departures"],

                "Remaining vehicles":results["remaining"],

                "Average queue":round(
                    results["average_queue"],2
                ),

                "Average waiting time":round(
                    results["average_waiting"],2
                ),

                "Vehicle crossing duration":crossing_duration,

                "Default playback speed":DEFAULT_SPEED

            })

        st.info(
            "Playback speed and vehicle crossing duration "
            "change only the visual replay. They do not "
            "change the underlying RL simulation metrics."
        )

        st.caption(
            "This is a software-based traffic simulation, "
            "not a live road monitoring system. Vehicle routes "
            "are illustrative and do not implement collision avoidance."
        )

    except Exception as error:

        st.error(
            f"Simulation failed: {error}"
        )

        st.exception(error)


else:

    st.info(
        "Configure your simulation and click "
        "'Generate & Run Simulation' to begin."
    )

    st.markdown("""
    ### Available capabilities

    - Deep Q-Network traffic signal controller
    - Fixed-time and vehicle-actuated baselines
    - Four-direction traffic simulation
    - Smooth vehicle animation
    - Straight, left and right movement paths
    - Pause, resume, replay and timeline controls
    - Five playback speeds
    - Directional queue monitoring
    - Traffic signal visualization
    - Simulation performance metrics
    """)

    st.markdown("""
    ### Recommended experiment

    | Setting | Value |
    |---|---:|
    | Controller | DQN |
    | Duration | 300 seconds |
    | Traffic demand | 1.0x |
    | Crossing duration | 5 seconds |
    | Playback speed | 1.0x |
    | Straight routes | 60% |
    | Left turns | 20% |
    | Right turns | 20% |
    """)
