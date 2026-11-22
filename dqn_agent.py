
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ======================================================
# 1. Q-NETWORK
# ======================================================

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)

        )

    def forward(self, state):

        return self.network(state)


# ======================================================
# 2. DQN AGENT
# ======================================================

class DQNAgent:

    def __init__(
        self,
        state_size=13,
        action_size=2,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.985,
        memory_size=20000,
        batch_size=64,
        double_dqn=False
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        # Select algorithm.
        self.double_dqn = double_dqn

        self.algorithm_name = (
            "Double DQN" if double_dqn else "Vanilla DQN"
        )

        # ----------------------------------------------
        # EXPLORATION SETTINGS
        # ----------------------------------------------

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # ----------------------------------------------
        # TRAINING SETTINGS
        # ----------------------------------------------

        self.batch_size = batch_size

        self.memory = deque(
            maxlen=memory_size
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ----------------------------------------------
        # POLICY NETWORK
        # ----------------------------------------------

        self.policy_net = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        # ----------------------------------------------
        # TARGET NETWORK
        # ----------------------------------------------

        self.target_net = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.update_target_network()

        self.target_net.eval()

        # ----------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.SmoothL1Loss()

        self.training_steps = 0

    # ==================================================
    # 3. ACTION SELECTION
    # ==================================================

    def select_action(self, state):

        state = np.asarray(
            state,
            dtype=np.float32
        )

        if state.shape != (self.state_size,):

            raise ValueError(
                f"Expected state shape ({self.state_size},), "
                f"received {state.shape}"
            )

        if random.random() < self.epsilon:

            return random.randrange(
                self.action_size
            )

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_net(
                state_tensor
            )

        return q_values.argmax(
            dim=1
        ).item()

    # ==================================================
    # 4. STORE EXPERIENCE
    # ==================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        terminated
    ):

        state = np.array(
            state,
            dtype=np.float32,
            copy=True
        )

        next_state = np.array(
            next_state,
            dtype=np.float32,
            copy=True
        )

        if state.shape != (self.state_size,):

            raise ValueError(
                f"Invalid state shape: {state.shape}"
            )

        if next_state.shape != (self.state_size,):

            raise ValueError(
                f"Invalid next state shape: {next_state.shape}"
            )

        self.memory.append(
            (
                state,
                int(action),
                float(reward),
                next_state,
                bool(terminated)
            )
        )

    # ==================================================
    # 5. EXPERIENCE REPLAY / LEARNING
    # ==================================================

    def learn(self):

        if len(self.memory) < self.batch_size:

            return None

        # ----------------------------------------------
        # SAMPLE EXPERIENCES
        # ----------------------------------------------

        batch = random.sample(
            self.memory,
            self.batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        states = torch.as_tensor(
            np.asarray(states),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.as_tensor(
            np.asarray(next_states),
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        # ----------------------------------------------
        # CURRENT Q-VALUES
        # ----------------------------------------------

        current_q_values = self.policy_net(
            states
        ).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # ----------------------------------------------
        # NEXT Q-VALUES
        # ----------------------------------------------

        with torch.no_grad():

            if self.double_dqn:

                # DOUBLE DQN:
                # Policy network selects the best action.

                next_actions = self.policy_net(
                    next_states
                ).argmax(
                    dim=1,
                    keepdim=True
                )

                # Target network evaluates the selected action.

                next_q_values = self.target_net(
                    next_states
                ).gather(
                    1,
                    next_actions
                ).squeeze(1)

            else:

                # VANILLA DQN:
                # Target network selects and evaluates the action.

                next_q_values = self.target_net(
                    next_states
                ).max(
                    dim=1
                )[0]

            # ------------------------------------------
            # BELLMAN TARGET
            # ------------------------------------------

            target_q_values = (
                rewards
                + self.gamma
                * next_q_values
                * (1.0 - dones)
            )

        # ----------------------------------------------
        # LOSS
        # ----------------------------------------------

        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # ----------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        self.training_steps += 1

        return loss.item()

    # ==================================================
    # 6. TARGET NETWORK UPDATE
    # ==================================================

    def update_target_network(self):

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    # ==================================================
    # 7. EPSILON DECAY
    # ==================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    # ==================================================
    # 8. SAVE MODEL
    # ==================================================

    def save(self, path):

        torch.save(
            self.policy_net.state_dict(),
            path
        )

        print(
            f"{self.algorithm_name} model saved: {path}"
        )

    # ==================================================
    # 9. LOAD MODEL
    # ==================================================

    def load(self, path):

        state_dict = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )

        self.policy_net.load_state_dict(
            state_dict
        )

        self.update_target_network()

        self.policy_net.eval()

        print(
            f"{self.algorithm_name} model loaded: {path}"
        )
