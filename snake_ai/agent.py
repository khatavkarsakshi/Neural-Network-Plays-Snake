import math
import random
from collections import deque

import numpy as np

from model import NeuralNetwork

MAX_MEMORY = 100_000
BATCH_SIZE = 1000


class Agent:
    """
    Q-learning agent. The Q-function is a NumPy neural network.

    Target:    y = r + gamma * max_a' Q_target(s', a')    (y = r if the game ended)
    Rewards:   +10 food, -10 crash, 0 otherwise (given by the game)
    Policy:    epsilon-greedy, epsilon decays with the number of games played
    """

    def __init__(self, hidden_size=64, lr=0.001, gamma=0.9,
                 eps_start=0.9, eps_min=0.01, eps_decay=0.97,
                 target_sync=200, seed=None):
        self.n_games = 0
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.target_sync = target_sync
        self.rng = random.Random(seed)

        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = NeuralNetwork(11, hidden_size, 3, lr, seed)
        self.target = NeuralNetwork(11, hidden_size, 3, lr, seed)
        self.target.copy_from(self.model)
        self.steps = 0

        # What the dashboard visualises
        self.last_state = np.zeros(11)
        self.last_h = np.zeros(hidden_size)
        self.last_q = np.zeros(3)
        self.last_action = 0
        self.last_random = False
        self.last_loss = 0.0

    @property
    def epsilon(self):
        return max(self.eps_min, self.eps_start * (self.eps_decay ** self.n_games))

    def games_until_epsilon(self, threshold=0.5):
        """Number of games until epsilon drops below `threshold` (the 'mostly random' phase)."""
        if self.eps_start <= threshold or self.eps_decay >= 1.0:
            return 0
        return int(math.ceil(math.log(threshold / self.eps_start) / math.log(self.eps_decay)))

    def get_action(self, state, greedy=False):
        """Returns an int: 0 = straight, 1 = left, 2 = right."""
        q = self.model.forward(state)
        self.last_state = np.asarray(state, dtype=np.float64)
        self.last_h = self.model.cache["a1"][0].copy()
        self.last_q = q[0].copy()

        if not greedy and self.rng.random() < self.epsilon:
            action = self.rng.randint(0, 2)       # explore
            self.last_random = True
        else:
            action = int(np.argmax(q[0]))         # exploit
            self.last_random = False
        self.last_action = action
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short(self, state, action, reward, next_state, done):
        """Learn from the most recent step."""
        self._learn([(state, action, reward, next_state, done)])

    def train_long(self):
        """Learn from a random batch of past experience (after each game)."""
        if not self.memory:
            return
        mem = list(self.memory)
        if len(mem) > BATCH_SIZE:
            batch = self.rng.sample(mem, BATCH_SIZE)
        else:
            batch = mem
        self._learn(batch)

    def _learn(self, batch):
        states = np.array([b[0] for b in batch], dtype=np.float64)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float64)
        next_states = np.array([b[3] for b in batch], dtype=np.float64)
        dones = np.array([b[4] for b in batch], dtype=np.float64)

        q_next = self.target.predict(next_states)
        y = rewards + self.gamma * q_next.max(axis=1) * (1.0 - dones)

        q_pred = self.model.forward(states)
        targets = q_pred.copy()                          # only the taken action gets an error
        targets[np.arange(len(batch)), actions] = y

        self.last_loss = self.model.fit_batch(q_pred, targets)

        self.steps += 1
        if self.steps % self.target_sync == 0:
            self.target.copy_from(self.model)