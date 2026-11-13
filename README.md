🐍 Neural Network That Plays Snake

A Snake-playing AI built **from scratch in NumPy**: no TensorFlow, no PyTorch. The forward pass, loss, backpropagation and Adam optimizer are all hand-written. The agent starts with random moves and learns to chase food and avoid walls and its own tail using Q-learning.

Includes a live Pygame dashboard that shows the network's neurons, the score curve and the math while it trains.

![Dashboard](assets/dashboard.png)
![Training curve](assets/snake_scores.png)

<img width="1280" height="728" alt="image" src="https://github.com/user-attachments/assets/67151d3a-8167-4573-a598-68c75dee2532" />
<img width="994" height="597" alt="image" src="https://github.com/user-attachments/assets/e26a03fa-edb9-4925-a851-778289116f80" />



 Features

- Pure NumPy neural network (11 → 64 → 3) with manual backpropagation
- Q-learning with epsilon-greedy exploration, experience replay and a target network
- Adam optimizer implemented by hand
- Live dashboard: game board, network activations, score chart, loss and formulas
- Dark-themed Matplotlib chart of "score per game while training"
- Watch mode to replay the trained network, plus CSV logging of every game

 How it works

**State (11 inputs)**

| Group | Inputs |
|---|---|
| Danger | straight, right, left |
| Direction | up, down, left, right |
| Food location | up, down, left, right |

**Actions (3 outputs):** straight, turn left, turn right

**Rewards:** `+10` eating food, `-10` crashing, `0` for a normal move

**Network**

```
h = max(0, x @ W1 + b1)        # hidden layer, ReLU
Q = h @ W2 + b2                # one Q-value per action
```

**Learning (Q-learning)**

```
y  = r + gamma * max Q_target(s')      # y = r if the game ended
dQ = 2 * (Q(s,a) - y)                  # loss gradient
dW2 = h.T @ dQ
dh  = (dQ @ W2.T) * (h > 0)
dW1 = x.T @ dh
```

**Policy:** with probability `epsilon` the snake plays a random move (explore). Otherwise it plays `argmax Q(s, a)` (exploit). Epsilon decays from 0.9 as games are played.

 Project structure

```
snake_ai/
├── snake_game.py   # game logic + 11-input state + rewards
├── model.py        # NumPy neural network, backprop, Adam
├── agent.py        # Q-learning, epsilon-greedy, replay memory, target network
├── dashboard.py    # Pygame dashboard (board, network view, live chart)
├── train.py        # training loop, terminal logs, Matplotlib plot
├── play.py         # watch the trained network play
└── assets/         # screenshots / demo GIF
```

 Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install pygame numpy matplotlib
```

Requires Python 3.9+.

 Usage

```bash
python train.py                              # live dashboard, 300 games
python train.py --episodes 400 --turbo       # start in turbo mode
python train.py --episodes 300 --no-render   # fastest, no window
python train.py --seed 42                    # reproducible run
python play.py                               # watch the trained model
```

Dashboard controls**

| Key | Action |
|---|---|
| `SPACE` | Pause / resume |
| `T` | Toggle turbo (train at full speed) |
| `UP` / `DOWN` | Change game speed |
| `ESC` | Quit (the plot is still saved) |

**Outputs:** `snake_scores.png` (chart), `scores.csv` (per-game log), `snake_model.npz` (weights from the best game), `snake_model_last.npz` (final weights).

Results

| Metric | Value |
|---|---|
| Games trained | [N] |
| Best score | [X] |
| Mean score (last 50 games) | [Y] |
| Training time | [Z min] |

The first ~20 games are mostly random moves. After that the score climbs as epsilon decays and the network starts exploiting what it learned.

 Hyperparameters

| Name | Value |
|---|---|
| Hidden units | 64 |
| Learning rate (Adam) | 0.001 |
| Discount (gamma) | 0.9 |
| Epsilon | 0.9 → 0.01 (×0.97 per game) |
| Replay memory / batch | 100,000 / 1,000 |
| Target network sync | every 200 updates |

# Ideas for improvement

- Add a longer-range view of the body to avoid trapping itself
- Replace the 11 features with a grid input and a small CNN
- Try Double DQN or prioritized replay
- Add reward shaping for moving toward the food





MIT# Neural-Network-Plays-Snake
Q-Learning from Scratch in NumPy | Python, NumPy, Pygame, Matplotlib
