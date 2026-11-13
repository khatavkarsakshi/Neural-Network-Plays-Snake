import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from dashboard import Dashboard, moving_average
from snake_game import SnakeGame


def plot_scores(scores, record, explore_games, path="snake_scores.png"):
    """Dark 'score per game while training' chart, styled like the post."""
    bg, panel, teal, light = "#0b0e13", "#10151c", "#2dd4bf", "#ccfbf1"
    x = np.arange(1, len(scores) + 1)
    y = np.array(scores)
    ma = moving_average(scores, 10)

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=bg)
    ax.set_facecolor(panel)
    ax.fill_between(x, y, color=teal, alpha=0.18)
    ax.plot(x, y, color=teal, linewidth=0.9, alpha=0.75, label="score")
    ax.plot(x, ma, color=light, linewidth=2.2, label="10-game average")

    if explore_games and len(scores) > explore_games:
        ax.axvline(explore_games, color="#475569", linewidth=0.8, linestyle="--")
        ax.text(explore_games + 2, max(record, 10) * 0.95, "mostly random moves",
                color="#94a3b8", fontsize=9, va="top")

    ax.set_title("score per game while training", color="#e2e8f0", loc="left", fontsize=13, pad=12)
    ax.text(1.0, 1.03, f"best game: {record}", transform=ax.transAxes,
            ha="right", color=teal, fontsize=10)
    ax.set_xlabel("game", color="#94a3b8")
    ax.set_ylabel("apples eaten", color="#94a3b8")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, max(len(scores), 1))
    ax.tick_params(colors="#94a3b8")
    ax.grid(color="#1e2633", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#1e2633")
    leg = ax.legend(facecolor=panel, edgecolor="#1e2633", loc="lower right")
    for t in leg.get_texts():
        t.set_color("#cbd5e1")

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=bg)
    print(f"Saved plot to {path}")
    plt.show()


def train(episodes, render, fps, turbo, seed):
    game = SnakeGame()
    agent = Agent(seed=seed)
    dash = Dashboard(fps=fps, turbo=turbo) if render else None

    scores, rows = [], []
    total_score, record = 0, 0
    stats = {"game": 0, "total": episodes, "score": 0, "record": 0, "mean": 0.0, "scores": scores}

    for ep in range(1, episodes + 1):
        game.reset()
        done = False
        score = 0

        while not done:
            state = game.get_state()
            action = agent.get_action(state)
            reward, done, score = game.step(action)
            next_state = game.get_state()

            agent.train_short(state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)

            if dash:
                stats["game"] = ep
                stats["score"] = score
                stats["record"] = max(record, score)
                dash.frame(game, agent, stats)
                if dash.quit:
                    break

        if dash and dash.quit:
            print("Window closed - stopping early.")
            break

        agent.n_games += 1
        agent.train_long()

        if score > record:
            record = score
            agent.model.save("snake_model.npz")

        scores.append(score)
        total_score += score
        mean = total_score / len(scores)
        stats["mean"] = mean
        stats["record"] = record
        rows.append([ep, score, record, round(mean, 3), round(agent.epsilon, 4)])

        print(f"Game {ep:4d} | Score: {score:3d} | High score: {record:3d} | "
              f"Mean: {mean:5.2f} | Epsilon: {agent.epsilon:.3f}")

    if dash:
        dash.close()

    agent.model.save("snake_model_last.npz")

    with open("scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["game", "score", "high_score", "mean_score", "epsilon"])
        writer.writerows(rows)
    print("Saved scores.csv, snake_model.npz (best game) and snake_model_last.npz (final weights)")

    if scores:
        plot_scores(scores, record, agent.games_until_epsilon(0.5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural network that plays Snake (NumPy Q-learning)")
    parser.add_argument("--episodes", type=int, default=300, help="number of games (100-500 recommended)")
    parser.add_argument("--no-render", action="store_true", help="train without a window (fastest)")
    parser.add_argument("--fps", type=int, default=30, help="game speed in the window")
    parser.add_argument("--turbo", action="store_true", help="start in turbo mode (press T to toggle)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible runs")
    args = parser.parse_args()

    train(args.episodes, render=not args.no_render, fps=args.fps, turbo=args.turbo, seed=args.seed)