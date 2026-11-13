import argparse

from agent import Agent
from dashboard import Dashboard
from snake_game import SnakeGame


def main():
    parser = argparse.ArgumentParser(description="Watch the trained network play Snake (no learning)")
    parser.add_argument("--model", default="snake_model_last.npz", help="weights file from train.py")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    agent = Agent()
    agent.model.load(args.model)
    game = SnakeGame()
    dash = Dashboard(fps=args.fps)

    scores, record, total = [], 0, 0
    stats = {"game": 0, "total": args.games, "score": 0, "record": 0, "mean": 0.0, "scores": scores}

    for ep in range(1, args.games + 1):
        game.reset()
        done = False
        score = 0
        while not done:
            state = game.get_state()
            action = agent.get_action(state, greedy=True)
            _, done, score = game.step(action)

            stats["game"] = ep
            stats["score"] = score
            stats["record"] = max(record, score)
            dash.frame(game, agent, stats)
            if dash.quit:
                break
        if dash.quit:
            break

        record = max(record, score)
        scores.append(score)
        total += score
        stats["mean"] = total / len(scores)
        stats["record"] = record
        print(f"Game {ep:3d} | Score: {score:3d} | High score: {record:3d}")

    dash.close()


if __name__ == "__main__":
    main()