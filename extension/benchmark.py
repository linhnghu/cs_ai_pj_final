"""Benchmark planning, adversarial play, reinforcement learning, and imitation."""

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import platform
import random
from statistics import mean, median, pstdev
import subprocess
import sys
from time import perf_counter, perf_counter_ns

from scenario import MAP_SEED, MOVES, SIZES, Maze, generate


ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
MAX_TURNS = 320
TEST_SEED = 300
TRAIN_SEED = 100
ROUTE_SEED = 2026
BASE_TRACKS = {"search": "pj1", "game": "pj2", "learning": "pj3", "ml": "pj5"}
EXTRA_TRACKS = {"scale": "pj1", "stress": "pj3", "hybrid": "pj5"}
TRACKS = BASE_TRACKS | EXTRA_TRACKS
SCALE_SIZES = ((21, 13), (41, 23), (61, 33))
SCALE_SEEDS = tuple(range(MAP_SEED, MAP_SEED + 6))


def walk(maze, start, actions):
    directions = {name: (dx, dy) for name, dx, dy in MOVES}
    cells = [start]
    for action in actions:
        x, y = cells[-1]
        dx, dy = directions[action]
        next_cell = (x + dx, y + dy)
        if next_cell not in maze.graph:
            raise ValueError(f"{action} enters a wall at {next_cell}.")
        cells.append(next_cell)
    return cells


def search_experiment(args):
    import search
    from searchAgents import manhattanHeuristic

    algorithms = {
        "DFS": search.dfs,
        "BFS": search.bfs,
        "UCS": search.ucs,
        "A*": lambda problem: search.astar(problem, manhattanHeuristic),
    }
    return {
        size: search_cases(Maze.load(size), args.routes, algorithms, ROUTE_SEED)
        for size in SIZES
    }


def search_cases(maze, count, algorithms, seed):
    import layout
    import pacman
    from searchAgents import PositionSearchProblem

    state = pacman.GameState()
    state.initialize(layout.Layout(maze.rows), 0)
    farthest = max(maze.distances(maze.start).items(), key=lambda item: item[1])[0]
    rng = random.Random(seed)
    cells = sorted(maze.graph)
    pairs = [(maze.start, farthest)] + [
        tuple(rng.sample(cells, 2)) for _ in range(count - 1)
    ]
    outcomes = {name: [] for name in algorithms}
    first_paths = {}
    names = list(algorithms)
    for index, (start, goal) in enumerate(pairs):
        optimum = maze.distances(goal)[start]
        # Rotate run order across cases to reduce warm-up bias.
        offset = index % len(names)
        for name in names[offset:] + names[:offset]:
            problem = PositionSearchProblem(
                state, start=start, goal=goal, warn=False, visualize=False
            )
            before = perf_counter_ns()
            actions = algorithms[name](problem)
            elapsed = perf_counter_ns() - before
            path = walk(maze, start, actions)
            if path[-1] != goal or len(actions) < optimum:
                raise RuntimeError(f"{name} returned an invalid route from {start} to {goal}.")
            outcomes[name].append({
                "start": start, "goal": goal, "steps": len(actions),
                "optimal_steps": optimum, "expanded": problem._expanded,
                "ratio": len(actions) / optimum, "ms": elapsed / 1_000_000,
            })
            if index == 0:
                first_paths[name] = path
    return {"outcomes": outcomes, "example_paths": first_paths}


def scale_experiment(args):
    import search
    from searchAgents import manhattanHeuristic

    algorithms = {"BFS": search.bfs,
                  "A*": lambda problem: search.astar(problem, manhattanHeuristic)}
    result = {}
    for width, height in SCALE_SIZES:
        tests = []
        for seed in SCALE_SEEDS:
            maze = generate(width, height, seed)
            run = search_cases(maze, args.routes, algorithms, ROUTE_SEED + seed)
            tests.append({"seed": seed, "open_cells": len(maze.graph), **run})
        result[f"{width}x{height}"] = tests
    return result


def ghost_streams(seed, count, ghost_policy="random"):
    from game import Directions
    from ghostAgents import DirectionalGhost, RandomGhost

    ghost_type = {"random": RandomGhost, "directional": DirectionalGhost}[ghost_policy]

    class SeededGhost(ghost_type):
        def __init__(self, index):
            super().__init__(index)
            self.random = random.Random(seed * 1000 + index)

        def getAction(self, state):
            actions = state.getLegalActions(self.index)
            if not actions:
                return Directions.STOP
            if ghost_policy == "random":
                return self.random.choice(actions)
            distribution = self.getDistribution(state)
            return self.random.choices(actions, [distribution[a] for a in actions])[0]

    return [SeededGhost(index) for index in range(1, count + 1)]


def run_game(board, agent, seed, timings=None, *, ghost_policy="random"):
    import pacman
    from textDisplay import NullGraphics

    class BoundedRules(pacman.ClassicGameRules):
        def process(self, state, game):
            super().process(state, game)
            if len(game.moveHistory) >= MAX_TURNS * len(game.agents):
                game.gameOver = True

    original_get_action = agent.getAction

    def wrapped_get_action(state):
        try:
            if timings is None:
                return original_get_action(state)
            start = perf_counter_ns()
            action = original_get_action(state)
            timings.append((perf_counter_ns() - start) / 1_000_000)
            return action
        finally:
            # The teaching scaffold retains explored states for its grader.
            pacman.GameState.explored.clear()

    random.seed(seed)
    agent.getAction = wrapped_get_action
    try:
        rules = BoundedRules()
        ghosts = ghost_streams(seed, board.getNumGhosts(), ghost_policy)
        game = rules.newGame(board, agent, ghosts, NullGraphics(), quiet=True)
        with redirect_stdout(StringIO()):
            game.run()
    finally:
        del agent.getAction
        pacman.GameState.explored.clear()

    state = game.state
    eaten = board.totalFood - state.getNumFood()
    return {
        "seed": seed, "win": state.isWin(), "loss": state.isLose(),
        "capped": not state.isWin() and not state.isLose(),
        "score": state.getScore(), "food_eaten": eaten, "food_total": board.totalFood,
        "turns": sum(index == 0 for index, _ in game.moveHistory),
    }


def evaluate_games(board, agent, games, *, ghost_policy="random"):
    timings = []
    rounds = [
        run_game(board, agent, TEST_SEED + index, timings, ghost_policy=ghost_policy)
        for index in range(games)
    ]
    return {
        "rounds": rounds,
        "median_decision_ms": median(timings),
        "decision_count": len(timings),
    }


def game_experiment(args):
    import layout
    from multiAgents import AlphaBetaAgent, ExpectimaxAgent, ReflexAgent

    board = layout.Layout(Maze.load("large").rows)
    agents = {
        "Reflex": ReflexAgent(),
        "Alpha-Beta": AlphaBetaAgent(depth=1, evalFn="better"),
        "Expectimax": ExpectimaxAgent(depth=1, evalFn="better"),
    }
    return {name: evaluate_games(board, agent, args.games) for name, agent in agents.items()}


def learning_experiment(args):
    import layout
    from qlearningAgents import ApproximateQAgent, PacmanQAgent

    board = layout.Layout(Maze.load("large").rows)
    agents = {
        "Q-learning": PacmanQAgent(numTraining=args.episodes),
        "Approximate Q": ApproximateQAgent(extractor="SimpleExtractor", numTraining=args.episodes),
    }
    results = {}
    for name, agent in agents.items():
        start = perf_counter()
        training = [run_game(board, agent, TRAIN_SEED + index) for index in range(args.episodes)]
        elapsed = perf_counter() - start
        if agent.episodesSoFar != args.episodes or agent.alpha != 0 or agent.epsilon != 0:
            raise RuntimeError("Evaluation started before the learned policy was frozen.")
        outcome = evaluate_games(board, agent, args.games)
        outcome["training"] = training
        outcome["training_seconds"] = elapsed
        outcome["learned_entries"] = len(agent.getWeights() if name == "Approximate Q" else agent.qValues)
        results[name] = outcome
    return results


def stress_experiment(args):
    import layout
    from qlearningAgents import ApproximateQAgent

    board = layout.Layout(Maze.load("large").rows)
    learner = ApproximateQAgent(extractor="SimpleExtractor", numTraining=args.episodes)
    training, checkpoints = [], {}
    completed = 0
    for checkpoint in sorted({0, args.episodes // 3, args.episodes}):
        for index in range(completed, checkpoint):
            training.append(run_game(board, learner, TRAIN_SEED + index))
        completed = checkpoint
        outcomes = {}
        for policy in ("random", "directional"):
            frozen = ApproximateQAgent(
                extractor="SimpleExtractor", numTraining=0, alpha=0.0, epsilon=0.0
            )
            frozen.weights = learner.getWeights().copy()
            outcomes[policy] = evaluate_games(board, frozen, args.games, ghost_policy=policy)
        checkpoints[str(checkpoint)] = {
            "weights": dict(learner.getWeights()), "outcomes": outcomes
        }
    return {"training": training, "checkpoints": checkpoints}


def ml_experiment(args):
    import torch
    from torch.nn import Linear, ReLU, Sequential
    from torch.nn.functional import cross_entropy
    from torch.utils.data import DataLoader, TensorDataset

    # PJ5's dense ReLU classifier and cross-entropy are applied to navigation.
    torch.set_num_threads(2)
    torch.manual_seed(ROUTE_SEED)
    maze = Maze.load("large")
    cells = sorted(maze.graph)
    rng = random.Random(ROUTE_SEED)
    goals = rng.sample(cells, 32)
    training_goals, testing_goals = goals[:24], goals[24:]
    legal = {cell: {index: destination for index, destination in edges}
             for cell, edges in maze.graph.items()}

    def features(cell, goal):
        x, y = cell
        gx, gy = goal
        w, h = maze.width - 1, maze.height - 1
        return [x/w, y/h, gx/w, gy/h, (gx-x)/w, (gy-y)/h,
                *(float(index in legal[cell]) for index in range(len(MOVES)))]

    inputs, labels = [], []
    for goal in training_goals:
        distances = maze.distances(goal)
        for cell in cells:
            if cell == goal:
                continue
            label = min(legal[cell], key=lambda index: distances[legal[cell][index]])
            inputs.append(features(cell, goal))
            labels.append(label)

    model = Sequential(Linear(10, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 4))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    train_data = TensorDataset(torch.tensor(inputs), torch.tensor(labels))
    loader = DataLoader(train_data, batch_size=128, shuffle=True)
    train_start = perf_counter()
    for _ in range(25):
        for batch, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy(model(batch), targets)
            loss.backward()
            optimizer.step()
    train_seconds = perf_counter() - train_start
    model.eval()

    trials = []
    with torch.inference_mode():
        for goal in testing_goals:
            distances = maze.distances(goal)
            starts = rng.sample([c for c in cells if c != goal], 10)
            # Evaluate the same deterministic policy repeatedly without re-running the net.
            logits = model(torch.tensor([features(c, goal) for c in cells]))
            policy = {cell: max(legal[cell], key=lambda index: float(row[index]))
                      for cell, row in zip(cells, logits, strict=True)}
            for start in starts:
                cell = start
                visited = set()
                steps = 0
                first_cell = legal[start][policy[start]]
                first_optimal = distances[first_cell] == distances[start] - 1
                while cell != goal and cell not in visited and steps < MAX_TURNS:
                    visited.add(cell)
                    cell = legal[cell][policy[cell]]
                    steps += 1
                reached = cell == goal
                fallback = [] if reached else maze.shortest_path(cell, goal)
                hybrid_steps = steps + max(0, len(fallback) - 1)
                trials.append({
                    "start": start, "goal": goal, "steps": steps, "reached": reached,
                    "first_optimal": first_optimal, "loop": not reached and cell in visited,
                    "optimal_steps": distances[start],
                    "fallback": fallback,
                    "hybrid_reached": reached or fallback[-1] == goal,
                    "hybrid_steps": hybrid_steps,
                })
    return {
        "training_seconds": train_seconds,
        "examples": len(train_data), "training_goals": training_goals,
        "testing_goals": testing_goals, "trials": trials,
        "torch": torch.__version__,
    }


EXPERIMENTS = {
    "search": search_experiment, "game": game_experiment,
    "learning": learning_experiment, "ml": ml_experiment,
    "scale": scale_experiment, "stress": stress_experiment, "hybrid": ml_experiment,
}


def describe():
    return {
        name: {
            "width": maze.width, "height": maze.height, "open": len(maze.graph),
            "food": sum(row.count(".") for row in maze.rows),
            "capsules": sum(row.count("o") for row in maze.rows),
            "ghosts": sum(row.count("1") + row.count("2") for row in maze.rows),
        }
        for name in SIZES
        for maze in [Maze.load(name)]
    }


def summarize_scale(data):
    summary = {}
    for size, layouts in data.items():
        by_algorithm = {}
        for algorithm in layouts[0]["outcomes"]:
            cases_by_layout = [layout["outcomes"][algorithm] for layout in layouts]
            cases = [case for rows in cases_by_layout for case in rows]
            map_means = [mean(row["expanded"] for row in rows) for rows in cases_by_layout]
            by_algorithm[algorithm] = {
                "cases": len(cases),
                "mean_expanded": mean(map_means),
                "std_map_mean_expanded": pstdev(map_means),
                "median_ms": median(row["ms"] for row in cases),
                "optimal": sum(row["steps"] == row["optimal_steps"] for row in cases),
            }
        summary[size] = by_algorithm
    return summary


def summarize(name, data):
    if name == "scale":
        return summarize_scale(data)
    if name == "stress":
        return {
            checkpoint: summarize("game", record["outcomes"])
            for checkpoint, record in data["checkpoints"].items()
        }
    if name == "search":
        return {
            size: {
                algorithm: {
                    "mean_steps": mean(row["steps"] for row in rows),
                    "mean_expanded": mean(row["expanded"] for row in rows),
                    "median_ms": median(row["ms"] for row in rows),
                    "mean_ratio": mean(row["ratio"] for row in rows),
                }
                for algorithm, rows in cases["outcomes"].items()
            }
            for size, cases in data.items()
        }
    if name in ("game", "learning"):
        return {
            agent: {
                "wins": sum(row["win"] for row in outcome["rounds"]),
                "losses": sum(row["loss"] for row in outcome["rounds"]),
                "capped": sum(row["capped"] for row in outcome["rounds"]),
                "mean_score": mean(row["score"] for row in outcome["rounds"]),
                "mean_food_fraction": mean(row["food_eaten"] / row["food_total"]
                                           for row in outcome["rounds"]),
                "mean_turns": mean(row["turns"] for row in outcome["rounds"]),
                "median_decision_ms": outcome["median_decision_ms"],
            }
            for agent, outcome in data.items()
        }
    trials = data["trials"]
    successful = [row for row in trials if row["reached"]]
    hybrid_successful = [row for row in trials if row["hybrid_reached"]]
    return {
        "trials": len(trials),
        "successes": len(successful),
        "loops": sum(row["loop"] for row in trials),
        "first_optimal": sum(row["first_optimal"] for row in trials),
        "mean_success_ratio": (
            mean(row["steps"] / row["optimal_steps"] for row in successful)
            if successful else None
        ),
        "hybrid_successes": len(hybrid_successful),
        "hybrid_fallbacks": sum(bool(row["fallback"]) for row in trials),
        "hybrid_mean_steps": mean(row["hybrid_steps"] for row in hybrid_successful),
        "bfs_mean_steps": mean(row["optimal_steps"] for row in trials),
        "hybrid_mean_ratio": mean(
            row["hybrid_steps"] / row["optimal_steps"] for row in hybrid_successful
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=45)
    parser.add_argument("--routes", type=int, default=30)
    parser.add_argument("--track", choices=["all", "extended", *TRACKS], default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=TRACKS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.games, args.episodes, args.routes) < 1:
        parser.error("games, episodes, and routes must be positive.")

    if args.worker:
        sys.path.insert(0, str(ROOT / TRACKS[args.worker]))
        print(json.dumps(EXPERIMENTS[args.worker](args)))
        return

    if args.track == "all":
        selected, filename = list(BASE_TRACKS), "results.json"
    elif args.track == "extended":
        selected, filename = list(EXTRA_TRACKS), "results-expanded.json"
    else:
        selected, filename = [args.track], f"results-{args.track}.json"
    output = args.output or HERE / filename
    results = {
        "config": {"map_seed": MAP_SEED, "route_seed": ROUTE_SEED, "games": args.games,
                   "episodes": args.episodes, "routes": args.routes, "max_turns": MAX_TURNS,
                   "training_seed_start": TRAIN_SEED, "testing_seed_start": TEST_SEED,
                   "python": platform.python_version()},
        "maps": describe(), "summary": {},
    }
    for name in selected:
        print(f"Running {name} ({TRACKS[name]})...", flush=True)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", name,
                   "--games", str(args.games), "--episodes", str(args.episodes),
                   "--routes", str(args.routes)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise SystemExit(f"{name} failed:\n{completed.stderr or completed.stdout}")
        results[name] = json.loads(completed.stdout)
        results["summary"][name] = summarize(name, results[name])
        print(f"Finished {name}.", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
