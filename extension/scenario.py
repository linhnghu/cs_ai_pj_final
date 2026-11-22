"""Build reproducible Pacman layouts and expose their navigable cells."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random


LAYOUTS = Path(__file__).resolve().parent / "layouts"
SIZES = {"small": (21, 13), "large": (41, 23)}
MAP_SEED = 188
MOVES = (("North", 0, 1), ("South", 0, -1), ("East", 1, 0), ("West", -1, 0))


@dataclass
class Maze:
    rows: tuple[str, ...]
    start: tuple[int, int]
    graph: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]]

    @property
    def width(self):
        return len(self.rows[0])

    @property
    def height(self):
        return len(self.rows)

    def distances(self, goal):
        distances = {goal: 0}
        frontier = deque([goal])
        while frontier:
            cell = frontier.popleft()
            for _, next_cell in self.graph[cell]:
                if next_cell not in distances:
                    distances[next_cell] = distances[cell] + 1
                    frontier.append(next_cell)
        return distances

    def shortest_path(self, start, goal):
        """Recover a shortest path by descending a BFS distance map."""
        distances = self.distances(goal)
        path = [start]
        while path[-1] != goal:
            _, next_cell = min(self.graph[path[-1]], key=lambda edge: distances[edge[1]])
            path.append(next_cell)
        return path

    @classmethod
    def parse(cls, rows):
        rows = tuple(rows)
        if not rows or len({len(row) for row in rows}) != 1:
            raise ValueError("A layout must be rectangular.")
        cells = {
            (x, y): char
            for y, row in enumerate(reversed(rows))
            for x, char in enumerate(row)
            if char != "%"
        }
        starts = [cell for cell, char in cells.items() if char == "P"]
        if len(starts) != 1:
            raise ValueError("A layout must contain exactly one Pacman start.")
        graph = {}
        for x, y in cells:
            graph[x, y] = [
                (i, (x + dx, y + dy))
                for i, (_, dx, dy) in enumerate(MOVES)
                if (x + dx, y + dy) in cells
            ]
        maze = cls(rows, starts[0], graph)
        if len(maze.distances(maze.start)) != len(cells):
            raise ValueError("All open cells must be connected.")
        return maze

    @classmethod
    def load(cls, size):
        return cls.parse((LAYOUTS / f"{size}.lay").read_text(encoding="ascii").splitlines())


def generate(width, height, seed):
    rng = random.Random(seed)
    grid = [["%"] * width for _ in range(height)]
    start = (1, 1)
    stack = [start]
    grid[1][1] = " "
    while stack:
        x, y = stack[-1]
        choices = [
            (x + 2 * dx, y + 2 * dy)
            for _, dx, dy in MOVES
            if 0 < x + 2 * dx < width - 1
            and 0 < y + 2 * dy < height - 1
            and grid[y + 2 * dy][x + 2 * dx] == "%"
        ]
        if choices:
            nx, ny = rng.choice(choices)
            grid[(y + ny) // 2][(x + nx) // 2] = " "
            grid[ny][nx] = " "
            stack.append((nx, ny))
        else:
            stack.pop()

    # Open rooms and cross-links so the arena has both corridors and cycles.
    for cx, cy in [(width // 4, height // 2), (width * 3 // 4, height // 3)]:
        for y in range(cy - 1, cy + 2):
            for x in range(cx - 2, cx + 3):
                grid[y][x] = " "
    links = [
        (x, y)
        for y in range(1, height - 1)
        for x in range(1, width - 1)
        if grid[y][x] == "%"
        and (grid[y][x-1] == grid[y][x+1] == " "
             or grid[y-1][x] == grid[y+1][x] == " ")
    ]
    for x, y in rng.sample(links, min(len(links), width * height // 25)):
        grid[y][x] = " "

    grid[start[1]][start[0]] = "P"
    maze = Maze.parse("".join(row) for row in reversed(grid))
    distances = maze.distances(start)
    by_distance = sorted(maze.graph, key=lambda p: (distances[p], p))
    ghost_positions = (by_distance[-1], by_distance[len(by_distance) * 2 // 3])
    available = [p for p in by_distance if p != start and p not in ghost_positions]
    capsules = [available[len(available) // 4], available[len(available) * 3 // 4]]
    food = rng.sample([p for p in available if p not in capsules], 10 if width < 30 else 22)
    for i, (x, y) in enumerate(ghost_positions, 1):
        grid[y][x] = str(i)
    for x, y in capsules:
        grid[y][x] = "o"
    for x, y in food:
        grid[y][x] = "."
    return Maze.parse("".join(row) for row in reversed(grid))


if __name__ == "__main__":
    LAYOUTS.mkdir(exist_ok=True)
    for name, (width, height) in SIZES.items():
        maze = generate(width, height, MAP_SEED)
        (LAYOUTS / f"{name}.lay").write_text("\n".join(maze.rows) + "\n", encoding="ascii")
