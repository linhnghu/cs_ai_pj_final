# Project 1 — Search
- `search.py` defines `SearchProblem` and DFS/BFS/UCS/A* entry points. Search returns action lists and consumes `(successor, action, stepCost)` triples. `searchAgents.py` adapts Pacman problems to those algorithms and contains goal/heuristic logic.
- Question map from `test_cases/`: q1 DFS, q2 BFS, q3 UCS, q4 A*, q5 corners problem, q6 corners heuristic, q7 food heuristic, q8 closest-dot search. `SearchAgent` resolves problem and function names from CLI arguments.
- Graph tests can check the expansion trace in addition to the action list. Preserve successor ordering and the expected point at which a state is considered expanded. Run from `pj1/` as described in `mem:task_completion`.
