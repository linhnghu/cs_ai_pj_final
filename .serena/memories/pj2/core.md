# Project 2 — Multiagent search
- `multiAgents.py` is the student file. Grader questions are q1 reflex evaluation, q2 minimax, q3 alpha-beta, q4 expectimax, q5 better state evaluation. Pacman uses agent index 0 and ghosts move in index order.
- One search depth unit covers Pacman and all ghost turns. Increment depth when control returns to Pacman. The grader checks generated nodes and actions, so preserve legal-action order and the expected expansion/pruning behavior.
- Run commands from `pj2/` using `mem:task_completion`.
