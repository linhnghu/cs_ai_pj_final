# Project 3 — Reinforcement learning
- `valueIterationAgents.py` handles planning against `mdp.py` (`q1` batch value iteration, `q4` prioritized sweeping). Batch updates read the previous iteration's values for every state. `gridworld.py` provides the concrete MDP; numbered cells are preterminal and take an `exit` action to the true terminal state.
- `analysis.py` contains Gridworld parameter answers (`q2`, `q3`) and the exploration question (`q7`). `qlearningAgents.py` contains tabular Q-learning (`q5`), epsilon-greedy policy (`q6`), Pacman training (`q8`), and approximate Q-learning (`q9`).
- `learningAgents.py` owns the agent/episode interface and training lifecycle. `featureExtractors.py` provides features used by `ApproximateQAgent`, which inherits action selection from `QLearningAgent`.
- Run `py -B autograder.py -q q1 --no-graphics` in `pj3/` for a focused value iteration check. See `mem:task_completion` for broader verification.
