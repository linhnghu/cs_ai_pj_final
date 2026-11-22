# Project 5 — Machine learning
- `models.py` holds `torch.nn.Module` implementations. `train.py` holds training entry points and `losses.py` contains grader-facing losses. Preserve model/output shapes and function names across those files.
- The grader registers tests in `autograder.py`: q1 perceptron, q2 regression, q3 digit classification, q4 language ID, q5 convolution, q6 causal attention. `gpt_model.py` and `chargpt.py` use the attention implementation in the separate character model.
- `backend.py` loads MNIST and language ID `.npz` datasets. Its lookup checks a sibling `data/` directory, then `pj5/data/`, then the module directory. The repository's copies are under `pj5/data/`.
- Run from `pj5/` with the dependencies in `mem:tech_stack` and commands in `mem:task_completion`.
