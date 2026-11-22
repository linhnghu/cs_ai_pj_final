# Runtime and dependencies
- `pj1`–`pj3` use Python 3 and the standard library. Pacman/Gridworld display helpers use `tkinter`, which some game modules import even when a text or headless run is selected.
- On this Windows setup, `py` resolves to CPython 3.14 and `python` resolves to a Microsoft Store alias. Use `py -0p` to list available interpreters. No repository Python version pin was found.
- `pj5` additionally imports PyTorch (`torch`), NumPy and Matplotlib. Check dependencies in the interpreter that runs the grader. The CPU build of PyTorch supports the current PJ5 scripts. There is no versioned dependency manifest in the current tree. See `mem:pj5/core` for data lookup.
