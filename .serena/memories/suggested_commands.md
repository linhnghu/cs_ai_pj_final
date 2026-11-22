# Windows PowerShell commands
- Enter the assignment directory first, e.g. `Set-Location .\pj3` from the workspace root. Local imports and default paths for test fixtures/layouts depend on that directory.
- `py -B autograder.py --help` shows supported options. `py -B autograder.py -q q1 --no-graphics` grades one question. For `pj1`–`pj3`, pass a fixture path from the assignment directory without the `.test` suffix, e.g. `py -B autograder.py -t test_cases/q1/1-tinygrid --no-graphics` in `pj3`. See `mem:task_completion` for the full check.
- `-B` prevents Python from writing bytecode. Some `__pycache__/` files are tracked here, so keep them out of unrelated changes.
- Example from `pj1`: `py -B pacman.py -l tinyMaze -p SearchAgent -a fn=bfs -q`. From `pj2`: `py -B pacman.py -l minimaxClassic -p MinimaxAgent -a depth=2 -q`.
- For `pj3` Gridworld logic, use the headless grader. The current `py -B gridworld.py -t -a value -i 5 -q` demo reaches the text table renderer and fails at `list(*newRows)` in `textGridworldDisplay.py`.
