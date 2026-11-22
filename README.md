# Đồ án Cơ sở Trí tuệ nhân tạo

Đồ án dựa trên bộ bài tập Pacman AI của UC Berkeley. Các phần thực hành bao gồm tìm đường trong mê cung, ra quyết định khi có đối thủ, học tăng cường và xây dựng mô hình học máy bằng PyTorch. Phần mở rộng dùng benchmark để so sánh các thuật toán trên nhiều kịch bản.

## Cấu trúc dự án

| Thư mục | Nội dung | Tệp chính |
| --- | --- | --- |
| [`pj1/`](pj1/) | Tìm kiếm với DFS, BFS, UCS và A*. Xây dựng heuristic cho bài toán đi qua các góc và thu thập thức ăn. | [`search.py`](pj1/search.py), [`searchAgents.py`](pj1/searchAgents.py) |
| [`pj2/`](pj2/) | Tìm kiếm đa tác tử với Reflex Agent, Minimax, cắt tỉa Alpha–Beta và Expectimax. | [`multiAgents.py`](pj2/multiAgents.py) |
| [`pj3/`](pj3/) | Mô hình MDP, Value Iteration, Prioritized Sweeping, Q-learning và Approximate Q-learning trên Gridworld và Pacman. | [`valueIterationAgents.py`](pj3/valueIterationAgents.py), [`qlearningAgents.py`](pj3/qlearningAgents.py) |
| [`pj5/`](pj5/) | Perceptron, hồi quy, phân loại chữ số MNIST, nhận diện ngôn ngữ, tích chập và causal attention. Có thêm chương trình GPT cấp ký tự. | [`models.py`](pj5/models.py), [`train.py`](pj5/train.py), [`chargpt.py`](pj5/chargpt.py) |
| [`extension/`](extension/) | Mê cung thử nghiệm, chương trình benchmark và kết quả JSON. | [`benchmark.py`](extension/benchmark.py), [`scenario.py`](extension/scenario.py) |
| [`report/`](report/) | Báo cáo Word cho từng bài và phần mở rộng. | [Báo cáo phần mở rộng](report/Bao-cao-Mo-rong-ban-day-du.docx) |

## Môi trường chạy

- Dùng Python 3 có `tkinter` cho các bài Pacman và Gridworld. Ví dụ dùng PowerShell trên Windows. Nếu dùng Bash trên macOS hoặc Linux, thay `py`, `Push-Location` và `Pop-Location` lần lượt bằng `python3`, `pushd` và `popd`.
- PJ5 cần PyTorch, NumPy và Matplotlib. Các benchmark học máy cũng dùng PyTorch. Dữ liệu MNIST và nhận diện ngôn ngữ có sẵn trong [`pj5/data/`](pj5/data/).

```powershell
py -m pip install torch numpy matplotlib
```

## Chạy Pacman và autograder

Mỗi bài có autograder riêng. Từ thư mục gốc, chạy ví dụ dưới đây để chấm câu `q1` của PJ1. Đổi `pj1` thành `pj2`, `pj3` hoặc `pj5` để chấm bài tương ứng. Bỏ `-q q1` để chấm toàn bộ câu của bài đó.

```powershell
Push-Location pj1
py -B autograder.py -q q1 --no-graphics
Pop-Location
```

Chạy thử BFS trên mê cung `tinyMaze` và Minimax trên `minimaxClassic`. Cờ `-q` của `pacman.py` chạy mô phỏng không mở cửa sổ đồ họa.

```powershell
Push-Location pj1
py -B pacman.py -l tinyMaze -p SearchAgent -a fn=bfs -q
Pop-Location

Push-Location pj2
py -B pacman.py -l minimaxClassic -p MinimaxAgent -a depth=2 -q
Pop-Location
```

## Benchmark phần mở rộng

Chạy các lệnh sau từ thư mục gốc. Chế độ `all` chạy các thí nghiệm tìm kiếm, chơi Pacman, học tăng cường và học máy. Chế độ `extended` kiểm tra khả năng mở rộng theo kích thước mê cung, độ ổn định của tác tử học tăng cường khi đổi hành vi của ma và phương pháp tìm đường kết hợp với mô hình học máy.

```powershell
py -B extension/benchmark.py --track all --output extension/results-local.json
py -B extension/benchmark.py --track extended --output extension/results-expanded-local.json
```

Để thử nhanh với hai tuyến đường, chạy:

```powershell
py -B extension/benchmark.py --track search --routes 2 --output extension/results-search-local.json
```

Kết quả đã lưu trong repo có tại [`results.json`](extension/results.json) và [`results-expanded.json`](extension/results-expanded.json). Các lệnh trên dùng `--output` để lưu kết quả mới vào tệp riêng.

## Nguồn gốc

Khung Pacman và autograder gốc thuộc bộ bài tập Pacman AI do UC Berkeley phát triển. Điều khoản sử dụng cho mục đích học tập nằm trong phần đầu các tệp mã nguồn gốc. [UC Berkeley CS 188](https://ai.berkeley.edu/)
