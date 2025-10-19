# HƯỚNG DẪN CHẠY PROJECT VRP-GA SYSTEM

## Tổng quan

VRP-GA System là một hệ thống giải quyết bài toán Vehicle Routing Problem (VRP) sử dụng thuật toán di truyền (Genetic Algorithm) với tối ưu hóa tìm kiếm cục bộ 2-opt.

## Yêu cầu hệ thống

- Python 3.8 trở lên
- Hệ điều hành: Windows, macOS, hoặc Linux
- RAM: Tối thiểu 4GB (khuyến nghị 8GB)
- Dung lượng ổ cứng: 500MB trống

## Cài đặt

### Bước 1: Kiểm tra Python

```bash
python --version
# hoặc
python3 --version
```

Nếu chưa có Python 3.8+, hãy tải về từ [python.org](https://python.org)

### Bước 2: Tải project

```bash
# Nếu có git
git clone <repository-url>
cd vrp-ga-system

# Hoặc tải file ZIP và giải nén
```

### Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 4: Chạy setup tự động (tùy chọn)

```bash
python setup.py
```

## Cách sử dụng cơ bản

### 1. Chạy với dữ liệu mẫu (Mockup Data)

```bash
# Tạo và giải bài toán với 20 khách hàng
python main.py --generate --customers 20 --capacity 100

# Tạo và giải bài toán với 50 khách hàng
python main.py --generate --customers 50 --capacity 200

# Tạo và giải bài toán với 100 khách hàng
python main.py --generate --customers 100 --capacity 300
```

### 2. Chạy với dữ liệu Solomon (nếu có)

```bash
# Sử dụng file dữ liệu Solomon
python main.py --solomon data/solomon_dataset/C1/C101.csv

# Với các file khác
python main.py --solomon data/solomon_dataset/C1/C102.csv
python main.py --solomon data/solomon_dataset/R1/R101.csv
```

## Các tùy chọn nâng cao

### Tùy chỉnh thuật toán di truyền

```bash
python main.py --generate --customers 50 \
               --generations 2000 \        # Số thế hệ
               --population 150 \           # Kích thước quần thể
               --crossover-prob 0.85 \      # Xác suất lai ghép
               --mutation-prob 0.2 \        # Xác suất đột biến
               --tournament-size 7 \        # Kích thước giải đấu
               --elitism-rate 0.2           # Tỷ lệ ưu tú
```

### Tùy chỉnh bài toán VRP

```bash
python main.py --generate --customers 50 \
               --capacity 200 \             # Sức chứa xe
               --vehicles 10 \              # Số xe tối đa
               --traffic-factor 1.2         # Hệ số giao thông
```

### Tùy chỉnh phương pháp phân cụm dữ liệu

```bash
# Phân cụm ngẫu nhiên
python main.py --generate --customers 50 --clustering random

# Phân cụm K-means
python main.py --generate --customers 50 --clustering kmeans

# Phân cụm theo hướng tâm
python main.py --generate --customers 50 --clustering radial
```

### Tùy chỉnh đầu ra

```bash
python main.py --generate --customers 50 \
               --output results/my_experiment \  # Thư mục kết quả
               --no-plots \                       # Không tạo biểu đồ
               --no-report \                      # Không tạo báo cáo
               --save-solution                    # Lưu dữ liệu giải pháp
```

### Tùy chỉnh thuật toán

```bash
python main.py --generate --customers 50 \
               --no-local-search \           # Bỏ qua tối ưu 2-opt
               --no-baseline                  # Bỏ qua thuật toán cơ sở
```

### Chế độ debug

```bash
python main.py --generate --customers 50 \
               --verbose \                   # Hiển thị chi tiết
               --seed 42                     # Đặt seed ngẫu nhiên
```

## Chạy demo và kiểm tra

### Demo nhanh

```bash
python demo.py --quick
```

### Demo đầy đủ

```bash
python demo.py
```

### Chạy test

```bash
python tests/run_tests.py
```

## Kết quả đầu ra

Sau khi chạy, hệ thống sẽ tạo các file kết quả trong thư mục `results/`:

### 1. Báo cáo văn bản

- `report.txt`: Báo cáo chi tiết với thống kê
- `report_data.json`: Dữ liệu đầy đủ dạng JSON

### 2. Hình ảnh trực quan

- `ga_routes.png`: Bản đồ tuyến đường của thuật toán di truyền
- `nn_routes.png`: Bản đồ tuyến đường của thuật toán cơ sở
- `comparison.png`: So sánh hai phương pháp
- `convergence.png`: Biểu đồ hội tụ của GA
- `ga_dashboard.png`: Bảng điều khiển KPI của GA
- `nn_dashboard.png`: Bảng điều khiển KPI của NN
- `comparison_chart.png`: Biểu đồ so sánh các chỉ số
- `improvements.png`: Phân tích cải thiện

### 3. Dữ liệu giải pháp

- `ga_solution_*.json`: Dữ liệu giải pháp GA
- `nn_solution_*.json`: Dữ liệu giải pháp NN

## Các ví dụ thực tế

### Ví dụ 1: Bài toán nhỏ (20 khách hàng)

```bash
python main.py --generate --customers 20 --capacity 100 --generations 500
```

**Thời gian chạy**: 10-30 giây

### Ví dụ 2: Bài toán trung bình (50 khách hàng)

```bash
python main.py --generate --customers 50 --capacity 200 --generations 1000
```

**Thời gian chạy**: 1-3 phút

### Ví dụ 3: Bài toán lớn (100 khách hàng)

```bash
python main.py --generate --customers 100 --capacity 300 --generations 2000 --population 200
```

**Thời gian chạy**: 5-15 phút

### Ví dụ 4: Sử dụng dữ liệu Solomon

```bash
python main.py --solomon data/solomon_dataset/C1/C101.csv --generations 1500
```

## Xử lý lỗi thường gặp

### Lỗi: "ModuleNotFoundError"

```bash
# Cài đặt lại thư viện
pip install -r requirements.txt

# Hoặc cài đặt từng thư viện
pip install numpy pandas matplotlib seaborn scikit-learn scipy pytest
```

### Lỗi: "FileNotFoundError" khi chạy Solomon

- Kiểm tra file dữ liệu có tồn tại không
- Đảm bảo đường dẫn đúng: `data/solomon_dataset/C1/C101.csv`

### Lỗi: "MemoryError" với bài toán lớn

- Giảm kích thước quần thể: `--population 50`
- Giảm số thế hệ: `--generations 500`
- Giảm số khách hàng: `--customers 30`

### Lỗi: "TimeoutError"

- Tăng thời gian chờ
- Giảm kích thước bài toán
- Sử dụng `--no-plots` để bỏ qua tạo hình ảnh

## Tối ưu hóa hiệu suất

### Cho bài toán nhỏ (< 50 khách hàng)

```bash
python main.py --generate --customers 30 \
               --generations 1000 \
               --population 100 \
               --crossover-prob 0.9 \
               --mutation-prob 0.15
```

### Cho bài toán lớn (> 100 khách hàng)

```bash
python main.py --generate --customers 150 \
               --generations 3000 \
               --population 300 \
               --crossover-prob 0.85 \
               --mutation-prob 0.2 \
               --tournament-size 10
```

## Cấu hình nâng cao

### Chỉnh sửa file config.py

```python
# Thay đổi các tham số mặc định
GA_CONFIG = {
    'population_size': 150,      # Tăng kích thước quần thể
    'generations': 2000,         # Tăng số thế hệ
    'crossover_prob': 0.85,      # Giảm xác suất lai ghép
    'mutation_prob': 0.2,        # Tăng xác suất đột biến
}

VRP_CONFIG = {
    'vehicle_capacity': 250,     # Tăng sức chứa xe
    'num_vehicles': 30,          # Tăng số xe
    'traffic_factor': 1.1,       # Thêm hệ số giao thông
}
```

## Hỗ trợ và góp ý

Nếu gặp vấn đề hoặc có góp ý:

1. Kiểm tra file README.md để biết thêm chi tiết
2. Chạy `python main.py --help` để xem tất cả tùy chọn
3. Kiểm tra log lỗi trong terminal
4. Thử chạy demo để kiểm tra hệ thống

## Lưu ý quan trọng

1. **Thời gian chạy**: Phụ thuộc vào kích thước bài toán và cấu hình máy
2. **Bộ nhớ**: Bài toán lớn cần nhiều RAM
3. **Kết quả**: Mỗi lần chạy có thể cho kết quả khác nhau do tính ngẫu nhiên
4. **Seed**: Sử dụng `--seed` để có kết quả tái lập được
5. **Dữ liệu**: Đảm bảo dữ liệu đầu vào hợp lệ

Chúc bạn sử dụng thành công hệ thống VRP-GA!
