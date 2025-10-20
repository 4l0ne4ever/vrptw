# HƯỚNG DẪN CHẠY PROJECT VRP-GA SYSTEM

## Tổng quan

VRP-GA System là một hệ thống giải quyết bài toán Vehicle Routing Problem (VRP) sử dụng thuật toán di truyền (Genetic Algorithm) với tối ưu hóa tìm kiếm cục bộ 2-opt. Hệ thống hỗ trợ:

- **Map Hà Nội tương tác** cho mockup datasets (tọa độ thực tế)
- **Visualization truyền thống** cho Solomon datasets (tọa độ giả lập)
- **Tính phí giao hàng thực tế** theo mô hình Ahamove với các phụ phí dịch vụ
- **Xuất kết quả chi tiết** bao gồm evolution data, optimal routes, và KPI comparison

## Yêu cầu hệ thống

- Python 3.8 trở lên
- Hệ điều hành: Windows, macOS, hoặc Linux
- RAM: Tối thiểu 4GB (khuyến nghị 8GB)
- Dung lượng ổ cứng: 500MB trống
- Kết nối internet (để tải Folium maps)

## Cài đặt

### Bước 1: Kiểm tra Python

```bash
python --version
# hoặc
python3 --version
```

Nếu chưa có Python 3.8+, hãy tải về từ [python.org](https://python.org)

### Bước 2: Tạo virtual environment (khuyến nghị)

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 4: Khởi tạo datasets

```bash
# Chuyển đổi Solomon datasets sang JSON
python main.py --convert-solomon

# Tạo mockup datasets mẫu
python main.py --create-samples
```

## Cách sử dụng cơ bản

### 1. Sử dụng Mockup Datasets (Map Hà Nội)

```bash
# Xem danh sách mockup datasets
python main.py --list-mockup

# Chạy với dataset nhỏ (10 khách hàng)
python main.py --mockup-dataset small_random --generations 100 --population 30

# Chạy với dataset trung bình (20 khách hàng)
python main.py --mockup-dataset medium_kmeans --generations 200 --population 50

# Chạy với dataset lớn (50 khách hàng)
python main.py --mockup-dataset large_kmeans --generations 500 --population 100
```

### 2. Sử dụng Solomon Datasets (Visualization truyền thống)

```bash
# Xem danh sách Solomon datasets
python main.py --list-solomon

# Chạy với dataset C101 (100 khách hàng)
python main.py --solomon-dataset C101 --generations 500 --population 100

# Chạy với dataset R101 (100 khách hàng)
python main.py --solomon-dataset R101 --generations 500 --population 100

# Chạy tất cả Solomon datasets trong batch mode
python main.py --solomon-batch --generations 100 --population 50
```

### 3. Auto-detect dataset type

```bash
# Hệ thống tự động phát hiện loại dataset
python main.py --dataset small_random --generations 100 --population 30
python main.py --dataset C101 --generations 500 --population 100
```

## Các tùy chọn nâng cao

### Tùy chỉnh thuật toán di truyền

```bash
python main.py --mockup-dataset medium_kmeans \
               --generations 2000 \        # Số thế hệ
               --population 150 \           # Kích thước quần thể
               --crossover-prob 0.85 \      # Xác suất lai ghép
               --mutation-prob 0.2 \        # Xác suất đột biến
               --tournament-size 7 \        # Kích thước giải đấu
               --elitism-rate 0.2           # Tỷ lệ ưu tú
```

### Tùy chỉnh đầu ra

```bash
python main.py --mockup-dataset small_random \
               --generations 100 \
               --population 30 \
               --no-plots \                 # Không tạo biểu đồ (chỉ cho Solomon)
               --no-report                  # Không tạo báo cáo
```

### Chế độ debug

```bash
python main.py --mockup-dataset small_random \
               --generations 50 \
               --population 20 \
               --verbose \                  # Hiển thị chi tiết
               --seed 42                    # Đặt seed ngẫu nhiên
```

## Quản lý Datasets

### Xem danh sách datasets

```bash
# Xem tất cả datasets
python main.py --list-datasets

# Xem chỉ mockup datasets
python main.py --list-mockup

# Xem chỉ Solomon datasets
python main.py --list-solomon
```

### Tạo datasets mới

```bash
# Tạo mockup datasets mẫu
python main.py --create-samples

# Chuyển đổi Solomon datasets sang JSON
python main.py --convert-solomon
```

## Kết quả đầu ra

### Files kết quả chi tiết (Tất cả datasets)

Hệ thống tự động tạo các file kết quả chi tiết:

#### 1. Evolution Data (Quá trình tiến hóa GA)

- `evolution_data_YYYYMMDD_HHMMSS.csv` - Dữ liệu tiến hóa qua các thế hệ
- **Nội dung**: generation, evaluated_individuals, min_fitness, max_fitness, avg_fitness, std_fitness, best_distance, avg_distance, diversity

#### 2. Optimal Routes (Lộ trình tối ưu)

- `optimal_routes_YYYYMMDD_HHMMSS.txt` - Lộ trình chi tiết từng xe
- **Nội dung**:
  - Xe 1: Depot → KH_5 → KH_12 → ... → Depot
  - Xe 2: Depot → KH_3 → KH_8 → ... → Depot
  - Tổng km, tải trọng mỗi xe, **phí giao hàng từng tuyến**

#### 3. KPI Comparison (So sánh GA vs Nearest Neighbor)

- `kpi_comparison_YYYYMMDD_HHMMSS.csv` - So sánh hiệu suất
- **Nội dung**: Tổng km, chi phí, số xe, thời gian tính toán, **phí giao hàng**, tỷ lệ cải thiện (%)

### Mockup Datasets (Map Hà Nội)

Sau khi chạy mockup dataset, hệ thống tạo:

#### 1. Bản đồ tương tác HTML

- `ga_hanoi_map_real.html` - GA solution với tuyến đường thực tế
- `ga_hanoi_map_straight.html` - GA solution với đường thẳng
- `nn_hanoi_map_real.html` - NN solution với tuyến đường thực tế
- `comparison_hanoi_map_real.html` - So sánh GA vs NN (tuyến thực tế)
- `comparison_hanoi_map_straight.html` - So sánh GA vs NN (đường thẳng)

#### 2. Báo cáo văn bản

- `report.txt` - Báo cáo chi tiết với thống kê

#### 3. Solomon Batch Summary (Chỉ khi chạy --solomon-batch)

- `solomon_summary_YYYYMMDD_HHMMSS.csv` - Tổng hợp tất cả Solomon datasets
- **Nội dung**: dataset, customers, capacity, vehicles, ga_distance, ga_cost, ga_routes, ga_utilization, ga_efficiency, ga_feasible

### Solomon Datasets (Visualization truyền thống)

Sau khi chạy Solomon dataset, hệ thống tạo:

#### 1. Hình ảnh trực quan PNG

- `ga_routes.png` - Bản đồ tuyến đường của GA
- `nn_routes.png` - Bản đồ tuyến đường của NN
- `comparison.png` - So sánh hai phương pháp
- `convergence.png` - Biểu đồ hội tụ của GA
- `ga_dashboard.png` - Bảng điều khiển KPI của GA
- `nn_dashboard.png` - Bảng điều khiển KPI của NN

#### 2. Báo cáo văn bản

- `report.txt` - Báo cáo chi tiết với thống kê

## Các ví dụ thực tế

### Ví dụ 1: Map Hà Nội - Bài toán nhỏ

```bash
python main.py --mockup-dataset small_random --generations 100 --population 30
```

**Kết quả**:

- 5 file HTML map tương tác (real routes + straight lines)
- 3 file CSV kết quả chi tiết (evolution, routes, KPI comparison)
- Report.txt với thống kê và **phí giao hàng**
  **Thời gian chạy**: 5-15 giây

### Ví dụ 2: Map Hà Nội - Bài toán trung bình

```bash
python main.py --mockup-dataset medium_kmeans --generations 200 --population 50
```

**Kết quả**:

- 5 file HTML map tương tác + 3 file CSV + report.txt
- **Phí giao hàng**: ~1,000,000-2,000,000 VND
  **Thời gian chạy**: 30-60 giây

### Ví dụ 3: Solomon - Bài toán lớn

```bash
python main.py --solomon-dataset C101 --generations 500 --population 100
```

**Kết quả**:

- 6+ file PNG + 3 file CSV + report.txt
- **Phí giao hàng**: Tính theo mô hình Ahamove
  **Thời gian chạy**: 1-3 phút

### Ví dụ 4: Solomon Batch Processing

```bash
python main.py --solomon-batch --generations 100 --population 50
```

**Kết quả**:

- File tổng hợp: `solomon_summary_YYYYMMDD_HHMMSS.csv`
- **55 Solomon datasets** được xử lý
- **Phí giao hàng** cho từng dataset
  **Thời gian chạy**: 10-30 phút

## Xử lý lỗi thường gặp

### Lỗi: "ModuleNotFoundError"

```bash
# Cài đặt lại thư viện
pip install -r requirements.txt

# Hoặc cài đặt từng thư viện
pip install numpy pandas matplotlib seaborn scikit-learn scipy pytest folium
```

### Lỗi: "Dataset not found"

```bash
# Kiểm tra datasets có sẵn
python main.py --list-datasets

# Tạo lại datasets nếu cần
python main.py --create-samples
python main.py --convert-solomon
```

### Lỗi: "MemoryError" với bài toán lớn

```bash
# Giảm kích thước quần thể
python main.py --solomon-dataset C101 --population 50

# Giảm số thế hệ
python main.py --solomon-dataset C101 --generations 200

# Sử dụng dataset nhỏ hơn
python main.py --mockup-dataset small_random
```

### Lỗi: "Map not loading"

- Kiểm tra kết nối internet (cần cho Folium)
- Đảm bảo file HTML được tạo trong thư mục results
- Mở file HTML bằng trình duyệt web

## Tối ưu hóa hiệu suất

### Cho Mockup Datasets (Map Hà Nội)

```bash
# Dataset nhỏ - chạy nhanh
python main.py --mockup-dataset small_random \
               --generations 50 \
               --population 20

# Dataset trung bình - cân bằng
python main.py --mockup-dataset medium_kmeans \
               --generations 200 \
               --population 50

# Dataset lớn - chất lượng cao
python main.py --mockup-dataset large_kmeans \
               --generations 500 \
               --population 100
```

### Cho Solomon Datasets (Visualization truyền thống)

```bash
# Dataset nhỏ - chạy nhanh
python main.py --solomon-dataset C101 \
               --generations 200 \
               --population 50 \
               --no-plots --no-report

# Dataset lớn - đầy đủ tính năng
python main.py --solomon-dataset C101 \
               --generations 1000 \
               --population 150
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

### Tùy chỉnh tọa độ Hà Nội

Chỉnh sửa file `src/data_processing/hanoi_coordinates.py`:

```python
# Thay đổi ranh giới Hà Nội
self.hanoi_bounds = {
    'min_lat': 20.5,   # Mở rộng về phía nam
    'max_lat': 21.5,   # Mở rộng về phía bắc
    'min_lon': 105.0,  # Mở rộng về phía tây
    'max_lon': 106.2   # Mở rộng về phía đông
}

# Thêm quận mới
self.districts['my_dong'] = {
    'lat': 21.0, 'lon': 105.8, 'radius': 0.1
}
```

## Tính năng đặc biệt

### 1. Map Hà Nội tương tác

- Tọa độ thực tế của Hà Nội
- Các quận: Hoàn Kiếm, Ba Đình, Đống Đa, Hai Bà Trưng, v.v.
- Landmarks: Hồ Hoàn Kiếm, Hồ Tây, Sân bay Nội Bài
- Zoom, pan, click để xem thông tin chi tiết

### 2. Dual Visualization System

- **Mockup datasets** → Map Hà Nội (HTML)
- **Solomon datasets** → Traditional plots (PNG)
- Auto-detection dựa trên tọa độ

### 3. JSON Dataset System

- Unified format cho tất cả datasets
- Auto-conversion từ Solomon CSV
- Metadata và catalog management

### 4. Tính phí giao hàng thực tế

Hệ thống tích hợp tính phí giao hàng theo mô hình **Ahamove**:

#### Công thức tính phí:

```
Cước phí = (Giá cơ bản × Số km) + Phụ phí dịch vụ
```

#### Các loại phí:

- **Phí cơ bản**: Theo khoảng cách (Express: 15,709 VND/2km đầu)
- **Phí điểm dừng**: 5,500 VND/điểm dừng thêm
- **Phí COD**: 0.6% giá trị đơn hàng
- **Phí chờ**: 60,000 VND/giờ sau 15 phút miễn phí

#### Ví dụ tính phí:

- Khoảng cách: 5km, 2 điểm dừng
- **Kết quả**: 51,645 VND (khớp với mô tả)

#### Xuất trong kết quả:

- **Optimal Routes**: Phí giao hàng từng tuyến
- **KPI Comparison**: Tổng phí giao hàng, phí/km, phí/khách hàng
- **Chi tiết**: Phân tích từng thành phần phí (cơ bản, COD, chờ, điểm dừng)

### 5. Batch Processing cho Solomon Datasets

- Chạy tất cả Solomon datasets cùng lúc: `--solomon-batch`
- Tạo file tổng hợp: `solomon_summary_YYYYMMDD_HHMMSS.csv`
- So sánh hiệu suất trên nhiều test cases

## Hỗ trợ và góp ý

Nếu gặp vấn đề hoặc có góp ý:

1. Kiểm tra file README.md để biết thêm chi tiết
2. Chạy `python main.py --help` để xem tất cả tùy chọn
3. Kiểm tra log lỗi trong terminal
4. Xem danh sách datasets: `python main.py --list-datasets`

## Lưu ý quan trọng

1. **Thời gian chạy**: Phụ thuộc vào kích thước bài toán và cấu hình máy
2. **Bộ nhớ**: Bài toán lớn cần nhiều RAM
3. **Kết quả**: Mỗi lần chạy có thể cho kết quả khác nhau do tính ngẫu nhiên
4. **Seed**: Sử dụng `--seed` để có kết quả tái lập được
5. **Internet**: Map Hà Nội cần kết nối internet để tải tiles
6. **Browser**: Mở file HTML bằng trình duyệt web để xem map

## Quick Start

```bash
# 1. Cài đặt
pip install -r requirements.txt

# 2. Khởi tạo datasets
python main.py --create-samples
python main.py --convert-solomon

# 3. Chạy map Hà Nội với phí giao hàng
python main.py --mockup-dataset small_random --generations 50 --population 20

# 4. Chạy Solomon với phí giao hàng
python main.py --solomon-dataset C101 --generations 100 --population 50

# 5. Chạy batch tất cả Solomon datasets
python main.py --solomon-batch --generations 50 --population 30

# 6. Xem kết quả trong thư mục results/
# - evolution_data_*.csv: Quá trình tiến hóa GA
# - optimal_routes_*.txt: Lộ trình + phí giao hàng
# - kpi_comparison_*.csv: So sánh GA vs NN + phí giao hàng
# - solomon_summary_*.csv: Tổng hợp Solomon datasets
# - *.html: Map Hà Nội tương tác
# - *.png: Visualization truyền thống
```

Chúc bạn sử dụng thành công hệ thống VRP-GA với map Hà Nội và tính phí giao hàng thực tế! 🗺️🚚💰
