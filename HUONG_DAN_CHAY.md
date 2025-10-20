# HƯỚNG DẪN CHẠY PROJECT VRP-GA SYSTEM

## Tổng quan

VRP-GA System là một hệ thống giải quyết bài toán Vehicle Routing Problem (VRP) sử dụng thuật toán di truyền (Genetic Algorithm) với tối ưu hóa tìm kiếm cục bộ 2-opt. Hệ thống hỗ trợ **hai loại visualization**:

- **Map Hà Nội tương tác** cho mockup datasets (tọa độ thực tế)
- **Visualization truyền thống** cho Solomon datasets (tọa độ giả lập)

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

### Mockup Datasets (Map Hà Nội)

Sau khi chạy mockup dataset, hệ thống tạo:

#### 1. Bản đồ tương tác HTML

- `ga_hanoi_map.html` - GA solution trên bản đồ Hà Nội
- `nn_hanoi_map.html` - NN solution trên bản đồ Hà Nội
- `comparison_hanoi_map.html` - So sánh GA vs NN

#### 2. Báo cáo văn bản

- `report.txt` - Báo cáo chi tiết với thống kê

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

**Kết quả**: 3 file HTML map tương tác + report.txt
**Thời gian chạy**: 5-15 giây

### Ví dụ 2: Map Hà Nội - Bài toán trung bình

```bash
python main.py --mockup-dataset medium_kmeans --generations 200 --population 50
```

**Kết quả**: 3 file HTML map tương tác + report.txt
**Thời gian chạy**: 30-60 giây

### Ví dụ 3: Solomon - Bài toán lớn

```bash
python main.py --solomon-dataset C101 --generations 500 --population 100
```

**Kết quả**: 6+ file PNG + report.txt
**Thời gian chạy**: 1-3 phút

### Ví dụ 4: So sánh hiệu suất

```bash
# Test với dataset nhỏ
python main.py --mockup-dataset small_random --generations 50 --population 20

# Test với dataset lớn
python main.py --solomon-dataset C101 --generations 100 --population 50
```

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

# 3. Chạy map Hà Nội
python main.py --mockup-dataset small_random --generations 50 --population 20

# 4. Chạy Solomon
python main.py --solomon-dataset C101 --generations 100 --population 50

# 5. Xem kết quả trong thư mục results/
```

Chúc bạn sử dụng thành công hệ thống VRP-GA với map Hà Nội! 🗺️🚚
