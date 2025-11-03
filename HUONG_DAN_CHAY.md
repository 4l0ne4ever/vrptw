# HƯỚNG DẪN CHẠY PROJECT VRP-GA SYSTEM

## Tổng quan

VRP-GA System là một hệ thống giải quyết bài toán Vehicle Routing Problem (VRP) sử dụng thuật toán di truyền (Genetic Algorithm) với tối ưu hóa tìm kiếm cục bộ 2-opt. Hệ thống hỗ trợ:

- **Map Hà Nội tương tác** cho mockup datasets (tọa độ thực tế)
- **Visualization truyền thống** cho Solomon datasets (tọa độ giả lập)
- **Tính phí giao hàng thực tế** theo mô hình Ahamove với các phụ phí dịch vụ
- **BKS Validation**: So sánh với Best-Known Solutions cho Solomon instances
- **Split Algorithm**: Thuật toán tối ưu phân chia tuyến theo Prins (2004)
- **Logging System**: Ghi log chi tiết quá trình chạy
- **Error Handling**: Custom exceptions cho error tracking tốt hơn
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

Có **2 cách** để làm việc với mockup data:

#### Cách 1: Sử dụng Mockup Datasets đã tạo sẵn (JSON)

```bash
# Xem danh sách mockup datasets
python main.py --list-mockup

# Chạy với dataset nhỏ (10 khách hàng) - sử dụng mặc định từ config.py
python main.py --mockup-dataset small_random
# Mặc định: generations=1000, population=100 (theo luận văn)

# Chạy với dataset trung bình (20 khách hàng) - sử dụng mặc định
python main.py --mockup-dataset medium_kmeans
# Mặc định: generations=1000, population=100

# Chạy với dataset lớn (50 khách hàng) - sử dụng mặc định
python main.py --mockup-dataset large_kmeans
# Mặc định: generations=1000, population=100

# Tùy chỉnh nếu cần (ví dụ: test nhanh với ít generations)
python main.py --mockup-dataset small_random --generations 100 --population 30
```

**Lưu ý**:

- **Mặc định**: Nếu không chỉ định `--generations` và `--population`, hệ thống sẽ dùng giá trị từ `config.py`:
  - `generations`: **1000** (theo luận văn - Table 2.18)
  - `population_size`: **100** (theo luận văn - Table 2.18)
- Datasets được lưu trong `data/datasets/mockup/` dạng JSON
- Data được tạo một lần bằng `--create-samples` và có thể tái sử dụng
- Phù hợp cho việc so sánh kết quả giữa các lần chạy

#### Cách 2: Generate Mockup Data ngay khi chạy (--generate)

```bash
# Generate và chạy ngay (không lưu vào JSON dataset) - sử dụng mặc định
python main.py --generate --customers 50 --capacity 200
# Mặc định: generations=1000, population=100

# Với tùy chọn clustering - sử dụng mặc định
python main.py --generate --customers 30 --capacity 200 --clustering kmeans
# Mặc định: generations=1000, population=100

# Với seed để tái lập được - sử dụng mặc định
python main.py --generate --customers 25 --seed 42
# Mặc định: generations=1000, population=100

# Tùy chỉnh nếu cần (ví dụ: test nhanh)
python main.py --generate --customers 50 --capacity 200 --generations 500 --population 100
python main.py --generate --customers 30 --capacity 200 --clustering kmeans --generations 200 --population 50
```

**Lưu ý**:

- **Mặc định**: Nếu không chỉ định `--generations` và `--population`, hệ thống sẽ dùng giá trị từ `config.py`:
  - `generations`: **1000** (theo luận văn - Table 2.18)
  - `population_size`: **100** (theo luận văn - Table 2.18)
- Data được **generate ngay tại thời điểm chạy**, không lưu vào JSON dataset
- Data được lưu tạm vào CSV trong `results/` để tham khảo
- Mỗi lần chạy `--generate` sẽ tạo data mới (trừ khi dùng `--seed`)
- Phù hợp cho việc test nhanh hoặc tạo data custom

### 2. Sử dụng Solomon Datasets (Visualization truyền thống)

```bash
# Xem danh sách Solomon datasets
python main.py --list-solomon

# Chạy với dataset C101 (100 khách hàng) - sử dụng mặc định
python main.py --solomon-dataset C101
# Mặc định: generations=1000, population=100 (theo luận văn)

# Chạy với dataset R101 (100 khách hàng) - sử dụng mặc định
python main.py --solomon-dataset R101
# Mặc định: generations=1000, population=100

# Chạy tất cả Solomon datasets trong batch mode - sử dụng mặc định
python main.py --solomon-batch
# Mặc định: generations=1000, population=100

# Tùy chỉnh nếu cần (ví dụ: test nhanh với batch mode)
python main.py --solomon-batch --generations 100 --population 50
```

### 3. Generate Mockup Data ngay khi chạy

```bash
# Generate data mới và chạy ngay (không lưu vào JSON dataset) - sử dụng mặc định
python main.py --generate --customers 50 --capacity 200
# Mặc định: generations=1000, population=100

# Với clustering method - sử dụng mặc định
python main.py --generate --customers 30 --clustering kmeans
# Mặc định: generations=1000, population=100

# Với seed để reproducible - sử dụng mặc định
python main.py --generate --customers 25 --seed 42
# Mặc định: generations=1000, population=100

# Tùy chỉnh nếu cần (ví dụ: test nhanh)
python main.py --generate --customers 50 --capacity 200 --generations 500 --population 100
python main.py --generate --customers 30 --clustering kmeans --generations 200 --population 50
```

**Khác biệt giữa `--generate` và `--mockup-dataset`**:

| Tùy chọn           | Mô tả                           | Data Location                          | Tái sử dụng                |
| ------------------ | ------------------------------- | -------------------------------------- | -------------------------- |
| `--mockup-dataset` | Load từ JSON dataset đã tạo     | `data/datasets/mockup/*.json`          | ✅ Có, data cố định        |
| `--generate`       | Generate data mới ngay khi chạy | `results/mockup_*_customers.csv` (tạm) | ❌ Không, data mới mỗi lần |

### 4. Auto-detect dataset type

```bash
# Hệ thống tự động phát hiện loại dataset - sử dụng mặc định
python main.py --dataset small_random
# Mặc định: generations=1000, population=100

python main.py --dataset C101
# Mặc định: generations=1000, population=100

# Tùy chỉnh nếu cần
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
- **Nội dung**:
  - dataset, customers, capacity, vehicles
  - ga_distance, ga_cost, ga_routes, ga_utilization, ga_efficiency, ga_feasible
  - **BKS data** (nếu có): bks_distance, bks_vehicles, gap_percent, vehicle_diff, quality
  - generations, population
- **BKS Statistics**: Tự động hiển thị average gap và quality distribution trong console

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
# Sử dụng mặc định (theo luận văn)
python main.py --mockup-dataset small_random
# Mặc định: generations=1000, population=100

# Hoặc test nhanh với ít generations
python main.py --mockup-dataset small_random --generations 100 --population 30
```

**Kết quả** (với mặc định generations=1000):

- 5 file HTML map tương tác (real routes + straight lines)
- 3 file CSV kết quả chi tiết (evolution, routes, KPI comparison)
- Report.txt với thống kê và **phí giao hàng**
  **Thời gian chạy**: 30-60 giây (với generations=1000)

**Lưu ý**: Với `--generations 100` thì thời gian chạy chỉ 5-15 giây (phù hợp cho test nhanh)

### Ví dụ 2: Map Hà Nội - Bài toán trung bình

```bash
# Sử dụng mặc định (theo luận văn)
python main.py --mockup-dataset medium_kmeans
# Mặc định: generations=1000, population=100

# Hoặc test nhanh
python main.py --mockup-dataset medium_kmeans --generations 200 --population 50
```

**Kết quả** (với mặc định generations=1000):

- 5 file HTML map tương tác + 3 file CSV + report.txt
- **Phí giao hàng**: ~1,000,000-2,000,000 VND
  **Thời gian chạy**: 1-2 phút (với generations=1000)

**Lưu ý**: Với `--generations 200` thì thời gian chạy chỉ 30-60 giây (phù hợp cho test nhanh)

### Ví dụ 3: Solomon - Bài toán lớn

```bash
# Sử dụng mặc định (theo luận văn)
python main.py --solomon-dataset C101
# Mặc định: generations=1000, population=100

# Hoặc test nhanh
python main.py --solomon-dataset C101 --generations 500 --population 100
```

**Kết quả** (với mặc định generations=1000):

- 6+ file PNG + 3 file CSV + report.txt
- **Phí giao hàng**: Tính theo mô hình Ahamove
- **BKS Comparison**: Tự động hiển thị gap và quality rating
  **Thời gian chạy**: 2-5 phút (với generations=1000)

**Lưu ý**: Với `--generations 500` thì thời gian chạy chỉ 1-3 phút (phù hợp cho test nhanh)

### Ví dụ 4: Solomon Batch Processing

```bash
# Sử dụng mặc định (theo luận văn)
python main.py --solomon-batch
# Mặc định: generations=1000, population=100

# Hoặc test nhanh với ít generations
python main.py --solomon-batch --generations 100 --population 50
```

**Kết quả** (với mặc định generations=1000):

- File tổng hợp: `solomon_summary_YYYYMMDD_HHMMSS.csv`
- **56 Solomon datasets** được xử lý
- **Phí giao hàng** cho từng dataset
- **BKS Comparison**: Tự động so sánh với Best-Known Solutions
  - Gap percentage cho mỗi instance
  - Quality rating (EXCELLENT/GOOD/ACCEPTABLE/POOR)
  - BKS statistics trong console và CSV
    **Thời gian chạy**: 2-6 giờ (với generations=1000 cho 56 datasets)

**Lưu ý**: Với `--generations 100` thì thời gian chạy chỉ 10-30 phút (phù hợp cho test nhanh)

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
# Thay đổi các tham số mặc định (theo luận văn - Table 2.18)
GA_CONFIG = {
    'population_size': 100,      # Kích thước quần thể (mặc định theo luận văn)
    'generations': 1000,         # Số thế hệ (mặc định theo luận văn)
    'crossover_prob': 0.9,       # Xác suất lai ghép
    'mutation_prob': 0.15,       # Xác suất đột biến
    'tournament_size': 5,        # Kích thước giải đấu
    'elitism_rate': 0.10,        # Tỷ lệ ưu tú (10%)
    'adaptive_mutation': False,   # Fixed mutation rate
    'convergence_threshold': 0.001,
    'stagnation_limit': 50,
    'use_split_algorithm': False  # Enable Split Algorithm (Prins 2004) - mặc định OFF
}

VRP_CONFIG = {
    'vehicle_capacity': 200,     # Sức chứa xe (theo luận văn)
    'num_vehicles': 25,          # Số xe mặc định
    'traffic_factor': 1.0,       # Hệ số giao thông (1.0 = không tắc nghẽn)
    'penalty_weight': 1000,      # Penalty cho constraint violations
    'use_waiting_fee': False,    # Phí chờ = 0 (theo luận văn)
    'cod_fee_rate': 0.006        # Phí COD = 0.6%
}

MOCKUP_CONFIG = {
    'n_customers': 50,           # Số khách hàng mặc định
    'demand_lambda': 7,          # Poisson(λ=7) cho demand
    'demand_min': 1,             # Demand tối thiểu
    'demand_max': 20,            # Demand tối đa
    'service_time': 600,         # Thời gian phục vụ = 10 phút (theo luận văn)
    'area_bounds': (0, 100),     # Không gian [0,100]×[0,100]
    'clustering': 'kmeans',      # Phương pháp clustering
    'n_clusters': 5,
    'seed': 42                   # Random seed
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

### 5. BKS Validation (Best-Known Solutions)

Hệ thống **TỰ ĐỘNG** so sánh kết quả với Best-Known Solutions từ literature:

- **BKS Data**: Lưu trong `data/solomon_bks.json` với 56 Solomon instances
- **Gap Calculation**: Tính phần trăm chênh lệch so với BKS
- **Quality Rating**: EXCELLENT (<1%), GOOD (<3%), ACCEPTABLE (<5%), POOR (>5%)
- **Automatic Comparison**: Tự động so sánh khi chạy:
  - ✅ **Single Solomon dataset**: Hiển thị BKS comparison trong console
  - ✅ **Solomon Batch Mode**: Thêm BKS data vào summary CSV và statistics

#### Sử dụng tự động:

```bash
# Chạy single Solomon dataset - tự động hiển thị BKS comparison
python main.py --solomon-dataset C101 --generations 500 --population 100

# Output sẽ bao gồm:
# BKS Comparison:
#   Instance: C101
#   Solution Distance: 828.94
#   BKS Distance: 828.94
#   Gap: 0.00%
#   Quality: EXCELLENT

# Chạy batch mode - tự động thêm BKS vào summary
python main.py --solomon-batch --generations 100 --population 50

# Output sẽ bao gồm:
# Batch Summary:
#   BKS Comparison (for X instances with BKS):
#     Average gap from BKS: X.XX%
#     Quality distribution:
#       EXCELLENT: X
#       GOOD: X
#       ...
```

#### Sử dụng trong code (nếu cần):

```python
# Sử dụng BKS validator trong code
from src.evaluation.bks_validator import BKSValidator

validator = BKSValidator('data/solomon_bks.json')
validation = validator.validate_solution('C101', solution)
print(f"Gap from BKS: {validation['gap_percent']:.2f}%")
print(f"Quality: {validation['quality']}")
```

### 6. Split Algorithm (Prins 2004)

Hệ thống triển khai Split Algorithm tối ưu theo Prins (2004) và **TỰ ĐỘNG** sử dụng khi được enable:

- **Dynamic Programming**: Tối ưu toàn cục cho việc phân chia tuyến
- **Optimal Route Splitting**: Tìm cách phân chia giant tour với chi phí nhỏ nhất
- **Capacity Respecting**: Đảm bảo mọi route đều thỏa mãn capacity constraint
- **Automatic Fallback**: Tự động fallback về greedy decoder nếu Split Algorithm fails

#### Cách enable Split Algorithm:

**Option 1**: Enable trong `config.py`

```python
GA_CONFIG = {
    # ... other config ...
    'use_split_algorithm': True,  # Enable Split Algorithm
}
```

**Option 2**: Enable khi tạo decoder trong code

```python
from src.algorithms.decoder import RouteDecoder

# Enable Split Algorithm
decoder = RouteDecoder(problem, use_split_algorithm=True)
routes = decoder.decode_chromosome(chromosome)
```

**Note**:

- Mặc định: **OFF** (sử dụng greedy decoder - nhanh hơn)
- Khi enable: Sử dụng Split Algorithm - tối ưu hơn nhưng chậm hơn
- Tự động fallback nếu có lỗi

#### Sử dụng trực tiếp (advanced):

```python
# Sử dụng Split Algorithm trực tiếp
from src.algorithms.split import SplitAlgorithm

splitter = SplitAlgorithm(problem)
routes, cost = splitter.split(giant_tour)
```

### 7. Logging System

Hệ thống có logging system chuyên nghiệp và **TỰ ĐỘNG** hoạt động:

- **Log Files**: Tự động tạo trong `logs/` với timestamp
- **Console + File**: Log ra cả console và file song song
- **Different Levels**: INFO, DEBUG, WARNING, ERROR
- **Automatic Logging**: Tự động log:
  - ✅ Application start/stop
  - ✅ Dataset loading
  - ✅ GA execution progress
  - ✅ Optimization results
  - ✅ Error messages với stack trace
  - ✅ BKS validation results
  - ✅ Export operations

#### Log Files Location:

- **Main log**: `logs/vrp_ga_YYYYMMDD_HHMMSS.log`
- **Batch log**: Tự động log trong batch mode
- **Optimization log**: Tự động log trong optimization process

#### Ví dụ log output:

```
2025-11-04 01:30:00 - vrp_ga - INFO - ============================================================
2025-11-04 01:30:00 - vrp_ga - INFO - VRP-GA System Starting
2025-11-04 01:30:00 - vrp_ga - INFO - ============================================================
2025-11-04 01:30:01 - vrp_ga.dataset - INFO - Loading JSON dataset: C101
2025-11-04 01:30:02 - vrp_ga.optimization - INFO - Starting optimization for: C101
2025-11-04 01:30:02 - vrp_ga.optimization - INFO - GA Configuration: generations=500, population=100
2025-11-04 01:30:05 - vrp_ga.optimization - INFO - Running GA for 500 generations...
2025-11-04 01:35:00 - vrp_ga.optimization - INFO - GA completed in 295.23 seconds
2025-11-04 01:35:01 - vrp_ga.optimization - INFO - BKS Validation - Instance: C101, Gap: 2.45%, Quality: GOOD
```

**Note**: Logging hoạt động tự động, không cần cấu hình thêm!

### 8. Error Handling

Custom exceptions cho error tracking tốt hơn và **TỰ ĐỘNG** được sử dụng:

- **Custom Exceptions**:

  - `CapacityViolationError`: Khi vượt quá capacity
  - `TimeWindowViolationError`: Khi vi phạm time window
  - `DistanceCalculationError`: Khi tính khoảng cách lỗi
  - `DatasetNotFoundError`: Khi không tìm thấy dataset (tự động raise khi load dataset fail)
  - `InvalidConfigurationError`: Khi config không hợp lệ (tự động validate trước khi chạy GA)
  - `DecodingError`: Khi decode chromosome fails
  - `InfeasibleSolutionError`: Khi solution không feasible

- **Automatic Error Handling**:
  - ✅ Validate GA config trước khi chạy
  - ✅ Raise specific exceptions với thông tin chi tiết
  - ✅ Log errors với stack trace vào log file
  - ✅ Hiển thị error messages rõ ràng trong console

#### Ví dụ error handling:

```bash
# Nếu dataset không tồn tại
Error: Dataset not found: 'invalid_dataset' (type: solomon) | Details: {'dataset_name': 'invalid_dataset', 'dataset_type': 'solomon'}

# Nếu config không hợp lệ
Error: Invalid GA configuration: GA_CONFIG: 'population_size' must be an integer >= 10.

# Tất cả errors được log vào log file với stack trace đầy đủ
```

### 9. Batch Processing cho Solomon Datasets

- Chạy tất cả Solomon datasets cùng lúc: `--solomon-batch`
- Tạo file tổng hợp: `solomon_summary_YYYYMMDD_HHMMSS.csv`
- So sánh hiệu suất trên nhiều test cases
- **Tự động so sánh với BKS** cho mỗi instance:
  - ✅ Hiển thị BKS gap trong console output
  - ✅ Thêm BKS data vào summary CSV
  - ✅ Hiển thị BKS statistics cuối batch (average gap, quality distribution)

#### Ví dụ batch output:

```bash
$ python main.py --solomon-batch --generations 100 --population 50

Running dataset 1/56: C101
Completed: Distance=828.94 (BKS: 828.94, Gap: 0.00%), Routes=10 (BKS: 10), Quality: EXCELLENT, Utilization=95.2%

Running dataset 2/56: C102
Completed: Distance=828.94 (BKS: 828.94, Gap: 0.00%), Routes=10 (BKS: 10), Quality: EXCELLENT, Utilization=94.8%

...

Batch Summary:
  Total datasets processed: 56
  Average distance: 987.65
  Average routes: 12.5
  Average utilization: 92.3%
  BKS Comparison (for 56 instances with BKS):
    Average gap from BKS: 2.45%
    Quality distribution:
      EXCELLENT: 15
      GOOD: 25
      ACCEPTABLE: 12
      POOR: 4
```

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

# 3. Chạy map Hà Nội với dataset có sẵn (sử dụng mặc định từ config.py)
python main.py --mockup-dataset small_random
# Mặc định: generations=1000, population=100 (theo luận văn - Table 2.18)

# 3b. Hoặc generate data mới và chạy (sử dụng mặc định)
python main.py --generate --customers 50 --capacity 200
# Mặc định: generations=1000, population=100

# 4. Chạy Solomon với phí giao hàng (mặc định: 1000 generations, 100 population)
python main.py --solomon-dataset C101
# Mặc định: generations=1000, population=100 (theo luận văn - Table 2.18)

# Lưu ý: Nếu muốn test nhanh, có thể override:
# python main.py --mockup-dataset small_random --generations 100 --population 30

# 5. Chạy batch tất cả Solomon datasets (sử dụng mặc định)
python main.py --solomon-batch
# Mặc định: generations=1000, population=100 (theo luận văn)

# Lưu ý: Nếu muốn test nhanh, có thể override:
# python main.py --solomon-batch --generations 100 --population 50

# 6. Xem kết quả trong thư mục results/
# - evolution_data_*.csv: Quá trình tiến hóa GA
# - optimal_routes_*.txt: Lộ trình + phí giao hàng
# - kpi_comparison_*.csv: So sánh GA vs NN + phí giao hàng
# - solomon_summary_*.csv: Tổng hợp Solomon datasets (với BKS gap nếu có)
# - *.html: Map Hà Nội tương tác
# - *.png: Visualization truyền thống

# 7. Xem log files trong thư mục logs/
# - vrp_ga_YYYYMMDD_HHMMSS.log: Log chi tiết quá trình chạy
#   * Tự động tạo khi chạy
#   * Ghi lại tất cả events: dataset loading, GA execution, BKS validation, errors
```

Chúc bạn sử dụng thành công hệ thống VRP-GA với map Hà Nội và tính phí giao hàng thực tế! 🗺️🚚💰
