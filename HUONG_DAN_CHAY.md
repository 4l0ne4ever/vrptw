# HƯỚNG DẪN CHẠY ỨNG DỤNG VRP-GA

## Mục Lục

1. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
2. [Cài Đặt](#cài-đặt)
3. [Chạy Ứng Dụng](#chạy-ứng-dụng)
4. [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
5. [Test Datasets](#test-datasets)
6. [Troubleshooting](#troubleshooting)

## Yêu Cầu Hệ Thống

### Phần Mềm Cần Thiết

- **Python**: 3.8 trở lên
- **pip**: Package manager cho Python
- **Virtual Environment**: Khuyến nghị sử dụng venv

### Hệ Điều Hành

- macOS (đã test trên macOS)
- Linux
- Windows (có thể cần điều chỉnh một số lệnh)

## Cài Đặt

### Bước 1: Navigate đến Project

```bash
cd "/Users/duongcongthuyet/Downloads/workspace/AI /optimize"
```

### Bước 2: Tạo Virtual Environment (nếu chưa có)

```bash
python3 -m venv venv
```

### Bước 3: Kích Hoạt Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### Bước 4: Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### Bước 5: Khởi Tạo Database (Tùy Chọn)

```bash
source venv/bin/activate
python app/database/init_db.py
```

**Lưu ý:** Database sẽ được tạo tự động khi chạy app lần đầu nếu chưa có.

## Chạy Ứng Dụng

### Cách 1: Quick Start Script (Dễ Nhất)

```bash
# Chạy script tự động
./start_app.sh
```

Script này sẽ:
- Tạo virtual environment nếu chưa có
- Cài đặt dependencies nếu chưa có
- Khởi tạo database nếu chưa có
- Chạy ứng dụng

### Cách 2: Chạy Main App Thủ Công

```bash
# Đảm bảo đang trong virtual environment
source venv/bin/activate

# Chạy ứng dụng
streamlit run app/streamlit_app.py
```

Ứng dụng sẽ tự động mở trong browser tại: `http://localhost:8501`

**Lưu ý:** 
- Nếu browser không tự mở, copy URL và paste vào browser
- Để dừng app, nhấn `Ctrl+C` trong terminal

### Cách 3: Chạy Trực Tiếp Từ Page

**Hanoi Mode:**
```bash
source venv/bin/activate
streamlit run app/pages/hanoi_mode.py
```

### Cách 4: Chạy Với Custom Port

```bash
source venv/bin/activate
streamlit run app/streamlit_app.py --server.port 8502
```

## Hướng Dẫn Sử Dụng

### Trang Chủ (Home)

Khi mở ứng dụng, bạn sẽ thấy trang chủ với:
- Welcome message
- Feature highlights
- Navigation buttons đến các pages

### Hanoi Mode - Tối Ưu Hóa Giao Hàng Hà Nội

#### Bước 1: Load Dataset

Có 4 cách để load dataset:

**1. Upload File:**
- Click tab "Upload File"
- Chọn file JSON, CSV, hoặc Excel từ `data/test_datasets/`
- File sẽ được parse và validate tự động
- Xem preview của dataset

**2. Manual Entry:**
- Click tab "Manual Entry"
- Nhập thông tin depot (tọa độ, capacity)
- Thêm customers một cách thủ công
- Xem map preview real-time
- Save dataset nếu muốn

**3. Load Saved:**
- Click tab "Load Saved"
- Chọn dataset từ danh sách đã lưu
- Dataset sẽ được load vào session

**4. Sample Data:**
- Click tab "Sample Data"
- Click "Generate Sample Dataset"
- Dataset mẫu sẽ được tạo tự động

#### Bước 2: Cấu Hình Parameters

Sau khi dataset được load:

1. **Chọn Preset:**
   - **Fast**: Population nhỏ, generations ít (nhanh, cho testing)
   - **Balanced**: Cân bằng (khuyến nghị cho production)
   - **Best Quality**: Population lớn, generations nhiều (chất lượng cao, chậm)
   - **Custom**: Tự điều chỉnh từng parameter

2. **Basic Parameters:**
   - **Population Size**: Kích thước quần thể (10-500)
   - **Generations**: Số thế hệ (10-5000)
   - **Number of Vehicles**: Số lượng xe

3. **Advanced Parameters** (trong collapsible section):
   - **Crossover Probability**: Xác suất lai ghép (0.5-1.0)
   - **Mutation Probability**: Xác suất đột biến (0.0-0.5)
   - **Tournament Size**: Kích thước tournament (2-10)
   - **Elitism Rate**: Tỷ lệ elitism (0.0-0.5)
   - **Use Split Algorithm**: Bật/tắt Split Algorithm (tối ưu route splitting)
   - **Penalty Weight**: Trọng số penalty cho constraint violations

4. **Runtime Estimation**: Xem ước tính thời gian chạy dựa trên parameters

#### Bước 3: Chạy Optimization

1. Click button **"Run Optimization"**
2. Chờ optimization hoàn thành (có thể mất vài giây đến vài phút tùy dataset và parameters)
3. Kết quả sẽ hiển thị tự động sau khi hoàn thành

**Lưu ý:**
- Có thể click **"Stop Optimization"** để dừng sớm
- UI sẽ bị block trong khi optimization đang chạy (đây là hạn chế của Streamlit)
- Progress sẽ được hiển thị sau khi hoàn thành

#### Bước 4: Xem Kết Quả

Sau khi optimization hoàn thành, bạn sẽ thấy:

**1. Key Metrics:**
- Total Distance: Tổng khoảng cách
- Number of Routes: Số lượng routes
- Fitness: Giá trị fitness
- Valid: Giải pháp có hợp lệ không

**2. Evolution Chart:**
- Biểu đồ tiến hóa qua các generations
- Best Distance line (màu xanh)
- Average Fitness line (màu xanh lá, nếu có)
- Có thể zoom và pan

**3. Visualization:**
- **Folium Map**: Interactive map với routes
  - Depot marker (màu đỏ, icon warehouse)
  - Customer markers với numbering (màu theo route)
  - Route polylines với different colors
  - Route visibility controls (checkboxes để hide/show routes)
  - Interactive popups với thông tin chi tiết
  - Download button để tải map HTML

**4. Solution Metrics:**
- **Basic Metrics**: Distance, routes, customers
- **Quality Metrics** (collapsible):
  - Capacity Utilization với progress bar
  - Load Balance Index với progress bar
  - Efficiency Score với progress bar
  - Feasibility status (✓ hoặc ✗)
- **Comparison Metrics** (collapsible):
  - GA vs Nearest Neighbor comparison
  - Improvement percentages
  - Visual indicators (success/warning/info)
- **Detailed Statistics** (collapsible):
  - GA statistics (generations, evaluations, execution time)
  - Solution quality (fitness, penalty, diversity)

**5. Comparison Chart:**
- Bar chart so sánh GA vs Nearest Neighbor
- Improvement percentages trên bars
- Color coding (green=GA tốt hơn, red=GA tệ hơn)

### History - Lịch Sử Chạy

- Xem tất cả optimization runs đã lưu
- Filter và search
- Xem chi tiết từng run
- Download results

### Datasets - Quản Lý Datasets

- Xem tất cả datasets đã lưu
- Delete datasets
- View dataset details
- Load dataset để sử dụng

### Help - Trợ Giúp

- Documentation
- FAQs
- Contact information

## Test Datasets

### Location

Tất cả test datasets nằm trong: `data/test_datasets/`

### Recommended Test Files

**For Quick Test (5 customers, < 5 seconds):**
```
JSON: data/test_datasets/hanoi_small_5_customers.json
CSV:  data/test_datasets/hanoi_small_5_customers.csv
Excel: data/test_datasets/hanoi_small_5_customers.xlsx
```

**For Standard Test (10 customers, < 30 seconds):**
```
JSON: data/test_datasets/hanoi_medium_10_customers.json
CSV:  data/test_datasets/hanoi_medium_10_customers.csv
Excel: data/test_datasets/hanoi_medium_10_customers.xlsx
```

**For Visualization Test (15 customers, clustered):**
```
JSON: data/test_datasets/hanoi_clustered_15_customers.json
Expected: Clear route visualization, multiple routes
```

**For File Upload Test:**
```
CSV:  data/test_datasets/hanoi_full_columns_10_customers.csv
Excel: data/test_datasets/hanoi_full_columns_10_customers.xlsx
```

### Test Workflow

**1. Quick Functionality Test:**
```
1. Upload: hanoi_small_5_customers.json
2. Preset: Fast
3. Click: Run Optimization
4. Expected: < 5 seconds, 1-2 routes
5. Check: Map displays correctly
```

**2. Standard Test:**
```
1. Upload: hanoi_medium_10_customers.json
2. Preset: Balanced
3. Click: Run Optimization
4. Expected: < 30 seconds, 2-3 routes
5. Check: All visualizations work
```

**3. Visualization Test:**
```
1. Upload: hanoi_clustered_15_customers.json
2. Preset: Balanced
3. Click: Run Optimization
4. Expected: Clear route visualization, multiple routes
5. Check: Route visibility controls work
6. Check: Map download works
```

**4. File Upload Test:**
```
1. Upload: hanoi_full_columns_10_customers.csv
2. Check: File parsed correctly
3. Check: Problem created successfully
4. Run optimization
5. Check: Results displayed correctly
```

## Troubleshooting

### Lỗi: ModuleNotFoundError

**Nguyên nhân:** Dependencies chưa được cài đặt

**Giải pháp:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Lỗi: Database Error

**Nguyên nhân:** Database chưa được khởi tạo

**Giải pháp:**
```bash
source venv/bin/activate
python app/database/init_db.py
```

Hoặc database sẽ được tạo tự động khi chạy app lần đầu.

### Lỗi: Port Already in Use

**Nguyên nhân:** Port 8501 đã được sử dụng

**Giải pháp 1: Sử dụng port khác**
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

**Giải pháp 2: Kill process đang dùng port**
```bash
# macOS/Linux
lsof -ti:8501 | xargs kill -9

# Hoặc tìm và kill manually
lsof -i:8501
```

### Lỗi: File Upload Failed

**Nguyên nhân:** 
- File format không đúng
- File quá lớn (> 10MB)
- File bị corrupt
- Missing required columns

**Giải pháp:**
- Kiểm tra file format (JSON, CSV, Excel)
- Kiểm tra file size (< 10MB)
- Sử dụng test datasets có sẵn
- Kiểm tra file có đầy đủ columns: id, x, y, demand

### Lỗi: Optimization Failed

**Nguyên nhân:**
- Dataset không hợp lệ
- Parameters không hợp lệ
- Memory issues
- Invalid coordinates

**Giải pháp:**
- Kiểm tra dataset structure
- Sử dụng parameters mặc định (Balanced preset)
- Giảm population size hoặc generations
- Sử dụng dataset nhỏ hơn
- Kiểm tra coordinates trong Hanoi bounds

### Lỗi: Map Not Displaying

**Nguyên nhân:**
- streamlit-folium chưa được cài đặt
- Network issues (loading map tiles)
- Invalid coordinates

**Giải pháp:**
```bash
pip install streamlit-folium
```

Kiểm tra internet connection (cần để load map tiles).

### Performance Issues

**Nếu optimization chậm:**
- Giảm population size (50-100)
- Giảm generations (100-500)
- Sử dụng "Fast" preset
- Sử dụng dataset nhỏ hơn (5-10 customers)

**Nếu map rendering chậm:**
- Giảm số lượng customers
- Sử dụng dataset nhỏ hơn
- Tắt route visibility cho routes không cần thiết

## Tips & Best Practices

### 1. Dataset Selection

- **Bắt đầu nhỏ**: Dùng dataset 5-10 customers để test
- **Tăng dần**: Sau khi hiểu workflow, tăng lên 20-50 customers
- **Real-world**: Sử dụng datasets thực tế cho production

### 2. Parameter Tuning

- **Fast Preset**: Cho testing nhanh (population=50, generations=500)
- **Balanced Preset**: Cho production (population=100, generations=1000) - **Khuyến nghị**
- **Best Quality Preset**: Cho kết quả tốt nhất (population=150, generations=1500) - chậm hơn
- **Custom**: Khi bạn hiểu rõ về GA parameters

### 3. Visualization

- **Route Visibility**: Tắt routes không cần để map rõ ràng hơn (khi có nhiều routes)
- **Download Map**: Tải map HTML để xem offline hoặc share
- **Comparison Chart**: So sánh với Nearest Neighbor để đánh giá chất lượng

### 4. Metrics Interpretation

- **Total Distance**: Càng thấp càng tốt
- **Number of Routes**: Cân bằng giữa số routes và distance
- **Capacity Utilization**: 70-90% là tốt (không quá thấp, không quá cao)
- **Load Balance**: Càng cao càng tốt (gần 1.0 = routes có load tương đương)
- **Efficiency Score**: Càng cao càng tốt (0-1 scale)
- **Feasibility**: Phải là "Yes" (solution hợp lệ)

### 5. Saving Results

- Save optimization runs để xem lại sau
- Save datasets để tái sử dụng
- Download results để báo cáo

## Quick Reference

### Commands

```bash
# Activate venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database (optional)
python app/database/init_db.py

# Run app
streamlit run app/streamlit_app.py

# Run specific page
streamlit run app/pages/hanoi_mode.py

# Quick start (all-in-one)
./start_app.sh
```

### File Locations

- **Main App**: `app/streamlit_app.py`
- **Pages**: `app/pages/`
- **Test Datasets**: `data/test_datasets/`
- **Database**: `data/database/vrp_app.db`
- **Logs**: `logs/`

### Key Features

- ✅ File upload (JSON, CSV, Excel)
- ✅ Manual data entry với map preview
- ✅ Sample data generation
- ✅ Parameter configuration với presets
- ✅ Real-time progress tracking
- ✅ Folium map visualization
- ✅ Evolution charts
- ✅ Comparison charts
- ✅ Metrics display với progress bars
- ✅ Download functionality (map HTML)

## Example Workflow

### Complete Example: Quick Test

```bash
# 1. Start app
source venv/bin/activate
streamlit run app/streamlit_app.py

# 2. Trong browser:
#    - Click "Hanoi Mode" hoặc navigate đến page
#    - Click tab "Upload File"
#    - Chọn: data/test_datasets/hanoi_small_5_customers.json
#    - Xem preview
#    - Chọn preset "Fast"
#    - Click "Run Optimization"
#    - Chờ kết quả (< 5 seconds)
#    - Xem map, charts, và metrics
```

### Complete Example: Standard Test

```bash
# 1. Start app
source venv/bin/activate
streamlit run app/streamlit_app.py

# 2. Trong browser:
#    - Navigate đến "Hanoi Mode"
#    - Click tab "Sample Data"
#    - Click "Generate Sample Dataset"
#    - Chọn preset "Balanced"
#    - Click "Run Optimization"
#    - Chờ kết quả (< 30 seconds)
#    - Explore visualizations
#    - Toggle route visibility
#    - Download map
#    - Xem metrics và comparison
```

## Support

Nếu gặp vấn đề:

1. Kiểm tra Troubleshooting section ở trên
2. Xem logs trong `logs/` directory
3. Kiểm tra file format và structure
4. Thử với test datasets có sẵn
5. Kiểm tra Python version (>= 3.8)

## Next Steps

Sau khi test thành công:

1. **Phase 5**: Save, Export, và History features (đang phát triển)
2. **Production**: Deploy ứng dụng
3. **Customization**: Điều chỉnh theo nhu cầu cụ thể

---

**Chúc bạn sử dụng ứng dụng thành công!**

