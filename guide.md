# Đề án: Phân tích Cây Quyết định trên Nhiều Bộ Dữ liệu

## Giai đoạn 1: Chuẩn bị và Tiền xử lý Dữ liệu (Mục 2.1)

### Chuẩn bị file
- Đảm bảo các file dữ liệu đã có trong thư mục `data/`:
  - `heart_disease.csv` https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
  
  - `palmer_penguins.csv` https://github.com/allisonhorst/palmerpenguins
  
  - `breast_cancer.csv` https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
  
  **NÊN THAM KHẢO THÊM CÁC LINK KHÁC ĐỂ PHÙ HỢP LINK TRÊN NÀY LÀ GỢI Í THÔI"**

### Làm việc trên Notebook (01_heart_disease_analysis.ipynb)
1. **Tải và Tiền xử lý**:
   - Đọc file `../data/heart_disease.csv`
   - Tiền xử lý:
     - Chuyển cột 'num' thành nhãn nhị phân
     - Xử lý categorical bằng One-Hot Encoding
     - Xử lý giá trị thiếu (NaN)

2. **Chia Dữ liệu**:
   - Tách features (X) và labels (y)
   - Dùng `train_test_split` tạo 4 tỷ lệ:
     - 40/60
     - 60/40
     - 80/20
     - 90/10
   - Lưu ý:
     - Shuffle dữ liệu
     - Chia phân tầng (stratified)

3. **Trực quan hóa**:
   - Vẽ biểu đồ phân phối lớp cho:
     - Dữ liệu gốc
     - Các tập train/test
   - Lưu ảnh vào `outputs/images/heart_disease/`

## Giai đoạn 2: Xây dựng và Đánh giá Mô hình (Mục 2.2 & 2.3)

1. **Xây dựng Cây Quyết định**:
   - Huấn luyện `DecisionTreeClassifier` với:
     - `criterion='entropy'` (information gain)
   - Trực quan hóa cây bằng Graphviz
   - Lưu ảnh vào `outputs/images/heart_disease/`

2. **Đánh giá Mô hình**:
   - Dự đoán trên tập test
   - Tạo `classification_report` và `confusion_matrix`
   - Ghi kết quả và nhận xét vào báo cáo .docx

## Giai đoạn 3: Phân tích Độ sâu Cây (Mục 2.4)

*(Chỉ thực hiện với tỷ lệ 80/20)*

1. **Thử nghiệm max_depth**:
   - Các giá trị: None, 2, 3, 4, 5, 6, 7

2. **Báo cáo**:
   - Trực quan hóa từng cây
   - Bảng `accuracy_score` theo max_depth
   - Biểu đồ độ chính xác ~ max_depth
   - Nhận xét vào báo cáo

## Giai đoạn 4: Lặp lại và Tổng hợp (Mục 2.5 & 3.1)

1. **Lặp lại**:
   - Thực hiện tương tự cho:
     - `02_penguins_analysis.ipynb`
     - `03_breast_cancer_analysis.ipynb`

2. **Viết báo cáo**:
   - So sánh kết quả 3 bộ dữ liệu
   - Phân tích ảnh hưởng của:
     - Số lớp
     - Số đặc trưng
     - Kích thước mẫu
   - Đầy đủ các mục:
     - Thông tin nhóm
     - Phân công công việc
     - Tự đánh giá
     - Tài liệu tham khảo
   - Xuất file PDF

