# Poker Hand Prediction

## 1. Giới thiệu dự án (Project Overview)
Dự án **Poker Hand Prediction** giải quyết bài toán phân loại đa lớp (Multi-class Classification) dựa trên bộ dữ liệu Poker Hand. Mục tiêu của bài toán là dự đoán phân loại tay bài Poker (từ 0 đến 9) dựa trên 5 lá bài được rút ra từ một bộ bài tiêu chuẩn 52 lá.

## 2. Thông tin Dữ liệu (Data Description)
Mỗi mẫu dữ liệu (sample) đại diện cho một tay bài gồm 5 lá bài.
- **RAW Data (Dữ liệu thô):** Bao gồm 10 đặc trưng (features) đại diện cho 5 lá bài. Mỗi lá bài được mô tả bởi 2 đặc trưng: Chất (Suit - ký hiệu S1 đến S5, giá trị từ 1-4) và Thứ hạng (Rank/Card - ký hiệu C1 đến C5, giá trị từ 1-13).
- **FE Data (Dữ liệu đã Feature Engineering):** Dữ liệu được bổ sung thêm các đặc trưng thủ công (hand-crafted features) được tính toán từ các lá bài như: số lượng rank duy nhất (unique ranks), rank xuất hiện nhiều lần nhất (max same rank), đếm số lượng cặp (count pair), v.v. để hỗ trợ quá trình phân loại của các mô hình học máy truyền thống.
- **Tập Train:** 25,010 mẫu.
- **Tập Test:** 1,000,000 mẫu.

**Phân tích độ quan trọng của đặc trưng (Feature Importance):**
![alt text](screenshots%20of%20benchmark/image-1.png)
> **Phân tích:** Trên dữ liệu RAW, các đặc trưng về thứ hạng của lá bài (C1 đến C5) chiếm độ quan trọng cao nhất (mỗi thuộc tính khoảng 13-14%). Chất của lá bài (S1 đến S5) ít quan trọng hơn (khoảng 6-7%).

![alt text](screenshots%20of%20benchmark/image-2.png)
> **Phân tích:** Sau khi thực hiện Feature Engineering (FE), các đặc trưng mới tạo ra (như `unique_rank_count`, `max_same_rank`) chi phối quyết định của mô hình, biến các tổ hợp bài phức tạp thành các con số phân tách cơ bản.

## 3. Phân phối dữ liệu (Data Imbalance)
Đặc điểm của bộ dữ liệu này là **sự mất cân bằng dữ liệu nghiêm trọng (Highly Imbalanced Data)**.

![alt text](screenshots%20of%20benchmark/image-3.png)
> **Phân tích phân phối:** Dữ liệu phân phối theo dạng hàm mũ, phản ánh xác suất thực tế của trò chơi Poker:
> - **Class 0 (Nothing in hand)** có 12,493 mẫu và **Class 1 (One pair)** có 10,599 mẫu, chiếm hơn 90% tổng số mẫu trong tập Train.
> - Các class cấp trung như **Class 2 (Two pairs)** (1,206 mẫu) hay **Class 3 (Three of a kind)** (513 mẫu) có số lượng giảm dần.
> - Các class cấp cao như **Class 8 (Straight flush)** hay **Class 9 (Royal flush)** có 5 mẫu cho mỗi class.
> Sự phân phối này gây khó khăn trong phân loại. Các mô hình học máy thông thường có xu hướng bỏ qua các tay bài hiếm (Class 4-9) và chỉ tập trung dự đoán Class 0, 1 để tối ưu hóa độ chính xác tổng thể (Accuracy).

## 4. Hiệu suất của các mô hình (Model Performance)
Các mô hình truyền thống đã được tinh chỉnh siêu tham số (Hyperparameter Tuning) sử dụng **GridSearchCV** kết hợp **Cross-Validation**. 

### 4.1. AdaBoost
- **RAW Data (Runtime: ~48s)**
  - Test Accuracy: `51.12%` (Incorrect Predictions: `488,800 / 1,000,000`)
  ```text
  --- Classification Report ---
                precision    recall  f1-score   support
             0       0.53      0.74      0.62    501209
             1       0.46      0.33      0.38    422498
             2       0.00      0.00      0.00     47622
             3       0.00      0.00      0.00     21121
             4       0.00      0.00      0.00      3885
             5       0.00      0.00      0.00      1996
             6       0.00      0.00      0.00      1424
             7       0.00      0.00      0.00       230
             8       0.00      0.00      0.00        12
             9       0.00      0.00      0.00         3
      accuracy                           0.51   1000000
     macro avg       0.10      0.11      0.10   1000000
  weighted avg       0.46      0.51      0.47   1000000
  ```
  ![alt text](screenshots%20of%20benchmark/image-4.png)
  > **Phân tích:** Trên dữ liệu thô, thuật toán AdaBoost bị ảnh hưởng nặng nề bởi sự mất cân bằng dữ liệu (skewed data). Mô hình chủ yếu dự đoán các class chiếm đa số (Class 0 và 1) để tối ưu độ chính xác chung, hoàn toàn không tự học được các quy luật (pattern) của bài Poker nên F1-score = 0 từ Class 2 trở đi.

- **FE Data (Runtime: ~48s)**
  - Test Accuracy: `99.25%` (Incorrect Predictions: `7,500 / 1,000,000`)
  ```text
  --- Classification Report ---
                precision    recall  f1-score   support
             0       0.99      1.00      0.99    501209
             1       1.00      1.00      1.00    422498
             2       1.00      1.00      1.00     47622
             3       0.93      1.00      0.96     21121
             4       0.00      0.00      0.00      3885
             5       0.00      0.00      0.00      1996
             6       0.00      0.00      0.00      1424
             7       0.00      0.00      0.00       230
             8       0.00      0.00      0.00        12
             9       0.00      0.00      0.00         3
      accuracy                           0.99   1000000
     macro avg       0.39      0.40      0.40   1000000
  weighted avg       0.99      0.99      0.99   1000000
  ```
  ![alt text](screenshots%20of%20benchmark/image-5.png)
  > **Phân tích:** Feature Engineering cung cấp các đặc trưng thủ công (như pattern tính sẵn), giúp AdaBoost phân tách tốt các class đa số, tăng Accuracy lên 99.25%. Tuy nhiên, do dữ liệu quá lệch, mô hình vẫn bó tay và không dự đoán được các tay bài cực hiếm từ Class 4 trở lên.

### 4.2. Ridge Regression
- **RAW Data (Runtime: ~45s)**
  - Test Accuracy (Rounded): `42.25%` (Incorrect Predictions: `577,500 / 1,000,000`)
  ```text
  --- Classification Report ---
                precision    recall  f1-score   support
             0       0.00      0.00      0.00    501209
             1       0.42      1.00      0.59    422498
             2       0.00      0.00      0.00     47622
             3       0.00      0.00      0.00     21121
             4       0.00      0.00      0.00      3885
             5       0.00      0.00      0.00      1996
             6       0.00      0.00      0.00      1424
             7       0.00      0.00      0.00       230
             8       0.00      0.00      0.00        12
             9       0.00      0.00      0.00         3
      accuracy                           0.42   1000000
     macro avg       0.04      0.10      0.06   1000000
  weighted avg       0.18      0.42      0.25   1000000
  ```
  ![alt text](screenshots%20of%20benchmark/image-6.png)
  > **Phân tích:** Hồi quy tuyến tính áp dụng trên dữ liệu lệch hoàn toàn không học được pattern của bài. Phần lớn dự đoán bị kéo tập trung về Class 1 (nhóm phổ biến), không phù hợp với yêu cầu phân loại rời rạc của trò chơi Poker.

- **FE Data (Runtime: ~45s)**
  - Test Accuracy (Rounded): `99.25%` (Incorrect Predictions: `7,500 / 1,000,000`)
  ```text
  --- Classification Report ---
                precision    recall  f1-score   support
             0       0.99      1.00      0.99    501209
             1       1.00      1.00      1.00    422498
             2       1.00      1.00      1.00     47622
             3       1.00      1.00      1.00     21121
             4       0.00      0.00      0.00      3885
             5       0.00      0.00      0.00      1996
             6       0.00      0.00      0.00      1424
             7       0.00      0.00      0.00       230
             8       0.00      0.00      0.00        12
             9       0.00      0.00      0.00         3
      accuracy                           0.99   1000000
     macro avg       0.40      0.40      0.40   1000000
  weighted avg       0.99      0.99      0.99   1000000
  ```
  ![alt text](screenshots%20of%20benchmark/image-7.png)
  > **Phân tích:** Dưới sự hỗ trợ của FE (mớm sẵn pattern), hồi quy tuyến tính phân loại được các class phổ biến. Tuy nhiên, sự mất cân bằng dữ liệu trầm trọng tiếp tục khiến mô hình thất bại trên các nhóm bài cao cấp (Class 4 trở lên).

### 4.3. Random Forest (Tuned)
- **RAW Data (Runtime: ~2 phút)**
  - Test Accuracy: `60.85%` (Incorrect Predictions: `~391,500 / 1,000,000`)
  ```text
  --- Classification Report ---
                precision    recall  f1-score   support
             0       0.63      0.82      0.71    501209
             1       0.57      0.47      0.52    422498
             2       0.36      0.00      0.00     47622
             3       0.56      0.00      0.00     21121
             4       0.22      0.00      0.00      3885
             5       0.97      0.04      0.07      1996
             6       0.00      0.00      0.00      1424
             7       0.00      0.00      0.00       230
             8       0.00      0.00      0.00        12
             9       0.00      0.00      0.00         3
      accuracy                           0.61   1000000
     macro avg       0.33      0.13      0.13   1000000
  weighted avg       0.59      0.61      0.57   1000000
  ```
  ![alt text](screenshots%20of%20benchmark/image-8.png)
  > **Phân tích:** Nhờ khả năng phân nhánh, Random Forest nhận diện được một phần Class 5 trên dữ liệu thô. Dù vậy, nó vẫn là mô hình ML truyền thống yếu thế trước dữ liệu skewed; việc thiếu hụt mẫu ở các class hiếm khiến mô hình không đủ dữ kiện để tự tìm ra pattern và hình thành quy tắc phân nhánh.

- **FE Data (Runtime: ~2 phút)**
  - Test Accuracy: `99.41%` (Incorrect Predictions: `~5,900 / 1,000,000`)
  ```text
  --- Classification Report ---
                precision    recall  f1-score   support
             0       0.99      1.00      0.99    501209
             1       1.00      1.00      1.00    422498
             2       1.00      1.00      1.00     47622
             3       1.00      1.00      1.00     21121
             4       0.00      0.00      0.00      3885
             5       1.00      0.01      0.02      1996
             6       1.00      1.00      1.00      1424
             7       1.00      1.00      1.00       230
             8       0.00      0.00      0.00        12
             9       0.00      0.00      0.00         3
      accuracy                           0.99   1000000
     macro avg       0.70      0.60      0.60   1000000
  weighted avg       0.99      0.99      0.99   1000000
  ```
  ![alt text](screenshots%20of%20benchmark/image-9.png)
  > **Phân tích:** Khi có FE, Random Forest học được pattern và phân loại đúng nhiều class hơn (F1-Score đạt 1.00 ở Class 6, 7). Dù vậy, với các class bị lệch cực đoan như Class 8 (12 mẫu) hay Class 9 (3 mẫu), mô hình vẫn hoàn toàn không thể nhận diện.

### 4.4. Multi-Layer Perceptron (MLP)
- **RAW Data (Runtime: ~5 phút)**
  - Test Accuracy: `99.71%` (Incorrect Predictions: `~2,900 / 1,000,000`)
  ```text
  --- Classification Report ---
                precision    recall  f1-score   support
             0       1.00      1.00      1.00    501209
             1       1.00      1.00      1.00    422498
             2       1.00      0.99      1.00     47622
             3       0.99      0.99      0.99     21121
             4       0.89      0.60      0.72      3885
             5       0.99      0.93      0.96      1996
             6       0.84      0.90      0.87      1424
             7       0.99      0.40      0.57       230
             8       0.14      0.33      0.20        12
             9       0.01      0.33      0.02         3
      accuracy                           1.00   1000000
     macro avg       0.79      0.75      0.73   1000000
  weighted avg       1.00      1.00      1.00   1000000
  ```
  ![alt text](screenshots%20of%20benchmark/image-12.png)
  > **Phân tích:** Khác với ML truyền thống, MLP thể hiện khả năng vượt trội trên dữ liệu skewed. Nó tự trích xuất và học được các "pattern" bản chất trực tiếp từ dữ liệu thô. Dù Class 8 và 9 chỉ có vài mẫu, mô hình vẫn nhận diện được (F1-score > 0), minh chứng cho việc Deep Learning có thể học logic quy luật thay vì phụ thuộc hoàn toàn vào số lượng mẫu phân bổ đồng đều.

- **FE Data (Runtime: ~5 phút)**
  - Test Accuracy: `99.65%` (Incorrect Predictions: `~3,500 / 1,000,000`)
  ```text
  --- Classification Report ---
                precision    recall  f1-score   support
             0       0.99      1.00      1.00    501209
             1       1.00      1.00      1.00    422498
             2       1.00      1.00      1.00     47622
             3       1.00      1.00      1.00     21121
             4       0.64      0.29      0.40      3885
             5       0.99      0.97      0.98      1996
             6       1.00      1.00      1.00      1424
             7       1.00      0.92      0.96       230
             8       0.04      0.17      0.07        12
             9       0.04      0.33      0.07         3
      accuracy                           1.00   1000000
     macro avg       0.77      0.77      0.75   1000000
  weighted avg       1.00      1.00      1.00   1000000
  ```
  ![alt text](screenshots%20of%20benchmark/image-13.png)
  > **Phân tích:** Việc dùng FE lại làm giảm độ chính xác chung của MLP. Khi nhào nặn các lá bài thành con số đếm, ta vô tình làm mất đi tín hiệu thô (raw signals) giữa các lá bài, khiến mạng nơ-ron bị hạn chế khả năng tự tìm ra pattern phức tạp. Do đó, khả năng nhận diện các mẫu thiểu số giảm sút so với việc học từ dữ liệu RAW.

### 4.5. Transformer Small
- **RAW Data (Runtime: 55.38s)**
  - Test Accuracy: `88.63%` (Incorrect Predictions: `114,000 / 1,000,000`)
  ```text
  --- Classification Report ---
                precision    recall  f1-score   support
             0       0.90      0.97      0.93    501209
             1       0.91      0.87      0.89    422498
             2       0.63      0.60      0.61     47622
             3       0.33      0.03      0.06     21121
             4       0.30      0.01      0.02      3885
             5       0.33      0.00      0.00      1996
             6       0.61      0.18      0.28      1424
             7       0.00      0.00      0.00       230
             8       0.00      0.00      0.00        12
             9       0.00      0.00      0.00         3
      accuracy                           0.89   1000000
     macro avg       0.40      0.27      0.28   1000000
  weighted avg       0.87      0.89      0.87   1000000
  ROC-AUC Score (Macro): 0.8378
  ```
  ![alt text](screenshots%20of%20benchmark/image-14.png)
  > **Phân tích:** Ở cấu hình tham số nhỏ, Transformer chưa đủ năng lực (capacity) để học toàn bộ pattern phức tạp trên dữ liệu lệch, dẫn đến dự đoán sai ở các class ít mẫu. Tuy nhiên, ROC-AUC cao cho thấy nó đang bắt đầu hiểu phân phối chung.

- **FE Data (Runtime: 118.44s)**
  - Test Accuracy: `99.62%` (Incorrect Predictions: `3,793 / 1,000,000`)
  ```text
  --- Classification Report ---
                precision    recall  f1-score   support
             0       0.99      1.00      1.00    501209
             1       1.00      1.00      1.00    422498
             2       1.00      1.00      1.00     47622
             3       1.00      1.00      1.00     21121
             4       0.58      0.20      0.30      3885
             5       0.98      0.98      0.98      1996
             6       1.00      1.00      1.00      1424
             7       1.00      0.74      0.85       230
             8       0.00      0.00      0.00        12
             9       0.20      1.00      0.33         3
      accuracy                           1.00   1000000
     macro avg       0.78      0.79      0.75   1000000
  weighted avg       1.00      1.00      1.00   1000000
  ROC-AUC Score (Macro): 0.9882
  ```
  ![alt text](screenshots%20of%20benchmark/image-15.png)
  > **Phân tích:** Với các biến đếm từ FE, Transformer Small được mớm sẵn các pattern tĩnh, bù đắp cho sự thiếu hụt năng lực để phân tách tốt hơn các class trung gian như Full house hay Four of a kind.

### 4.6. Transformer Medium 
- **RAW Data (Runtime: 416.33s)**
  - Test Accuracy: `99.98%` (Incorrect Predictions: `239 / 1,000,000`)
  ```text
  --- Classification Report ---
                precision    recall  f1-score   support
             0       1.00      1.00      1.00    501209
             1       1.00      1.00      1.00    422498
             2       1.00      1.00      1.00     47622
             3       1.00      1.00      1.00     21121
             4       1.00      0.99      1.00      3885
             5       1.00      1.00      1.00      1996
             6       1.00      1.00      1.00      1424
             7       1.00      0.14      0.25       230
             8       1.00      0.83      0.91        12
             9       0.60      1.00      0.75         3
      accuracy                           1.00   1000000
     macro avg       0.96      0.90      0.89   1000000
  weighted avg       1.00      1.00      1.00   1000000
  ROC-AUC Score (Macro): 0.9985
  ```
  ![alt text](screenshots%20of%20benchmark/image-17.png)
  > **Phân tích:** Transformer Medium xuất sắc xử lý vấn đề skewed data cực đoan, nhận diện thành công tổ hợp hiếm như Straight flush (F1: 0.91) và Royal flush (F1: 0.75) chỉ với 12 và 3 mẫu. **Tại sao Transformer lại tốt hơn MLP một chút?** Điểm mấu chốt là cơ chế **Self-Attention**. MLP dàn phẳng 5 lá bài (10 đặc trưng) thành 1 vector để tìm pattern toàn cục. Trong khi đó, Transformer dùng Self-Attention để liên tục "nhìn chéo" (pairwise tương tác) giữa từng lá bài với nhau. Việc rà soát xem các lá có cùng chất (Flush) hay liên tiếp (Straight) chính là cốt lõi của bài Poker, và Self-Attention sinh ra để làm chính xác việc này, giúp nó học pattern hoàn hảo hơn MLP.

- **FE Data (Runtime: 458.30s)**
  - Test Accuracy: `99.95%` (Incorrect Predictions: `451 / 1,000,000`)
  ```text
  --- Classification Report ---
                precision    recall  f1-score   support
             0       1.00      1.00      1.00    501209
             1       1.00      1.00      1.00    422498
             2       1.00      1.00      1.00     47622
             3       1.00      1.00      1.00     21121
             4       0.99      0.90      0.94      3885
             5       1.00      1.00      1.00      1996
             6       1.00      1.00      1.00      1424
             7       1.00      0.95      0.98       230
             8       0.70      0.58      0.64        12
             9       0.38      1.00      0.55         3
      accuracy                           1.00   1000000
     macro avg       0.91      0.94      0.91   1000000
  weighted avg       1.00      1.00      1.00   1000000
  ROC-AUC Score (Macro): 0.9997
  ```
  ![alt text](screenshots%20of%20benchmark/image-16.png)
  > **Phân tích:** Tương tự MLP, khi truyền FE vào Transformer Medium, thông tin chi tiết từng lá bài bị "nén" lại, vô tình làm yếu đi sức mạnh so sánh tương tác của cơ chế Self-Attention. Hậu quả là hiệu suất nhận diện class thiểu số giảm nhẹ so với học trực tiếp từ dữ liệu thô.

![alt text](screenshots%20of%20benchmark/image-10.png)
> **Phân tích (RAW Data):** Biểu đồ hiển thị lỗi trên tập thô chỉ ra mạng nơ-ron (MLP và Transformer) phân loại chính xác hơn các thuật toán truyền thống.

![alt text](screenshots%20of%20benchmark/image-11.png)
> **Phân tích (FE Data):** Feature Engineering giúp cân bằng chung mức sai số. Tuy nhiên, tập gốc chưa qua xử lý (RAW) khi kết hợp với cấu trúc nơ-ron đủ lớn vẫn tạo ra kết quả có sai số thấp nhất.

## 5. Tổng hợp So sánh (Comparison Tables)

### Bảng 1: Hiệu suất trên RAW Data (Dữ liệu chưa qua xử lý)
| Model | Test Accuracy (%) | Incorrect Predictions | Macro F1 | ROC-AUC | Runtime |
| --- | :---: | :---: | :---: | :---: | :---: |
| Ridge Regression | 42.25% | 577,500 | 0.04 | N/A | ~45s |
| AdaBoost | 51.12% | 488,800 | 0.10 | N/A | ~48s |
| Random Forest | 60.85% | 391,500 | 0.13 | N/A | ~2m |
| Transformer Small | 88.63% | 114,000 | 0.28 | 0.8378 | 55.38s |
| **MLP** | **99.71%** | **2,900** | **0.73** | **N/A** | **~5m** |
| **Transformer Medium** | **99.98%** | **239** | **0.89** | **0.9985** | **416.33s** |

### Bảng 2: Hiệu suất trên FE Data (Dữ liệu đã qua Feature Engineering)
| Model | Test Accuracy (%) | Incorrect Predictions | Macro F1 | ROC-AUC | Runtime |
| --- | :---: | :---: | :---: | :---: | :---: |
| Ridge Regression | 99.25% | 7,500 | 0.40 | N/A | ~45s |
| AdaBoost | 99.25% | 7,500 | 0.40 | N/A | ~48s |
| Random Forest | 99.41% | 5,900 | 0.60 | N/A | ~2m |
| Transformer Small | 99.62% | 3,793 | 0.75 | 0.9882 | 118.44s |
| **MLP** | **99.65%** | **3,500** | **0.75** | **N/A** | **~5m** |
| **Transformer Medium** | **99.95%** | **451** | **0.91** | **0.9997** | **458.30s** |

**Kết luận tổng quát:** 
- **Mô hình truyền thống (AdaBoost, Ridge, Random Forest):** Hoạt động kém trên dữ liệu thô bị skewed. Chúng không tự học được "pattern" của trò chơi mà chủ yếu bị thiên lệch, chỉ đoán các class đa số. Cần có Feature Engineering (FE) mớm sẵn quy luật thì mới phân loại ổn các class trung bình, nhưng vẫn thất bại hoàn toàn trên các class cực hiếm.
- **Deep Learning (MLP, Transformer):** Thể hiện sự vượt trội khi xử lý dữ liệu skewed. Cả hai tự học được logic, "pattern" của bài Poker trực tiếp từ tập RAW data bất chấp sự chênh lệch dữ liệu cực đoan, dự đoán thành công các class chỉ có 3-12 mẫu.
- **FE vs RAW trên Deep Learning:** Trái với ML truyền thống, FE lại làm *giảm* sức mạnh của MLP và Transformer. Việc gộp đặc trưng thủ công làm mất đi chi tiết thô của từng lá bài, hạn chế khả năng tự tìm quy luật của mạng.
- **Transformer so với MLP:** Transformer nhỉnh hơn ở khả năng nhận diện xuất sắc các class khó nhờ cơ chế **Self-Attention**. Thay vì nhìn tổng thể 1 chiều như MLP, Self-Attention liên tục tự động so sánh tương tác chéo giữa từng lá bài với nhau (cùng chất, cùng số, độ liên tiếp) - điều này khớp hoàn hảo với bản chất cách xác định bộ bài Poker, giúp Transformer học pattern sâu sắc và tạo ra ranh giới quyết định chính xác nhất.
