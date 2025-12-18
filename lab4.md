# Lab 4 – Report
BÁO CÁO THỰC HÀNH LAB 4: PHÂN LOẠI VĂN BẢN (TEXT CLASSIFICATION)

Môn học: Xử lý dữ liệu lớn / NLP Nội dung: Xây dựng mô hình phân loại cảm xúc (Sentiment Analysis) sử dụng Scikit-learn và PySpark.


Cấu trúc thư mục dự án (Directory Structure)

├── data/                                # Thư mục chứa dữ liệu (Dataset)
│  
├── notebook/                            # Thư mục chứa Jupyter Notebooks (Mã nguồn chính)
│   ├── Lab4.pdf                         # Thực hành 
│   
├── report/                              # Thư mục chứa báo cáo
│   └── lap4.md                          # File báo cáo chi tiết này
│
├── src/                                 # Mã nguồn Python (Modules/Classes tái sử dụng)
│   └── Lap4
├── .gitignore                           # File cấu hình bỏ qua file rác (tmp, __pycache__)
└── README.md                            # Hướng dẫn chạy và tổng quan dự án



1. Các bước triển khai (Implementation Steps)
Quy trình thực hiện được chia thành 3 giai đoạn chính: Xây dựng Baseline với Scikit-learn, Triển khai trên PySpark, và Thử nghiệm mô hình cải tiến.

Bước 1: Xây dựng Baseline Model (Scikit-learn)

Tiền xử lý: Sử dụng TfidfVectorizer để chuyển đổi văn bản thành vector đặc trưng.



Mô hình: Xây dựng class TextClassifier bao bọc thuật toán LogisticRegression (với solver=liblinear).



Huấn luyện & Đánh giá:

Chia dữ liệu sentiments.csv thành tập train và valid.

Đổi tên cột text thành sentence và loại bỏ các giá trị null.


Huấn luyện mô hình và đo lường các chỉ số: Accuracy, Precision, Recall, F1-score .

Bước 2: Triển khai với PySpark (Big Data approach)
Khởi tạo SparkSession và load dữ liệu từ file CSV .

Xây dựng Pipeline xử lý dữ liệu gồm các bước tuần tự:


Tokenizer: Tách câu thành từ.


StopWordsRemover: Loại bỏ từ dừng (stop words).


HashingTF: Chuyển đổi từ thành vector tần suất (số lượng features = 10,000).


IDF: Tính toán trọng số Inverse Document Frequency.


LogisticRegression: Mô hình phân loại.

Bước 3: Thử nghiệm mô hình cải tiến (Improved Model)
Chuyển đổi sang thuật toán Naive Bayes (MultinomialNB) để so sánh với Logistic Regression.


Thay đổi cách trích xuất đặc trưng trong TfidfVectorizer:

Thêm stop_words="english" để loại bỏ từ nhiễu.

Sử dụng ngram_range=(1,2) để bắt các cụm 2 từ (bigrams) thay vì chỉ từ đơn.

2. Hướng dẫn chạy code (Code Execution Guide)
Môi trường: Google Colab.

Chuẩn bị dữ liệu:

Cần file sentiments.csv.

Sử dụng lệnh files.upload() để tải file lên môi trường Colab.

Thực thi:

Chạy tuần tự các cell từ trên xuống dưới.

Đoạn code Spark yêu cầu cài đặt môi trường Java/Spark (thường đã tích hợp sẵn hoặc cần cài thêm trên Colab).

Hàm main() chạy pipeline của Spark.

Hàm test_model_improvement() chạy mô hình Naive Bayes cải tiến.

3. Phân tích kết quả (Result Analysis)
Dưới đây là so sánh hiệu năng giữa mô hình cơ sở (Baseline) và mô hình thử nghiệm cải tiến.

A. Kết quả của Baseline Model (Logistic Regression)

Accuracy: ~79.02%.


F1-Score (Macro): ~0.666.

Nhận xét: Mô hình đạt độ chính xác khá tốt (gần 80%). Tuy nhiên, chỉ số Recall (0.619) thấp hơn Precision (0.785), cho thấy mô hình có xu hướng bỏ sót một số trường hợp của lớp thiểu số hoặc khó phân loại.

B. Kết quả của PySpark Model

Accuracy: 73.94%.

Nhận xét: Độ chính xác thấp hơn so với Scikit-learn (73.9% vs 79%). Nguyên nhân có thể do HashingTF gây ra va chạm (collision) khi ánh xạ từ vựng vào không gian vector cố định, làm mất mát thông tin so với TfidfVectorizer chuẩn.

C. Kết quả của Improved Model (MultinomialNB + N-grams)

Accuracy: ~70.66%.

Phân tích hiệu quả (Why it was NOT effective):

Mặc dù áp dụng N-grams và loại bỏ stop words, độ chính xác giảm từ 79% xuống 70.6%.

Lý do:

Giả định độc lập: Naive Bayes giả định các từ độc lập với nhau, điều này không hoàn toàn đúng với văn bản, đặc biệt khi dùng N-grams.


Mất cân bằng dữ liệu: Bảng báo cáo phân loại (Classification Report) cho thấy sự chênh lệch lớn về dữ liệu hỗ trợ (support): Lớp 0 có 427 mẫu, Lớp 1 có 732 mẫu.

Recall kém ở lớp 0: Mô hình nhận diện lớp 0 rất tệ (Recall chỉ đạt 0.23), trong khi lớp 1 đạt 0.98. Điều này cho thấy việc loại bỏ stop words có thể đã vô tình loại bỏ các từ phủ định quan trọng (ví dụ: "not", "no") khiến mô hình dự đoán sai cảm xúc tiêu cực thành tích cực.

4. Khó khăn và Giải pháp (Challenges and Solutions)
Khó khăn: Cảnh báo về tham số multi_class trong LogisticRegression.


Chi tiết: FutureWarning: 'multi_class' was deprecated....

Giải pháp: Mặc định Scikit-learn sẽ tự động chọn (auto), nhưng để code tường minh và tránh cảnh báo, nên thiết lập rõ ràng multi_class='auto' hoặc ovr khi khởi tạo model.

Khó khăn: Xử lý dữ liệu văn bản thô.

Chi tiết: Dữ liệu có thể chứa giá trị Null hoặc tên cột không khớp.


Giải pháp: Thực hiện đổi tên cột chuẩn hóa (rename) và loại bỏ dòng thiếu dữ liệu bằng dropna(subset=["sentence", "label"]) trước khi đưa vào huấn luyện .

Khó khăn: Định dạng dữ liệu nhãn (Label).

Chi tiết: Nhãn trong file CSV có thể ở dạng float hoặc chuỗi.


Giải pháp: Ép kiểu về số nguyên astype(int) để đảm bảo tương thích với các hàm đánh giá của Scikit-learn.

5. Tài liệu tham khảo (References)
Scikit-learn Documentation: sklearn.linear_model.LogisticRegression, sklearn.feature_extraction.text.TfidfVectorizer, sklearn.naive_bayes.MultinomialNB.

PySpark Documentation: pyspark.ml.Pipeline, pyspark.ml.feature.HashingTF, IDF.

Pandas Documentation: pd.read_csv, dropna.