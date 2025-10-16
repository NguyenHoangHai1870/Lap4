# Báo cáo Lab: Word Embedding với Gensim và Spark

1.Giải thích các bước thực hiện

Sử dụng model pre-trained (Gensim)
   - Tải model `glove-wiki-gigaword-50` bằng Gensim.  
   - Lấy vector của từ đơn (`get_vector`), tính độ tương đồng (`get_similarity`) và tìm từ đồng nghĩa (`get_most_similar`).

Nhúng câu/văn bản 
   - Triển khai hàm `embed_document()` bằng cách lấy trung bình vector các từ trong câu.  
   - Ví dụ: `"The queen rules the country."` → vector 50 chiều.

Huấn luyện Word2Vec trên dữ liệu nhỏ (Gensim) 
   - Sử dụng tập `en_ewt-ud-train.txt`.  
   - Huấn luyện model với `vector_size=100`, `window=5`, `min_count=2`.  
   - Lưu và tải lại model.

Huấn luyện Word2Vec trên dữ liệu lớn (Spark MLlib)
   - Đọc dữ liệu `c4-train.00000-of-01024-30K.json.gz`.  
   - Tiền xử lý: chữ thường, loại bỏ ký tự không phải chữ cái, token hóa.  
   - Huấn luyện Word2Vec (`vectorSize=100`, `minCount=5`) và lưu model.

Trực quan hóa embedding  
   - Lấy sample 1% vocab để tránh OOM.  
   - Giảm chiều vector từ 100D xuống 2D bằng PCA.  
   - Vẽ scatter plot với annotation các từ.

2.Hướng dẫn chạy code

Cài đặt các thư viện:

pip install gensim numpy matplotlib pyspark scikit-learn

Chạy từng cell trong notebook Colab:

Tải model pre-trained, thử nghiệm vector, similarity, most similar.

Huấn luyện Word2Vec trên tập dữ liệu nhỏ.

Chạy Spark để huấn luyện Word2Vec trên dữ liệu lớn.

Trực quan hóa embedding bằng PCA.

Kết quả sẽ được in ra console và biểu đồ scatter plot hiển thị các cụm từ.


3. Phân tích kết quả


3.1. Độ tương đồng và từ đồng nghĩa

Vector for 'king' (10 phần tử đầu):

[ 0.50451 0.68607 -0.59517 -0.022801 0.60046 -0.13498 -0.08813 0.47377 -0.61798 -0.31012 ]

Similarity:

king vs queen: 0.7839

king vs man: 0.5309

Most similar to 'computer' (GloVe pre-trained):

[('computers', 0.9165), ('software', 0.8815), ('technology', 0.8526),
 ('electronic', 0.8126), ('internet', 0.8060), ('computing', 0.8026),
 ('devices', 0.8016), ('digital', 0.7992), ('applications', 0.7913), ('pc', 0.7883)]

Most similar to 'computer' (Spark Word2Vec, dữ liệu lớn):

Word	       Similarity
desktop      	0.7051
laptop      	0.6597
uwowned      	0.6361
computers   	0.6353
usb	         0.6087

Nhận xét:

   Pre-trained GloVe cho độ tương đồng cao, phản ánh mối quan hệ ngữ nghĩa rõ ràng.

   Spark Word2Vec nhận dạng từ đồng nghĩa trong dữ liệu lớn, nhưng similarity thấp hơn, do corpus domain-specific khác với pre-trained corpus.

3.2. Trực quan hóa embedding

   PCA giảm chiều từ 100D → 2D.

Các cụm từ cùng chủ đề nằm gần nhau:
   'king' – 'queen'

   'computer' – 'laptop' – 'desktop'

   Biểu đồ scatter plot thể hiện mối quan hệ ngữ nghĩa giữa các từ, xác nhận model học được embedding hiệu quả.

3.3. So sánh model pre-trained và self-trained

Tiêu chí	                  (GloVe)	                 (Spark/Gensim)
Độ phủ từ vựng	            Rất lớn	               Giới hạn theo corpus
Từ đồng nghĩa chính xác   	Cao	                  Tốt với từ phổ biến
Ứng dụng domain-specific	Hạn chế	               Tốt nếu dữ liệu đúng domain
Kích thước dữ liệu	      Lớn (6B token)	         Phụ thuộc dữ liệu

4. Khó khăn và giải pháp

Khó khăn:

   Dữ liệu lớn dễ bị OOM khi trực quan hóa.

   Một số từ OOV trong pre-trained model.

   Xử lý JSON nén tốn thời gian.

Giải pháp:

   Sample 1% vocab trước khi vẽ scatter plot.

   Thêm vector zeros cho từ OOV.

   Sử dụng Spark để xử lý song song dữ liệu lớn.

5. Trích dẫn tài liệu
   Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for Word Representation. EMNLP.

   Gensim Documentation: https://radimrehurek.com/gensim/

   Apache Spark MLlib: https://spark.apache.org/docs/latest/ml-features.html#word2vec

Scikit-learn PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
