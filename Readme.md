Báo cáo Lab: Word Embedding với Gensim và Spark
1. Giải thích các bước thực hiện

  Sử dụng model pre-trained (Gensim)
    Tải model glove-wiki-gigaword-50 bằng Gensim.
    Lấy vector của từ đơn (get_vector), tính độ tương đồng giữa các từ (get_similarity) và tìm các từ đồng nghĩa (get_most_similar).

  Nhúng câu/văn bản
    Triển khai hàm embed_document() bằng cách lấy trung bình vector các từ trong câu.
    Ví dụ: "The queen rules the country." được nhúng thành một vector 50 chiều.

  Huấn luyện Word2Vec trên dữ liệu nhỏ (Gensim)
    Sử dụng tập en_ewt-ud-train.txt.
    Huấn luyện model với vector size = 100, window = 5, min_count = 2.
    Lưu và tải lại model đã huấn luyện.

  Huấn luyện Word2Vec trên dữ liệu lớn (Spark MLlib)
    Cài đặt Spark, đọc dữ liệu c4-train.00000-of-01024-30K.json.gz.
    Tiền xử lý: chuyển về chữ thường, loại bỏ ký tự không phải chữ cái, token hóa.
    Huấn luyện Word2Vec (vectorSize=100, minCount=5) và lưu model.

  Trực quan hóa embedding
    Lấy sample 1% vocab để tránh OOM.
    Giảm chiều từ 100D xuống 2D bằng PCA.
    Vẽ scatter plot với annotation các từ.

2. Hướng dẫn chạy code

Cài đặt các thư viện:
  pip install gensim numpy matplotlib pyspark scikit-learn

Tải model pre-trained và thử nghiệm vector, similarity, most similar.
Huấn luyện Word2Vec trên tập dữ liệu nhỏ (Gensim) và lưu lại model.
Chạy Spark để huấn luyện Word2Vec trên dữ liệu lớn.
Trực quan hóa embedding bằng PCA.

Kết quả sẽ được in ra console, và biểu đồ scatter plot hiển thị các cụm từ.

3. Phân tích kết quả
3.1. Độ tương đồng và từ đồng nghĩa

Vector for 'king':
[ 0.50451   0.68607  -0.59517  -0.022801  0.60046  -0.13498  -0.08813
  0.47377  -0.61798  -0.31012 ]

Similarity:
king vs queen: 0.7839043
king vs man: 0.53093773

Most similar to 'computer'(GloVe):
[('computers', 0.9165045022964478), ('software', 0.8814992904663086), ('technology', 0.852556049823761), ('electronic', 0.812586784362793), ('internet', 0.8060455322265625), ('computing', 0.802603542804718), ('devices', 0.8016185760498047), ('digital', 0.7991793751716614), ('applications', 0.7912740707397461), ('pc', 0.7883159518241882)]

Document embedding:
[ 0.06438001  0.43381    -0.779435    0.0075025   0.07915     0.20077899
 -0.2454325  -0.05369498 -0.00951262 -0.68774253]


Most similar words to 'computer' (Spark Word2Vec, dữ liệu lớn):

+---------+------------------+
|word     |similarity        |
+---------+------------------+
|desktop  |0.7051            |
|laptop   |0.6597            |
|uwowned  |0.6361            |
|computers|0.6353            |
|usb      |0.6087            |
+---------+------------------+


Nhận xét:
  Pre-trained GloVe cho kết quả tương đồng cao, phản ánh mối quan hệ ngữ nghĩa rõ ràng.
  Spark Word2Vec nhận dạng từ đồng nghĩa trong dữ liệu lớn, nhưng similarity thấp hơn, vì dữ liệu domain-specific khác với corpus pre-trained.

3.2. Trực quan hóa embedding
  PCA giúp giảm chiều từ 100D xuống 2D.
  Các cụm từ cùng chủ đề (ví dụ: 'king' – 'queen', 'computer' – 'laptop' – 'desktop') nằm gần nhau.
  Biểu đồ thể hiện mối quan hệ ngữ nghĩa giữa các từ, xác nhận model học được embedding hiệu quả.
  Các từ có mối quan hệ ngữ nghĩa gần gũi thường nằm gần nhau trên mặt phẳng 2D, cho thấy PCA đã bảo toàn được một phần cấu trúc lân cận từ không gian 1000D.
  
3.3. So sánh model pre-trained và model tự huấn luyện
  Tiêu chí                  	     (GloVe)	                   (Spark/Gensim)
  Độ phủ từ vựng              	  Rất lớn	                   Giới hạn theo corpus
  Từ đồng nghĩa chính xác	        Cao                 	Tốt với từ phổ biến trong dataset
  Ứng dụng domain-specific    	  Hạn chế                	Tốt nếu dữ liệu đúng domain
  Kích thước dữ liệu          	  Lớn (6B token)	       Phụ thuộc dữ liệu của người dùng
3.4. Khó khăn và giải pháp

Khó khăn:
  Dữ liệu lớn dễ bị OOM khi trực quan hóa.
  Một số từ OOV trong pre-trained model.
  Xử lý dữ liệu JSON nén tốn thời gian.

Giải pháp:
  Sample 1% vocab trước khi vẽ scatter plot.
  Thêm vector zeros cho từ OOV.
  Sử dụng Spark để xử lý song song dữ liệu lớn.

3.5. Trích dẫn tài liệu

  Model Pre-trained: GloVe: Global Vectors for Word Representation (Stanford, Jeffrey Pennington et al.)
  Thư viện: Gensim Documentation (Word2Vec, API load model)
  Công cụ: Apache Spark MLlib (Word2Vec, PCA)
  Thư viện: Scikit-learn (PCA)
