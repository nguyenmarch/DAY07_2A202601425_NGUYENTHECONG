# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store (K3)

**Nhóm:** Lmao

**Thành viên:** Nguyễn Tuấn Anh (2A202601669 — Trưởng nhóm), Nguyễn Thị Lý (2A202601962), Đỗ Hùng Anh (2A202601175), Nguyễn Thế Công (2A202601425)

**Ngày:** 2026-08-03

> Kết quả trong báo cáo được tái tạo bằng `scripts/evaluate_hust_retrieval.py`, FastEmbed 0.8.0, model ONNX `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Mock embedding không được dùng để kết luận chất lượng retrieval.

---

## 1. Lựa chọn tài liệu (10 điểm)

### Phạm vi

Nhóm tập trung vào **đăng ký học tập và thủ tục học vụ công khai của HUST**: kế hoạch đăng ký lớp theo học kỳ, học phần tương đương/thay thế, điều kiện đồ án tốt nghiệp, lớp đầy và học phần không mở. Một thông báo sau đại học được giữ làm distractor để kiểm tra metadata filtering theo `audience`.

### Data inventory

| # | Tài liệu | Nguồn chính thức | Ngày lấy / phiên bản | Ký tự nội dung | Metadata chính |
|---|---|---|---|---:|---|
| 1 | Kế hoạch đăng ký lớp 2026.1 | `ctt.hust.edu.vn`, kế hoạch 29240 | 2026-08-03 / 2026-07-20 | 6.353 | `student`, `academic-affairs`, `course-registration`, `2026.1` |
| 2 | Kế hoạch đăng ký lớp hè 2025.3 | `ctt.hust.edu.vn`, kế hoạch 29239 | 2026-08-03 / 2026-06-20 | 5.753 | `student`, `academic-affairs`, `course-registration`, `2025.3` |
| 3 | Đăng ký kế hoạch học tập 2025.2 | `ctt.hust.edu.vn`, bài viết 45580 | 2026-08-03 / `not-stated` | 2.957 | `student`, `academic-affairs`, `study-plan`, `2025.2` |
| 4 | Quy trình học phần tương đương/thay thế | `soict.hust.edu.vn` | 2026-08-03 / 2024-03-21 | 1.930 | `student`, `soict`, `equivalent-course` |
| 5 | Điều kiện đăng ký đồ án tốt nghiệp | `soict.hust.edu.vn` | 2026-08-03 / 2025-10-21 | 4.080 | `student`, `soict`, `graduation-project`, `2025.2` |
| 6 | Đăng ký bổ sung học phần Toán khi lớp đầy | `fami.hust.edu.vn` | 2026-08-03 / `not-stated` | 603 | `student`, `fami`, `full-class-registration`, `2025.2` |
| 7 | Hướng dẫn khi học phần không mở | `fami.hust.edu.vn` | 2026-08-03 / `not-stated` | 1.618 | `student`, `fami`, `unavailable-course` |
| 8 | Đăng ký học phần sau đại học 2026.1 | `sdh.hust.edu.vn` | 2026-08-03 / `not-stated` | 1.693 | `postgraduate`, `academic-affairs`, `course-registration`, `2026.1` |

Toàn bộ URL đầy đủ và căn cứ sử dụng nằm trong `data/hust_academic/sources.csv`. Corpus có 8 file Markdown, tổng 24.987 ký tự nội dung.

### Quản trị dữ liệu

- [x] Crawler chỉ truy cập 8 URL được chỉ định, kiểm tra `robots.txt` và chờ tối thiểu một giây giữa request.
- [x] Nguồn đều là trang công khai trên subdomain chính thức của HUST.
- [x] Không lấy trang yêu cầu đăng nhập, hồ sơ cá nhân hoặc dữ liệu nội bộ.
- [x] Tên/email cá nhân của cán bộ được loại khỏi bản làm sạch; chỉ giữ hướng dẫn liên hệ theo vai trò khi cần.
- [x] `sources.csv` khớp một-một với 8 `doc_id` và mỗi file đủ metadata bắt buộc.

### Metadata schema

| Trường | Kiểu | Ví dụ | Công dụng |
|---|---|---|---|
| `doc_id` | string | `hust-course-registration-2026-1` | Truy vết, cập nhật và xóa toàn bộ chunks của một tài liệu. |
| `source_url` | string | URL CTT HUST | Kiểm chứng tại nguồn chính thức. |
| `retrieved_at` | date string | `2026-08-03` | Biết thời điểm thu thập. |
| `document_version` | string | `2026-07-20` | Phân biệt phiên bản/thời điểm hiệu lực. |
| `audience` | enum string | `student`, `postgraduate` | Tránh trộn quy định đại học và sau đại học. |
| `department` | string | `academic-affairs`, `soict` | Lọc theo đơn vị phụ trách. |
| `category` | string | `course-registration` | Lọc theo loại thủ tục. |
| `academic_period` | string | `2026.1`, `2025.3` | Tránh nhầm giữa kỳ chính và kỳ hè. |
| `student_cohort` | string | `K70-or-earlier` | Chọn đúng thông báo theo khóa. |
| `language` | string | `vi` | Hỗ trợ corpus đa ngôn ngữ trong tương lai. |

---

## 2. Thiết kế chiến lược (15 điểm)

### Baseline trên ba tài liệu

Chạy `ChunkingStrategyComparator().compare(text, chunk_size=500)`:

| Tài liệu | Strategy | Chunk count | Avg. length | Nhận xét |
|---|---|---:|---:|---|
| Đăng ký lớp 2026.1 | Fixed | 13 | 488,7 | Dễ cắt giữa bảng/mục. |
| Đăng ký lớp 2026.1 | Sentence | 20 | 314,8 | Giữ câu tốt, nhưng dòng bảng có thể bị gom. |
| Đăng ký lớp 2026.1 | Recursive | 14 | 451,9 | Giữ đoạn tốt hơn fixed. |
| Học phần tương đương | Fixed | 4 | 482,5 | Có thể cắt giữa các bước. |
| Học phần tương đương | Sentence | 3 | 640,7 | Giữ câu nhưng có chunk vượt 500 ký tự. |
| Học phần tương đương | Recursive | 6 | 320,0 | Chia tương đối tốt theo bước/đoạn. |
| Lớp Toán đầy | Fixed | 2 | 301,5 | Thông báo ngắn bị chia đôi. |
| Lớp Toán đầy | Sentence | 1 | 601,0 | Giữ trọn thông báo. |
| Lớp Toán đầy | Recursive | 2 | 300,5 | Hai đoạn vẫn dễ đọc. |

### Chiến lược từng thành viên

- **Nguyễn Tuấn Anh:** `SentenceChunker(max_sentences_per_chunk=4)`. Mục tiêu là giữ câu điều kiện và mốc thời gian trọn vẹn.
- **Nguyễn Thị Lý:** `RecursiveChunker(chunk_size=500)`. Ưu tiên đoạn, dòng, câu rồi từ/ký tự.
- **Đỗ Hùng Anh:** `FixedSizeChunker(chunk_size=500, overlap=50)`. Đây là baseline đơn giản và có kích thước ổn định.
- **Nguyễn Thế Công:** `HeadingSectionChunker(chunk_size=500)`. Tách tại Markdown heading, mục số La Mã và tiểu mục đánh số; nếu section quá dài thì recursive split và lặp lại heading để giữ ngữ cảnh.

### Kết quả A/B retrieval

Cùng model FastEmbed, cùng 8 tài liệu, cùng 5 query và `top_k=3`. Một hit chỉ được tính khi top-3 chứa **đúng chunk có chuỗi bằng chứng**, không chỉ đúng `doc_id`.

| Thành viên | Strategy | Relevant chunk trong top-3 | Quy đổi coverage (/10) | Điểm mạnh | Điểm yếu |
|---|---|---:|---:|---|---|
| Nguyễn Tuấn Anh | Sentence | 5/5 | 10/10 | Giữ trọn câu, tốt nhất ngang heading. | Kích thước chunk không đồng đều. |
| Nguyễn Thị Lý | Recursive | 3/5 | 6/10 | Tôn trọng đoạn và giới hạn kích thước. | Một số bảng/mục bị tách khỏi heading. |
| Đỗ Hùng Anh | Fixed + overlap | 3/5 | 6/10 | Đơn giản, ổn định. | Cắt ranh giới section/bảng. |
| Nguyễn Thế Công | Heading/section | 5/5 | 10/10 | Giữ tên mục cùng nội dung; query mốc thời gian lên đúng section ở rank 1. | Regex heading cần điều chỉnh nếu format nguồn thay đổi. |

Sentence và heading cùng đạt 5/5 top-3 coverage. Nhóm chọn **HeadingSectionChunker** cho demo vì ngoài độ phủ, chunk còn giữ tên mục giúp người xem truy vết lý do kết quả liên quan.

---

## 3. Benchmark và chất lượng retrieval (10 điểm)

### Năm câu hỏi và gold answer

| # | Query | Gold answer | Nguồn bằng chứng |
|---|---|---|---|
| 1 | Các mốc bắt đầu và kết thúc của ba đợt đăng ký chính thức, đăng ký điều chỉnh và đăng ký thêm kỳ 2026.1 là khi nào? | Chính thức: 16h00 22/07–14h00 03/08/2026; điều chỉnh: 16h00 03/08–14h00 15/08/2026; đăng ký thêm: 16h00 15/08–16h00 22/08/2026. | `hust-course-registration-2026-1`, mục “Các đợt đăng ký lớp chính”. |
| 2 | Sinh viên có được rút học phần trong kỳ hè 2025.3 không? | Không; kỳ hè tự nguyện nên không có rút học phần. | `hust-summer-registration-2025-3`, mục hủy/mở/rút học phần. |
| 3 | Nếu học phần thay thế đã có trong danh sách được phê duyệt thì có cần làm đơn đăng ký không? | Không cần đơn đăng ký; sau khi học xong vẫn làm đơn công nhận học phần tương đương/thay thế. | `hust-equivalent-course-process`, Bước 1 và Bước 5. |
| 4 | Đăng ký bổ sung vào lớp Toán đã đầy bằng cách nào và nhóm nào được ưu tiên? | Điền “Đơn xin đăng ký vào lớp đầy” trên QLDT; ưu tiên K67 trở về trước. | `hust-full-math-class-registration`; lọc `category=full-class-registration`. |
| 5 | Kỳ 2026.1, sinh viên đại học bình thường chương trình chuẩn được đăng ký tối đa bao nhiêu TC? | Tối đa 24 TC, tối thiểu 12 TC. | `hust-course-registration-2026-1`, mục khối lượng tín chỉ; lọc `audience=student`. |

File máy đọc tương ứng: `data/hust_academic/benchmarks.json`.

### Kết quả HeadingSectionChunker

| # | Relevant rank | Score của relevant chunk | Top-3 có bằng chứng? | Câu trả lời grounded |
|---|---:|---:|---|---|
| 1 | 1 | 0,7742 | Có | Đủ ba khoảng thời gian theo gold answer. |
| 2 | 2 | 0,7673 | Có | Không được rút vì kỳ hè là tự nguyện. |
| 3 | 3 | 0,7199 | Có | Không cần đơn đăng ký nếu đã trong danh sách; vẫn cần bước công nhận. |
| 4 | 2 | 0,5758 | Có | Nộp biểu mẫu trên QLDT; ưu tiên K67 trở về trước. |
| 5 | 2 | 0,7112 | Có | Sinh viên bình thường chương trình chuẩn: 12–24 TC. |

Top-3 coverage đạt **5/5**. Theo rubric chặt, chỉ câu 1 có bằng chứng ở rank 1; bốn câu còn lại cần agent tổng hợp từ top-3. Nhóm không ghi điểm “ảo” cho một LLM bên ngoài: repo dùng `demo_llm`, còn độ đúng của answer được kiểm tra thủ công bằng gold answer và chunk nguồn.

### Tác dụng metadata filter

- Q4 lọc `category=full-class-registration`, loại các thông báo đăng ký lớp chung và giữ đúng tài liệu lớp Toán đầy.
- Q5 lọc `audience=student`, loại thông báo `audience=postgraduate` có mức tối đa 12 TC để tránh trả lời nhầm hệ đào tạo.

---

## 4. Failure analysis và demo (5 điểm)

### Failure case

Với FixedSizeChunker, Q1 và Q5 thất bại: chunk có câu trả lời không nằm top-3. Nguyên nhân là bảng tín chỉ và ba mốc đăng ký bị cắt tại biên ký tự, trong khi các chunk khác của cùng tài liệu lặp nhiều từ “đăng ký/thời gian/tín chỉ” và được xếp cao hơn. RecursiveChunker cũng chỉ đạt 3/5 vì không luôn giữ heading bên cạnh phần bảng.

HeadingSectionChunker cải thiện Q1: toàn bộ ba mốc nằm cùng chunk “1. Các đợt đăng ký lớp chính” và lên rank 1. Q5 có đúng chunk ở rank 2; hướng cải thiện tiếp theo là đưa `section_title` vào metadata hoặc áp dụng hybrid lexical+dense search cho số liệu/bảng.

### Nội dung demo

1. Chạy 42 unit tests.
2. Giới thiệu front matter và `sources.csv` của corpus HUST.
3. Chạy cùng 5 query với fixed và heading để thấy 3/5 → 5/5.
4. So sánh Q5 có/không có `audience=student`.
5. Mở chunk nguồn để kiểm chứng câu trả lời, không chỉ nhìn score.

### Bài học

Chunking là quyết định về đơn vị ngữ nghĩa, không chỉ chia đều ký tự. Metadata giúp loại nhiễu giữa các đối tượng/học kỳ, còn gold answer và chuỗi bằng chứng ngăn việc đánh giá “đúng tài liệu nhưng sai đoạn”. Nếu làm lại, nhóm sẽ chuẩn hóa heading ngay khi crawl và thêm hybrid BM25/dense retrieval cho câu hỏi chứa mã, ngày và số tín chỉ.

---

## Tự đánh giá

| Tiêu chí | Tự đánh giá |
|---|---:|
| Chất lượng bộ tài liệu | 10/10 |
| Thiết kế chiến lược | 15/15 |
| Chất lượng retrieval | 9/10 |
| Demo và bài học | 5/5 |
| **Tổng phần nhóm** | **39/40** |
