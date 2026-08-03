# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** [Nguyễn Thế Công (MSSV: 2A202601425)]
**Nhóm:** [Lmao ]
**Ngày:** [03/08/2026 ]

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai văn bản có cosine similarity cao khi các vector embedding của chúng hướng gần giống nhau. Điều đó thường cho thấy chúng có chủ đề hoặc ý nghĩa ngữ nghĩa tương đồng, dù không nhất thiết dùng chính xác cùng từ ngữ.

**Ví dụ có độ tương tự CAO:**
- Câu A: Sinh viên phải đăng ký học phần trước thời hạn.
- Câu B: Người học cần hoàn tất việc chọn môn trước ngày đóng cổng đăng ký.
- Tại sao tương đồng: Hai câu dùng từ khác nhau nhưng cùng diễn đạt yêu cầu hoàn thành đăng ký môn đúng hạn.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Thư viện cho phép sinh viên gia hạn sách trực tuyến.
- Câu B: Mưa lớn khiến nhiều chuyến bay bị hoãn.
- Tại sao khác: Hai câu nói về hai chủ đề và hai loại sự kiện không liên quan.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine similarity so sánh hướng của hai vector nên ít bị ảnh hưởng bởi độ lớn của vector, tập trung tốt hơn vào quan hệ ngữ nghĩa. Khoảng cách Euclid còn phụ thuộc độ lớn và có thể coi hai vector cùng hướng nhưng khác độ dài là xa nhau; điều này thường không mong muốn khi so sánh text embedding.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Phép tính: `ceil((10,000 - 50) / (500 - 50)) = ceil(9,950 / 450) = ceil(22.11...)`.
> Đáp án: **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap bằng 100, số chunk là `ceil((10,000 - 100) / (500 - 100)) = ceil(24.75) = 25`, tức tăng từ 23 lên 25. Overlap lớn hơn giúp giữ lại ngữ cảnh nằm sát ranh giới giữa hai chunk, nhưng đổi lại làm tăng dữ liệu lưu trữ, số lần embedding và khả năng trả về nội dung trùng lặp.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi dùng regex `(?<=[.!?])(?:[ \t]+|\n+)` để tách tại khoảng trắng hoặc xuống dòng theo sau dấu kết thúc câu, đồng thời giữ dấu câu trong câu đứng trước. Các câu được strip, bỏ phần rỗng và gom theo `max_sentences_per_chunk`; văn bản rỗng hoặc chỉ có khoảng trắng trả về danh sách rỗng.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán thử lần lượt các separator từ ranh giới lớn đến nhỏ: đoạn, dòng, câu, từ rồi ký tự. Nếu đoạn hiện tại đã không vượt `chunk_size` thì trả về ngay; nếu separator không xuất hiện thì thử separator tiếp theo, còn phần vẫn quá dài được đưa vào lời gọi đệ quy. Khi hết separator hoặc gặp separator rỗng, hàm cắt trực tiếp theo số ký tự để luôn có fallback.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Mỗi `Document` được chuẩn hóa thành record gồm ID duy nhất, content, bản sao metadata và embedding. Store luôn giữ một bản in-memory ổn định cho lab, đồng thời đồng bộ sang ChromaDB nếu thư viện khả dụng. Khi search, query được embedding một lần, tính dot product với từng record, sắp xếp score giảm dần rồi lấy tối đa `top_k` kết quả.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` lọc record theo tất cả cặp key-value trong metadata trước khi tính similarity, nhờ đó kết quả chỉ được xếp hạng trong đúng nhóm đối tượng cần tìm. `delete_document` tìm và xóa tất cả record có `metadata["doc_id"]` trùng ID yêu cầu, đồng thời xóa các ID tương ứng khỏi ChromaDB nếu backend này đang hoạt động.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Agent lấy top-k chunks từ store, đánh số từng context và ghép chúng vào prompt cùng câu hỏi. Prompt yêu cầu LLM chỉ dùng context được cung cấp và nói không biết nếu thông tin không đủ, qua đó giảm khả năng tạo câu trả lời không có căn cứ. Nếu store không có kết quả, prompt cũng ghi rõ rằng không tìm thấy context liên quan.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
$ .venv/bin/python -m pytest tests/ -q
..........................................                               [100%]
42 passed in 0.06s
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | cao / thấp | | |
| 2 | | | cao / thấp | | |
| 3 | | | cao / thấp | | |
| 4 | | | cao / thấp | | |
| 5 | | | cao / thấp | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> *Viết 2-3 câu:*

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** __ / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | / 5 |
| Kết quả truy xuất của tôi (Competition Results) | / 10 |
| **Tổng phần cá nhân hiện đã hoàn thành** | **45 / 60** |
