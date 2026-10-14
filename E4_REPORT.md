# E4 — Đánh giá final checkpoint (rank)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút train | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E4 | E3; rank=4 | 118,935,552 | 11.5923 | 54.8010 | 0.4296 | [0.3910; 0.4679] | 0.9228 | 0.2831 | 0.3664 | 0.6667 |

Đánh giá checkpoint final E4 biến **rank**, baseline **E3**, dataset `d10-21e4070301`, split **test**. Notebook chỉ suy luận/chấm, không huấn luyện lại.

Cấu hình ghi trong run: seed 42, lr 0.0002, batch 2, grad_accum 4, 300 bước train. Các cấu hình ngoài biến E4 đã đối chiếu với run baseline.

Giá trị **4**: F1 **0.4296**, CI95 **[0.39095022624434395, 0.46792452830188674]**, precision **0.4533**, recall **0.4082**, TP/FP/FN **238/287/345**; parse_valid_rate **1.0000**, FP/ảnh **0.9228**.
Tài nguyên train từ run/evidence: **118,935,552** tham số, **11.5923 GB**, **54.8010 phút**. Chấm từ checkpoint E4 đã cung cấp.

**Chưa đủ hai giá trị theo GUIDE: còn thiếu 64.** Kết quả trên hoàn thiện cho checkpoint đã có; chưa thể kết luận so sánh ablation. Thêm final/evidence nhánh còn lại rồi chạy lại notebook; không tự train nhánh thiếu.

Test chuẩn có 311 ảnh, 3 nhóm session. CI bootstrap theo ảnh có thể đánh giá thấp phụ thuộc giữa frame, chưa phải CI ghép cặp của hiệu F1; mới có một seed. Không suy rộng một lần chạy thành bảo đảm cho phiên quay mới.

Sáu ảnh của **E4_RANK_4**, split **test**: xanh là GT, đỏ là prediction. TP/FP/FN theo scorer IoU ≥ 0,5, chọn từ JSONL thật theo số TP/FP/FN. Đây là kết quả E4.

![GT và prediction](figures/e4_rank_test_predictions.png)

