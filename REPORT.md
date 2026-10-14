Họ tên: Nguyễn Văn Giáp
`dataset_version`: `d10-21e4070301`

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

Số E0/E1 lấy trực tiếp từ [e0.json](report/evidence/e0.json) và [e1.json](report/evidence/e1.json), cùng test 311 ảnh, không thiếu dự đoán. Log [work/e1/run.json](work/e1/run.json) khớp cả ba trường tài nguyên E1. Chưa có evidence E2/E3/E4; dấu “—” biểu thị chưa có số đo. E0 không train, chưa có log VRAM hoặc thời gian suy luận.

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút train | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | 0 | 0,0601 | [0,0397; 0,0837] | 1,3408 | 0,1375 | 0,0034 | 0,1343 |
| E1 | LoRA LLM | 33.030.144 | 8,5783 | 45,4397 | 0,2563 | [0,2201; 0,2936] | 1,3408 | 0,1836 | 0,1813 | 0,5156 |
| E2 | + projector | — | — | — | — | — | — | — | — | — |
| E3 | + LoRA ViT | — | — | — | — | — | — | — | — | — |
| E4 | Chưa có ablation | — | — | — | — | — | — | — | — | — |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

Đếm trên kiến trúc Qwen3-VL-4B ở thiết bị `meta`, dùng các kích thước trong [config model chính thức](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/raw/main/config.json), cho kết quả:

| Khối | Số tham số |
|---|---|
| LLM, gồm embedding | 4.022.468.096 |
| ViT, gồm projector | 415.347.712 |
| ViT ngoài projector | 306.242.560 |
| `visual.merger` | 27.271.680 |
| `visual.deepstack_merger_list` | 81.833.472 |
| Projector, tổng hai nhóm trên | 109.105.152 |
| Tổng model | 4.437.815.808 |

Projector nằm trong ViT nên không cộng thêm vào tổng ViT. `lm_head` chia sẻ trọng số với embedding, không đếm hai lần. Đây là số tham số kiến trúc, không phải dung lượng trọng số 4-bit hay số tham số đã train.

- **E1:** cập nhật LoRA của attention/MLP LLM; đóng băng trọng số gốc LLM, ViT và projector. Run thực tế có **33.030.144** tham số trainable.
- **E2:** như E1, mở thêm toàn bộ projector fp32. Với cùng rank/targets, trainable dự kiến là **142.135.296 = 33.030.144 + 109.105.152**; đây là phép tính theo setup, chưa phải số từ run E2.
- **E3:** như E2, thêm LoRA trên attention/MLP các block ViT; trọng số gốc các block ViT vẫn đóng băng. LLM có 36 × 7 = **252** module LoRA; E3 thêm 24 × 4 = **96**, tổng **348**. Chưa có số trainable thực nghiệm E3.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

E0 có `parse_valid_rate = 0,6655948553`, tức **66,56% = 207/311** ảnh được parser chấp nhận; **104/311** không được chấp nhận. `missing_predictions = 0`, nên output không hợp lệ không phải do thiếu dòng dự đoán.

Hai nhóm vấn đề thấy được ở mức tổng hợp:

1. **Output không đáp ứng parser:** 104 ảnh. Parser kiểm tra cả cấu trúc và box hợp lệ, nên chưa thể quy tất cả thành lỗi cú pháp JSON.
2. **Nhận diện/định vị chưa đạt IoU 0,5:** **31 TP, 417 FP, 552 FN**, F1 **0,0601**. FN còn bao gồm GT trên ảnh có output không hợp lệ, nên hai nhóm này không độc lập để cộng lại.

Crack đặc biệt yếu: **1 TP, 288 FP, 298 FN**, F1 **0,0034**. FP chưa chứng minh model bịa box: box lệch, sai lớp hoặc box dư đều có thể tạo FP.

**Phần chưa đủ bằng chứng:** không có `work/e0.jsonl` hay output thô trong notebook. Chưa thể xếp hạng hai lỗi cụ thể trong sai thang/bịa box/sai lớp/JSON hỏng hoặc trích một output thật. Cần lấy lại JSONL từ phiên đã chạy; không dựng ví dụ thay cho output model.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

Đã kiểm chứng trên **311 ảnh test, 583 box GT**: chuyển nhãn thật thành JSON bằng `objects_to_answer()` của notebook, rồi gọi `evaluate()` trong `scripts/score.py` với hai lựa chọn `coord`.

| Thang output tổng hợp từ GT | Thang chấm | F1@IoU0,5 |
|---|---|---|
| norm1000 | norm1000 | **1,0000** |
| norm1000 | pixel | **0,0000** |
| pixel | norm1000 | **0,0000** |
| pixel | pixel | **1,0000** |

Ảnh là 640×360: `x_px = x_norm × 640/1000`, `y_px = y_norm × 360/1000`. Coi trực tiếp tọa độ norm1000 là pixel phóng tọa độ x khoảng **1,5625 lần**, y khoảng **2,7778 lần** so với đúng vị trí. Chiều ngược lại thu nhỏ x còn **0,64 lần**, y còn **0,36 lần**. JSON vẫn có thể hợp lệ nhưng box không đạt IoU yêu cầu.

Chỉ sai cách diễn giải tọa độ đã làm F1 từ 1 xuống 0 trong kiểm tra này. **Đây là output tổng hợp từ GT, không phải F1 mới của E0/E1 hoặc E4 coord_scale.** Thiếu JSONL dự đoán nên chưa đo được mức giảm riêng của hai model. Kiểm tra này không chạy lại model hoặc chọn checkpoint bằng test.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

Mục tiêu là học `P(JSON đáp án | ảnh, prompt)`. Prompt và token ảnh được cung cấp làm điều kiện, nên đặt `labels = -100` ở đó; chỉ đáp án và token kết thúc được tính loss. Mask loss vẫn cho phép gradient từ đáp án truyền qua projector/LoRA ViT khi setup mở những khối này.

Kiểm tra ảnh train **`20250216_164325`**, 640×360, với processor E1 đã lưu và `encode_example()` hiện tại:

- Chuỗi đầy đủ có **599 token**, gồm **220 token ảnh**. `image_grid_thw = [1,22,40]`; sau merge 2×2 có `22 × 40 / 4 = 220` token.
- **349 token** prompt/ngữ cảnh được mask; **250 token** được tính loss; **0 token ảnh** được tính loss.
- Prompt là tiền tố của full chat. Giải mã `labels != -100` thu đúng JSON **9 box**, rồi `<|im_end|>` và xuống dòng.

Nếu bỏ mask trong ví dụ này, cả **599 token** đều bị giám sát, gồm **220 token ảnh**. Loss còn tối ưu việc dự đoán lại phần điều kiện và placeholder ảnh, thay đổi mục tiêu và tăng trọng số cho phần không phải đáp án. Loss tổng có thể giảm mà định vị chưa cải thiện tương ứng. Chưa chạy ablation train bỏ mask nên không gán một mức giảm F1 cụ thể.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

E2 cần cập nhật trực tiếp **109.105.152** tham số projector. Nếu các linear của projector thành `Linear4bit` với `Params4bit`, trọng số là dạng nén dùng cho QLoRA; đường train này giữ trọng số gốc cố định và cập nhật adapter, không full fine-tune trọng số nén bằng AdamW thông thường.

Lỗi gặp trong lab là skip `"visual"` trong khi module thật bắt đầu bằng `"model.visual"`. Với Transformers 5.19, prefix ngắn này không bảo vệ các lớp con. Đã sửa hàm nạp model thành:

```python
llm_int8_skip_modules=["model.visual", "lm_head"]
```

`verify_base_precision()` kiểm tra cả lớp `Linear4bit`/`Linear8bitLt`, kiểu `Params4bit`/`Int8Params` và dtype của các trọng số được bảo vệ trước khi train. Dtype đơn thuần chưa đủ vì `Params4bit` có thể là float trước khi pack. Phải nạp lại model; `.float()` trên trọng số nén không khôi phục trọng số pretrained fp32 gốc. Đã đối chiếu [mã so khớp của Transformers 5.19](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/quantizers/quantizers_utils.py#L24).

Vòng train của lab giữ projector trainable ở **fp32** để lưu/cập nhật tham số và gradient ổn định hơn fp16, đồng thời tương thích với bước unscale của GradScaler. Forward vẫn autocast fp16. Đây là lựa chọn cho vòng train hiện tại, không có nghĩa mọi phương pháp huấn luyện đều bắt buộc dùng fp32 cho projector. ViT ngoài projector và `lm_head` giữ fp16; LLM gốc dùng NF4 4-bit.

Kiểm thử trước đó trên kiến trúc `meta` xác nhận E2 khớp **252 module LoRA**, mở đúng **109.105.152** tham số projector fp32. Đây là kiểm tra setup, chưa phải run/F1 E2. Projector được lưu riêng trong `extra_trainable.pt` và nạp lại cùng adapter khi suy luận.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

Frame gần nhau trong video có cảnh nền và đối tượng gần giống nhau. Chia ngẫu nhiên theo ảnh dễ đưa các frame gần trùng vào train lẫn test, làm F1 có thể cao hơn khả năng tổng quát sang phiên quay mới. Chia nguyên nhóm session giúp giảm rò rỉ này.

Đếm trực tiếp các JSONL và đối chiếu `data/splits/meta.json`:

| Split | Ảnh | Nhóm session | pothole | crack | manhole | Tổng box |
|---|---|---|---|---|---|---|
| train | **1.422** | **13** | **1.074** | **1.750** | **691** | **3.515** |
| val | **276** | **2** | **66** | **470** | **103** | **639** |
| test | **311** | **3** | **121** | **299** | **163** | **583** |
| Tổng | **2.009** | **18** | **1.261** | **2.519** | **957** | **4.737** |

Giao nhau session ở train/val, train/test và val/test đều **rỗng**. Đây là các nhóm trong trường `session`, gồm cả nhóm `num-*`. Val chỉ có **66 box pothole** so với **470 crack**, nên cần đọc F1 từng lớp. Chưa chạy random split để đo mức tăng F1; giải thích rò rỉ dựa trên cấu trúc dữ liệu, không phải phép so sánh random/session đã chạy.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?

Hiện chỉ so sánh được E0/E1: F1 tăng **0,1962**, tức **19,62 điểm phần trăm**, từ **0,0601** lên **0,2563**. Parse rate tăng **66,56% → 100%**; TP tăng **31 → 147**, FN giảm **552 → 436**. FP vẫn **417**, FP/ảnh vẫn **1,3408**: E1 cải thiện recall và output hợp lệ nhưng chưa giảm tổng FP.

E1 có **33.030.144** tham số trainable, VRAM đỉnh **8,5783 GB**, train **45,4397 phút** cho **300 bước tối ưu**. Cả 300 loss hữu hạn; bước đầu **1,2049**, bước cuối **0,5528**. Trung bình 20 bước đầu **0,8481**, 20 bước cuối **0,5682**. Loss có xu hướng giảm nhưng dao động, không giảm đơn điệu; loss train không chứng minh tổng quát hóa.

**Chưa kết luận được lợi ích mở projector/LoRA ViT:** cần evidence E2/E3 với cùng split, lịch train và phạm vi lượng tử hóa, rồi so F1/CI95 với VRAM/thời gian. E1 là setup train duy nhất có kết quả hiện tại, chưa phải setup tốt nhất trong E1–E3.

Hạn chế về nguồn: evidence E0/E1 và run E1 chưa ghi danh sách skip hoặc kiểm tra dtype ViT; notebook hiện cũng không có output thực thi. Chưa xác nhận các lượt đã lưu dùng bản sửa `model.visual` hay chưa. Cần xác nhận điều này trước khi so sánh với E2/E3 dùng cấu hình mới.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

| Setup | F1 | CI95 | Độ rộng CI | Độ rộng theo điểm phần trăm |
|---|---|---|---|---|
| E0 | **0,0601** | **[0,0397; 0,0837]** | **0,0440** | **4,40** |
| E1 | **0,2563** | **[0,2201; 0,2936]** | **0,0735** | **7,35** |

Hai CI không chồng nhau: cận dưới E1 **0,2201** cao hơn cận trên E0 **0,0837**. Mức cải thiện trong evidence hiện có lớn so với độ rộng CI, nhưng vẫn cần giữ hạn chế về cấu hình lượng tử hóa ở câu 7.

`score.py` bootstrap **1.000 lần theo ảnh**, cộng TP/FP/FN rồi tính micro-F1. Test chỉ có **3 nhóm session**, và frame cùng session có thể tương quan. Vì vậy 311 ảnh không tương đương 311 cảnh độc lập; CI theo ảnh có thể đánh giá thấp bất định khi tổng quát sang phiên quay mới.

Không kết luận setup hơn nhau chỉ vì F1 chênh vài điểm phần trăm. CI riêng từng setup chưa phải CI của hiệu F1; vì dùng cùng test, cần bootstrap ghép cặp theo cùng ảnh hoặc phân tích theo session để đánh giá chênh lệch trực tiếp. Thiếu output/TP-FP-FN từng ảnh nên chưa làm được phân tích đó; cũng chưa có CI E2/E3/E4.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

E0 yếu nhất ở **crack**, F1 **0,0034**. Với E1, crack thấp nhất về điểm ước lượng (**0,1813**), nhưng gần pothole (**0,1836**); chênh **0,0023** chưa đủ cơ sở kết luận crack khó hơn pothole một cách chắc chắn. Manhole đạt **0,5156**.

| Lớp | Box train | Box test | TP/FP/FN E1 | F1 E1 | Trung vị diện tích box test | Tỷ lệ box test dưới 1% ảnh |
|---|---|---|---|---|---|---|
| pothole | **1.074** | **121** | **19 / 67 / 102** | **0,1836** | **0,4193%** | **77,69%** |
| crack | **1.750** | **299** | **62 / 323 / 237** | **0,1813** | **1,9813%** | **22,74%** |
| manhole | **691** | **163** | **66 / 27 / 97** | **0,5156** | **0,6688%** | **71,17%** |

Pothole có nhiều box nhỏ, trung vị **0,4193%** ảnh, nên sai vài pixel dễ giảm IoU. Crack có nhiều mẫu train nhất và box test lớn hơn hai lớp khác: dữ liệu không ủng hộ giải thích crack kém vì ít mẫu hoặc box nhỏ nhất. Ảnh GT cho thấy crack có thể là vết nứt mảnh hoặc vùng gồm nhiều đường nứt; khoanh vùng và phân biệt với texture mặt đường là giả thuyết cần kiểm tra. Manhole thường có biên hình học rõ hơn, nhưng thiếu output từng ảnh để chứng minh đó là nguyên nhân model làm tốt hơn.

Hình dưới lấy **6 ảnh test thật**, chọn box nhỏ và gần trung vị cho mỗi lớp. Đỏ: pothole; vàng: crack; xanh: manhole. Tên ảnh, session và diện tích được ghi trong hình và JSON phân tích.

![Sáu ảnh test với nhãn ground truth](report/figures/test_ground_truth_examples.png)

**Đây là box GT, không phải box dự đoán.** Hình minh họa độ khó dữ liệu, chưa đáp ứng yêu cầu ảnh dự đoán đúng/sai có prediction và GT trong RUBRIC. Cần lấy lại `work/e1.jsonl` để chọn FP/FN/TP và kiểm tra các giả thuyết trên.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

Chưa có `e4*.json` hoặc checkpoint E4, nên chưa có kết quả ablation. Notebook mặc định chọn **rank**: hai nhánh **r=4, alpha=8** và **r=64, alpha=128**. Đây là cấu hình dự kiến, không chứng minh đã chạy.

Để hoàn thiện cần chốt baseline sau E1/E2/E3, giữ dữ liệu, seed, lịch train và phạm vi lượng tử hóa, rồi chạy hai giá trị của đúng một biến. Báo cáo F1, CI95 và tài nguyên cả hai nhánh. Với data_size/coord_scale, có thể tái dùng nhánh tương đương baseline nhưng phải ghi rõ nguồn. Kiểm tra GT ở câu 3 không thay thế E4 vì không huấn luyện theo hai điều kiện nhãn/prompt.

## Nhìn lại

Lỗi đã gặp là ViT/projector bị lượng tử hóa do tên skip không khớp; E2 cần cập nhật toàn bộ projector nên lỗi lộ rõ khi setup/train. Cần kiểm tra lớp trọng số sau khi tải, lưu riêng projector và nạp lại cùng adapter.

E1 cải thiện JSON hợp lệ và TP nhưng FP vẫn **417**: đúng định dạng chưa đồng nghĩa định vị tốt. Thống kê lớp cho thấy không nên quy mọi lỗi cho ít mẫu hoặc box nhỏ.

Scorer tổng hợp chưa giải thích vì sao từng box sai, chưa xác nhận dtype/skip của run, và bootstrap theo ảnh chưa thể hiện đầy đủ phụ thuộc giữa frame cùng session. Cần log và output từng ảnh để kiểm tra.

Phần còn thiếu để nộp đầy đủ: họ tên; xác nhận cấu hình E0/E1 đã lưu; output thô E0/E1; evidence/log E2/E3; hai nhánh E4; ảnh có GT lẫn prediction. Không dùng số dự kiến hoặc thí nghiệm tổng hợp thay cho kết quả chạy thật.
