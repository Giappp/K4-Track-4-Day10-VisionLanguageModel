"""Phân tích bằng chứng hiện có, chạy trên CPU từ thư mục gốc repo.

Chạy: python report/analyze_current_data.py
Không suy luận model, không huấn luyện và không ghi đè evidence E0–E4.
Kiểm tra thang tọa độ dùng output tổng hợp từ nhãn GT, không phải output model.
"""

import ast
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont
from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import evaluate


def notebook_definitions():
    """Dùng đúng hàm chuyển nhãn/mask của notebook; bỏ qua ô chạy thí nghiệm."""
    import torch

    nb = json.loads((ROOT / "lab_day10.ipynb").read_text(encoding="utf-8"))
    names = {"yolo_to_box", "objects_to_answer", "make_prompt", "user_message", "encode_example"}
    namespace = {"json": json, "torch": torch, "CLASSES": CLASSES}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code" or any(s.lstrip().startswith("!") for s in cell["source"]):
            continue
        tree = ast.parse("".join(cell["source"]))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<notebook>", "exec"), namespace)
    assert names <= namespace.keys()
    return namespace


def architecture_counts():
    import torch
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    # Các kích thước có ảnh hưởng đến số tham số lấy từ config model chính thức:
    # https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/raw/main/config.json
    config = Qwen3VLConfig(
        text_config={"vocab_size": 151936, "hidden_size": 2560, "intermediate_size": 9728,
                     "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8,
                     "head_dim": 128, "tie_word_embeddings": True},
        vision_config={"depth": 24, "hidden_size": 1024, "intermediate_size": 4096,
                       "num_heads": 16, "out_hidden_size": 2560,
                       "deepstack_visual_indexes": [5, 11, 17], "num_position_embeddings": 2304},
        tie_word_embeddings=True,
    )
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)
    count = lambda module: sum(param.numel() for param in module.parameters())
    visual = model.model.visual
    merger, deepstack = count(visual.merger), count(visual.deepstack_merger_list)
    return {"method": "Đếm tham số trên kiến trúc meta; không tải trọng số pretrained/GPU.",
            "config_source": "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/raw/main/config.json",
            "total": count(model), "llm_with_embeddings": count(model.model.language_model),
            "visual_including_projector": count(visual), "merger": merger,
            "deepstack_mergers": deepstack, "projector_total": merger + deepstack,
            "visual_without_projector": count(visual) - merger - deepstack,
            "lm_head_tied_to_embeddings": model.lm_head.weight is model.model.language_model.embed_tokens.weight}


def mask_check(functions, sample):
    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(ROOT / "work/e1/processor", local_files_only=True)
    processor.tokenizer.padding_side = "left"
    with Image.open(ROOT / "data" / sample["image_path"]) as source:
        image = source.convert("RGB")
    answer = functions["objects_to_answer"](sample["objects"])
    prompt = functions["make_prompt"]()
    encoded = functions["encode_example"](image, prompt, answer, processor)
    supervised = encoded["labels"] != -100
    decoded = processor.tokenizer.decode(encoded["input_ids"][supervised], skip_special_tokens=False)
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_mask = encoded["input_ids"] == image_id
    assert decoded.strip() == answer + "<|im_end|>"
    assert (encoded["labels"][image_mask] == -100).all()
    assert int(image_mask.sum()) == 220
    labels_without_mask = encoded["input_ids"].clone()
    return {"method": "CPU, processor E1 đã lưu và encode_example hiện tại trong notebook.",
            "image_id": sample["image_id"], "image_size": list(image.size),
            "image_grid_thw": encoded["image_grid_thw"].tolist(),
            "image_tokens": int(image_mask.sum()), "total_tokens": encoded["input_ids"].numel(),
            "supervised_tokens": int(supervised.sum()),
            "masked_tokens": int((~supervised).sum()),
            "supervised_image_tokens": int((supervised & image_mask).sum()),
            "without_mask_supervised_tokens": int((labels_without_mask != -100).sum()),
            "without_mask_supervised_image_tokens": int((labels_without_mask[image_mask] != -100).sum()),
            "target": answer, "decoded_supervised_tokens": decoded,
            "prompt_is_prefix": True, "answer_and_end_token_verified": True,
            "limitation": "Không phải ablation huấn luyện có/không mask; không suy ra F1 của nhánh bỏ mask."}


def ground_truth_gallery(entries, destination):
    """Chọn sáu trường hợp nhãn thật; không dựng box dự đoán thay cho output bị thiếu."""
    def area(obj):
        return obj["yolo"][2] * obj["yolo"][3]

    selected = []
    used = set()
    for label in CLASSES:
        candidates = sorted(((area(obj), row["image_id"], row, obj)
                             for row in entries for obj in row["objects"] if obj["label"] == label),
                            key=lambda item: (item[0], item[1]))
        for target in (0, len(candidates) // 2):
            ordered = sorted(range(len(candidates)), key=lambda index: abs(index - target))
            index = next(i for i in ordered if candidates[i][1] not in used)
            box_area, image_id, row, obj = candidates[index]
            used.add(image_id)
            selected.append({"row": row, "focus_label": label, "focus_yolo": obj["yolo"],
                             "area_pct": box_area * 100,
                             "selection": "smallest_available" if target == 0 else "near_median"})

    colors = {"pothole": "#ff6961", "crack": "#ffd166", "manhole": "#53d8fb"}
    canvas = Image.new("RGB", (1304, 1320), "#142034")
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "TEST GROUND TRUTH ONLY | pothole=red, crack=yellow, manhole=cyan", fill="white", font=font)
    for index, item in enumerate(selected):
        row = item["row"]
        with Image.open(ROOT / "data" / row["image_path"]) as source:
            image = source.convert("RGB")
        painter = ImageDraw.Draw(image)
        for obj in row["objects"]:
            xc, yc, w, h = obj["yolo"]
            box = ((xc - w / 2) * image.width, (yc - h / 2) * image.height,
                   (xc + w / 2) * image.width, (yc + h / 2) * image.height)
            painter.rectangle(box, outline=colors[obj["label"]], width=2)
            painter.text((max(0, box[0]), max(0, box[1] - 12)), obj["label"], fill=colors[obj["label"]], font=font)
        x, y = 12 + index % 2 * 652, 42 + index // 2 * 422
        canvas.paste(image, (x, y))
        draw.text((x, y + 367), row["image_id"], fill="white", font=font)
        draw.text((x, y + 383), f"Focus: {item['focus_label']}; area={item['area_pct']:.4f}%; {item['selection']}",
                  fill="white", font=font)
        draw.text((x, y + 399), f"GT boxes: {len(row['objects'])}; session={row['session']}", fill="white", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return [{key: value for key, value in item.items() if key != "row"}
            | {"image_id": item["row"]["image_id"], "session": item["row"]["session"],
               "n_gt_boxes": len(item["row"]["objects"])} for item in selected]


def training_summary(path, score, n_train_images):
    run = json.loads(path.read_text())
    assert run["name"] == score["name"]
    config = run["config"]
    for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
        assert run[key] == score["meta"][key], (path, key)
    for key in ("setup", "seed", "train_fraction"):
        if key in score["meta"]:
            assert config[key] == score["meta"][key], (path, key)
    if "lora_r" in score["meta"]:
        assert config["lora"]["r"] == score["meta"]["lora_r"]
    steps = run.get("step", run.get("completed_steps"))
    assert steps == config["max_steps"]
    history = run["loss_history"]
    assert [row["step"] for row in history] == list(range(1, steps + 1))
    assert all(math.isfinite(row["loss"]) for row in history)
    if "n_train_images" in run:
        assert run["n_train_images"] == int(n_train_images * config.get("train_fraction", 1))
    matched = run.get("matched_modules")
    if matched is not None:
        expected = 348 if run["name"] == "E3" else 252
        assert len(matched) == len(set(matched)) == expected
        llm = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
        vit = r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"
        assert sum(bool(re.fullmatch(llm, name)) for name in matched) == 252
        assert sum(bool(re.fullmatch(vit, name)) for name in matched) == expected - 252
    summary = {"source": str(path.relative_to(ROOT)), "config": config,
               "completed_steps": steps, "n_train_images": run.get("n_train_images"),
               "matched_module_count": len(matched) if matched is not None else None,
               "first_loss": history[0]["loss"], "last_loss": history[-1]["loss"],
               "min_loss": min(row["loss"] for row in history),
               "max_loss": max(row["loss"] for row in history),
               "mean_first_20": statistics.mean(row["loss"] for row in history[:20]),
               "mean_last_20": statistics.mean(row["loss"] for row in history[-20:]),
               "all_losses_finite": True, "resource_metadata_matches_evidence": True}
    for key in ("lr", "grad_norm"):
        if all(key in row for row in history):
            assert all(math.isfinite(row[key]) for row in history)
            summary[key] = {"first": history[0][key], "last": history[-1][key],
                            "min": min(row[key] for row in history),
                            "max": max(row[key] for row in history)}
    return summary


def main():
    functions = notebook_definitions()
    splits = {name: read_jsonl(ROOT / "data/splits" / f"{name}.jsonl") for name in ("train", "val", "test")}
    meta = json.loads((ROOT / "data/splits/meta.json").read_text())
    stats, sessions = {}, {}
    for name, entries in splits.items():
        sessions[name] = {row["session"] for row in entries}
        counts = Counter(obj["label"] for row in entries for obj in row["objects"])
        areas = {label: [obj["yolo"][2] * obj["yolo"][3] * 100
                         for row in entries for obj in row["objects"] if obj["label"] == label]
                 for label in CLASSES}
        stats[name] = {"images": len(entries), "total_boxes": sum(counts.values()),
                       "boxes_per_class": dict(counts), "sessions": len(sessions[name]),
                       "median_box_area_pct": {label: statistics.median(values) for label, values in areas.items()},
                       "fraction_box_area_under_1pct": {label: sum(v < 1 for v in values) / len(values)
                                                        for label, values in areas.items()}}
        assert meta["counts"][name]["images"] == len(entries)
        assert meta["counts"][name]["boxes"] == dict(counts)
    overlaps = {f"{a}/{b}": sorted(sessions[a] & sessions[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    assert not any(overlaps.values())
    evidence_paths = {path.stem: path for path in sorted((ROOT / "report/evidence").glob("e[0-3].json"))}
    evidence = {name: json.loads(path.read_text()) for name, path in evidence_paths.items()}
    for name, score in evidence.items():
        assert score["name"] == name.upper() and score["score_version"] == 1
        assert score["dataset_version"] == meta["dataset_version"] and score["split"] == "test"
        assert score["n_images"] == len(splits["test"]) and score["missing_predictions"] == 0
        assert score["coord"] == "norm1000"
        for counts in [score["micro"], *score["per_class"].values()]:
            denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
            assert math.isclose(counts["f1"], 2 * counts["tp"] / denominator if denominator else 0)
        assert math.isclose(score["fp_per_image"], score["micro"]["fp"] / score["n_images"])
        if name != "e0":
            for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
                value = score["meta"][key]
                assert isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
    run_paths = {}
    training = {}
    for name in evidence:
        if name == "e0":
            continue
        candidates = [ROOT / f"work/{name}/run.json", ROOT / f"work/{name}/final/run.json"]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is not None:
            run_paths[name] = path
            training[name.upper()] = training_summary(path, evidence[name], len(splits["train"]))
    common_keys = ("model_id", "quantization", "lr", "batch_size", "grad_accum", "max_steps", "seed")
    common_training_config = {}
    for key in common_keys:
        values = [summary["config"][key] for summary in training.values()]
        assert all(value == values[0] for value in values), key
        common_training_config[key] = values[0]

    comparisons = {}
    for before, after in (("e0", "e1"), ("e1", "e2"), ("e2", "e3"), ("e1", "e3")):
        if before not in evidence or after not in evidence:
            continue
        a, b = evidence[before], evidence[after]
        difference = {"f1_delta": b["micro"]["f1"] - a["micro"]["f1"],
                      "f1_delta_percentage_points": 100 * (b["micro"]["f1"] - a["micro"]["f1"]),
                      "ci95_gap": b["f1_ci95"][0] - a["f1_ci95"][1],
                      "per_class_f1_delta_percentage_points": {
                          label: 100 * (b["per_class"][label]["f1"] - a["per_class"][label]["f1"])
                          for label in CLASSES}}
        for key in ("tp", "fp", "fn"):
            difference[f"{key}_delta"] = b["micro"][key] - a["micro"][key]
        for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
            if key in a["meta"] and key in b["meta"]:
                difference[f"{key}_delta"] = b["meta"][key] - a["meta"][key]
                difference[f"{key}_increase_pct"] = 100 * (b["meta"][key] / a["meta"][key] - 1)
        comparisons[f"{after}_vs_{before}"] = difference

    coordinate_check = {}
    for answer_coord in ("norm1000", "pixel"):
        predictions = {row["image_id"]: functions["objects_to_answer"](
            row["objects"], answer_coord, row["width"], row["height"])
            for row in splits["test"]}
        for score_coord in ("norm1000", "pixel"):
            result = evaluate(splits["test"], predictions, coord=score_coord)
            coordinate_check[f"answer_{answer_coord}__score_{score_coord}"] = result

    gallery = ground_truth_gallery(splits["test"], ROOT / "report/figures/test_ground_truth_examples.png")
    files = ["data/splits/meta.json", "data/splits/train.jsonl", "data/splits/val.jsonl",
             "data/splits/test.jsonl",
             "lab_day10.ipynb", "work/e1/processor/processor_config.json",
             "work/e1/processor/tokenizer_config.json", "work/e1/processor/tokenizer.json"]
    files.extend(str(path.relative_to(ROOT)) for path in evidence_paths.values())
    files.extend(str(path.relative_to(ROOT)) for path in run_paths.values())
    available = [name.upper() for name in evidence]
    if list((ROOT / "report/evidence").glob("e4*.json")):
        available.append("E4")
    analysis = {
        "dataset_version": meta["dataset_version"],
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
        "available_evidence": available,
        "missing_evidence": [name for name in ("E0", "E1", "E2", "E3", "E4") if name not in available],
        "raw_predictions_available": bool(list((ROOT / "work").rglob("*.jsonl"))),
        "splits": stats, "session_overlap": overlaps, "architecture": architecture_counts(),
        "evidence": evidence, "comparisons": comparisons,
        "best_setup_by_test_f1": max((name for name in evidence if name != "e0"),
                                     key=lambda name: evidence[name]["micro"]["f1"]).upper(),
        "standalone_training_log_available": {
            name.upper(): name in run_paths
            for name in evidence if name != "e0"},
        "training": training, "common_training_config": common_training_config,
        "mask_check": mask_check(functions, splits["train"][0]),
        "coordinate_check": {"method": "Output tổng hợp từ GT test, chuyển bằng hàm notebook, chấm với evaluate của đề.",
                             "limitation": "Không suy luận model, không phải E4, không đo lại E0/E1 trên test.",
                             "cases": coordinate_check},
        "ground_truth_gallery": gallery,
    }
    destination = ROOT / "report/analysis/current_data.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Đã ghi:", destination.relative_to(ROOT))
    print("Mask check:", {key: analysis["mask_check"][key] for key in ("total_tokens", "image_tokens", "supervised_tokens", "masked_tokens")})
    print("Coordinate check:", {key: value["micro"]["f1"] for key, value in coordinate_check.items()})


if __name__ == "__main__":
    main()
