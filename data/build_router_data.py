import json
import random

BASE       = "/home/sclab/moe_model/processed_v2/"
DOMAIN_MAP = {"LAW": 0, "MED": 1, "SOC": 2, "STEM": 3}
SEED       = 42
CROSS_MULT = 5
random.seed(SEED)

# ── 소프트 레이블 생성 ───────────────────────────
def build_label(item, domain):
    label = [0.0] * 4

    if domain == "CROSS":
        crossed = [d for d in item.get("crossed_domains", []) if d in DOMAIN_MAP]
        if crossed:
            score = round(1.0 / len(crossed), 4)
            for d in crossed:
                label[DOMAIN_MAP[d]] = score
    else:
        label[DOMAIN_MAP[domain]] = 1.0

    return label

# ── 데이터 로드 및 변환 ──────────────────────────
def build_dataset():
    files = {
        "LAW":   "law.json",
        "MED":   "med.json",
        "SOC":   "soc.json",
        "STEM":  "stem.json",
        "CROSS": "cross_rewritten.json",
    }

    print("=" * 50)
    print("데이터 로드 및 소프트 레이블 변환")
    print("=" * 50)

    non_cross = []
    cross_data = []

    for domain, fname in files.items():
        with open(BASE + fname, encoding="utf-8") as f:
            raw = json.load(f)

        samples = []
        skipped = 0

        for item in raw:
            instruction = item.get("instruction", "").strip()
            if not instruction:
                skipped += 1
                continue

            label = build_label(item, domain)

            if domain == "CROSS" and sum(label) == 0.0:
                skipped += 1
                continue

            samples.append({
                "instruction": instruction,
                "label":       label,
                "domain":      domain
            })

        print(f"  {domain}: {len(samples)}개 (스킵: {skipped}개)")
        print(f"    레이블 예시: {samples[0]['label']}")

        if domain == "CROSS":
            cross_data = samples
        else:
            non_cross.extend(samples)

    # CROSS 오버샘플링
    cross_oversampled = cross_data * CROSS_MULT
    all_data = non_cross + cross_oversampled
    random.shuffle(all_data)

    print(f"\n단일 도메인: {len(non_cross)}개")
    print(f"CROSS: {len(cross_data)}개 × {CROSS_MULT} = {len(cross_oversampled)}개")
    print(f"총합:  {len(all_data)}개")

    # Train / Val / Test (8:1:1)
    n     = len(all_data)
    train = all_data[:int(n * 0.8)]
    val   = all_data[int(n * 0.8):int(n * 0.9)]
    test  = all_data[int(n * 0.9):]

    for split, name in [(train, "train"), (val, "val"), (test, "test")]:
        path = BASE + f"router_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        print(f"저장: {path} ({len(split)}개)")

    # 최종 샘플 확인
    print("\n=== 레이블 변환 확인 ===")
    for domain in ["LAW", "MED", "SOC", "STEM", "CROSS"]:
        sample = next(x for x in all_data if x["domain"] == domain)
        print(f"[{domain}] {sample['label']} | {sample['instruction'][:50]}...")

if __name__ == "__main__":
    build_dataset()